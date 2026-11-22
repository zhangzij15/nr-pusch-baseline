"""Draw measured BLER-versus-SNR curves for fixed NR PUSCH MCS values."""

from __future__ import annotations

from collections import defaultdict
from math import ceil, expm1, isfinite, log, nan
from pathlib import Path


# CP-OFDM MCS table modulation groups in TS 38.214 Tables 5.1.3.1-1 to -3.
_MCS_GROUP_ENDS = {
    1: ((9, 2), (16, 4), (28, 6)),
    2: ((4, 2), (10, 4), (19, 6), (27, 8)),
    3: ((14, 2), (20, 4), (28, 6)),
}
_MODULATION = {
    1: ("BPSK", ("#4B5563", "#14213D")),
    2: ("QPSK", ("#0072B2", "#034875", "#376BC0", "#1978A6")),
    4: ("16-QAM", ("#D55E00", "#A33F00", "#AD6800", "#C9432F")),
    6: ("64-QAM", ("#008267", "#3C7541", "#156054", "#5E7308")),
    8: ("256-QAM", ("#9B4D81", "#743A87", "#8E3759", "#694698")),
    10: ("1024-QAM", ("#495999", "#27336A")),
}
_MARKERS = ("o", "s", "^", "D", "p", "X", "P", "*", "<", ">", "d", "H", "h", "8", "x")
_LINE_STYLES = ("-", "--", "-.", ":")


def _group_rows(rows, mcs_table):
    curves = defaultdict(list)
    modulation_bits = {}
    measured_pairs = set()
    for row in rows:
        mcs, snr = int(row["mcs_index"]), float(row["snr_db"])
        num_tb, errors = int(row["num_tb"]), int(row["tb_errors"])
        if not isfinite(snr) or num_tb < 1 or not 0 <= errors <= num_tb:
            raise ValueError("Waterfall points require finite SNR and valid transport block counts")
        if (mcs, snr) in measured_pairs:
            raise ValueError(f"Duplicate waterfall point for MCS {mcs} at {snr:g} dB")
        measured_pairs.add((mcs, snr))
        bits = row.get("qam_bits_per_symbol")
        if bits is None or bits == "":
            bits = next((bits for last, bits in _MCS_GROUP_ENDS.get(mcs_table, ()) if 0 <= mcs <= last), None)
        if bits is None or int(bits) not in _MODULATION:
            raise ValueError(f"Missing or unsupported modulation order for MCS {mcs}")
        if modulation_bits.setdefault(mcs, int(bits)) != int(bits):
            raise ValueError(f"Inconsistent modulation order for MCS {mcs}")
        curves[mcs].append((snr, num_tb, errors))
    if not curves:
        raise ValueError("At least one measured MCS point is required for a waterfall figure")
    groups = defaultdict(list)
    for mcs in sorted(curves):
        groups[modulation_bits[mcs]].append((mcs, sorted(curves[mcs])))
    return sorted(groups.items())


def _waterfall_figure(rows, mcs_table, *, subtitle: str | None = None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterMathtext, MaxNLocator

    groups = _group_rows(rows, mcs_table)
    figure = Figure(figsize=(11, 6.2), dpi=240, layout="constrained")
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(1, 2, width_ratios=(4.7, 1.25))
    axes = figure.add_subplot(layout[0, 0])
    legend_layout = layout[0, 1].subgridspec(
        len(groups), 1, height_ratios=[1 + ceil(len(curves) / 2) for _, curves in groups],
    )

    target = axes.axhline(0.1, color="#526173", linestyle="--", linewidth=1, label="10% BLER", zorder=1)
    has_zero_errors = False
    for group_index, (bits, curves) in enumerate(groups):
        modulation, colors = _MODULATION[bits]
        handles = []
        for order, (mcs, points) in enumerate(curves):
            color = colors[order % len(colors)]
            line, = axes.plot(
                [snr for snr, _, _ in points],
                [errors / count if errors else nan for _, count, errors in points],
                label=f"MCS {mcs}", color=color, linewidth=1.45,
                linestyle=_LINE_STYLES[(order // len(colors)) % len(_LINE_STYLES)],
                marker=_MARKERS[order % len(_MARKERS)], markersize=3.8,
                markerfacecolor="white", markeredgewidth=0.95, zorder=3,
            )
            handles.append(line)
            zero_points = [(snr, -expm1(log(0.05) / count)) for snr, count, errors in points if errors == 0]
            if zero_points:
                snr, upper = zero_points[0]
                axes.plot(
                    [snr], [upper], marker="v", linestyle="none", color=color,
                    markersize=4.5, markerfacecolor="none", markeredgewidth=1, zorder=4,
                )
                has_zero_errors = True
        legend_axes = figure.add_subplot(legend_layout[group_index, 0])
        legend_axes.set_axis_off()
        legend_axes.legend(
            handles=handles, title=modulation, loc="upper left", mode="expand",
            frameon=False, ncols=2, borderaxespad=0.1, columnspacing=0.7,
            title_fontsize=10.5, fontsize=9.0, handlelength=2.35, handletextpad=0.55,
        )

    guide_handles = [target]
    if has_zero_errors:
        guide_handles.append(Line2D(
            [], [], marker="v", linestyle="none", color="#526173", markerfacecolor="none",
            label="First zero-error point (nominal 95% binomial upper bound)",
        ))
    axes.legend(handles=guide_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncols=len(guide_handles), fontsize=8.5, frameon=False)
    axes.set_yscale("log")
    axes.set_ylim(bottom=min(axes.get_ylim()[0], 0.01), top=1.12)
    axes.set_axisbelow(True)
    axes.set_xlabel("Reference SNR (dB)", fontsize=11)
    axes.set_ylabel("Transport block error rate (BLER)", fontsize=11)
    sample_counts = {count for _, curves in groups for _, points in curves for _, count, _ in points}
    if subtitle:
        axes.set_title(subtitle, loc="left", fontsize=10, color="#526173")
    elif len(sample_counts) == 1:
        axes.set_title(f"{next(iter(sample_counts)):,} TBs per measured point",
                       loc="left", fontsize=10, color="#526173")
    axes.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    axes.yaxis.set_major_formatter(LogFormatterMathtext())
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(which="major", color="#DCE2E8", linewidth=0.75)
    axes.grid(which="minor", axis="y", color="#EBEFF3", linewidth=0.5)
    figure.suptitle(f"NR PUSCH BLER waterfall · MCS table {mcs_table}", fontsize=15, fontweight="semibold")
    return figure


def write_mcs_waterfall_figure(rows, output_path: str | Path, *, mcs_table: int = 1,
                               subtitle: str | None = None) -> Path:
    """Save one measured BLER curve per MCS and mark zero-error bounds separately."""
    output_path = Path(output_path)
    _waterfall_figure(rows, mcs_table, subtitle=subtitle).savefig(output_path)
    return output_path
