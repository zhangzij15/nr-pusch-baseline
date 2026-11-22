"""Store paired codebook sweeps and plot their measured throughput and BLER."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from .outputs import (_observed_throughput_field, _plot_bler_targets, _plot_throughput_annotations,
                      _throughput_legend, write_csv)
from .throughput_statistics import paired_throughput_differences


# Nested open markers stay distinguishable when measurements overlap.
_CURVES = {
    "full": ("Full NR codebook", "#0072B2", "-", "o", 9, 2.6),
    "partial_noncoherent": ("Partial + non-coherent", "#D55E00", "--", "s", 6.5, 1.9),
    "noncoherent": ("Non-coherent only", "#008267", ":", "D", 4, 1.7),
}

_THROUGHPUT_LABELS = {
    "thp_wall_clock_mbps": "Throughput over all TDD slots (Mbit/s)",
    "thp_including_srs_mbps": "Throughput over SRS and PUSCH slots (Mbit/s)",
    "thp_duty_cycle_mbps": "Duty-cycle-adjusted throughput (Mbit/s)",
    "thp_scheduled_mbps": "Throughput over PUSCH slots (Mbit/s)",
}


def _comparison_figure(runs, metric, field):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(7.2, 4.6), dpi=180, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    all_points = [point for run in runs.values() for point in run["points"]]
    for level, (name, style) in enumerate(_CURVES.items(), start=3):
        label, color, line, marker, size, width = style
        points = sorted(runs[name]["points"], key=lambda point: point["snr_db"])
        snr = [point["snr_db"] for point in points]
        axes.plot(
            snr, [point[field] for point in points], label=label, color=color, linestyle=line, linewidth=width,
            marker=marker, markersize=size, markerfacecolor="none", markeredgewidth=1.25,
            zorder=level,
        )
        if metric == "thp":
            _plot_throughput_annotations(axes, points, field, color=color, zorder=level)
    axes.set_xlabel("Reference SNR (dB)")
    if metric == "bler":
        _plot_bler_targets(axes, {point["target_bler"] for point in all_points
                                 if point.get("target_bler") is not None})
        axes.set_title("MCS-adaptive PUSCH block error rate")
        axes.set_ylabel("Observed BLER")
        axes.set_ylim(-0.045, 1.045)
        axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        normalized = all(run.get("metadata", {}).get("adaptation", {}).get("pusch_total_data_re_power") == 1
                         for run in runs.values())
        axes.set_title("MCS-adaptive uplink throughput" + (" at equal transmit power" if normalized else ""))
        axes.set_ylabel(_THROUGHPUT_LABELS[field])
        axes.set_ylim(bottom=0)
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_axisbelow(True)
    axes.grid(visible=True, axis="y", color="#DDE3EA", linewidth=0.7)
    legend_style = {"framealpha": 0.95, "edgecolor": "#DDE3EA"}
    if metric == "thp":
        _throughput_legend(axes, all_points, **legend_style)
    else:
        axes.legend(**legend_style)
    return figure


def write_comparison(output_dir: str | Path, runs: dict[str, dict]) -> Path:
    """Write the comparison plots, measurements and available paired differences."""
    if runs.keys() != _CURVES.keys():
        raise ValueError("A comparison requires full, partial_noncoherent and noncoherent runs")
    if any(not run["points"] for run in runs.values()):
        raise ValueError("Every codebook run must contain at least one completed SNR point")
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")

    ordered_runs = {name: runs[name] for name in _CURVES}
    rows = [{"codebook": name, **point} for name, run in ordered_runs.items() for point in run["points"]]
    throughput_field = _observed_throughput_field(rows)
    differences = paired_throughput_differences(ordered_runs, throughput_field)
    summary = {"throughput_field": throughput_field, "runs": runs}
    if differences:
        summary["paired_throughput_differences"] = differences
    summary_json = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)
    plots = {}
    for filename, metric, field in (("THP_vs_SNR.png", "thp", throughput_field),
                                    ("BLER_vs_SNR.png", "bler", "bler")):
        figure = _comparison_figure(runs, metric, field)
        with BytesIO() as stream:
            figure.canvas.print_png(stream)
            plots[filename] = stream.getvalue()

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "summary.json").write_text(summary_json + "\n", encoding="utf-8")
    write_csv(directory / "results.csv", rows)
    if differences:
        write_csv(directory / "throughput_differences.csv", differences)
    for filename, png in plots.items():
        (directory / filename).write_bytes(png)
    return directory
