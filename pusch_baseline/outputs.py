"""Transport-block statistics and SNR sweep artifacts."""

from __future__ import annotations

import csv
from io import BytesIO
import json
import math
from operator import index
from pathlib import Path


TARGET_THROUGHPUT = "thp_at_target_mbps"


def target_throughput_report(point: dict) -> str | None:
    status = point.get("target_status")
    if status is None:
        return None
    throughput = point[_observed_throughput_field([point])]
    report = f"THP {throughput:.3f} Mbps ({status})"
    interval = point.get("throughput_ci95_mbps")
    if interval is not None:
        report += f" | 95% CI [{interval[0]:.3f}, {interval[1]:.3f}] Mbps"
    return report


def _observed_throughput_field(points):
    if all("thp_wall_clock_mbps" in point for point in points):
        return "thp_wall_clock_mbps"
    if any(point.get("ul_duty_cycle", 1.0) != 1.0 for point in points):
        return "thp_duty_cycle_mbps"
    if all("thp_including_srs_mbps" in point for point in points):
        return "thp_including_srs_mbps"
    return "thp_scheduled_mbps"


def _plot_bler_targets(axes, targets):
    for target in sorted(targets):
        axes.axhline(target, color="0.35", linestyle="--", linewidth=1.1,
                     label=f"Target BLER {100 * target:g}%")


def _plot_throughput_annotations(axes, points, field, *, color, zorder):
    intervals = [(point["snr_db"], *point["throughput_ci95_mbps"]) for point in points
                 if point.get("throughput_ci95_mbps") is not None]
    if intervals:
        snr, low, high = zip(*intervals)
        axes.vlines(snr, low, high, color=color, linewidth=1.0, alpha=0.65, zorder=zorder - 0.5)
    qualified = [(point["snr_db"], point[field]) for point in points
                 if point.get("target_status") == "at_target"]
    if qualified:
        snr, thp = zip(*qualified)
        axes.scatter(snr, thp, s=6, color=color, linewidths=0, zorder=zorder + 0.5)


def _throughput_legend(axes, points, **kwargs):
    from matplotlib.lines import Line2D

    handles, labels = axes.get_legend_handles_labels()
    if any(point.get("target_status") == "at_target" for point in points):
        handles.append(Line2D([], [], marker="o", linestyle="none", color="0.25", markersize=3))
        labels.append("BLER target met")
    elif any("target_status" in point for point in points):
        handles.append(Line2D([], [], marker="o", linestyle="none", color="0.25",
                              markerfacecolor="none", markersize=5))
        labels.append("No point met the BLER target")
    if any(point.get("throughput_ci95_mbps") is not None for point in points):
        handles.append(Line2D([], [], marker="|", linestyle="none", color="0.25", markersize=10))
        labels.append("95% SRS bootstrap interval")
    axes.legend(handles, labels, **kwargs)


def summarize_point(
    snr_db: float,
    num_tb: int,
    tb_errors: int,
    tb_size_bits: int,
    slot_seconds: float,
    ul_duty_cycle: float,
    stop_reason: str,
) -> dict:
    """Summarize a completed SNR point with one transport block per slot."""
    snr_db = float(snr_db)
    num_tb = index(num_tb)
    tb_errors = index(tb_errors)
    tb_size_bits = index(tb_size_bits)
    slot_seconds = float(slot_seconds)
    ul_duty_cycle = float(ul_duty_cycle)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if num_tb < 1 or not 0 <= tb_errors <= num_tb:
        raise ValueError("Expected num_tb >= 1 and 0 <= tb_errors <= num_tb")
    if tb_size_bits < 1 or not math.isfinite(slot_seconds) or slot_seconds <= 0:
        raise ValueError("Expected positive TB size and finite positive slot duration")
    if not 0.0 <= ul_duty_cycle <= 1.0:
        raise ValueError("ul_duty_cycle must be between 0 and 1")

    bler = tb_errors / num_tb
    z = 1.959963984540054
    z_squared = z**2
    denominator = 1.0 + z_squared / num_tb
    center = (bler + z_squared / (2.0 * num_tb)) / denominator
    radius = z * math.sqrt(bler * (1.0 - bler) / num_tb + (z / (2.0 * num_tb)) ** 2)
    radius /= denominator
    zero_error_upper = -math.expm1(math.log(0.05) / num_tb) if tb_errors == 0 else None
    successful_tb = num_tb - tb_errors
    peak_mbps = tb_size_bits / (slot_seconds * 1e6)
    scheduled_mbps = peak_mbps * successful_tb / num_tb

    return {
        "snr_db": snr_db,
        "num_tb": num_tb,
        "tb_errors": tb_errors,
        "successful_tb": successful_tb,
        "tb_size_bits": tb_size_bits,
        "slot_seconds": slot_seconds,
        "ul_duty_cycle": ul_duty_cycle,
        "bler": bler,
        "bler_wilson95_low": 0.0 if tb_errors == 0 else max(0.0, center - radius),
        "bler_wilson95_high": 1.0 if tb_errors == num_tb else min(1.0, center + radius),
        "bler_zero_error_upper95": zero_error_upper,
        "thp_scheduled_mbps": scheduled_mbps,
        "thp_duty_cycle_mbps": scheduled_mbps * ul_duty_cycle,
        "thp_peak_mbps": peak_mbps,
        "stop_reason": stop_reason,
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write heterogeneous dict rows to UTF-8 CSV, JSON-encoding nested structures."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        } for row in rows)


def write_results(
    output_dir: str | Path,
    config: dict,
    metadata: dict,
    points: list[dict],
    metric: str = "both",
) -> Path:
    """Write all tabular results and selected headless plots to a fresh directory."""
    if metric not in {"both", "bler", "thp"}:
        raise ValueError("metric must be one of both, bler, thp")
    if not points:
        raise ValueError("At least one completed SNR point is needed to write results")
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")

    summary = json.dumps(
        {"config": config, "metadata": metadata, "points": points},
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    plots = _render_plots(points, metric)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "summary.json").write_text(summary + "\n", encoding="utf-8")
    write_csv(directory / "results.csv", points)
    for filename, png in plots.items():
        (directory / filename).write_bytes(png)
    return directory


def _render_plots(points: list[dict], metric: str) -> dict[str, bytes]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    points = sorted(points, key=lambda point: point["snr_db"])
    snr_db = [point["snr_db"] for point in points]
    targets = {point["target_bler"] for point in points if point.get("target_bler") is not None}
    plots = {}
    if metric in {"both", "bler"}:
        figure = Figure(figsize=(7.2, 4.6), layout="constrained")
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot()
        if targets:
            ax.plot(snr_db, [point["bler"] for point in points], "o-", label="Evaluation BLER")
            _plot_bler_targets(ax, targets)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_ylim(-0.045, 1.045)
        else:
            _plot_log_bler(ax, snr_db, points)
        ax.set(xlabel="Reference SNR (dB)", ylabel="Transport block error rate")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        with BytesIO() as stream:
            canvas.print_png(stream)
            plots["BLER_vs_SNR.png"] = stream.getvalue()

    if metric in {"both", "thp"}:
        figure = Figure(figsize=(7.2, 4.6), layout="constrained")
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot()
        if targets:
            field = _observed_throughput_field(points)
            color = "#0072B2"
            ax.plot(snr_db, [point[field] for point in points], "o-", markerfacecolor="none",
                    color=color, zorder=3, label="Measured throughput")
            _plot_throughput_annotations(ax, points, field, color=color, zorder=3)
        else:
            _plot_throughput(ax, snr_db, points)
            if "throughput_srs_clusters" in points[0]:
                _plot_throughput_annotations(ax, points, _observed_throughput_field(points),
                                             color="#D55E00", zorder=3)
        ax.set(xlabel="Reference SNR (dB)", ylabel="Goodput (Mbit/s)", ylim=(0, None))
        if not targets and all(point["thp_scheduled_mbps"] == 0.0 for point in points):
            ax.set_ylim(0, max(point["thp_peak_mbps"] for point in points))
        ax.grid(True, alpha=0.25)
        _throughput_legend(ax, points)
        with BytesIO() as stream:
            canvas.print_png(stream)
            plots["THP_vs_SNR.png"] = stream.getvalue()

    return plots


def _plot_log_bler(ax, snr_db, points):
    positive_bler = [point["bler"] if point["bler"] > 0 else math.nan for point in points]
    if any(point["tb_errors"] > 0 for point in points):
        ax.semilogy(snr_db, positive_bler, "o-", label="Observed BLER")
    zero_error_points = [point for point in points if point.get("bler_zero_error_upper95") is not None]
    if zero_error_points:
        ax.semilogy(
            [point["snr_db"] for point in zero_error_points],
            [point["bler_zero_error_upper95"] for point in zero_error_points],
            "v", markerfacecolor="none", color="tab:orange", label="Zero errors (95% upper bound)",
        )
    observed_zeros = [point for point in points
                      if point["tb_errors"] == 0 and "bler_zero_error_upper95" not in point]
    if observed_zeros:
        display_floor = min(0.01, min((p["bler"] for p in points if p["bler"] > 0), default=1.0) / 2)
        ax.semilogy(
            [p["snr_db"] for p in observed_zeros], [display_floor] * len(observed_zeros),
            "v", markerfacecolor="none", color="tab:orange", label="Zero observed errors (display floor)",
        )
    ax.set_ylim(top=1.12)


def _plot_throughput(ax, snr_db, points):
    ax.plot(snr_db, [point["thp_scheduled_mbps"] for point in points], "o-", label="PUSCH slots")
    if "thp_wall_clock_mbps" in points[0]:
        ax.plot(snr_db, [point["thp_wall_clock_mbps"] for point in points],
                "^-", label="All physical slots")
    elif "thp_including_srs_mbps" in points[0]:
        ax.plot(snr_db, [point["thp_including_srs_mbps"] for point in points],
                "^-", label="Including SRS slots")
    if any(point["ul_duty_cycle"] != 1.0 for point in points):
        ax.plot(snr_db, [point["thp_duty_cycle_mbps"] for point in points],
                "s--", label="UL duty-cycle adjusted")
