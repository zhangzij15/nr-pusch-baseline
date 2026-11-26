"""Run a paired 2x2 (SRS ICE/RCE x DMRS ICE/RCE) PUSCH throughput sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys

from pusch_baseline import load_config, simulate
from pusch_baseline.__main__ import _parse_snr
from pusch_baseline.outputs import _observed_throughput_field, _plot_bler_targets, write_csv
from pusch_baseline.simulation import _CODEBOOK_SIZES, configure_worker_env
from pusch_baseline.throughput_statistics import paired_throughput_differences


PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_DIR / "pusch_baseline"
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "cdl_c_4t64r_codebooks.json"
DEFAULT_SNRS = (-26.0, -24.0, -22.0, -20.0, -16.0, -12.0, -8.0, -6.0, -4.0, 0.0)

VARIANTS = (
    ("srs_ice_dmrs_ice", "SRS ICE + DMRS ICE", "perfect", "perfect"),
    ("srs_rce_dmrs_ice", "SRS RCE + DMRS ICE", "real", "perfect"),
    ("srs_ice_dmrs_rce", "SRS ICE + DMRS RCE", "perfect", "lmmse"),
    ("srs_rce_dmrs_rce", "SRS RCE + DMRS RCE", "real", "lmmse"),
)

_STYLES = {
    "srs_ice_dmrs_ice": ("SRS ICE + DMRS ICE", "#0072B2", "-", "o", 9.0, 2.5, 3),
    "srs_rce_dmrs_ice": ("SRS RCE + DMRS ICE", "#009E73", "--", "D", 5.2, 1.8, 4),
    "srs_ice_dmrs_rce": ("SRS ICE + DMRS RCE", "#D55E00", "-", "s", 8.2, 2.4, 5),
    "srs_rce_dmrs_rce": ("SRS RCE + DMRS RCE", "#CC79A7", ":", "^", 5.4, 1.9, 6),
}


def _merge_variant_shards(shards: dict[str, dict[float, dict]], snrs) -> dict[str, dict]:
    snr_list = list(snrs)
    runs = {}
    for key, label, *_ in VARIANTS:
        first = shards[key][snr_list[0]]
        base_metadata = first["metadata"]
        if any(shards[key][snr]["metadata"] != base_metadata for snr in snr_list):
            raise RuntimeError(f"Inconsistent metadata or channel fingerprints in {key} SNR shards")
        runs[key] = {
            "config": {
                **first["config"],
                "sweep": {**first["config"]["sweep"], "snr_db": snr_list},
            },
            "metadata": {**base_metadata, "variant_label": label},
            "points": [shards[key][snr]["points"][0] for snr in snr_list],
        }
    return runs


def _run_shard(cfg, variant_key: str, srs_est: str, dmrs_est: str, snr_db: float):
    shard_cfg = replace(
        cfg,
        channel=replace(cfg.channel, estimator=dmrs_est),
        sweep=replace(cfg.sweep, snr_db=(snr_db,)),
        adaptation=replace(cfg.adaptation, srs_estimator=srs_est),
    )
    return variant_key, snr_db, simulate(shard_cfg)


def _render_ice_rce_figures(runs: dict, throughput_field: str) -> dict[str, bytes]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    all_points = [p for run in runs.values() for p in run["points"]]
    snrs = sorted({p["snr_db"] for p in all_points})

    fig = Figure(figsize=(8.4, 5.2), dpi=200, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    for key, (label, color, ls, marker, ms, lw, z) in _STYLES.items():
        pts = sorted(runs[key]["points"], key=lambda p: p["snr_db"])
        xs = [p["snr_db"] for p in pts]
        ys = [p[throughput_field] for p in pts]
        ax.plot(
            xs, ys, label=label, color=color, linestyle=ls, linewidth=lw,
            marker=marker, markersize=ms, markerfacecolor="none", markeredgewidth=1.35, zorder=z,
        )
        ivs = [(p["snr_db"], *p["throughput_ci95_mbps"]) for p in pts if p.get("throughput_ci95_mbps")]
        if ivs:
            sx, lo, hi = zip(*ivs)
            ax.vlines(sx, lo, hi, color=color, linewidth=0.95, alpha=0.55, zorder=z - 0.5)
        met = [(p["snr_db"], p[throughput_field]) for p in pts if p.get("target_status") == "at_target"]
        if met:
            mx, my = zip(*met)
            ax.scatter(mx, my, s=7, color=color, linewidths=0, zorder=z + 0.5)

    evolution = {
        run.get("metadata", {}).get("adaptation", {}).get("channel_evolution")
        for run in runs.values()
    }
    evo_tag = " | Continuous CIR" if evolution == {"continuous"} else ""
    ax.set_title(f"PUSCH Throughput vs SNR | SRS & DM-RS ICE vs RCE{evo_tag}", fontsize=11.5, pad=10)
    ax.set_xlabel("Reference SNR (dB)")
    ax.set_ylabel("Throughput over all TDD slots (Mbit/s)"
                  if throughput_field == "thp_wall_clock_mbps" else "Goodput (Mbit/s)")
    ax.set_xticks(snrs)
    if len(snrs) > 1:
        ax.set_xlim(snrs[0] - 0.8, snrs[-1] + 0.8)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(visible=True, axis="y", color="#DDE3EA", linewidth=0.7)
    ax.grid(visible=True, axis="x", color="#EEF2F6", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if any(p.get("target_status") == "at_target" for p in all_points):
        handles.append(Line2D([], [], marker="o", linestyle="none", color="0.25", markersize=3))
        labels.append("BLER target met")
    if any(p.get("throughput_ci95_mbps") is not None for p in all_points):
        handles.append(Line2D([], [], marker="|", linestyle="none", color="0.25", markersize=9))
        labels.append("95% SRS bootstrap CI")
    ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#DDE3EA", fontsize=8.5)

    low_snrs = [s for s in snrs if s <= snrs[0] + 6.5]
    if len(low_snrs) >= 3 and len(snrs) >= 6:
        axins = ax.inset_axes([0.07, 0.54, 0.38, 0.39])
        for key, (_, color, ls, marker, ms, lw, z) in _STYLES.items():
            pts = [p for p in sorted(runs[key]["points"], key=lambda p: p["snr_db"]) if p["snr_db"] in low_snrs]
            axins.plot(
                [p["snr_db"] for p in pts], [p[throughput_field] for p in pts],
                color=color, linestyle=ls, linewidth=lw * 0.85,
                marker=marker, markersize=ms * 0.72, markerfacecolor="none", markeredgewidth=1.1, zorder=z,
            )
        ys_low = [p[throughput_field] for p in all_points if p["snr_db"] in low_snrs]
        axins.set_xlim(low_snrs[0] - 0.6, low_snrs[-1] + 0.6)
        axins.set_ylim(max(0.0, min(ys_low) - 0.35), max(ys_low) + 0.45)
        axins.set_xticks(low_snrs)
        axins.tick_params(labelsize=7.5)
        axins.set_title(f"Low-SNR zoom ({low_snrs[0]:g} to {low_snrs[-1]:g} dB)", fontsize=8, pad=3)
        axins.grid(visible=True, color="#E2E8F0", linewidth=0.5)
        axins.spines[["top", "right"]].set_visible(False)
        ax.indicate_inset_zoom(axins, edgecolor="#94A3B8", linestyle="--", linewidth=0.75)

    with BytesIO() as buf:
        fig.canvas.print_png(buf)
        thp_png = buf.getvalue()

    fig_b = Figure(figsize=(8.0, 4.6), dpi=200, layout="constrained")
    FigureCanvasAgg(fig_b)
    axb = fig_b.add_subplot()
    for key, (label, color, ls, marker, ms, lw, z) in _STYLES.items():
        pts = sorted(runs[key]["points"], key=lambda p: p["snr_db"])
        axb.plot(
            [p["snr_db"] for p in pts], [p["bler"] for p in pts],
            label=label, color=color, linestyle=ls, linewidth=lw,
            marker=marker, markersize=ms, markerfacecolor="none", markeredgewidth=1.3, zorder=z,
        )
    _plot_bler_targets(axb, {p["target_bler"] for p in all_points if p.get("target_bler") is not None})
    axb.set_title(f"PUSCH Observed BLER vs SNR | SRS & DM-RS ICE vs RCE{evo_tag}", fontsize=11)
    axb.set_xlabel("Reference SNR (dB)")
    axb.set_ylabel("Observed BLER")
    axb.set_xticks(snrs)
    axb.set_ylim(-0.02, max(0.25, max(p["bler"] for p in all_points) + 0.05))
    axb.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axb.spines[["top", "right"]].set_visible(False)
    axb.grid(visible=True, axis="y", color="#DDE3EA", linewidth=0.7)
    axb.legend(framealpha=0.95, edgecolor="#DDE3EA", fontsize=8.5)
    with BytesIO() as buf:
        fig_b.canvas.print_png(buf)
        bler_png = buf.getvalue()

    return {"THP_vs_SNR_ice_rce.png": thp_png, "BLER_vs_SNR_ice_rce.png": bler_png}


def write_ice_rce_comparison(output_dir: str | Path, runs: dict[str, dict]) -> Path:
    """Write plots, CSVs, paired differences, and JSON summary for the 4-curve comparison."""
    expected_keys = {key for key, *_ in VARIANTS}
    if runs.keys() != expected_keys:
        raise ValueError("ICE/RCE comparison requires all four SRS/DMRS combinations")
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"Output directory already exists: {directory}")

    ordered = {key: runs[key] for key, *_ in VARIANTS}
    rows = [{"variant": key, **point} for key, run in ordered.items() for point in run["points"]]
    throughput_field = _observed_throughput_field(rows)
    differences = paired_throughput_differences(ordered, throughput_field)
    summary = {
        "throughput_field": throughput_field,
        "variants": {k: {"label": lbl, "srs_estimator": se, "dmrs_estimator": de} for k, lbl, se, de in VARIANTS},
        "runs": ordered,
    }
    if differences:
        summary["paired_throughput_differences"] = differences

    plots = _render_ice_rce_figures(ordered, throughput_field)
    summary_json = json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2)

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "summary.json").write_text(summary_json + "\n", encoding="utf-8")
    write_csv(directory / "results.csv", rows)
    if differences:
        write_csv(directory / "paired_differences.csv", differences)
    for filename, png in plots.items():
        (directory / filename).write_bytes(png)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare SRS/DMRS ICE vs RCE throughput across SNR.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Adaptive Rank-1 codebook JSON config (default: cdl_c_4t64r_codebooks.json).")
    parser.add_argument("--output-dir", type=Path, help="New directory for the outputs.")
    parser.add_argument("--snr", type=_parse_snr, help="Comma-separated SNR list in dB.")
    parser.add_argument("--warmup-tb", type=int, help="Override OLLA warmup TB count per SNR.")
    parser.add_argument("--num-tb", type=int, help="Override evaluation TB count per SNR.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel (variant, SNR) worker processes.")
    parser.add_argument("--tx-power-mode", choices=("full_power", "per_port"),
                        help="Override adaptive PUSCH transmit power mode.")
    parser.add_argument("--ue-yaw-offsets", type=_parse_snr,
                        help="Comma-separated CDL UE yaw offsets in degrees, sampled per SRS period.")
    parser.add_argument("--channel-evolution", choices=("per_period", "continuous"),
                        help="Override CIR continuity across SRS periods.")
    parser.add_argument("--seed", type=int, help="Override the channel, SRS-noise and payload seed.")
    args = parser.parse_args(argv)

    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        cfg = load_config(args.config)
        if cfg.adaptation is None or cfg.pusch.num_antenna_ports != 4 or cfg.pusch.precoding != "codebook":
            raise ValueError("ICE/RCE comparison requires an adaptive four-port codebook configuration")
        if cfg.pusch.rank != 1:
            raise ValueError("DMRS RCE (lmmse) requires Rank 1")

        snrs = args.snr if args.snr is not None else DEFAULT_SNRS
        sweep_overrides = {"snr_db": snrs}
        if args.num_tb is not None:
            sweep_overrides["num_tb_per_snr"] = args.num_tb
        if args.seed is not None:
            sweep_overrides["seed"] = args.seed
        cfg = replace(cfg, sweep=replace(cfg.sweep, **sweep_overrides))
        if args.ue_yaw_offsets is not None:
            cfg = replace(cfg, channel=replace(cfg.channel, ue_yaw_offsets_deg=args.ue_yaw_offsets))

        adaptation = cfg.adaptation
        if adaptation.rank_tpmi_candidates is None:
            full_tpmis = tuple((1, tpmi) for tpmi in range(_CODEBOOK_SIZES[1][0]))
            adaptation = replace(adaptation, rank_tpmi_candidates=full_tpmis)
        if args.tx_power_mode is not None:
            adaptation = replace(adaptation, tx_power_mode=args.tx_power_mode)
        if args.channel_evolution is not None:
            adaptation = replace(adaptation, channel_evolution=args.channel_evolution)
        if args.warmup_tb is not None and adaptation.olla is None:
            raise ValueError("--warmup-tb requires adaptation.olla in the configuration")
        if adaptation.olla is not None and (args.warmup_tb is not None or args.num_tb is not None):
            adaptation = replace(
                adaptation, olla=adaptation.olla.fit_budget(cfg.sweep.num_tb_per_snr, args.warmup_tb),
            )
        cfg = replace(cfg, adaptation=adaptation)
        cfg.validate()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = args.output_dir or PACKAGE_DIR / "results" / f"ice_rce_{stamp}"
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")

        configure_worker_env(args.workers)
        tasks = [(key, srs_e, dmrs_e, snr) for key, _, srs_e, dmrs_e in VARIANTS for snr in snrs]
        shards = {key: {} for key, *_ in VARIANTS}
        print(f"Running {len(tasks)} paired (variant, SNR) shards with {args.workers} worker(s)...", flush=True)

        if args.workers == 1:
            for done, (key, srs_e, dmrs_e, snr) in enumerate(tasks, start=1):
                _, _, run = _run_shard(cfg, key, srs_e, dmrs_e, snr)
                shards[key][snr] = run
                p = run["points"][0]
                print(f"[{done:2d}/{len(tasks)}] {key:16s} | SNR {snr:+5g} dB | "
                      f"BLER {p['bler']:.3f} ({p['tb_errors']}/{p['num_tb']}) | "
                      f"THP {p[_observed_throughput_field([p])]:.3f} Mbps", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
                futures = {pool.submit(_run_shard, cfg, *task): task for task in tasks}
                for done, fut in enumerate(as_completed(futures), start=1):
                    key, snr, run = fut.result()
                    shards[key][snr] = run
                    p = run["points"][0]
                    print(f"[{done:2d}/{len(tasks)}] {key:16s} | SNR {snr:+5g} dB | "
                          f"BLER {p['bler']:.3f} ({p['tb_errors']}/{p['num_tb']}) | "
                          f"THP {p[_observed_throughput_field([p])]:.3f} Mbps", flush=True)

        runs = _merge_variant_shards(shards, snrs)

        saved_dir = write_ice_rce_comparison(output_dir, runs)
        print(f"Saved {saved_dir.resolve()}")
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ICE/RCE comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
