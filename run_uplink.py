"""Run a configured NR uplink sweep and save its plots."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from pusch_baseline import load_config, simulate
from pusch_baseline.__main__ import _parse_snr
from pusch_baseline.outputs import target_throughput_report, write_results
from pusch_baseline.simulation import simulate_codebook_comparison


PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_DIR / "pusch_baseline"
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "cdl_c_4t64r_codebooks.json"


def _report(point: dict, *, name: str | None = None) -> None:
    prefix = f"[{name}] " if name else ""
    throughput = point.get("thp_wall_clock_mbps", point.get("thp_duty_cycle_mbps", point["thp_scheduled_mbps"]))
    thp_report = target_throughput_report(point) or f"THP {throughput:.3f} Mbps"
    print(
        f"{prefix}SNR {point['snr_db']:>6g} dB | "
        f"BLER {point['bler']:.5g} ({point['tb_errors']}/{point['num_tb']}) | "
        f"{thp_report}",
        flush=True,
    )


def _comparison_shard(cfg, snr_db):
    settings = replace(cfg.sweep, snr_db=(snr_db,))
    return simulate_codebook_comparison(replace(cfg, sweep=settings))


def _parallel_comparison(cfg, workers):
    """Run SNR shards with identical seeds and merge them in configured order."""
    snrs = cfg.sweep.snr_db
    shards = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(snrs))) as executor:
        pending = {executor.submit(_comparison_shard, cfg, snr): snr for snr in snrs}
        for job in as_completed(pending):
            snr = pending[job]
            shard = job.result()
            shards[snr] = shard
            for name, run in shard.items():
                _report(run["points"][0], name=name)

    combined = shards[snrs[0]]
    for name, run in combined.items():
        run["config"]["sweep"]["snr_db"] = snrs
        run["points"] = [shards[snr][name]["points"][0] for snr in snrs]
        if any(shards[snr][name]["metadata"] != run["metadata"] for snr in snrs):
            raise RuntimeError(f"Inconsistent metadata or channel fingerprints in {name} SNR shards")
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and plot an NR PUSCH SNR sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON configuration (default: cdl_c_4t64r_codebooks.json).")
    parser.add_argument("--single", action="store_true",
                        help="Run one sweep using the configured grant candidates.")
    parser.add_argument("--output-dir", type=Path, help="New directory for the outputs.")
    parser.add_argument("--snr", type=_parse_snr, help="Override the configured SNR list, e.g. --snr=-16,-8,0.")
    parser.add_argument("--num-tb", type=int, help="Override the evaluation TB count per SNR.")
    parser.add_argument("--warmup-tb", type=int, help="Override the OLLA calibration TB count per SNR.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel paired SNR shards for a codebook sweep.")
    parser.add_argument("--ue-yaw-offsets", type=_parse_snr,
                        help="Comma-separated CDL UE yaw offsets in degrees, sampled per SRS period.")
    parser.add_argument("--tx-power-mode", choices=("full_power", "per_port"),
                        help="Override the adaptive PUSCH and SRS power model.")
    parser.add_argument("--channel-evolution", choices=("per_period", "continuous"),
                        help="Override CIR continuity across SRS periods.")
    args = parser.parse_args(argv)

    try:
        if args.workers < 1 or (args.single and args.workers != 1):
            raise ValueError("--workers must be positive; --single requires one worker")
        cfg = load_config(args.config)
        sweep_overrides = {}
        if args.snr is not None:
            sweep_overrides["snr_db"] = args.snr
        if args.num_tb is not None:
            sweep_overrides["num_tb_per_snr"] = args.num_tb
        cfg = replace(cfg, sweep=replace(cfg.sweep, **sweep_overrides))
        if args.ue_yaw_offsets is not None:
            cfg = replace(cfg, channel=replace(cfg.channel, ue_yaw_offsets_deg=args.ue_yaw_offsets))
        if args.tx_power_mode is not None:
            if cfg.adaptation is None:
                raise ValueError("--tx-power-mode requires an adaptive configuration")
            cfg = replace(cfg, adaptation=replace(cfg.adaptation, tx_power_mode=args.tx_power_mode))
        if args.channel_evolution is not None:
            if cfg.adaptation is None:
                raise ValueError("--channel-evolution requires an adaptive configuration")
            cfg = replace(cfg, adaptation=replace(cfg.adaptation, channel_evolution=args.channel_evolution))
        if args.warmup_tb is not None:
            if cfg.adaptation is None or cfg.adaptation.olla is None:
                raise ValueError("--warmup-tb requires adaptation.olla in the configuration")
            olla = cfg.adaptation.olla
            max_window = args.warmup_tb
            if olla.max_mcs_spread is not None:
                max_window = min(args.warmup_tb // 3, cfg.sweep.num_tb_per_snr // 3)
            olla = replace(olla, warmup_tb=args.warmup_tb, window_tb=max(1, min(olla.window_tb, max_window)))
            cfg = replace(cfg, adaptation=replace(cfg.adaptation, olla=olla))
        cfg.validate()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = args.output_dir or PACKAGE_DIR / "results" / f"uplink_{stamp}"
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")

        if cfg.adaptation is not None and cfg.adaptation.olla is not None:
            olla = cfg.adaptation.olla
            print(f"Target BLER {olla.target_bler:.1%} | {olla.warmup_tb} warmup TBs and "
                  f"{cfg.sweep.num_tb_per_snr} measurement TBs per SNR", flush=True)

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(os.cpu_count() or 1, 2 if args.workers > 1 else 4)))
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1" if args.workers > 1 else "2")
        if args.workers > 1:
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(name, "1")
        if args.single:
            saved_dir = write_results(output_dir, **simulate(cfg, on_point=_report))
        else:
            from pusch_baseline.comparison_outputs import write_comparison

            def progress(phase, done, total):
                if total >= 200:
                    print(f"{phase} | {done}/{total} TB per SNR", flush=True)

            if args.workers > 1 and len(cfg.sweep.snr_db) > 1:
                runs = _parallel_comparison(cfg, args.workers)
            else:
                runs = simulate_codebook_comparison(
                    cfg, on_point=lambda name, point: _report(point, name=name), on_progress=progress,
                )
            saved_dir = write_comparison(output_dir, runs)
        print(f"Saved {saved_dir.resolve()}")
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"PUSCH simulation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
