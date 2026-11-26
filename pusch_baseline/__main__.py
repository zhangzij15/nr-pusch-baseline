"""Command-line entry point for the standalone PUSCH baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

from . import load_config, simulate
from .outputs import target_throughput_report, write_results
from .simulation import configure_worker_env


def _parse_snr(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(point) for point in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc


def _report(point: dict) -> None:
    throughput = target_throughput_report(point)
    if throughput is None:
        throughput = f"THP {point['thp_scheduled_mbps']:.3f} Mbps"
        if "thp_wall_clock_mbps" in point:
            throughput += f" | wall-clock {point['thp_wall_clock_mbps']:.3f} Mbps"
        elif "thp_including_srs_mbps" in point:
            throughput += f" | incl. SRS {point['thp_including_srs_mbps']:.3f} Mbps"
    print(
        f"SNR {point['snr_db']:>6g} dB | BLER {point['bler']:.5g} "
        f"({point['tb_errors']}/{point['num_tb']}) | "
        f"{throughput}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep SNR for 5G NR CP-OFDM PUSCH.")
    parser.add_argument("--config", type=Path, help="JSON profile. Default: AWGN SISO.")
    parser.add_argument("--snr", type=_parse_snr, help="Example: --snr=-8,-4,0,4")
    parser.add_argument("--num-tb", type=int, help="Maximum TB count per SNR.")
    parser.add_argument("--warmup-tb", type=int, help="Override the OLLA calibration TB count per SNR.")
    parser.add_argument("--batch-size", type=int, help="TBs per batch for fixed-grant simulations.")
    parser.add_argument("--seed", type=int, help="Random seed for the full sweep.")
    parser.add_argument("--metric", choices=("both", "bler", "thp"), default="both",
                        help="Choose which plots to save. CSV/JSON include both metrics.")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run.")
    parser.add_argument("--check", action="store_true", help="Validate JSON and overrides.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        overrides = {
            key: value for key, value in {
                "snr_db": args.snr,
                "num_tb_per_snr": args.num_tb,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }.items() if value is not None
        }
        cfg = replace(cfg, sweep=replace(cfg.sweep, **overrides))
        if args.warmup_tb is not None and (cfg.adaptation is None or cfg.adaptation.olla is None):
            raise ValueError("--warmup-tb requires adaptation.olla in the configuration")
        if cfg.adaptation is not None and cfg.adaptation.olla is not None and (
            args.warmup_tb is not None or args.num_tb is not None
        ):
            olla = cfg.adaptation.olla.fit_budget(cfg.sweep.num_tb_per_snr, args.warmup_tb)
            cfg = replace(cfg, adaptation=replace(cfg.adaptation, olla=olla))
        cfg.validate()
        if args.check:
            print("Configuration is valid.")
            return 0
        if cfg.channel.kind == "custom":
            raise ValueError('Use the Python simulate API with channel.kind="custom".')
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = args.output_dir or Path("results") / f"pusch_{stamp}"
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        configure_worker_env(1)
        run = simulate(cfg, on_point=_report)
        saved_dir = write_results(output_dir, **run, metric=args.metric)
        print(f"Saved {saved_dir.resolve()}")
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"PUSCH simulation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
