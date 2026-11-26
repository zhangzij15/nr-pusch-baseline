"""Measure and plot all usable rows of NR PUSCH MCS Table 1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


PROJECT = Path(__file__).resolve().parent
PACKAGE = PROJECT / "pusch_baseline"
DEFAULT_CONFIG = PACKAGE / "configs" / "cdl_c_4t64r_codebooks.json"
MCS_ROWS = tuple(range(29))
# The CIR clock, TPMI scoring and transport-block coding affect saved measurements.
WATERFALL_REVISION = 2


def _run_curve(config, bank, mcs, batch_size, fixed_snrs, directory, min_snr_step=1.0):
    from pusch_baseline.simulation import apply_tensorflow_threads

    apply_tensorflow_threads()
    from pusch_baseline.mcs_waterfall import simulate_mcs_curve

    point_files = directory / f"{mcs:02d}"
    point_files.mkdir(exist_ok=True)
    completed = [json.loads(path.read_text(encoding="utf-8")) for path in point_files.glob("*.json")]

    def report(point):
        path = point_files / f"{float(point['snr_db'])}.json"
        staged = path.with_suffix(".tmp")
        staged.write_text(json.dumps(point, allow_nan=False), encoding="utf-8")
        staged.replace(path)
        print(f"MCS {mcs:02d} | {point['snr_db']:+6g} dB | "
              f"{point['tb_errors']:3d}/{point['num_tb']} errors", flush=True)

    return simulate_mcs_curve(
        config, bank, mcs, batch_size, on_point=report, snr_points=fixed_snrs,
        completed_points=completed, min_snr_step=min_snr_step,
    )


def _load_fixed_grid(summary_path, required=MCS_ROWS):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    points = {mcs: set() for mcs in MCS_ROWS}
    for point in data["points"]:
        mcs = int(point["mcs_index"])
        snr = float(point["snr_db"])
        if mcs not in points or not isfinite(snr) or snr in points[mcs]:
            raise ValueError("Fixed SNR grid has an invalid or duplicate measurement")
        points[mcs].add(snr)
    missing = [mcs for mcs in required if not points[mcs]]
    if missing:
        if tuple(required) == MCS_ROWS:
            raise ValueError("Fixed SNR grid must cover all 29 MCS rows")
        missing_rows = ", ".join(str(mcs) for mcs in missing)
        raise ValueError(f"Fixed SNR grid is missing requested MCS row(s): {missing_rows}")
    return {mcs: tuple(sorted(points[mcs])) for mcs in required}


def _restore_legacy_manifest_defaults(manifest):
    """Fill fields absent before fixed-yaw, full-power runs became configurable."""
    config = manifest["config"]
    config["channel"].setdefault("ue_yaw_offsets_deg", [])
    adaptation = config["adaptation"]
    adaptation.setdefault("tx_power_mode", "full_power")
    if adaptation.get("olla") is not None:
        adaptation["olla"].setdefault("mcs_axis", "index")
    return manifest


def _restore_legacy_bank_defaults(bank):
    return {"tx_power_mode": "full_power", "srs_estimator": "real", "srs_power_per_active_re": 1.0,
            "ue_yaw_offsets_deg": [], **bank}


def _tpmi_policy(bank_metadata: dict) -> str:
    sounding = "ideal noise-free SRS" if bank_metadata.get("srs_estimator") == "perfect" else "noisy SRS"
    tpmis = bank_metadata.get("tpmi_candidates")
    count = bank_metadata.get("num_tpmi_candidates", 28)
    if tpmis is None and count == 28:
        subset = "the full set of rank-one codebook precoders"
    else:
        subset = f"{count} configured rank-one codebook precoder(s)"
    return f"{sounding} selects from {subset}"


def _write_summary(directory, manifest, bank_metadata, rows):
    from pusch_baseline.mcs_waterfall_plot import write_mcs_waterfall_figure

    metadata = {
        **manifest,
        "physical_channel_bank": bank_metadata,
        "snr_reference": "unit reference PUSCH data RE power / noise variance per receive antenna RE",
        "tpmi_policy": _tpmi_policy(bank_metadata),
        "sampling": "fixed MCS per curve; paired physical samples and standardized receiver noise across SNR",
        "decoder": "Sionna PUSCH 20-iteration TB decoding with an RV0 rate-matching extension at low rates",
    }
    (directory / "summary.json").write_text(
        json.dumps({"metadata": metadata, "points": rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subtitle = f"{manifest['num_tb_per_point']:,} TBs per measured point"
    if manifest["config"].get("adaptation", {}).get("channel_evolution") == "continuous":
        subtitle += " · continuous CIR"
    for suffix in ("png", "pdf"):
        write_mcs_waterfall_figure(
            rows, directory / f"BLER_vs_SNR_MCS_table1.{suffix}", mcs_table=1, subtitle=subtitle,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-tb", type=int, default=64, help="Common measured TB count per (MCS, SNR).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mcs", help="Comma-separated row subset for a partial sweep or verification.")
    snr_grid = parser.add_mutually_exclusive_group()
    snr_grid.add_argument("--snr-from", type=Path, help="Reuse the per-MCS SNR points of a waterfall summary.json.")
    snr_grid.add_argument("--snr-db", help="Use these comma-separated SNR points for every requested MCS.")
    parser.add_argument("--estimator", choices=("ls", "lmmse", "perfect"),
                        help="Override the configured DM-RS estimator.")
    parser.add_argument("--channel-evolution", choices=("per_period", "continuous"),
                        help="Override CIR continuity across SRS periods.")
    parser.add_argument("--snr-step", type=float, default=1.0,
                        help="Minimum adaptive SNR refinement step in dB (1, 0.5, 0.25, or 0.125).")
    parser.add_argument("--seed", type=int, help="Override the paired channel and pilot noise seed.")
    args = parser.parse_args()
    if min(args.num_tb, args.batch_size, args.workers) <= 0:
        parser.error("TB, batch, and worker counts must be positive")
    try:
        required = MCS_ROWS if args.mcs is None else tuple(sorted({int(mcs) for mcs in args.mcs.split(",")}))
    except ValueError:
        parser.error("--mcs expects a comma-separated list of integers")
    if not required or not set(required).issubset(MCS_ROWS):
        parser.error("--mcs supports Table 1 indices 0 through 28")

    from pusch_baseline.simulation import apply_tensorflow_threads, configure_worker_env

    configure_worker_env(args.workers, intra_threads=2, inter_threads=1)
    apply_tensorflow_threads()

    from pusch_baseline.config import load_config
    from pusch_baseline.mcs_waterfall import waterfall_snr_grid
    from pusch_baseline.mcs_waterfall_channel import write_channel_bank

    config = args.config.resolve()
    cfg = load_config(config)
    if args.estimator is not None:
        cfg = replace(cfg, channel=replace(cfg.channel, estimator=args.estimator))
    if args.channel_evolution is not None:
        if cfg.adaptation is None:
            parser.error("--channel-evolution requires an adaptive configuration")
        cfg = replace(cfg, adaptation=replace(cfg.adaptation, channel_evolution=args.channel_evolution))
    if args.seed is not None:
        cfg = replace(cfg, sweep=replace(cfg.sweep, seed=args.seed))
    grant = cfg.pusch
    if (grant.rank, grant.num_antenna_ports, grant.mcs_table, grant.precoding) != (1, 4, 1, "codebook"):
        parser.error("the waterfall requires Rank 1, four TX ports, and MCS Table 1 codebook PUSCH")
    if cfg.adaptation is None or cfg.adaptation.tdd_pattern is None:
        parser.error("the waterfall requires a TDD/SRS profile for the physical channel bank")
    try:
        cfg.validate()
        fixed_grid = _load_fixed_grid(args.snr_from, required) if args.snr_from is not None else {}
        if args.snr_db is not None:
            points = tuple(sorted(float(snr) for snr in args.snr_db.split(",")))
            if len(set(points)) != len(points) or not all(map(isfinite, points)):
                raise ValueError("--snr-db requires distinct finite SNR points")
            fixed_grid = dict.fromkeys(required, points)
        default_snrs = waterfall_snr_grid(args.snr_step)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    snrs = sorted({snr for group in fixed_grid.values() for snr in group}) if fixed_grid else default_snrs

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (args.output_dir or PACKAGE / "results" / f"mcs_waterfall_{stamp}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "simulation_revision": WATERFALL_REVISION,
        "config": json.loads(json.dumps(cfg.to_dict())),
        "num_tb_per_point": args.num_tb,
        "batch_size": args.batch_size,
        "snr_grid_db": list(snrs),
        "mcs_table": 1,
    }
    if fixed_grid:
        manifest["fixed_snr_by_mcs"] = {str(mcs): list(grid) for mcs, grid in fixed_grid.items()}
    manifest_file = directory / "settings.json"
    if manifest_file.exists():
        stored_manifest = _restore_legacy_manifest_defaults(json.loads(manifest_file.read_text(encoding="utf-8")))
        if stored_manifest.get("simulation_revision") != WATERFALL_REVISION:
            parser.error("this waterfall was measured with a different simulator revision; use a new output directory")
        if stored_manifest != manifest:
            parser.error("output directory contains a waterfall with different parameters")
    else:
        if any(directory.iterdir()):
            parser.error("output directory must be empty or an existing matching waterfall")
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checkpoint = directory / "mcs_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    pending = [mcs for mcs in reversed(required) if not (checkpoint / f"{mcs:02d}.json").exists()]
    bank_record = directory / "channel_bank.json"
    if pending:
        with TemporaryDirectory(prefix="pusch_mcs_bank_") as temp:
            print(f"Preparing {args.num_tb} common PUSCH channel samples", flush=True)
            write_channel_bank(
                cfg, snrs, args.num_tb, Path(temp),
                on_progress=lambda done, total: print(f"Shared channels {done}/{total}", flush=True),
            )
            bank = json.loads((Path(temp) / "bank.json").read_text(encoding="utf-8"))
            if bank_record.exists():
                stored_bank = _restore_legacy_bank_defaults(json.loads(bank_record.read_text(encoding="utf-8")))
                if stored_bank != bank:
                    raise RuntimeError("rerun produced a different common channel bank")
            else:
                bank_record.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                jobs = {
                    executor.submit(
                        _run_curve, cfg, temp, mcs, args.batch_size, fixed_grid.get(mcs), checkpoint, args.snr_step,
                    ): mcs
                    for mcs in pending
                }
                failures = {}
                for job in as_completed(jobs):
                    mcs = jobs[job]
                    try:
                        points = job.result()
                    except Exception as exc:
                        failures[mcs] = exc
                        print(f"MCS {mcs:02d} failed: {exc}", file=sys.stderr, flush=True)
                        continue
                    result_file = checkpoint / f"{mcs:02d}.json"
                    staged = result_file.with_suffix(".tmp")
                    staged.write_text(json.dumps(points, allow_nan=False), encoding="utf-8")
                    staged.replace(result_file)
                    print(f"Completed MCS {mcs:02d} ({len(points)} SNR points)", flush=True)
                if failures:
                    failed_rows = ", ".join(str(mcs) for mcs in sorted(failures))
                    raise RuntimeError(f"Incomplete waterfall MCS rows: {failed_rows}") from next(iter(failures.values()))

    rows = [
        point
        for stored in sorted(checkpoint.glob("*.json"))
        for point in json.loads(stored.read_text(encoding="utf-8"))
    ]
    stored_bank = _restore_legacy_bank_defaults(json.loads(bank_record.read_text(encoding="utf-8")))
    _write_summary(directory, manifest, stored_bank, rows)
    print(f"Saved {len(set(row['mcs_index'] for row in rows))} MCS curves to {directory}", flush=True)


if __name__ == "__main__":
    main()
