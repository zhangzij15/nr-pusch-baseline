"""Tests for the 4-curve SRS/DMRS ICE vs RCE comparison outputs."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pusch_baseline import SimulationConfig
from pusch_baseline.config import AdaptationSettings, OllaSettings
from run_ice_rce import VARIANTS, write_ice_rce_comparison


class IceRceComparisonTests(unittest.TestCase):
    def test_paired_shard_matches_independent_runs_with_one_cir_per_epoch(self):
        import run_ice_rce
        from pusch_baseline.continuous_channel import ContinuousCir

        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4),
            pusch=replace(base.pusch, num_antenna_ports=4, precoding="codebook", mcs_index=5),
            channel=replace(base.channel, kind="tdl", num_rx_ant=2),
            sweep=replace(base.sweep, snr_db=(-3.0,), num_tb_per_snr=3),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDD", srs_period_slots=5, channel_evolution="continuous",
                rank_tpmi_candidates=((1, 0), (1, 3)), mcs_indices=(5, 6),
                olla=OllaSettings(warmup_tb=3, window_tb=1, max_mcs_spread=1.0),
            ),
        )
        snr = cfg.sweep.snr_db[0]
        expected = {
            key: run_ice_rce._run_shard(cfg, key, srs, dmrs, snr)[1][key]
            for key, _, srs, dmrs in VARIANTS
        }
        sample = ContinuousCir.sample
        with patch.object(ContinuousCir, "sample", autospec=True, side_effect=sample) as calls:
            _, paired = run_ice_rce._run_paired_shard(cfg, (snr,))
        self.assertEqual(calls.call_count, 6)
        self.assertEqual(paired, expected)

    def _sample_runs(self):
        snrs = (-26.0, -24.0, -22.0, -20.0, -16.0, -8.0)
        runs = {}
        for offset, (key, label, srs_e, dmrs_e) in enumerate(VARIANTS):
            runs[key] = {
                "config": {"pusch": {"rank": 1}},
                "metadata": {"variant_label": label, "srs_estimator": srs_e, "dmrs_estimator": dmrs_e},
                "points": [{
                    "snr_db": snr,
                    "bler": 0.1,
                    "target_bler": 0.1,
                    "target_status": "at_target",
                    "slot_seconds": 0.0005,
                    "ul_duty_cycle": 1.0,
                    "thp_wall_clock_mbps": max(0.5, 6.0 + 0.2 * snr - 0.15 * offset),
                    "throughput_ci95_mbps": [
                        max(0.3, 5.8 + 0.2 * snr - 0.15 * offset),
                        max(0.7, 6.2 + 0.2 * snr - 0.15 * offset),
                    ],
                    "throughput_srs_clusters": [
                        {"srs_slot_index": 1, "num_slots": 10, "delivered_payload_bits": 10000 - 500 * offset},
                        {"srs_slot_index": 11, "num_slots": 10, "delivered_payload_bits": 12000 - 500 * offset},
                    ],
                } for snr in snrs],
            }
        return runs

    def test_write_ice_rce_creates_plots_csvs_and_paired_differences(self):
        runs = self._sample_runs()
        with TemporaryDirectory() as temp:
            out = write_ice_rce_comparison(Path(temp) / "ice_rce", runs)
            self.assertEqual(
                {p.name for p in out.iterdir()},
                {"THP_vs_SNR_ice_rce.png", "BLER_vs_SNR_ice_rce.png",
                 "results.csv", "paired_differences.csv", "summary.json"},
            )
            for name in ("THP_vs_SNR_ice_rce.png", "BLER_vs_SNR_ice_rce.png"):
                self.assertEqual((out / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["throughput_field"], "thp_wall_clock_mbps")
            self.assertEqual(len(summary["paired_throughput_differences"]), 6 * 6)
            with (out / "results.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 24)
            with (out / "paired_differences.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 36)

    def test_rejects_incomplete_variant_set(self):
        runs = self._sample_runs()
        del runs["srs_rce_dmrs_rce"]
        with TemporaryDirectory() as temp, self.assertRaises(ValueError):
            write_ice_rce_comparison(Path(temp) / "incomplete", runs)

    def test_continuous_cir_metadata_renders_plots(self):
        runs = self._sample_runs()
        for run in runs.values():
            run["metadata"]["adaptation"] = {"channel_evolution": "continuous"}
        with TemporaryDirectory() as temp:
            out = write_ice_rce_comparison(Path(temp) / "ice_rce_continuous", runs)
            for name in ("THP_vs_SNR_ice_rce.png", "BLER_vs_SNR_ice_rce.png"):
                self.assertEqual((out / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_merge_variant_shards_validates_cross_snr_metadata(self):
        from run_ice_rce import _merge_variant_shards

        snrs = (-20.0, -12.0)
        shards = {}
        for key, _, srs_e, dmrs_e in VARIANTS:
            shards[key] = {
                snr: {
                    "config": {"sweep": {"snr_db": [snr]}},
                    "metadata": {"srs": srs_e, "dmrs": dmrs_e, "channel_sha256": "abc123"},
                    "points": [{"snr_db": snr, "bler": 0.1}],
                }
                for snr in snrs
            }
        merged = _merge_variant_shards(shards, snrs)
        self.assertEqual(merged["srs_ice_dmrs_ice"]["config"]["sweep"]["snr_db"], [-20.0, -12.0])
        self.assertEqual(merged["srs_ice_dmrs_ice"]["metadata"]["variant_label"], "SRS ICE + DMRS ICE")
        self.assertNotIn("variant_label", shards["srs_ice_dmrs_ice"][-20.0]["metadata"])

        shards["srs_rce_dmrs_rce"][-12.0]["metadata"] = {
            "srs": "real", "dmrs": "lmmse", "channel_sha256": "mismatch",
        }
        with self.assertRaisesRegex(RuntimeError, "Inconsistent metadata or channel fingerprints"):
            _merge_variant_shards(shards, snrs)


if __name__ == "__main__":
    unittest.main()
