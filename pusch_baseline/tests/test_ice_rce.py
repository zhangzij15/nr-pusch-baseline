"""Tests for the 4-curve SRS/DMRS ICE vs RCE comparison outputs."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from run_ice_rce import VARIANTS, write_ice_rce_comparison


class IceRceComparisonTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
