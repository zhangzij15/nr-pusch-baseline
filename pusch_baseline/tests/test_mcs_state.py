"""Tests for continuous MCS-index feedback and configuration."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pusch_baseline.config import OllaSettings, load_config
from pusch_baseline.olla import McsIndexController


class McsStateTests(unittest.TestCase):
    def test_ack_and_nack_track_fractional_index_and_cross_exact_boundary(self):
        controller = McsIndexController(OllaSettings(), 14, range(5, 25))

        controller.observe(True, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.mcs_state, 13.1)
        self.assertEqual(controller.mcs_index, 13)
        for _ in range(9):
            controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.mcs_state, 14.0)
        self.assertEqual(controller.mcs_index, 14)

    def test_gapped_ladder_floors_to_highest_available_mcs_and_clips(self):
        controller = McsIndexController(OllaSettings(), 13, (14, 5, 9))

        self.assertEqual(controller.mcs_state, 13)
        self.assertEqual(controller.mcs_index, 9)
        for _ in range(11):
            controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertEqual(controller.mcs_state, 14)
        self.assertEqual(controller.mcs_index, 14)
        for _ in range(20):
            controller.observe(True, lowest_mcs=False, highest_mcs=False)
        self.assertEqual(controller.mcs_state, 5)
        self.assertEqual(controller.mcs_index, 5)

    def test_ladder_axis_passes_each_gap_in_one_feedback_step_unit(self):
        controller = McsIndexController(OllaSettings(mcs_axis="ladder"), 9, (8, 9, 11, 12))
        for _ in range(9):
            controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertEqual(controller.mcs_index, 9)
        controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertEqual(controller.mcs_index, 11)
        self.assertAlmostEqual(controller.mcs_state, 11)

    def test_summary_preserves_calibration_window_with_mcs_telemetry(self):
        settings = OllaSettings(warmup_tb=10, window_tb=10)
        controller = McsIndexController(settings, 14, range(5, 25))

        self.assertEqual(controller.summary()["calibration_status"], "insufficient_samples")
        for failed in [True] + [False] * 9:
            controller.observe(failed, lowest_mcs=False, highest_mcs=False)
        summary = controller.summary()

        self.assertEqual(summary["calibration_status"], "converged")
        self.assertEqual(summary["calibration_bler"], 0.1)
        self.assertEqual(summary["calibration_total_tb"], 10)
        self.assertAlmostEqual(summary["mcs_state"], 14.0)
        self.assertEqual(summary["next_mcs_index"], 14)
        self.assertNotIn("margin_offset_db", summary)

    def test_default_direct_config_and_legacy_margin_config_round_trip(self):
        cases = [
            ({"olla": {}, "mcs_indices": [5, 6, 7]}, "mcs_index", (5, 6, 7)),
            ({"olla": {}}, "sinr_margin", None),
            ({"olla": {"step_mcs": 0.5}}, "mcs_index", None),
            ({"olla": {"step_db": 2.0}}, "sinr_margin", None),
            ({"olla": {"control": "mcs_index", "step_db": 2.0}}, "mcs_index", None),
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for settings, expected_control, expected_indices in cases:
                with self.subTest(settings=settings):
                    path.write_text(json.dumps({"adaptation": settings}), encoding="utf-8")
                    cfg = load_config(path)
                    self.assertEqual(cfg.adaptation.olla.control, expected_control)
                    self.assertEqual(cfg.adaptation.mcs_indices, expected_indices)
                    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
                    self.assertEqual(load_config(path), cfg)

    def test_rejects_invalid_direct_ladders(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for indices in ([], [5, 5], [5, 29]):
                with self.subTest(indices=indices):
                    path.write_text(json.dumps({"adaptation": {"olla": {}, "mcs_indices": indices}}),
                                    encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "mcs_indices"):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
