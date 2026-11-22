"""Tests for target-BLER calibration and its configuration."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pusch_baseline.config import AdaptationSettings, OllaSettings, SimulationConfig, load_config
from pusch_baseline.olla import OllaController


class OllaControllerTests(unittest.TestCase):
    def test_failure_raises_margin_and_success_lowers_it_with_clipping(self):
        controller = OllaController(OllaSettings(step_db=2.0, max_offset_db=2.0))

        controller.observe(True, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.margin_offset_db, 1.8)
        controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.margin_offset_db, 1.6)
        controller.observe(True, lowest_mcs=False, highest_mcs=False)
        self.assertEqual(controller.margin_offset_db, 2.0)

        for _ in range(30):
            controller.observe(False, lowest_mcs=True, highest_mcs=False)
        self.assertEqual(controller.margin_offset_db, -2.0)

    def test_summary_checks_the_recent_window_after_warmup(self):
        controller = OllaController(OllaSettings(warmup_tb=20, window_tb=10))
        self.assertEqual(controller.summary()["calibration_status"], "insufficient_samples")
        self.assertIsNone(controller.summary()["calibration_bler"])

        for _ in range(10):
            controller.observe(True, lowest_mcs=True, highest_mcs=False)
        self.assertEqual(controller.summary()["calibration_status"], "insufficient_samples")

        for failed in [True] + [False] * 9:
            controller.observe(failed, lowest_mcs=False, highest_mcs=False)
        summary = controller.summary()

        self.assertEqual(summary["calibration_status"], "converged")
        self.assertEqual(summary["calibration_bler"], 0.1)
        self.assertEqual(summary["calibration_num_tb"], 10)
        self.assertEqual(summary["calibration_total_tb"], 20)
        self.assertEqual(summary["margin_offset_db"], controller.margin_offset_db)

    def test_reports_ceiling_and_out_of_target_windows(self):
        settings = OllaSettings(warmup_tb=10, window_tb=10)

        for errors, lowest, highest, expected in [
            (0, False, True, "at_mcs_ceiling"),
            (0, False, False, "below_target"),
            (3, True, False, "above_target"),
        ]:
            with self.subTest(errors=errors, highest=highest):
                controller = OllaController(settings)
                for tb in range(10):
                    controller.observe(tb < errors, lowest_mcs=lowest, highest_mcs=highest)

                summary = controller.summary()
                self.assertEqual(summary["calibration_status"], expected)
                self.assertEqual(summary["calibration_all_lowest_mcs"], lowest)
                self.assertEqual(summary["calibration_all_highest_mcs"], highest)


class OllaSettingsTests(unittest.TestCase):
    def test_loads_optional_nested_settings_and_keeps_existing_configs(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.json"
            for raw, expected in [
                ({}, None),
                ({"adaptation": {"olla": None}}, None),
                ({"adaptation": {"olla": {"warmup_tb": 30, "window_tb": 10}}},
                 OllaSettings(warmup_tb=30, window_tb=10, control="sinr_margin")),
            ]:
                with self.subTest(raw=raw):
                    path.write_text(json.dumps(raw), encoding="utf-8")
                    cfg = load_config(path)
                    self.assertEqual(cfg.adaptation.olla if cfg.adaptation else None, expected)

    def test_rejects_invalid_targets_bounds_and_windows(self):
        settings = OllaSettings()
        for invalid in [
            replace(settings, target_bler=0.0),
            replace(settings, target_bler=1.0),
            replace(settings, warmup_tb=0),
            replace(settings, window_tb=settings.warmup_tb + 1),
            replace(settings, tolerance=-0.01),
            replace(settings, step_db=0.0),
            replace(settings, max_offset_db=float("inf")),
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                invalid.validate()

    def test_rejects_early_stop_when_olla_is_enabled(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            sweep=replace(base.sweep, target_block_errors=1),
            adaptation=AdaptationSettings(olla=OllaSettings()),
        )

        with self.assertRaisesRegex(ValueError, "target_block_errors"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
