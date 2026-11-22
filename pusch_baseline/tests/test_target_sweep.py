"""Regression checks for shared fading and post-calibration measurements."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings, OllaSettings


class TargetSweepTests(unittest.TestCase):
    def test_adaptive_snr_points_share_one_cir_trajectory(self):
        import tensorflow as tf

        base = SimulationConfig()
        cfg = replace(
            base, adaptation=AdaptationSettings(),
            channel=replace(base.channel, kind="custom", normalize=False),
            sweep=replace(base.sweep, snr_db=(-25.0, 25.0), num_tb_per_snr=3),
        )
        cir_calls = []

        def factory(_cfg, _grid):
            def channel(batch_size, samples, _frequency):
                cir_calls.append(samples)
                gains = tf.ones((batch_size, 1, 1, 1, 1, 1, samples), dtype=tf.complex64)
                return gains, tf.zeros((batch_size, 1, 1, 1))

            return channel

        result = simulate(cfg, channel_model_factory=factory)
        self.assertEqual(cir_calls, [56])
        self.assertEqual([p["tb_errors"] for p in result["points"]], [3, 0])

    def test_target_sweep_uses_disjoint_training_and_measurement_windows(self):
        from pusch_baseline.config import OllaSettings

        base = SimulationConfig()
        cfg = replace(
            base,
            adaptation=AdaptationSettings(
                srs_period_slots=4, mcs_thresholds_db=((5, 2.0),),
                olla=OllaSettings(warmup_tb=10, window_tb=10),
            ),
            sweep=replace(base.sweep, snr_db=(-30.0, 30.0), num_tb_per_snr=10),
        )
        low, high = simulate(cfg)["points"]
        self.assertEqual((low["num_tb"], high["num_tb"]), (10, 10))
        self.assertEqual((low["calibration_total_tb"], high["calibration_total_tb"]), (10, 10))
        self.assertEqual((low["num_slots"], high["num_slots"]), (14, 14))
        self.assertEqual(low["target_status"], "calibration_not_converged")
        self.assertIsNone(low["thp_at_target_mbps"])
        self.assertEqual(high["target_status"], "at_mcs_ceiling")
        self.assertIsNone(high["thp_at_target_mbps"])
        self.assertGreater(high["thp_duty_cycle_mbps"], 0)

    def test_throughput_requires_both_target_windows(self):
        from pusch_baseline.adaptive_simulation import _add_target_result
        from pusch_baseline.config import OllaSettings
        from pusch_baseline.olla import OllaController

        settings = OllaSettings(warmup_tb=10, window_tb=10)
        controller = OllaController(settings)
        for index in range(10):
            controller.observe(index == 0, lowest_mcs=False, highest_mcs=False)
        point = {"bler": 0.1, "num_tb": 10, "grant_counts": [{"mcs_index": 5}],
                 "thp_duty_cycle_mbps": 3.0}
        _add_target_result(point, controller.summary(), settings, highest_mcs=14)
        self.assertEqual(point["target_status"], "at_target")
        self.assertEqual(point["thp_at_target_mbps"], 3.0)

        point["bler"] = 0.2
        _add_target_result(point, controller.summary(), settings, highest_mcs=14)
        self.assertEqual(point["target_status"], "above_target")
        self.assertIsNone(point["thp_at_target_mbps"])

    def test_ceiling_calibration_is_preserved_when_evaluation_is_inside_window(self):
        from pusch_baseline.adaptive_simulation import _add_target_result

        point = {"bler": 0.1, "num_tb": 10, "grant_counts": [{"mcs_index": 14}],
                 "thp_duty_cycle_mbps": 3.0}
        _add_target_result(point, {"calibration_status": "at_mcs_ceiling"},
                           OllaSettings(window_tb=10), highest_mcs=14)
        self.assertEqual(point["target_status"], "at_mcs_ceiling")
        self.assertIsNone(point["thp_at_target_mbps"])

    def test_stability_gate_uses_warmup_and_independent_measurement(self):
        from pusch_baseline.link import PuschLink

        cfg = SimulationConfig()
        cfg = replace(
            cfg,
            sweep=replace(cfg.sweep, snr_db=(-5.0, 5.0), num_tb_per_snr=30),
            pusch=replace(cfg.pusch, mcs_index=14),
            adaptation=AdaptationSettings(
                mcs_indices=(13, 14, 15),
                olla=OllaSettings(warmup_tb=40, window_tb=10, tolerance=0.04, max_mcs_spread=1.0),
            ),
        )
        attempts = {False: 0, True: 0}

        def feedback(_link, _batch_size, noise, *, channel_response):
            weak = float(noise) > 1
            attempts[weak] += 1
            return int(weak or attempts[weak] % 10 == 0)

        with patch.object(PuschLink, "count_block_errors", feedback):
            run = simulate(cfg)
        weak, strong = run["points"]
        self.assertEqual(weak["target_status"], "at_mcs_floor")
        self.assertIsNone(weak["thp_at_target_mbps"])
        self.assertEqual(strong["target_status"], "at_target")
        self.assertGreater(strong["thp_at_target_mbps"], 0)
        self.assertEqual(strong["calibration_total_tb"], 40)
        self.assertEqual(strong["num_tb"], 30)
        self.assertEqual(strong["calibration_bler"], 0.1)
        self.assertEqual(strong["measurement_stability"]["pooled_bler"], 0.1)
        self.assertTrue(strong["calibration_stability"]["stable"])
        self.assertTrue(strong["measurement_stability"]["stable"])
        hashes = run["metadata"]["channel_fingerprints_sha256"]
        self.assertNotEqual(hashes["warmup"], hashes["measurement"])
