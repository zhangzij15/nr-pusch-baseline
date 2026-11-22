"""Convergence checks for long OLLA simulations."""

import json
import unittest

from pusch_baseline.mcs_stability import assess_mcs_stability


class McsStabilityTests(unittest.TestCase):
    def assess(self, samples):
        return assess_mcs_stability(
            samples, window_tb=10, target_bler=0.1, bler_tolerance=0.05, max_mcs_spread=0.5,
        )

    def test_uses_last_three_disjoint_windows_and_accepts_a_spread_on_the_boundary(self):
        calibration = self.assess(
            [(4, True)] * 15
            + [(10, True)] + [(10, False)] * 9
            + [(10, False)] * 5 + [(11, False)] * 5
            + [(10, True)] * 2 + [(10, False)] * 8
        )

        self.assertTrue(calibration["stable"])
        self.assertEqual(calibration["status"], "stable")
        self.assertEqual(calibration["mcs_spread"], 0.5)
        self.assertEqual(calibration["pooled_bler"], 0.1)
        self.assertEqual(calibration["windows"], [
            {"mean_mcs": 10.0, "bler": 0.1},
            {"mean_mcs": 10.5, "bler": 0.0},
            {"mean_mcs": 10.0, "bler": 0.2},
        ])
        json.dumps(calibration, allow_nan=False)

    def test_nominal_bler_can_coexist_with_mcs_drift(self):
        calibration = self.assess([
            (mcs, k == 0)
            for mcs in (9, 10, 11)
            for k in range(10)
        ])

        self.assertFalse(calibration["stable"])
        self.assertEqual(calibration["status"], "mcs_drift")
        self.assertEqual(calibration["pooled_bler"], 0.1)
        self.assertEqual(calibration["mcs_spread"], 2.0)

    def test_constant_mcs_with_out_of_target_bler_is_rejected(self):
        calibration = self.assess([(24, False)] * 30)

        self.assertFalse(calibration["stable"])
        self.assertEqual(calibration["status"], "bler_outside_window")
        self.assertEqual(calibration["mcs_spread"], 0.0)
        self.assertEqual(calibration["pooled_bler"], 0.0)

    def test_requires_all_three_windows(self):
        calibration = self.assess([(9, False)] * 29)

        self.assertFalse(calibration["stable"])
        self.assertEqual(calibration["status"], "insufficient_samples")
        self.assertIsNone(calibration["pooled_bler"])
        self.assertIsNone(calibration["mcs_spread"])
        self.assertEqual(calibration["windows"], [])


if __name__ == "__main__":
    unittest.main()
