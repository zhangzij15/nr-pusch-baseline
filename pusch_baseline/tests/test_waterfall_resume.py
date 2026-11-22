"""Read historical waterfall checkpoint metadata with equivalent defaults."""

from copy import deepcopy
import unittest

from run_mcs_waterfall import _restore_legacy_bank_defaults, _restore_legacy_manifest_defaults


class WaterfallResumeTests(unittest.TestCase):
    def test_legacy_settings_match_an_explicitly_equivalent_configuration(self):
        legacy = {
            "config": {"channel": {"estimator": "ls"}, "adaptation": {"olla": {"control": "mcs_index"}}},
            "num_tb_per_point": 512,
        }
        current = {
            "config": {
                "channel": {"estimator": "ls", "ue_yaw_offsets_deg": []},
                "adaptation": {"tx_power_mode": "full_power", "olla": {"control": "mcs_index", "mcs_axis": "index"}},
            },
            "num_tb_per_point": 512,
        }
        restored = _restore_legacy_manifest_defaults(deepcopy(legacy))
        self.assertEqual(restored, current)
        current["config"]["channel"]["estimator"] = "lmmse"
        self.assertNotEqual(restored, current)

    def test_legacy_bank_matches_unchanged_scaling_and_fingerprint(self):
        legacy = {"physical_sha256": "original", "mean_data_re_tx_power": 1.0}
        current = {
            **legacy, "tx_power_mode": "full_power", "ue_yaw_offsets_deg": [], "srs_power_per_active_re": 1.0,
        }
        self.assertEqual(_restore_legacy_bank_defaults(legacy), current)
        self.assertNotEqual(_restore_legacy_bank_defaults(legacy), {**current, "tx_power_mode": "per_port"})
        self.assertNotEqual(_restore_legacy_bank_defaults(legacy), {**current, "physical_sha256": "different"})

    def test_waterfall_snr_grid_supports_power_of_two_sub_db_steps(self):
        from pusch_baseline.mcs_waterfall import WATERFALL_SNRS, waterfall_snr_grid

        self.assertEqual(waterfall_snr_grid(1.0), WATERFALL_SNRS)
        grid = waterfall_snr_grid(0.25)
        self.assertEqual((grid[0], grid[1], grid[-1], len(grid)), (-36.0, -35.75, 12.0, 193))
        with self.assertRaises(ValueError):
            waterfall_snr_grid(0.3)
