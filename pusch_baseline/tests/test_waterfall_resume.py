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
            **legacy, "tx_power_mode": "full_power", "srs_estimator": "real",
            "ue_yaw_offsets_deg": [], "srs_power_per_active_re": 1.0,
        }
        self.assertEqual(_restore_legacy_bank_defaults(legacy), current)
        self.assertNotEqual(_restore_legacy_bank_defaults(legacy), {**current, "tx_power_mode": "per_port"})
        self.assertNotEqual(_restore_legacy_bank_defaults(legacy), {**current, "srs_estimator": "perfect"})
        self.assertNotEqual(_restore_legacy_bank_defaults(legacy), {**current, "physical_sha256": "different"})

    def test_waterfall_snr_grid_supports_power_of_two_sub_db_steps(self):
        from pusch_baseline.mcs_waterfall import WATERFALL_SNRS, waterfall_snr_grid

        self.assertEqual(waterfall_snr_grid(1.0), WATERFALL_SNRS)
        grid = waterfall_snr_grid(0.25)
        self.assertEqual((grid[0], grid[1], grid[-1], len(grid)), (-36.0, -35.75, 12.0, 193))
        with self.assertRaises(ValueError):
            waterfall_snr_grid(0.3)

    def test_fixed_grid_supports_partial_mcs_subsets_and_rejects_missing_rows(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from run_mcs_waterfall import _load_fixed_grid

        partial = {
            "points": [
                {"mcs_index": 9, "snr_db": -20.0},
                {"mcs_index": 9, "snr_db": -22.0},
                {"mcs_index": 10, "snr_db": -19.0},
            ],
        }
        with TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.json"
            summary_path.write_text(json.dumps(partial), encoding="utf-8")
            self.assertEqual(
                _load_fixed_grid(summary_path, required=(9, 10)),
                {9: (-22.0, -20.0), 10: (-19.0,)},
            )
            with self.assertRaisesRegex(ValueError, "missing requested MCS"):
                _load_fixed_grid(summary_path, required=(9, 11))
            with self.assertRaisesRegex(ValueError, "all 29 MCS rows"):
                _load_fixed_grid(summary_path)

    def test_tpmi_policy_reflects_srs_estimator_and_candidate_subset(self):
        from run_mcs_waterfall import _tpmi_policy

        self.assertEqual(
            _tpmi_policy({"srs_estimator": "real", "num_tpmi_candidates": 28}),
            "noisy SRS selects from the full set of rank-one codebook precoders",
        )
        self.assertEqual(
            _tpmi_policy({"srs_estimator": "perfect", "num_tpmi_candidates": 28}),
            "ideal noise-free SRS selects from the full set of rank-one codebook precoders",
        )
        self.assertEqual(
            _tpmi_policy({"srs_estimator": "real", "num_tpmi_candidates": 4, "tpmi_candidates": [0, 1, 2, 3]}),
            "noisy SRS selects from 4 configured rank-one codebook precoder(s)",
        )
