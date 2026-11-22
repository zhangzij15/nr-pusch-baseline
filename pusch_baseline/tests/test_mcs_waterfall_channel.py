"""Check physical replay and normalized SRS-selected precoders in a tiny bank."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pusch_baseline import load_config
from pusch_baseline.mcs_waterfall_channel import write_channel_bank


class McsWaterfallChannelTests(unittest.TestCase):
    def test_reordering_snr_preserves_channels_and_reorders_srs_weights(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks_long.json"
        cfg = load_config(config_path)
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, bs_panel=replace(cfg.channel.bs_panel, rows=2, cols=2)),
        )
        with TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            write_channel_bank(cfg, [-12, -6], 4, first)
            write_channel_bank(cfg, [-6, -12], 4, second)
            meta = json.loads((first / "bank.json").read_text(encoding="utf-8"))
            other_meta = json.loads((second / "bank.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["snrs"], [-12, -6])
            self.assertEqual(meta["num_tb"], 4)
            self.assertEqual(meta["physical_sha256"], other_meta["physical_sha256"])
            channel = np.load(first / "channel.npy")
            self.assertEqual(channel.shape, (4, 4, 4, 14, 48))
            np.testing.assert_array_equal(channel, np.load(second / "channel.npy"))
            np.testing.assert_array_equal(np.load(first / "slots.npy"), [2, 7, 12, 17])
            weights = np.load(first / "weights.npy")
            self.assertEqual(weights.shape, (2, 4, 4))
            np.testing.assert_array_equal(weights, np.load(second / "weights.npy")[::-1])
            np.testing.assert_allclose(np.sum(np.abs(weights)**2, axis=-1), 1.0, atol=1e-7)
            np.testing.assert_array_equal(weights[:, :1], weights[:, 1:2])
            np.testing.assert_array_equal(weights[:, 2:3], weights[:, 3:4])

    def test_power_mode_uses_paired_channel_with_the_same_sampled_yaws(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks_long.json"
        cfg = load_config(config_path)
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, ue_yaw_offsets_deg=(0, 90, 180, 270),
                            bs_panel=replace(cfg.channel.bs_panel, rows=2, cols=2)),
        )
        with TemporaryDirectory() as temp:
            full_dir = Path(temp) / "full"
            capped_dir = Path(temp) / "capped"
            full = write_channel_bank(cfg, [-12], 4, full_dir)
            capped_cfg = replace(cfg, adaptation=replace(cfg.adaptation, tx_power_mode="per_port"))
            capped = write_channel_bank(capped_cfg, [-12], 4, capped_dir)
            self.assertEqual(capped["srs_power_per_active_re"], 1.0)
            self.assertEqual(full["physical_sha256"], capped["physical_sha256"])
            power_by_port = np.abs(np.load(capped_dir / "weights.npy"))**2
            np.testing.assert_allclose(np.max(power_by_port, axis=-1), 0.25, atol=1e-7)
            self.assertAlmostEqual(capped["mean_data_re_tx_power"], np.sum(power_by_port, axis=-1).mean())
