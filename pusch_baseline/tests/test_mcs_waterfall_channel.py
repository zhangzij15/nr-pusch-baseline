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
    def test_short_grants_decode_from_a_full_slot_channel_bank(self):
        from pusch_baseline.mcs_waterfall import simulate_mcs_curve

        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks.json")
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, kind="tdl", num_rx_ant=2, estimator="perfect"),
        )
        with TemporaryDirectory() as temp:
            bank = write_channel_bank(cfg, [30.0], 2, Path(temp))
            self.assertEqual(bank["shape"][-2], 14)
            for length in (7, 10):
                with self.subTest(length=length):
                    short = replace(cfg, pusch=replace(cfg.pusch, symbol_allocation=(0, length)))
                    point, = simulate_mcs_curve(short, temp, 5, 2, snr_points=[30.0])
                    self.assertEqual((point["num_tb"], point["tb_errors"]), (2, 0))

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

    def test_precoder_is_held_for_every_pusch_slot_of_one_srs_epoch(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks_long.json"
        cfg = load_config(config_path)
        # A 10-slot period on DSUDD serves two U slots (2 and 7) per SRS.
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, bs_panel=replace(cfg.channel.bs_panel, rows=2, cols=2)),
            adaptation=replace(cfg.adaptation, srs_period_slots=10),
        )
        with TemporaryDirectory() as temp:
            write_channel_bank(cfg, [-12, -6], 4, Path(temp))
            np.testing.assert_array_equal(np.load(Path(temp) / "slots.npy"), [2, 7, 12, 17])
            weights = np.load(Path(temp) / "weights.npy")
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

    def test_ideal_srs_bank_selects_noise_free_precoders_on_the_same_channels(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks_long.json"
        cfg = load_config(config_path)
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, bs_panel=replace(cfg.channel.bs_panel, rows=2, cols=2)),
        )
        with TemporaryDirectory() as temp:
            real_dir = Path(temp) / "real"
            ideal_dir = Path(temp) / "ideal"
            real = write_channel_bank(cfg, [-30, 10], 4, real_dir)
            ideal_cfg = replace(cfg, adaptation=replace(cfg.adaptation, srs_estimator="perfect"))
            ideal = write_channel_bank(ideal_cfg, [-30, 10], 4, ideal_dir)
            self.assertEqual((real["srs_estimator"], ideal["srs_estimator"]), ("real", "perfect"))
            np.testing.assert_array_equal(np.load(real_dir / "channel.npy"), np.load(ideal_dir / "channel.npy"))
            # Noise-free sounding picks the same precoder at every SNR.
            weights = np.load(ideal_dir / "weights.npy")
            np.testing.assert_array_equal(weights[0], weights[1])

    def test_configured_rank_one_tpmi_subset_restricts_selected_precoders(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/cdl_c_4t64r_codebooks_long.json"
        cfg = load_config(config_path)
        cfg = replace(
            cfg,
            carrier=replace(cfg.carrier, n_size_grid=4),
            channel=replace(cfg.channel, bs_panel=replace(cfg.channel.bs_panel, rows=2, cols=2)),
            adaptation=replace(cfg.adaptation, rank_tpmi_candidates=((1, 0), (1, 1), (1, 2), (1, 3))),
        )
        with TemporaryDirectory() as temp:
            meta = write_channel_bank(cfg, [-12, 0], 4, Path(temp) / "noncoherent")
            self.assertEqual(meta["num_tpmi_candidates"], 4)
            self.assertEqual(meta["tpmi_candidates"], [0, 1, 2, 3])
            weights = np.load(Path(temp) / "noncoherent" / "weights.npy")
            # Non-coherent Rank-1 TPMIs 0..3 activate exactly one TX port per slot.
            active_ports = np.count_nonzero(np.abs(weights) > 0, axis=-1)
            np.testing.assert_array_equal(active_ports, 1)

            rank2_cfg = replace(
                cfg,
                channel=replace(cfg.channel, estimator="ls"),
                adaptation=replace(cfg.adaptation, rank_tpmi_candidates=((1, 0), (2, 0))),
            )
            with self.assertRaises(ValueError):
                write_channel_bank(rank2_cfg, [-12], 2, Path(temp) / "rank2")
