"""Tests for continuous cross-SRS-period CIR evolution (CDL, TDL, and custom)."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tensorflow as tf
from sionna.phy import config as sionna_config
from sionna.phy.channel.tr38901 import CDL, TDL, PanelArray

from pusch_baseline import SimulationConfig, load_config, simulate
from pusch_baseline.adaptive_simulation import SoundingChannel
from pusch_baseline.config import AdaptationSettings, OllaSettings
from pusch_baseline.continuous_channel import ContinuousCir
from pusch_baseline.simulation import simulate_codebook_comparison
from pusch_baseline.srs import SrsEstimator


class ContinuousChannelTests(unittest.TestCase):
    def test_continuous_cdl_matches_single_long_sionna_trajectory(self):
        ut = PanelArray(num_rows_per_panel=2, num_cols_per_panel=2, polarization="single",
                        polarization_type="V", antenna_pattern="omni", carrier_frequency=6.7e9,
                        element_vertical_spacing=0.5, element_horizontal_spacing=0.5, precision="single")
        bs = PanelArray(num_rows_per_panel=2, num_cols_per_panel=2, polarization="single",
                        polarization_type="V", antenna_pattern="38.901", carrier_frequency=6.7e9,
                        element_vertical_spacing=1.5, element_horizontal_spacing=0.5, precision="single")
        fs = 28_000.0
        for profile in ("C", "D"):
            with self.subTest(profile=profile):
                cdl = CDL(profile, 100e-9, 6.7e9, ut, bs, "uplink",
                          min_speed=30 / 3.6, max_speed=30 / 3.6, precision="single")
                seed = 20261112
                # Reference: run Sionna CDL once with the same isolated channel seed.
                caller = sionna_config.tf_rng.state.read_value()
                sionna_config.tf_rng.reset(
                    tf.random.Generator.from_seed(seed ^ 0x9E3779B9, alg=sionna_config.tf_rng.algorithm).state
                )
                try:
                    a_ref, tau_ref = cdl(1, 210, fs)
                finally:
                    sionna_config.tf_rng.reset(caller)

                stream = ContinuousCir(cdl, seed)
                for chunk in (range(0, 14), range(70, 84), range(180, 210)):
                    a_chunk, tau_chunk = stream.sample(list(chunk), fs)
                    np.testing.assert_allclose(a_chunk.numpy(), a_ref[..., chunk.start:chunk.stop].numpy(),
                                               atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(tau_chunk.numpy(), tau_ref.numpy(), atol=1e-7)

    def test_continuous_tdl_matches_single_long_sionna_trajectory(self):
        fs = 28_000.0
        for profile in ("C", "D"):
            with self.subTest(profile=profile):
                tdl = TDL(profile, 100e-9, 3.5e9, min_speed=30 / 3.6, max_speed=30 / 3.6,
                          num_rx_ant=2, num_tx_ant=2, precision="single")
                seed = 12345
                caller = sionna_config.tf_rng.state.read_value()
                sionna_config.tf_rng.reset(
                    tf.random.Generator.from_seed(seed ^ 0x9E3779B9, alg=sionna_config.tf_rng.algorithm).state
                )
                try:
                    a_ref, tau_ref = tdl(1, 180, fs)
                finally:
                    sionna_config.tf_rng.reset(caller)

                stream = ContinuousCir(tdl, seed)
                for chunk in (range(0, 14), range(98, 112), range(150, 180)):
                    a_chunk, tau_chunk = stream.sample(list(chunk), fs)
                    np.testing.assert_allclose(a_chunk.numpy(), a_ref[..., chunk.start:chunk.stop].numpy(),
                                               atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(tau_chunk.numpy(), tau_ref.numpy(), atol=1e-7)

    def test_continuous_sounding_is_temporally_correlated_across_srs_periods(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4, subcarrier_spacing_khz=30),
            pusch=replace(base.pusch, num_antenna_ports=4, rank=1, precoding="codebook", tpmi=0),
            channel=replace(
                base.channel, kind="cdl", profile="C", carrier_frequency_ghz=6.7, ue_speed_kmh=3.0,
                normalize=False,
                ue_panel=replace(base.channel.ue_panel, rows=2, cols=2),
                bs_panel=replace(base.channel.bs_panel, rows=2, cols=2),
            ),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDD", srs_period_slots=5, srs_start_symbol=10,
                channel_evolution="continuous",
            ),
        )
        est = SrsEstimator(4, 4, 0, start_symbol=10)
        continuous = SoundingChannel(cfg, None, est)
        h_slot1 = np.asarray(next(continuous.slots((0, 1, 6), start_slot=1)))
        h_slot11 = np.asarray(next(continuous.slots((0, 1, 6), start_slot=11)))
        # 10 slots = 5 ms at 3 km/h, 6.7 GHz (f_D ~ 18.6 Hz -> J0(2*pi*f_D*5ms) ~ 0.92).
        corr_cont = abs(np.vdot(h_slot1, h_slot11)) / (np.linalg.norm(h_slot1) * np.linalg.norm(h_slot11))
        self.assertGreater(corr_cont, 0.75)

        per_period = SoundingChannel(
            replace(cfg, adaptation=replace(cfg.adaptation, channel_evolution="per_period")),
            None, est,
        )
        p1 = np.asarray(next(per_period.slots((0, 1, 6), start_slot=1)))
        p2 = np.asarray(next(per_period.slots((0, 1, 6), start_slot=11)))
        corr_indep = abs(np.vdot(p1, p2)) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        self.assertLess(corr_indep, corr_cont)

    def test_continuous_tdd_simulation_spans_warmup_and_measurement_without_reset(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4),
            pusch=replace(base.pusch, num_antenna_ports=4, rank=1, precoding="codebook", mcs_index=14),
            channel=replace(base.channel, kind="custom", num_rx_ant=1, normalize=False, estimator="perfect"),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDD", srs_period_slots=5, srs_start_symbol=10,
                channel_evolution="continuous",
                mcs_indices=(5, 14),
                olla=OllaSettings(control="mcs_index", warmup_tb=2, window_tb=2),
            ),
            sweep=replace(base.sweep, snr_db=(25.0,), num_tb_per_snr=4, seed=42),
        )
        seen_calls = []

        def factory(_cfg, _grid):
            def cir(batch_size, num_time_steps, _fs):
                seen_calls.append((batch_size, num_time_steps))
                # Slot 0..9 (warmup) + slot 10..27 (measurement) = 28 slots -> 392 symbols.
                # Make port 0 active in warmup + first meas epoch (slots < 20), port 3 after (slots >= 20).
                slot = tf.range(num_time_steps) // 14
                p0 = tf.cast(slot < 20, tf.complex64)
                p3 = tf.cast(slot >= 20, tf.complex64)
                zeros = tf.zeros_like(p0)
                gains = tf.stack([p0, zeros, zeros, p3], axis=0)
                return tf.reshape(gains, (1, 1, 1, 1, 4, 1, num_time_steps)), tf.zeros((1, 1, 1, 1))

            return cir

        runs = simulate_codebook_comparison(cfg, channel_model_factory=factory)
        # Single continuous CIR draw across both warmup and measurement.
        self.assertEqual(len(seen_calls), 1)
        self.assertEqual(runs["full"]["metadata"]["adaptation"]["channel_evolution"], "continuous")
        full_point, = runs["full"]["points"]
        # Measurement epoch 0 is at physical slot 11 (< 20 -> port 0, TPMI 0); epoch 1 at slot 21 (>= 20 -> port 3, TPMI 3).
        self.assertEqual([g["tpmi"] for g in full_point["srs_grants"]], [0, 3])
        self.assertEqual(full_point["tb_errors"], 0)

    def test_continuous_normalize_holds_single_drop_rms_across_periods(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4),
            pusch=replace(base.pusch, num_antenna_ports=1, rank=1),
            channel=replace(base.channel, kind="custom", num_rx_ant=1, normalize=True),
            adaptation=AdaptationSettings(srs_period_slots=4, channel_evolution="continuous"),
        )

        def factory(_cfg, _grid):
            def cir(_batch, num_steps, _fs):
                # Slot 0 has amplitude 2.0 (RMS = 2.0); slot 4 has amplitude 4.0 -> normalized should be 2.0, not 1.0!
                slot = tf.range(num_steps) // 14
                amp = tf.where(slot < 4, 2.0, 4.0)
                gains = tf.cast(tf.reshape(amp, (1, 1, 1, 1, 1, 1, num_steps)), tf.complex64)
                return gains, tf.zeros((1, 1, 1, 1))

            return cir

        est = SrsEstimator(1, 4, 0)
        stream = SoundingChannel(cfg, factory, est, total_slots=8)
        s0 = np.asarray(next(stream.slots((0, 1), start_slot=0)))
        s4 = np.asarray(next(stream.slots((0, 1), start_slot=4)))
        np.testing.assert_allclose(np.abs(s0), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.abs(s4), 2.0, atol=1e-6)

    def test_config_round_trips_channel_evolution_and_omits_default(self):
        base = SimulationConfig(adaptation=AdaptationSettings())
        self.assertNotIn("channel_evolution", base.to_dict()["adaptation"])
        cont = SimulationConfig(adaptation=AdaptationSettings(channel_evolution="continuous"))
        self.assertEqual(cont.to_dict()["adaptation"]["channel_evolution"], "continuous")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "cfg.json"
            path.write_text(json.dumps(cont.to_dict()), encoding="utf-8")
            self.assertEqual(load_config(path), cont)


if __name__ == "__main__":
    unittest.main()
