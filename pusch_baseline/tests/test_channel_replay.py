"""Paired channel trajectories for the fixed-grant SNR sweep."""

from dataclasses import replace
import unittest

from pusch_baseline import SimulationConfig, simulate


class ChannelReplayTests(unittest.TestCase):
    def test_cir_rewinds_without_advancing_the_callers_random_stream(self):
        from sionna.phy import config
        from pusch_baseline.link import PairedCir

        rng = config.tf_rng
        caller_state = rng.state.read_value()
        self.addCleanup(rng.reset, caller_state)

        def model(batch_size, num_symbols, _sample_rate):
            return rng.uniform((batch_size, num_symbols)).numpy().tolist()

        replay = PairedCir(model, 144)
        first = replay(1, 4, 30000.0)
        second = replay(1, 4, 30000.0)
        self.assertNotEqual(first, second)
        self.assertEqual(rng.state.numpy().tolist(), caller_state.numpy().tolist())

        rng.uniform((9,))
        advanced_state = rng.state.numpy().tolist()
        replay.rewind()
        self.assertEqual(replay(1, 4, 30000.0), first)
        self.assertEqual(replay(1, 4, 30000.0), second)
        self.assertEqual(rng.state.numpy().tolist(), advanced_state)

    def test_fixed_grant_replays_same_cir_batches_at_each_snr(self):
        import tensorflow as tf
        from sionna.phy import config

        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=8),
            channel=replace(base.channel, kind="custom", normalize=False),
            sweep=replace(base.sweep, snr_db=(25.0, 30.0), num_tb_per_snr=2, batch_size=1),
        )
        samples = []

        def factory(_cfg, _grid):
            def model(batch_size, num_symbols, _sample_rate):
                gain = config.tf_rng.uniform((batch_size,), minval=0.8, maxval=1.2)
                samples.append(gain.numpy().tolist())
                gains = tf.cast(tf.reshape(gain, (-1, 1, 1, 1, 1, 1, 1)), tf.complex64)
                paths = tf.broadcast_to(gains, (batch_size, 1, 1, 1, 1, 1, num_symbols))
                return paths, tf.zeros((batch_size, 1, 1, 1))

            return model

        result = simulate(cfg, channel_model_factory=factory)
        self.assertNotEqual(samples[0], samples[1])
        self.assertEqual(samples[:2], samples[2:])
        self.assertEqual([point["num_tb"] for point in result["points"]], [2, 2])

    def test_cdl_yaw_changes_between_trajectories_and_replays_with_the_cir(self):
        import numpy as np

        from pusch_baseline.link import PairedCir, build_cir_model

        base = SimulationConfig()
        fixed_channel = replace(
            base.channel, kind="cdl", ue_panel=replace(base.channel.ue_panel, pattern="38.901"),
        )
        cfg = replace(base, channel=replace(fixed_channel, ue_yaw_offsets_deg=(0, 90, 180, 270)))
        replay = PairedCir(build_cir_model(cfg, None, None), cfg.sweep.seed)
        fixed = PairedCir(build_cir_model(replace(cfg, channel=fixed_channel), None, None), cfg.sweep.seed)
        moving_samples = [replay(1, 3, 30e3)[0].numpy() for _ in range(8)]
        fixed_samples = [fixed(1, 3, 30e3)[0].numpy() for _ in range(8)]
        matching = [np.array_equal(moving, stationary) for moving, stationary in zip(moving_samples, fixed_samples)]
        self.assertIn(False, matching)
        self.assertIn(True, matching)
        replay.rewind()
        for sample in moving_samples:
            np.testing.assert_array_equal(replay(1, 3, 30e3)[0].numpy(), sample)
