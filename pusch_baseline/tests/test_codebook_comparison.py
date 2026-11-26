"""End-to-end regression for paired fixed-rank codebook sweeps."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from pusch_baseline import SimulationConfig


class CodebookComparisonTests(unittest.TestCase):
    def test_codebooks_share_one_cir_and_restrict_the_selected_tpmi(self):
        import tensorflow as tf

        from pusch_baseline.config import AdaptationSettings
        from pusch_baseline.link import PuschLink
        from pusch_baseline.simulation import simulate, simulate_codebook_comparison

        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4),
            pusch=replace(base.pusch, num_antenna_ports=4, precoding="codebook"),
            channel=replace(base.channel, kind="custom", normalize=False, estimator="perfect"),
            sweep=replace(base.sweep, snr_db=(25.0,), num_tb_per_snr=1, seed=144),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDD", srs_period_slots=5, sinr_margin_db=0.0,
                mcs_thresholds_db=((5, 0.0), (9, 27.0), (14, 29.5)),
            ),
        )
        calls = []

        def factory(_config, _grid):
            def channel(batch_size, num_symbols, _sample_rate):
                calls.append(num_symbols)
                gains = tf.ones((batch_size, 1, 1, 1, 4, 1, num_symbols), tf.complex64)
                return gains, tf.zeros((batch_size, 1, 1, 1))

            return channel

        payloads = []
        noises = []
        count_errors = PuschLink.count_block_errors

        def record(link, batch_size, no, **kwargs):
            with (patch.object(link, "_receiver", wraps=link._receiver) as rx,
                  patch.object(link, "_transmitter", wraps=link._transmitter) as tx,
                  patch.object(link, "_apply_ofdm", wraps=link._apply_ofdm) as channel):
                errors = count_errors(link, batch_size, no, **kwargs)
            clean = link._apply_ofdm(*channel.call_args.args[:2])
            noises.append((rx.call_args.args[0] - clean).numpy() / np.sqrt(float(no)))
            payloads.append(tx.call_args.args[0].numpy().ravel())
            return errors

        with patch.object(PuschLink, "count_block_errors", record):
            runs = simulate_codebook_comparison(cfg, channel_model_factory=factory)
            self.assertEqual(calls, [28])
            # Re-run one independent ICE shard at a different SNR with only
            # the smallest TB size. It must use the same slot noise and bits.
            single = replace(cfg, sweep=replace(cfg.sweep, snr_db=(26.0,)),
                             adaptation=replace(cfg.adaptation, srs_estimator="perfect",
                                                mcs_thresholds_db=((5, 0.),)))
            simulate(single, channel_model_factory=factory)
        self.assertEqual(len(noises), 4)
        longest = max(payloads, key=len)
        self.assertEqual(len(set(map(len, payloads))), 3)
        for bits, noise in zip(payloads, noises):
            np.testing.assert_array_equal(bits, longest[:len(bits)])
            np.testing.assert_allclose(noise, noises[0], rtol=5e-6, atol=5e-6)
        self.assertAlmostEqual(float(np.mean(np.abs(noises[0])**2)), 1., delta=.15)
        for name, expected_mcs, num_tpmi in (
            ("full", 14, 28), ("partial_noncoherent", 9, 12), ("noncoherent", 5, 4),
        ):
            with self.subTest(name=name):
                run = runs[name]
                point, = run["points"]
                grant, = point["srs_grants"]
                self.assertEqual(run["metadata"]["codebook_tpmi"], list(range(num_tpmi)))
                self.assertEqual((grant["slot_index"], grant["rank"], grant["mcs_index"]),
                                 (1, 1, expected_mcs))
                self.assertIn(grant["tpmi"], range(num_tpmi))
                self.assertEqual((point["num_tb"], point["num_slots"]), (1, 3))

    def test_pusch_noise_and_bits_vary_across_slots_and_replay_by_slot(self):
        import tensorflow as tf
        from pusch_baseline.adaptive_simulation import _slot_pusch_inputs

        def inputs(slot):
            return _slot_pusch_inputs(2**32 - 1, slot, 100, (1, 1, 2, 7, 48), tf.float64)

        first = inputs(2)
        second = inputs(7)
        replay = inputs(2)
        for a, b, c in zip(first, second, replay):
            self.assertFalse(np.array_equal(a, b))
            np.testing.assert_array_equal(a, c)
