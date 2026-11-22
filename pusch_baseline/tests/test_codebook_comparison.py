"""End-to-end regression for paired fixed-rank codebook sweeps."""

from dataclasses import replace
import unittest

from pusch_baseline import SimulationConfig


class CodebookComparisonTests(unittest.TestCase):
    def test_codebooks_share_one_cir_and_restrict_the_selected_tpmi(self):
        import tensorflow as tf

        from pusch_baseline.config import AdaptationSettings
        from pusch_baseline.simulation import simulate_codebook_comparison

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

        runs = simulate_codebook_comparison(cfg, channel_model_factory=factory)
        self.assertEqual(calls, [28])
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
