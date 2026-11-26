"""Check graph execution against eager PUSCH and adaptive slot results."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
from sionna.phy.nr import PUSCHReceiver, PUSCHTransmitter

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings, OllaSettings
from pusch_baseline.link import PuschLink, _pusch_config
from pusch_baseline.low_rate_tb import low_rate_plan


class AdaptivePhyGraphTests(unittest.TestCase):
    def test_explicit_inputs_match_eager_native_and_rv0_links(self):
        from pusch_baseline.link import make_adaptive_phy_graph_cache

        base = SimulationConfig()
        cache = make_adaptive_phy_graph_cache()
        decoder_cache = {}
        antenna_channels = tf.constant(
            [1 + .2j, -.8 + .5j, .3 - .4j, .6 + .2j, .4 + .3j, .9 - .3j, -.7 + .3j, .9 + .8j],
            tf.complex64,
        )
        response = tf.broadcast_to(tf.reshape(antenna_channels, (1, 1, 2, 1, 4, 1, 1)), (1, 1, 2, 1, 4, 14, 48))
        noise_shape = (1, 1, 2, 14, 48)
        noise = tf.complex(
            tf.random.stateless_normal(noise_shape, (170, 47)),
            tf.random.stateless_normal(noise_shape, (310, 90)),
        ) / tf.constant(2**0.5, tf.complex64)

        cases = (
            (0, "lmmse", 1, "single", 3, 12),
            (14, "perfect", 1, "single", 3, 12),
            (14, "perfect", 1, "single", 7, 0),
            (14, "ls", 2, "double", 3, 0),
        )
        for mcs, estimator, rank, precision, slot, tpmi in cases:
            with self.subTest(mcs=mcs, estimator=estimator, rank=rank, slot=slot, tpmi=tpmi):
                cfg = replace(
                    base,
                    carrier=replace(base.carrier, n_size_grid=4, slot_number=slot),
                    pusch=replace(base.pusch, mcs_index=mcs, rank=rank,
                                  num_antenna_ports=4, precoding="codebook", tpmi=tpmi),
                    channel=replace(base.channel, kind="custom", num_rx_ant=2, estimator=estimator),
                    precision=precision,
                )
                self.assertEqual(low_rate_plan(_pusch_config(cfg)) is not None, mcs == 0)
                eager = PuschLink(cfg, external_channel=True, tb_decoder_cache=decoder_cache)
                graph = PuschLink(cfg, external_channel=True, tb_decoder_cache=decoder_cache, phy_graph_cache=cache)
                rdtype = tf.float64 if precision == "double" else tf.float32
                cdtype = tf.complex128 if precision == "double" else tf.complex64
                channel_response = tf.cast(response, cdtype)
                standard_noise = tf.cast(noise, cdtype)
                payload = tf.cast(tf.random.stateless_uniform(
                    (1, 1, eager.tb_size_bits), (110, 40), minval=0, maxval=2, dtype=tf.int32,
                ), rdtype)
                eager_signal = eager.noiseless_signal(payload, channel_response)
                shared_signal = graph.noiseless_signal(payload, channel_response)
                np.testing.assert_allclose(shared_signal, eager_signal, rtol=1e-6, atol=1e-7)

                for snr, expected in ((35.0, 0), (-40.0, 1)):
                    no = graph.noise_variance(snr)
                    inputs = dict(channel_response=channel_response, payload=payload, standard_noise=standard_noise)
                    self.assertEqual(eager.count_block_errors(1, no, **inputs), expected)
                    self.assertEqual(graph.count_block_errors(1, no, **inputs), expected)
                    self.assertEqual(graph.count_block_errors(1, no, noiseless_signal=shared_signal, **inputs), expected)

    def test_adaptive_sweep_uses_graph_and_preserves_slot_decisions(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4, subcarrier_spacing_khz=15),
            pusch=replace(base.pusch, num_antenna_ports=4, precoding="codebook", mcs_index=5),
            channel=replace(base.channel, kind="tdl", estimator="perfect"),
            sweep=replace(base.sweep, snr_db=(-40.0, 35.0), num_tb_per_snr=3),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDDDDDDD", srs_period_slots=10, channel_evolution="continuous",
                mcs_indices=(5, 14), olla=OllaSettings(warmup_tb=3, window_tb=1, max_mcs_spread=1.0),
            ),
        )
        expected = simulate(cfg)
        self.assertEqual([point["tb_errors"] for point in expected["points"]], [3, 0])
        inside_graph = set()

        def record_execution(component, call):
            def execute(block, *args, **kwargs):
                if tf.inside_function():
                    inside_graph.add(component)
                return call(block, *args, **kwargs)

            return execute

        with (
            patch.object(PUSCHTransmitter, "call", record_execution("tx", PUSCHTransmitter.call)),
            patch.object(PUSCHReceiver, "call", record_execution("rx", PUSCHReceiver.call)),
        ):
            actual = simulate(cfg, compile_phy=True)

        self.assertEqual(inside_graph, {"tx", "rx"})
        self.assertEqual(actual["points"], expected["points"])
        self.assertEqual(actual["metadata"]["channel_fingerprints_sha256"],
                         expected["metadata"]["channel_fingerprints_sha256"])
