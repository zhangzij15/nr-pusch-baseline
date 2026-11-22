"""Verify the decoder graph can be reused across slots and spatial grants."""

from dataclasses import replace
import unittest

import numpy as np
import tensorflow as tf

from pusch_baseline.config import SimulationConfig
from pusch_baseline.link import PuschLink


class CompiledTbDecoderTests(unittest.TestCase):
    def test_shared_graph_matches_native_crc_and_successful_payload(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=6, cell_id=41),
            pusch=replace(base.pusch, mcs_index=9, num_antenna_ports=4, precoding="codebook"),
        )
        cache = {}
        native = PuschLink(cfg, external_channel=True)
        first = PuschLink(cfg, external_channel=True, tb_decoder_cache=cache)
        shifted = replace(cfg, carrier=replace(cfg.carrier, slot_number=3),
                          pusch=replace(cfg.pusch, tpmi=12))
        other_slot = PuschLink(shifted, external_channel=True, tb_decoder_cache=cache)
        eager = native._receiver._tb_decoder
        compiled = first._receiver._tb_decoder
        self.assertIs(compiled, other_slot._receiver._tb_decoder)
        self.assertEqual(len(cache), 1)

        payload = tf.random.stateless_uniform((1, 1, native.tb_size_bits), seed=(6, 12),
                                              minval=0, maxval=2, dtype=tf.int32)
        encoded = native._transmitter._tb_encoder(tf.cast(payload, tf.float32))
        good_llrs = 16 * (2 * encoded - 1)
        noisy_llrs = tf.random.stateless_normal(tf.shape(encoded), seed=(3, 2))
        data = tf.concat((good_llrs, noisy_llrs), axis=0)
        expected_bits, expected_crc = eager(data)
        bits, crc = compiled(data)
        np.testing.assert_array_equal(crc, expected_crc)
        np.testing.assert_array_equal(crc[:, 0], [True, False])
        np.testing.assert_array_equal(bits[0], expected_bits[0])
        np.testing.assert_array_equal(bits[0], payload[0])

        other_id = replace(cfg, pusch=replace(cfg.pusch, data_scrambling_id=492))
        changed = PuschLink(other_id, external_channel=True, tb_decoder_cache=cache)
        self.assertIsNot(changed._receiver._tb_decoder, compiled)
        self.assertEqual(len(cache), 2)
