"""Equivalence to Sionna's complete linear receiver for the single-stream case."""

from dataclasses import replace
import unittest

import numpy as np
import tensorflow as tf
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LinearDetector

from pusch_baseline.config import SimulationConfig
from pusch_baseline.link import PuschLink
from pusch_baseline.rank_one_detector import RankOneDetector


class RankOneDetectorTests(unittest.TestCase):
    def test_llrs_match_native_receiver_with_broadcast_and_per_antenna_noise(self):
        base = SimulationConfig()
        cfg = replace(base, carrier=replace(base.carrier, n_size_grid=6))
        grid = PuschLink(cfg, external_channel=True).resource_grid
        fast = RankOneDetector(grid, 4, precision="single")
        native = LinearDetector(
            "lmmse", "bit", "maxlog", grid, StreamManagement(np.ones((1, 1), bool), 1), "qam", 4,
        )

        rng = np.random.default_rng(23)
        shape = (2, 1, 8, 14, grid.fft_size)
        h = tf.constant(rng.normal(size=shape) + 1j * rng.normal(size=shape), tf.complex64)
        y = tf.constant(rng.normal(size=shape) + 1j * rng.normal(size=shape), tf.complex64)
        h = h[:, :, :, None, None]
        err_var = tf.constant(rng.uniform(0.1, 0.4, size=h.shape[1:]), tf.float32)
        branch_noise = tf.constant(rng.uniform(0.2, 1.2, size=shape[:3]), tf.float32)

        for error, noise in ((tf.constant(0.0), tf.constant(0.6)), (err_var, branch_noise)):
            with self.subTest(noise_shape=noise.shape):
                expected = native(y, h, error, noise)
                got = fast(y, h, error, noise)
                np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)

    def test_zero_effective_channel_has_erasure_llrs(self):
        base = SimulationConfig()
        cfg = replace(base, carrier=replace(base.carrier, n_size_grid=6))
        grid = PuschLink(cfg, external_channel=True).resource_grid
        detector = RankOneDetector(grid, 2, precision="single")
        y = tf.ones((1, 1, 8, 14, grid.fft_size), tf.complex64)
        h = tf.zeros((1, 1, 8, 1, 1, 14, grid.fft_size), tf.complex64)
        llrs = detector(y, h, tf.constant(0.1), tf.constant(0.5))
        np.testing.assert_array_equal(llrs, 0)

