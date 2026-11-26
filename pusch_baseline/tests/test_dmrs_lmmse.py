"""Regression tests for the Rank 1 DM-RS channel estimator."""

import unittest
from dataclasses import replace

import numpy as np
import tensorflow as tf
from scipy.special import j0
from sionna.phy.nr import PUSCHTransmitter

from pusch_baseline.config import SimulationConfig
from pusch_baseline.link import _pusch_config
from pusch_baseline.dmrs_lmmse import DmrsLmmseEstimator


class DmrsLmmseTests(unittest.TestCase):
    def test_estimate_and_error_match_dense_posterior(self):
        base = SimulationConfig(precision="double")
        dmrs = replace(base.pusch.dmrs, config_type=2, length=2, port_set=(8,))
        cfg = replace(base, carrier=replace(base.carrier, n_size_grid=2),
                      pusch=replace(base.pusch, dmrs=dmrs))
        grid = PUSCHTransmitter(_pusch_config(cfg), precision=cfg.precision).resource_grid
        delay_s, doppler_hz = 3e-6, 1100
        estimator = DmrsLmmseEstimator(grid, delay_s, doppler_hz, precision=cfg.precision)
        mask = grid.pilot_pattern.mask.numpy()[0, 0]
        pilots = grid.pilot_pattern.pilots.numpy()[0, 0]
        pilot_re = np.argwhere(mask)[np.abs(pilots) > 0]
        pilot_values = pilots[np.abs(pilots) > 0]
        num_symbols, num_subcarriers = mask.shape
        all_re = np.argwhere(np.ones(mask.shape))

        def covariance(left, right):
            delta = left[:, None] - right[None, :]
            time = j0(2 * np.pi * doppler_hz * grid.ofdm_symbol_duration * delta[..., 0])
            freq = 1 / (1 + 2j * np.pi * delay_s * grid.subcarrier_spacing * delta[..., 1])
            return time * freq

        prior = covariance(pilot_re, pilot_re)
        cross = covariance(all_re, pilot_re)
        rng = np.random.default_rng(12)
        color = np.linalg.cholesky(prior + 1e-13 * np.eye(len(prior)))
        sample = rng.standard_normal((2, len(prior), 2)).view(np.complex128)[..., 0]
        sample = sample @ color.T
        noise = np.asarray([0.3, 0.8])
        pilot_noise = noise / np.abs(pilot_values[0])**2
        y = np.zeros((1, 1, 2, num_symbols, num_subcarriers), dtype=np.complex128)
        y[..., pilot_re[:, 0], pilot_re[:, 1]] = sample * pilot_values

        power = np.mean(np.einsum("ai,ij,aj->a", sample.conj(), prior, sample).real
                        - np.trace(prior).real * pilot_noise) / np.sum(np.abs(prior)**2)
        expected_mean = []
        expected_error = []
        for branch, branch_noise in zip(sample, pilot_noise, strict=True):
            inverse = np.linalg.inv(power * prior + branch_noise * np.eye(len(prior)))
            expected_mean.append(power * cross @ inverse @ branch)
            expected_error.append(power - power**2 *
                                  np.sum(((cross @ inverse) * cross.conj()).real, axis=1))

        estimate, variance = tf.function(estimator)(tf.constant(y), tf.constant(noise[None, None]))
        np.testing.assert_allclose(estimate.numpy().reshape(2, -1), expected_mean, atol=2e-5)
        np.testing.assert_allclose(variance.numpy().reshape(2, -1), expected_error, atol=2e-5)

    def test_flat_noiseless_estimate_recovers_complex_gain(self):
        cfg = SimulationConfig()
        tx = PUSCHTransmitter(_pusch_config(cfg), return_bits=True, precision=cfg.precision)
        estimator = DmrsLmmseEstimator(tx.resource_grid, 0, 0, precision=cfg.precision)
        gains = tf.constant([1 + 0.3j, -0.25 + 0.5j], tf.complex64)
        x, _ = tx(1)
        y = x * gains[None, None, :, None, None]
        estimate, variance = estimator(y, tf.constant(0.0))
        np.testing.assert_allclose(estimate, tf.broadcast_to(
            gains[None, None, :, None, None, None, None], estimate.shape
        ), atol=1e-6)
        np.testing.assert_allclose(variance, 0, atol=1e-6)

    def test_link_passes_slot_invariant_symbol_duration_to_lmmse(self):
        from pusch_baseline.link import PuschLink

        base = SimulationConfig(precision="double")
        cfg0 = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=4, slot_number=0),
            channel=replace(base.channel, kind="tdl", profile="C", ue_speed_kmh=120.0, estimator="lmmse"),
        )
        cfg2 = replace(cfg0, carrier=replace(cfg0.carrier, slot_number=2))
        est0 = PuschLink(cfg0, external_channel=True)._receiver._channel_estimator
        est2 = PuschLink(cfg2, external_channel=True)._receiver._channel_estimator
        np.testing.assert_allclose(est0._grid_time.numpy(), est2._grid_time.numpy(), atol=1e-12)
