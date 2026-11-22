"""Tests for frequency-domain NR sounding and LS estimation."""

import unittest

import numpy as np

from pusch_baseline.srs import SrsEstimator


class SrsTests(unittest.TestCase):
    def test_comb_and_resource_bandwidth_follow_nr_table(self):
        estimator = SrsEstimator(num_tx_ports=4, num_prbs=51, sequence_id=0)
        self.assertEqual(estimator.num_sounded_prbs, 48)
        self.assertEqual(estimator.symbol_indices, (10, 11, 12, 13))
        self.assertEqual(estimator.cyclic_shifts, (0, 2, 4, 6))
        self.assertEqual(estimator.power_per_active_re, 1.0)
        self.assertEqual(estimator.power_per_port_active_re, 0.25)
        np.testing.assert_array_equal(estimator.subcarrier_indices, np.arange(0, 48 * 12, 2))

    def test_long_reference_sequence_uses_nr_root_and_wraps(self):
        estimator = SrsEstimator(num_tx_ports=1, num_prbs=8, sequence_id=0)
        expected = [1.0, np.exp(-4j * np.pi / 47), np.exp(-12j * np.pi / 47), 1.0]
        np.testing.assert_allclose(estimator.reference_sequence[[0, 1, 2, 47]], expected)
        np.testing.assert_allclose(np.abs(estimator.reference_sequence), 1.0)

    def test_four_prb_resource_uses_short_sequence_table(self):
        estimator = SrsEstimator(num_tx_ports=1, num_prbs=4, sequence_id=55)
        phase = np.array([-3, 1, -1, -3, 1, -1, -3]) * np.pi / 4
        np.testing.assert_allclose(
            estimator.reference_sequence[[0, 1, 2, 3, 21, 22, 23]], np.exp(1j * phase)
        )

    def test_multiport_sequences_match_matlab_nrsrs(self):
        # MATLAB R2024b nrSRS with KTC=2, KBarTC=0, CyclicShift=0, BSRS=0, Repetition=NumSRSPorts.
        est4 = SrsEstimator(num_tx_ports=4, num_prbs=48, sequence_id=1)
        expected_4 = np.array([
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.921202 - 0.389084j, 0.389084 + 0.921202j, -0.921202 + 0.389084j, -0.389084 - 0.921202j],
            [0.363372 - 0.931644j, -0.363372 + 0.931644j, 0.363372 - 0.931644j, -0.363372 + 0.931644j],
            [-0.735922 - 0.677067j, -0.677067 + 0.735922j, 0.735922 + 0.677067j, 0.677067 - 0.735922j],
            [-0.6564 + 0.7544j, 0.7544 + 0.6564j, 0.6564 - 0.7544j, -0.7544 - 0.6564j],
        ])
        np.testing.assert_allclose(est4.port_sequences[:, [0, 1, 2, 3, 287]].T, expected_4, atol=1e-4)

        est2 = SrsEstimator(num_tx_ports=2, num_prbs=4, sequence_id=25)
        self.assertEqual(est2.cyclic_shifts, (0, 4))
        s = np.sqrt(0.5)
        expected_2 = np.array([
            [-s - s * 1j, -s - s * 1j],
            [s + s * 1j, -s - s * 1j],
            [s - s * 1j, s - s * 1j],
            [-s - s * 1j, s + s * 1j],
        ])
        np.testing.assert_allclose(est2.port_sequences[:, [0, 1, 2, 3]].T, expected_2, atol=1e-6)

    def test_known_channel_recovers_distinct_ports_via_cyclic_shift_cdm(self):
        for num_ports in (1, 2, 4):
            with self.subTest(num_ports=num_ports):
                estimator = SrsEstimator(num_tx_ports=num_ports, num_prbs=4, sequence_id=25)
                channel = np.zeros((3, num_ports, 14, 48), dtype=np.complex64)
                rng = np.random.default_rng(10 + num_ports)
                port_gains = (
                    rng.standard_normal((3, num_ports)) + 1j * rng.standard_normal((3, num_ports))
                ).astype(np.complex64)
                expected = np.broadcast_to(port_gains[None, :, :], (48, 3, num_ports))
                channel[:, :, estimator.symbol_indices, :] = port_gains[:, :, None, None]
                estimate = estimator.estimate(channel, noise_variance=0.0, rng=np.random.default_rng(1))
                np.testing.assert_allclose(estimate, expected, atol=1e-6, rtol=1e-6)
                self.assertEqual(estimate.dtype, np.dtype("complex64"))

    def test_single_port_recovers_linear_slope_and_holds_band_edge(self):
        estimator = SrsEstimator(num_tx_ports=1, num_prbs=4, sequence_id=25)
        channel = np.zeros((3, 1, 14, 48), dtype=np.complex64)
        base = np.array([[1.0], [0.2j], [2.0]], dtype=np.complex64)
        slope = (0.01 + 0.02j) * np.minimum(np.arange(48), 46)
        expected = base[None, :, :] + slope[:, None, None].astype(np.complex64)
        channel[:, 0, 13, :] = expected[:, :, 0].T
        estimate = estimator.estimate(channel, noise_variance=0.0, rng=np.random.default_rng(1))
        np.testing.assert_allclose(estimate, expected, atol=1e-6, rtol=1e-6)

    def test_noisy_cdm_ls_is_seeded_uncorrelated_across_ports_and_matches_error_variance(self):
        estimator = SrsEstimator(num_tx_ports=4, num_prbs=24, sequence_id=0)
        channel = np.zeros((64, 4, 14, 288), dtype=np.complex128)
        noise_variance = 0.08
        estimate = estimator.estimate(channel, noise_variance, np.random.default_rng(42))
        replayed = estimator.estimate(channel, noise_variance, np.random.default_rng(42))
        np.testing.assert_array_equal(estimate, replayed)
        expected_var = estimator.error_variance(noise_variance)
        # Verify average variance across tones and cross-port independence.
        observed_cov = np.einsum("krp,krq->pq", estimate.conj(), estimate) / (288 * 64)
        mean_var = float(np.mean(expected_var))
        np.testing.assert_allclose(np.diag(observed_cov).real, mean_var, rtol=0.08)
        off_diag = observed_cov - np.diag(np.diag(observed_cov))
        self.assertLess(np.max(np.abs(off_diag)), 0.05 * mean_var)
        # At block centers (3, 11, ...), variance equals noise_variance / N_symb = 0.02.
        centers = np.arange(3, 288, 8)
        np.testing.assert_allclose(expected_var[centers], noise_variance / 4)
        self.assertAlmostEqual(float(np.mean(np.abs(estimate[centers])**2)), noise_variance / 4, delta=0.002)
        self.assertEqual(estimate.dtype, np.dtype("complex128"))

    def test_interpolation_noise_variance_accounts_for_pilots_and_held_band_edge(self):
        estimator = SrsEstimator(num_tx_ports=1, num_prbs=9, sequence_id=0)
        variance = estimator.error_variance(0.08)
        self.assertEqual(variance.shape, (108,))
        np.testing.assert_allclose(variance[:94], np.tile([0.08, 0.04], 47))
        np.testing.assert_allclose(variance[94:], np.full(14, 0.08))

    def test_total_power_scaling_changes_ls_error_with_the_same_standardized_noise(self):
        full = SrsEstimator(4, 4, 0)
        capped = SrsEstimator(4, 4, 0, power_per_active_re=0.25)
        pilots = np.zeros((2, 4, 4, 24), dtype=np.complex64)
        full_estimate = full.estimate_from_pilots(pilots, 0.1, np.random.default_rng(3))
        capped_estimate = capped.estimate_from_pilots(pilots, 0.1, np.random.default_rng(3))
        np.testing.assert_allclose(capped_estimate, 2 * full_estimate, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(capped.error_variance(0.1), 4 * full.error_variance(0.1))


if __name__ == "__main__":
    unittest.main()
