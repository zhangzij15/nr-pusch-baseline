"""Unit tests for the SRS-driven uplink grant policy."""

import math
import unittest

import numpy as np

from pusch_baseline.adaptation import select_grant, select_grants


class LinkAdaptationTests(unittest.TestCase):
    def test_siso_predicts_frequency_selective_sinr_and_applies_margin(self):
        channel = np.repeat(np.array([[[1.0]], [[2.0j]]]), 12, axis=0)
        weaker = (1, 0, 3)
        stronger = (1, 0, 8)
        precoder = np.ones((1, 1))
        candidates = {weaker: (precoder, 128), stronger: (precoder, 256)}
        thresholds = {3: -2.0, 8: 3.0}

        key, sinr_db = select_grant(channel, 1.0, candidates, thresholds, 0.0)
        self.assertEqual(key, stronger)
        self.assertAlmostEqual(sinr_db, 10 * math.log10(math.sqrt(10.0) - 1.0))
        key, _ = select_grant(channel, 1.0, candidates, thresholds, 1.0)
        self.assertEqual(key, weaker)

    def test_rank_and_tpmi_respond_to_spatial_structure(self):
        first = (1, 0, 5)
        second = (1, 1, 5)
        two_layers = (2, 0, 5)
        candidates = {
            first: (np.array([[1.0], [0.0]]), 128),
            second: (np.array([[0.0], [1.0]]), 128),
            two_layers: (np.eye(2) / math.sqrt(2.0), 256),
        }
        # Two orthogonal streams at 10 dB before splitting TX power.
        channel = np.array([math.sqrt(10.0) * np.eye(2)])
        key, sinr_db = select_grant(channel, 1.0, candidates, {5: 6.0}, 0.0)
        self.assertEqual(key, two_layers)
        self.assertAlmostEqual(sinr_db, 10 * math.log10(5.0))
        # The other TX port fades out. Rank 1 on the second port now wins.
        channel = np.array([np.diag([0.0, math.sqrt(10.0)])])
        key, sinr_db = select_grant(channel, 1.0, candidates, {5: 6.0}, 0.0)
        self.assertEqual(key, second)
        self.assertAlmostEqual(sinr_db, 10.0)

    def test_nonorthogonal_layers_include_inter_stream_interference(self):
        channel = np.array([[[1.0, 1.0], [0.0, 0.0]]])
        choice = (2, 0, 2)
        key, sinr_db = select_grant(
            channel, 1.0, {choice: (np.eye(2) / math.sqrt(2.0), 256)}, {2: -9.0}, 0.0
        )
        self.assertEqual(key, choice)
        self.assertAlmostEqual(sinr_db, 10 * math.log10(1.0 / 3.0))

    def test_fallback_prefers_lowest_mcs_and_breaks_ties_consistently(self):
        channel = np.zeros((3, 1, 2), dtype=complex)
        precoder = np.array([[1.0], [0.0]])
        chosen = (1, 1, 2)
        candidates = {
            (1, 3, 2): (precoder, 128),
            (1, 0, 5): (precoder, 256),
            chosen: (precoder, 128),
        }
        key, sinr_db = select_grant(channel, 1.0, candidates, {2: 0.0, 5: 5.0}, 0.0)
        self.assertEqual(key, chosen)
        self.assertLess(sinr_db, -200.0)
        self.assertTrue(math.isfinite(sinr_db))

    def test_srs_error_debiases_noise_only_massive_mimo(self):
        # This deterministic observation has one unit of error energy per RX.
        channel = np.ones((12, 64, 1), dtype=complex)
        robust = (1, 0, 3)
        aggressive = (1, 0, 14)
        precoder = np.ones((1, 1))
        candidates = {robust: (precoder, 128), aggressive: (precoder, 512)}
        thresholds = {3: -2.0, 14: 10.0}

        key, sinr_db = select_grant(channel, 1.0, candidates, thresholds, 0.0)
        self.assertEqual(key, aggressive)
        self.assertAlmostEqual(sinr_db, 10.0 * math.log10(64.0))
        key, sinr_db = select_grant(
            channel, 1.0, candidates, thresholds, 0.0, srs_error_variance=1.0
        )
        self.assertEqual(key, robust)
        self.assertLess(sinr_db, -200.0)

    def test_srs_gram_averages_per_prb_with_partial_prb_weights(self):
        # After subtracting per-tone error, the first PRB has mean power 2.
        # Its final one-tone neighbor has power 8 and represents 1/13 of the REs.
        error_variances = np.arange(13) / 4.0
        power = np.array([1.0, 3.0] * 6 + [8.0])
        channel = np.sqrt(power + error_variances).reshape((13, 1, 1))
        choice = (1, 0, 3)
        _, sinr_db = select_grant(
            channel, 1.0, {choice: (np.ones((1, 1)), 128)}, {3: 0.0}, 0.0,
            srs_error_variance=error_variances,
        )
        predicted = (3.0 ** (12.0 / 13.0)) * (9.0 ** (1.0 / 13.0)) - 1.0
        self.assertAlmostEqual(sinr_db, 10.0 * math.log10(predicted))

    def test_perfect_and_zero_error_srs_select_the_same_precoder(self):
        # The first port has greater PRB-average power but six deep fades.
        # Mixing per-tone and per-PRB MI scoring would reverse the decision.
        channel = np.zeros((12, 1, 2), dtype=complex)
        channel[:6, 0, 0] = math.sqrt(6)
        channel[:, 0, 1] = math.sqrt(2)
        candidates = {(1, i, 5): (np.eye(2)[:, i:i+1], 128) for i in range(2)}
        ideal = select_grants(channel, 1., candidates, None, 0., {"full": candidates})
        real = select_grants(channel, 1., candidates, None, 0., {"full": candidates},
                             srs_error_variance=np.zeros(12))
        self.assertEqual(ideal, real)
        key, score = ideal["full"]
        self.assertEqual(key, (1, 0, 5))
        self.assertAlmostEqual(score, 10 * math.log10(3))

    def test_shared_scores_match_independent_codebook_decisions(self):
        channel = np.tile([[[1.0, 2.0]]], (12, 1, 1))
        port0 = np.array([[1.0], [0.0]])
        port1 = np.array([[0.0], [1.0]])
        candidates = {
            (1, 0, 5): (port0, 128),
            (1, 0, 14): (port0, 256),
            (1, 1, 5): (port1, 128),
            (1, 1, 14): (port1, 256),
        }
        subsets = {"full": tuple(candidates), "restricted": ((1, 0, 5), (1, 0, 14))}
        thresholds = {5: 2.0, 14: 8.0}

        decisions = select_grants(
            channel, 0.25, candidates, thresholds, 1.0, subsets, srs_error_variance=0.125,
        )
        self.assertEqual(decisions["full"][0], (1, 1, 14))
        self.assertEqual(decisions["restricted"][0], (1, 0, 5))
        for name, keys in subsets.items():
            independent = select_grant(
                channel, 0.25, {key: candidates[key] for key in keys}, thresholds, 1.0,
                srs_error_variance=0.125,
            )
            self.assertEqual(decisions[name], independent)

    def test_thresholdless_selection_uses_sum_rate_across_ranks(self):
        first = (1, 0, 10)
        second = (1, 1, 10)
        two_layers = (2, 0, 10)
        candidates = {
            first: (np.array([[1.0], [0.0]]), 2000),
            second: (np.array([[0.0], [1.0]]), 2000),
            two_layers: (np.eye(2) / math.sqrt(2.0), 4000),
        }
        # High-SNR orthogonal 2x2 channel: Rank 2 delivers nearly 2x sum capacity.
        channel = np.broadcast_to(np.eye(2, dtype=complex)[None], (12, 2, 2))
        choice, sinr_db = select_grants(channel, 0.01, candidates, None, 0.0, {None: list(candidates)})[None]
        self.assertEqual(choice, two_layers)
        self.assertAlmostEqual(sinr_db, 10.0 * math.log10(50.0))

        # When one port fades out, Rank 1 on the surviving port maximizes sum capacity.
        faded = np.broadcast_to(np.diag([0.0, 1.0]).astype(complex)[None], (12, 2, 2))
        choice, sinr_db = select_grants(faded, 0.01, candidates, None, 0.0, {None: list(candidates)})[None]
        self.assertEqual(choice, second)
        self.assertAlmostEqual(sinr_db, 20.0)


if __name__ == "__main__":
    unittest.main()
