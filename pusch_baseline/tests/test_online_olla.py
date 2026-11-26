"""Regress per-TTI outer-loop feedback and measurement accounting."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings, OllaSettings
from pusch_baseline.olla import OllaController


class OnlineOllaTests(unittest.TestCase):
    def test_default_feedback_steps_target_ten_percent_bler(self):
        controller = OllaController(OllaSettings())

        controller.observe(True, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.margin_offset_db, 0.9)
        for _ in range(9):
            controller.observe(False, lowest_mcs=False, highest_mcs=False)
        self.assertAlmostEqual(controller.margin_offset_db, 0.0)

    def test_nack_changes_next_mcs_without_waiting_for_srs_in_either_phase(self):
        from pusch_baseline.link import PuschLink

        base = SimulationConfig()
        cfg = replace(
            base,
            adaptation=AdaptationSettings(
                srs_period_slots=4,
                mcs_thresholds_db=((5, 2.0), (14, 29.65)),
                sinr_margin_db=0.0,
                olla=OllaSettings(control="sinr_margin", warmup_tb=10, window_tb=10),
            ),
            sweep=replace(base.sweep, snr_db=(30.0,), num_tb_per_snr=10),
        )
        # MCS 5 uses QPSK and MCS 14 uses 16-QAM in table 1. Use real link
        # construction and SRS-based selection with deterministic decoder feedback.
        outcomes = [1] + [0] * 9 + [1] + [0] * 8 + [1]
        modulation_orders = []

        def decode(link, batch_size, _noise, *, channel_response, **_):
            self.assertEqual(batch_size, 1)
            self.assertIsNotNone(channel_response)
            outcome = outcomes[len(modulation_orders)]
            modulation_orders.append(link.grant_metadata["qam_bits_per_symbol"])
            return outcome

        with patch.object(PuschLink, "count_block_errors", decode):
            point, = simulate(cfg)["points"]

        self.assertEqual(len(modulation_orders), 20)
        # All three slots in each slice are scheduled by the same SRS epoch.
        self.assertEqual(modulation_orders[:3], [4, 2, 2])
        self.assertEqual(modulation_orders[10:13], [4, 2, 2])
        self.assertEqual((point["calibration_total_tb"], point["num_tb"], point["tb_errors"]),
                         (10, 10, 2))
        self.assertAlmostEqual(point["calibration_bler"], 0.1)
        self.assertAlmostEqual(point["margin_offset_db"], 0.0)
        self.assertAlmostEqual(point["measurement_margin_offset_db"], 1.0)
        self.assertEqual(point["target_status"], "above_target")
        self.assertIsNone(point["thp_at_target_mbps"])

    def test_rank_stays_fixed_between_srs_epochs_when_mcs_drops(self):
        import tensorflow as tf

        from pusch_baseline.link import PuschLink

        base = SimulationConfig()
        cfg = replace(
            base,
            pusch=replace(base.pusch, num_antenna_ports=2, precoding="codebook"),
            channel=replace(base.channel, kind="custom", num_rx_ant=2, normalize=False),
            adaptation=AdaptationSettings(
                srs_period_slots=4, rank_tpmi_candidates=((1, 0), (2, 0)),
                mcs_thresholds_db=((5, 2.0), (14, 26.65)), sinr_margin_db=0.0,
                olla=OllaSettings(control="sinr_margin", warmup_tb=3, window_tb=3),
            ),
            sweep=replace(base.sweep, snr_db=(30.0,), num_tb_per_snr=3),
        )
        grants = []

        def factory(_cfg, _grid):
            def channel(batch_size, samples, _frequency):
                identity = tf.reshape(tf.eye(2, dtype=tf.complex64), (1, 1, 2, 1, 2, 1, 1))
                gains = tf.broadcast_to(identity, (batch_size, 1, 2, 1, 2, 1, samples))
                return gains, tf.zeros((batch_size, 1, 1, 1))

            return channel

        def decode(link, _batch_size, _noise, *, channel_response, **_):
            grant = link.grant_metadata
            grants.append((len(grant["dmrs_ports"]), grant["qam_bits_per_symbol"]))
            return int(len(grants) == 1)

        with patch.object(PuschLink, "count_block_errors", decode):
            simulate(cfg, channel_model_factory=factory)

        # During the first epoch, feedback lowers the MCS of the same rank-2 grant.
        self.assertEqual(grants[:3], [(2, 4), (2, 2), (2, 2)])
        self.assertEqual(grants[3], (1, 4))


if __name__ == "__main__":
    unittest.main()
