"""Exercise online MCS-index updates in the PUSCH scheduler."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings, OllaSettings


class McsIndexAdaptationTests(unittest.TestCase):
    def test_low_rate_rv0_is_used_by_the_adaptive_link(self):
        base = SimulationConfig()
        cfg = replace(
            base,
            pusch=replace(base.pusch, mcs_index=0),
            channel=replace(base.channel, estimator="lmmse"),
            adaptation=AdaptationSettings(
                mcs_indices=(0,),
                olla=OllaSettings(warmup_tb=2, window_tb=2),
            ),
            sweep=replace(base.sweep, snr_db=(25.0,), num_tb_per_snr=2),
        )
        point, = simulate(cfg)["points"]
        self.assertEqual((point["num_tb"], point["tb_errors"]), (2, 0))
        self.assertEqual({grant["mcs_index"] for grant in point["grant_counts"]}, {0})
        self.assertEqual(point["delivered_payload_bits"], point["attempted_payload_bits"])

    def test_per_tti_updates_survive_srs_and_carry_into_measurement(self):
        from pusch_baseline.link import PuschLink

        base = SimulationConfig()
        cfg = replace(
            base,
            pusch=replace(base.pusch, mcs_index=14),
            adaptation=AdaptationSettings(
                srs_period_slots=4,
                mcs_indices=(13, 14, 15),
                mcs_thresholds_db=((13, 100.0), (14, 100.0), (15, 100.0)),
                olla=OllaSettings(control="mcs_index", step_mcs=1.0, warmup_tb=11, window_tb=10),
            ),
            sweep=replace(base.sweep, snr_db=(-15.0, 25.0), num_tb_per_snr=10),
        )
        low_errors = [0, 1] + [0] * 9 + [1] + [0] * 9
        low_grants = []
        high_grants = []
        original_init = PuschLink.__init__

        def record_grant(link, grant_cfg, *args, **kwargs):
            original_init(link, grant_cfg, *args, **kwargs)
            link.test_mcs = grant_cfg.pusch.mcs_index

        def decode(link, batch_size, noise_variance, *, channel_response):
            self.assertEqual(batch_size, 1)
            self.assertIsNotNone(channel_response)
            if float(noise_variance) < 1.0:
                high_grants.append(link.test_mcs)
                return 0
            outcome = low_errors[len(low_grants)]
            low_grants.append(link.test_mcs)
            return outcome

        with (
            patch.object(PuschLink, "__init__", record_grant),
            patch.object(PuschLink, "count_block_errors", decode),
        ):
            low, high = simulate(cfg)["points"]

        # Feedback changes the third grant before the next SRS and survives later SRSs.
        self.assertEqual(low_grants[:11], [14, 14] + [13] * 8 + [14])
        # The extra ACK in warmup advances the fractional state to 14.1, so after
        # the measurement NACK it takes only eight ACKs to return to MCS 14.
        self.assertEqual(low_grants[11:], [14] + [13] * 8 + [14])
        self.assertAlmostEqual(low["mcs_state"], 14.1)
        self.assertAlmostEqual(low["measurement_mcs_state"], 14.1)
        self.assertEqual((low["calibration_total_tb"], low["num_tb"], low["tb_errors"]), (11, 10, 1))
        self.assertAlmostEqual(low["calibration_bler"], 0.1)
        self.assertAlmostEqual(low["bler"], 0.1)
        self.assertEqual(low["target_status"], "at_target")
        self.assertAlmostEqual(low["thp_at_target_mbps"], low["thp_duty_cycle_mbps"])

        # Both SNRs use the same radio timeline, but the ACK-only point climbs independently.
        self.assertEqual(high_grants[:11], [14] * 10 + [15])
        self.assertEqual(high_grants[11:], [15] * 10)
        self.assertAlmostEqual(high["mcs_state"], 15.0)
        self.assertAlmostEqual(high["measurement_mcs_state"], 15.0)
        self.assertEqual(high["tb_errors"], 0)
        self.assertIsNone(high["thp_at_target_mbps"])


if __name__ == "__main__":
    unittest.main()
