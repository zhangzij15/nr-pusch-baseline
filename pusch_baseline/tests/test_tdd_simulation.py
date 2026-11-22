"""End-to-end checks for SRS and PUSCH on a TDD slot timeline."""

from dataclasses import replace
from math import isfinite
import unittest

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings


class TddSimulationTests(unittest.TestCase):
    def setUp(self):
        base = SimulationConfig()
        self.config = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=8),
            pusch=replace(base.pusch, num_antenna_ports=4, precoding="codebook"),
            channel=replace(base.channel, kind="custom", normalize=False, estimator="perfect"),
            adaptation=AdaptationSettings(
                tdd_pattern="DSUDD",
                srs_start_symbol=10,
                srs_period_slots=5,
                rank_tpmi_candidates=((1, 0), (1, 3)),
                mcs_thresholds_db=((5, 2.0),),
            ),
            sweep=replace(base.sweep, snr_db=(30.0,), num_tb_per_snr=5, seed=144),
        )

    def test_tdd_counts_elapsed_slots_and_indicates_codeword_after_each_srs(self):
        import tensorflow as tf

        cir_lengths = []

        def factory(_config, grid):
            self.assertEqual(grid.num_ofdm_symbols, 14)

            def cir(batch_size, num_time_steps, sampling_frequency):
                self.assertEqual(batch_size, 1)
                self.assertEqual(sampling_frequency, 1 / grid.ofdm_symbol_duration)
                # Native four-port rank-one TPMI 0 and 3 select ports 0 and 3.
                active_port = 0 if len(cir_lengths) % 2 == 0 else 3
                cir_lengths.append(num_time_steps)
                port = tf.cast(tf.one_hot(active_port, 4), tf.complex64)
                slot = tf.range(num_time_steps) // 14
                # During sounding all four ports share the four-symbol SRS resource (10..13).
                symbol = tf.range(num_time_steps) % 14
                sounding = (slot == 0) & (symbol >= 10)
                gain = tf.cast(sounding | (slot == 1) | (slot == 6), tf.complex64)
                gains = tf.reshape(port[:, None] * gain, (1, 1, 1, 1, 4, 1, num_time_steps))
                return gains, tf.zeros((1, 1, 1, 1))

            return cir

        result = simulate(self.config, channel_model_factory=factory)
        point, = result["points"]

        # srs_period_slots=5 sounds every S slot of DSUDD, once per 5-slot period.
        indicated = point["srs_grants"]
        self.assertEqual([grant["slot_index"] for grant in indicated], [1, 6, 11, 16, 21])
        self.assertEqual([grant["tpmi"] for grant in indicated], [0, 3, 0, 3, 0])
        self.assertTrue(all(isfinite(grant["effective_sinr_db"]) for grant in indicated))
        self.assertEqual((point["num_tb"], point["num_srs_slots"], point["num_slots"]),
                         (5, 5, 23))
        self.assertEqual(point["num_idle_slots"], 13)
        self.assertEqual(point["tb_errors"], 0)
        self.assertEqual({grant["tpmi"]: grant["num_tb"] for grant in point["grant_counts"]},
                         {0: 3, 3: 2})
        self.assertAlmostEqual(point["thp_including_srs_mbps"],
                               point["thp_scheduled_mbps"] * 5 / 23)
        self.assertEqual(point["thp_wall_clock_mbps"], point["thp_including_srs_mbps"])
        clusters = point["throughput_srs_clusters"]
        self.assertEqual([cluster["srs_slot_index"] for cluster in clusters], [1, 6, 11, 16, 21])
        self.assertEqual([cluster["num_slots"] for cluster in clusters], [6, 5, 5, 5, 2])
        self.assertEqual(sum(cluster["delivered_payload_bits"] for cluster in clusters),
                         point["delivered_payload_bits"])
        self.assertEqual(point["throughput_ci95_num_clusters"], 5)
        low, high = point["throughput_ci95_mbps"]
        self.assertLess(low, point["thp_wall_clock_mbps"])
        self.assertGreater(high, point["thp_wall_clock_mbps"])
        self.assertEqual(cir_lengths, [28] * 5)
        self.assertEqual(result["metadata"]["adaptation"]["srs_sounded_prbs"], 8)
        self.assertEqual(result["metadata"]["adaptation"]["srs_symbols"], (10, 11, 12, 13))
        self.assertEqual(result["metadata"]["adaptation"]["srs_cyclic_shifts"], (0, 2, 4, 6))
        self.assertEqual(result["metadata"]["adaptation"]["srs_power_per_active_re"], 1.0)
        self.assertEqual(result["metadata"]["adaptation"]["srs_power_per_port_active_re"], 0.25)
        self.assertEqual(result["metadata"]["adaptation"]["srs_estimator"], "real")

    def test_ideal_srs_estimator_uses_noiseless_full_band_sounding(self):
        import tensorflow as tf

        ideal_cfg = replace(
            self.config,
            adaptation=replace(self.config.adaptation, srs_estimator="perfect"),
            sweep=replace(self.config.sweep, snr_db=(-30.0,), num_tb_per_snr=2),
        )

        def factory(_config, _grid):
            def cir(batch_size, num_time_steps, _sampling_frequency):
                port = tf.cast(tf.one_hot(3, 4), tf.complex64)
                gains = tf.broadcast_to(
                    tf.reshape(port, (1, 1, 1, 1, 4, 1, 1)),
                    (batch_size, 1, 1, 1, 4, 1, num_time_steps),
                )
                return gains, tf.zeros((batch_size, 1, 1, 1))

            return cir

        result = simulate(ideal_cfg, channel_model_factory=factory)
        point, = result["points"]
        self.assertEqual(result["metadata"]["adaptation"]["srs_estimator"], "perfect")
        # Two TBs on DSUDD with a 5-slot period use two SRS epochs.
        self.assertEqual([grant["tpmi"] for grant in point["srs_grants"]], [3, 3])

    def test_tdd_validation_rejects_missing_slots_partial_band_or_double_duty(self):
        config = self.config
        config.validate()
        invalid = {
            "missing S": replace(config, adaptation=replace(config.adaptation, tdd_pattern="DDUDD")),
            "missing U": replace(config, adaptation=replace(config.adaptation, tdd_pattern="DSDDD")),
            "partial band": replace(config, carrier=replace(config.carrier, n_size_grid=51)),
            "double duty": replace(config, sweep=replace(config.sweep, ul_duty_cycle=0.5)),
            "two ports": replace(config, pusch=replace(config.pusch, num_antenna_ports=2)),
            "SRS overlaps slot end": replace(config, adaptation=replace(config.adaptation, srs_start_symbol=11)),
        }
        for case, candidate in invalid.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                candidate.validate()
