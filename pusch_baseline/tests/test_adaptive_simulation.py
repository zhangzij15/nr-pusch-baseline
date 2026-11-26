"""End-to-end checks for sounding, grant selection and PUSCH accounting."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from pusch_baseline import SimulationConfig, simulate
from pusch_baseline.config import AdaptationSettings


class AdaptiveSimulationTests(unittest.TestCase):
    def test_sounding_projection_matches_sionna_with_common_or_per_port_delays(self):
        import numpy as np
        import tensorflow as tf
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        from pusch_baseline.adaptive_simulation import SoundingChannel
        from pusch_baseline.srs import SrsEstimator

        base = SimulationConfig()
        random = np.random.default_rng(144)
        shape = (1, 1, 2, 1, 4, 3, 42)
        gains = random.standard_normal(shape) + 1j * random.standard_normal(shape)
        base_delays = np.array([0, 70e-9, 340e-9]).reshape(1, 1, 1, 3)
        per_port_delays = (
            base_delays.reshape(1, 1, 1, 1, 1, 3)
            + 1e-9 * np.arange(8).reshape(1, 1, 2, 1, 4, 1)
        )
        for precision, normalize, delays in (
            ("single", False, base_delays),
            ("double", True, base_delays),
            ("single", False, per_port_delays),
        ):
            with self.subTest(precision=precision, normalize=normalize, delay_rank=delays.ndim):
                config = replace(
                    base,
                    precision=precision,
                    carrier=replace(base.carrier, n_size_grid=4),
                    pusch=replace(base.pusch, num_antenna_ports=4),
                    channel=replace(base.channel, kind="custom", num_rx_ant=2, normalize=normalize),
                )
                dtype = tf.complex128 if precision == "double" else tf.complex64
                gains_tensor = tf.constant(gains, dtype=dtype)
                delays_tensor = tf.constant(delays, dtype=dtype.real_dtype)

                def factory(_config, _grid):
                    return lambda *_: (gains_tensor, delays_tensor)

                estimator = SrsEstimator(4, 4, 1, start_symbol=10)
                frequencies = subcarrier_frequencies(
                    config.carrier.n_size_grid * 12,
                    config.carrier.subcarrier_spacing_khz * 1000,
                    precision=precision,
                )
                reference = cir_to_ofdm_channel(frequencies, gains_tensor, delays_tensor)
                sounding = reference[..., :14, :]
                data = reference[..., 28:42, :]
                if normalize:
                    rms = tf.cast(tf.sqrt(tf.reduce_mean(tf.square(tf.abs(sounding)))), dtype)
                    sounding, data = sounding / rms, data / rms
                with patch("pusch_baseline.adaptive_simulation.cir_to_ofdm_channel",
                           wraps=cir_to_ofdm_channel) as native_projection:
                    pilot_response, data_response = list(SoundingChannel(config, factory, estimator).slots((0, 2)))
                self.assertEqual(native_projection.call_count, 2 if delays.ndim == 6 else 0)
                np.testing.assert_allclose(pilot_response, estimator.sample_pilots(sounding[0, 0, :, 0]),
                                           rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(data_response, data, rtol=1e-6, atol=1e-6)

    def test_srs_drives_grants_and_counts_sounding_time(self):
        config = SimulationConfig()
        config = replace(
            config,
            adaptation=AdaptationSettings(
                srs_period_slots=4, mcs_thresholds_db=((5, 2.0), (14, 10.0)),
            ),
            sweep=replace(
                config.sweep, snr_db=(-25.0, 25.0), num_tb_per_snr=4, seed=144,
            ),
        )
        result = simulate(config)
        low, high = result["points"]

        self.assertEqual(low["grant_counts"][0]["mcs_index"], 5)
        self.assertEqual(high["grant_counts"][0]["mcs_index"], 14)
        self.assertEqual((low["tb_errors"], high["tb_errors"]), (4, 0))
        for point in (low, high):
            self.assertEqual((point["num_tb"], point["num_srs_slots"], point["num_slots"]),
                             (4, 2, 6))
            self.assertEqual(sum(grant["num_tb"] for grant in point["grant_counts"]), 4)
        self.assertEqual(high["delivered_payload_bits"], high["attempted_payload_bits"])
        self.assertAlmostEqual(high["thp_including_srs_mbps"],
                               high["thp_scheduled_mbps"] * 4 / 6)

    def test_srs_and_pusch_consume_successive_slots_of_one_cir(self):
        import tensorflow as tf

        config = SimulationConfig()
        config = replace(
            config,
            channel=replace(config.channel, kind="custom", normalize=False),
            adaptation=AdaptationSettings(),
            sweep=replace(config.sweep, snr_db=(25.0,), num_tb_per_snr=3, seed=144),
        )
        cir_calls = []

        def factory(_config, grid):
            self.assertEqual(grid.num_ofdm_symbols, 14)
            self.assertGreater(grid.cyclic_prefix_length, 0)

            def cir(batch_size, num_time_steps, sampling_frequency):
                self.assertEqual(sampling_frequency, 14 / _config.slot_duration_s)
                cir_calls.append((batch_size, num_time_steps))
                # The channel is strong during sounding and in outage during PUSCH.
                gains = tf.cast(tf.range(num_time_steps) < 14, tf.complex64)
                return tf.reshape(gains, (1, 1, 1, 1, 1, 1, num_time_steps)), tf.zeros((1, 1, 1, 1))

            return cir

        result = simulate(config, channel_model_factory=factory)
        point, = result["points"]
        self.assertEqual(cir_calls, [(1, 56)])
        self.assertEqual(point["grant_counts"][0]["mcs_index"], 14)
        self.assertEqual(point["tb_errors"], 3)
        self.assertEqual(point["delivered_payload_bits"], 0)

    def test_zero_sounding_reference_does_not_erase_later_data_slots(self):
        import tensorflow as tf

        config = SimulationConfig()
        config = replace(
            config,
            channel=replace(config.channel, kind="custom", normalize=True),
            adaptation=AdaptationSettings(),
            sweep=replace(config.sweep, snr_db=(25.0,), num_tb_per_snr=3, seed=144),
        )

        def factory(_config, _grid):
            def cir(_batch_size, num_time_steps, _sampling_frequency):
                gains = tf.cast(tf.range(num_time_steps) >= 14, tf.complex64)
                return tf.reshape(gains, (1, 1, 1, 1, 1, 1, num_time_steps)), tf.zeros((1, 1, 1, 1))

            return cir

        result = simulate(config, channel_model_factory=factory)
        point, = result["points"]
        self.assertEqual(point["grant_counts"][0]["mcs_index"], 5)
        self.assertEqual((point["num_tb"], point["tb_errors"]), (3, 0))
