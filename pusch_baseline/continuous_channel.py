"""Continuous cross-period CIR samplers matching Sionna CDL and TDL physics."""

from __future__ import annotations

import tensorflow as tf
from sionna.phy import PI, config as sionna_config
from sionna.phy.channel.tr38901 import CDL, TDL, Rays, Topology
from sionna.phy.utils import flatten_last_dims, insert_dims, split_dim

from .link import _OrientationSampledCdl


class ContinuousCir:
    """Evolve a single CIR realization continuously over arbitrary symbol indices.

    For built-in Sionna CDL and TDL models, drop-level random parameters
    (velocity, ray coupling, initial phases, LoS phase) are drawn once from a
    dedicated Sionna RNG stream, then evaluated analytically at any physical
    OFDM symbol indices without allocating the full timeline in memory.
    """

    def __init__(self, model, seed: int, *, total_symbols: int | None = None):
        self._raw_model = model
        self._rng = sionna_config.tf_rng
        channel_seed = seed ^ 0x9E3779B9
        self._initial_state = tf.random.Generator.from_seed(
            channel_seed, alg=self._rng.algorithm,
        ).state.read_value()
        self._total_symbols = total_symbols
        self.rewind()

    def rewind(self) -> None:
        self._cached_custom = None
        caller_state = self._rng.state.read_value()
        self._rng.reset(self._initial_state)
        try:
            model = self._raw_model
            if isinstance(model, _OrientationSampledCdl):
                model.rewind()
                model = model._models[model._rng.integers(len(model._models))]
            self._model = model
            if isinstance(model, CDL):
                self._kind = "cdl"
                self._state = self._init_cdl(model)
            elif isinstance(model, TDL):
                self._kind = "tdl"
                self._state = self._init_tdl(model)
            else:
                self._kind = "custom"
                self._state = self._rng.state.read_value()
        finally:
            self._rng.reset(caller_state)

    @staticmethod
    def _init_cdl(cdl: CDL):
        v_r = sionna_config.tf_rng.uniform([1, 1], cdl._min_speed, cdl._max_speed, dtype=cdl.rdtype)
        v_phi = sionna_config.tf_rng.uniform([1, 1], 0.0, 2.0 * PI, dtype=cdl.rdtype)
        v_theta = sionna_config.tf_rng.uniform([1, 1], 0.0, PI, dtype=cdl.rdtype)
        velocities = tf.stack([
            v_r * tf.cos(v_phi) * tf.sin(v_theta),
            v_r * tf.sin(v_phi) * tf.sin(v_theta),
            v_r * tf.cos(v_theta),
        ], axis=-1)
        topology = Topology(
            velocities=velocities,
            moving_end=cdl._moving_end,
            los_aoa=cdl._los_aoa,
            los_zoa=cdl._los_zoa,
            los_aod=cdl._los_aod,
            los_zod=cdl._los_zod,
            los=tf.fill([1, 1, 1], cdl._los),
            distance_3d=tf.zeros([1, 1, 1], cdl.rdtype),
            tx_orientations=insert_dims(cdl._tx_orientation, 2, 0),
            rx_orientations=insert_dims(cdl._rx_orientation, 2, 0),
        )
        aoa, aod, zoa, zod = cdl._random_coupling(cdl._aoa, cdl._aod, cdl._zoa, cdl._zod)
        rays = Rays(
            delays=cdl._delays * cdl._delay_spread,
            powers=cdl._powers,
            aoa=aoa, aod=aod, zoa=zoa, zod=zod, xpr=cdl._xpr,
        )
        phi = cdl._cir_sampler._step_10(tf.shape(rays.aoa))
        return phi, topology, cdl._k_factor, rays

    @staticmethod
    def _init_tdl(tdl: TDL):
        doppler = sionna_config.tf_rng.uniform(
            [1, 1, 1, 1, 1, 1, 1, 1], tdl._min_doppler, tdl._max_doppler, tdl.rdtype,
        )
        theta = sionna_config.tf_rng.uniform(
            [1, 1, 1, 1, 1, tdl._num_clusters, 1, tdl._num_sinusoids],
            -PI / tf.cast(tdl._num_sinusoids, tdl.rdtype),
            PI / tf.cast(tdl._num_sinusoids, tdl.rdtype),
            tdl.rdtype,
        )
        alpha = tdl._alpha_const + theta
        phi = sionna_config.tf_rng.uniform(
            [1, 1, tdl._num_rx_ant, 1, tdl._num_tx_ant, tdl._num_clusters, 1, tdl._num_sinusoids],
            -PI, PI, tdl.rdtype,
        )
        phi_0 = (
            sionna_config.tf_rng.uniform([1, 1, 1, 1, 1, 1, 1], -PI, PI, tdl.rdtype)
            if tdl._los else None
        )
        delays = insert_dims(tdl.delays, 3, 0)
        return doppler, alpha, phi, phi_0, delays

    def sample(self, symbol_indices, sampling_frequency: float):
        """Return (path_gains, delays) evaluated at absolute OFDM symbol indices."""
        indices = tf.convert_to_tensor(symbol_indices, dtype=tf.int32)
        if self._kind == "cdl":
            phi, topology, k_factor, rays = self._state
            sample_times = tf.cast(indices, self._model.rdtype) / sampling_frequency
            h, delays = self._model._cir_sampler._step_11(phi, topology, k_factor, rays, sample_times, None)
            h = tf.stop_gradient(tf.transpose(h, [0, 2, 4, 1, 5, 3, 6]))
            delays = tf.stop_gradient(tf.transpose(delays, [0, 2, 1, 3]))
            return h, delays

        if self._kind == "tdl":
            return self._sample_tdl(indices, sampling_frequency)

        if self._cached_custom is None:
            needed = max(self._total_symbols or 0, int(tf.reduce_max(indices).numpy()) + 1)
            caller_state = self._rng.state.read_value()
            self._rng.reset(self._state)
            try:
                self._cached_custom = self._model(1, needed, sampling_frequency)
                self._state = self._rng.state.read_value()
            finally:
                self._rng.reset(caller_state)
        path_gains, delays = self._cached_custom
        available = int(path_gains.shape[-1])
        if int(tf.reduce_max(indices).numpy()) >= available:
            raise ValueError(
                f"Custom continuous CIR holds {available} OFDM symbols; pass total_slots to "
                "SoundingChannel (or total_symbols to ContinuousCir) to cover the whole timeline"
            )
        return tf.gather(path_gains, indices, axis=-1), delays

    def _sample_tdl(self, indices, sampling_frequency: float):
        tdl = self._model
        doppler, alpha, phi, phi_0, delays = self._state
        sample_times = tf.cast(indices, tdl.rdtype) / sampling_frequency
        sample_times_8d = tf.expand_dims(insert_dims(sample_times, 6, 0), -1)

        argument = doppler * sample_times_8d * tf.cos(alpha) + phi
        h = tf.complex(tf.cos(argument), tf.sin(argument))
        norm = tf.complex(1.0 / tf.sqrt(tf.cast(tdl._num_sinusoids, tdl.rdtype)),
                          tf.constant(0.0, tdl.rdtype))
        h = norm * tf.reduce_sum(h, axis=-1)
        mean_powers = tf.expand_dims(insert_dims(tdl._mean_powers, 5, 0), -1)
        h = tf.cast(tf.sqrt(mean_powers), tdl.cdtype) * h

        if tdl._los:
            doppler_7d = tf.squeeze(doppler, axis=-1)
            times_7d = tf.squeeze(sample_times_8d, axis=-1)
            arg_spec = doppler_7d * times_7d * tf.cos(tdl._los_angle_of_arrival) + phi_0
            h_spec = tf.complex(tf.cos(arg_spec), tf.sin(arg_spec))
            los_scale = tf.cast(tf.sqrt(tdl._los_power), tdl.cdtype)
            h = tf.concat([h_spec * los_scale + h[:, :, :, :, :, :1, :], h[:, :, :, :, :, 1:, :]], axis=5)

        if tdl._spatial_corr_mat_sqrt is not None:
            h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
            h = tf.expand_dims(flatten_last_dims(h, 2), axis=-1)
            h = tf.squeeze(tf.matmul(tdl._spatial_corr_mat_sqrt, h), axis=-1)
            h = split_dim(h, [tdl._num_rx_ant, tdl._num_tx_ant], tf.rank(h) - 1)
            h = tf.transpose(h, [0, 1, 5, 2, 6, 3, 4])
        elif tdl._rx_corr_mat_sqrt is not None or tdl._tx_corr_mat_sqrt is not None:
            h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
            if tdl._rx_corr_mat_sqrt is not None:
                h = tf.matmul(tdl._rx_corr_mat_sqrt, h)
            if tdl._tx_corr_mat_sqrt is not None:
                h = tf.matmul(h, tdl._tx_corr_mat_sqrt, adjoint_b=True)
            h = tf.transpose(h, [0, 1, 5, 2, 6, 3, 4])

        return tf.stop_gradient(h), tf.stop_gradient(delays)
