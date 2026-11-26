"""Rank 1 DM-RS estimation with a separable time-frequency covariance prior."""

from functools import lru_cache

import numpy as np
import tensorflow as tf
from scipy.special import j0
from sionna.phy import Block
from sionna.phy.ofdm import RemoveNulledSubcarriers
from sionna.phy.utils import expand_to_rank


@lru_cache(maxsize=32)
def _covariance_basis(size, positions, scale, axis):
    """Cache the dominant pilot eigenmodes and their projection to the grid."""
    delta = np.arange(size)[:, None] - np.asarray(positions)[None, :]
    covariance = j0(scale * delta) if axis == "time" else 1 / (1 + 1j * scale * delta)
    values, vectors = np.linalg.eigh(covariance[list(positions)])
    # Tiny modes contribute negligible power and are poorly conditioned.
    active = values > values[-1] * 1e-6
    vectors = vectors[:, active]
    return values[active], vectors, covariance @ vectors


class DmrsLmmseEstimator(Block):
    """Use independent rank-one pilot observations and an empirical channel power.

    A causal exponential power-delay profile gives frequency covariance.
    The time covariance uses Jakes' model at the configured maximum Doppler.
    The posterior mean and marginal error variance account for pilot noise
    when the receiver computes the channel gain and demapper reliability.
    """

    def __init__(self, resource_grid, delay_spread_s, doppler_hz, *, precision, symbol_duration_s=None):
        super().__init__(precision=precision)
        pattern = resource_grid.pilot_pattern
        mask = pattern.mask.numpy()[0, 0].astype(bool)
        pilots = pattern.pilots.numpy()[0, 0]
        occupied = np.zeros(mask.shape, dtype=pilots.dtype)
        occupied[mask] = pilots
        active = occupied != 0
        symbols = np.flatnonzero(active.any(axis=1))
        subcarriers = np.flatnonzero(active.any(axis=0))
        pilot_grid = occupied[np.ix_(symbols, subcarriers)]
        power = np.abs(pilot_grid)**2
        if not power.size or not np.allclose(power, power.flat[0]) or power.flat[0] <= 0:
            raise ValueError("Rank 1 LMMSE requires a rectangular, constant-power DM-RS grid.")

        symbol_duration = resource_grid.ofdm_symbol_duration if symbol_duration_s is None else float(symbol_duration_s)
        self._remove_nulls = RemoveNulledSubcarriers(resource_grid, precision=precision)
        self._symbols = tf.constant(symbols)
        self._subcarriers = tf.constant(subcarriers)
        self._pilot_inverse = tf.constant(1 / pilot_grid, self.cdtype)
        self._pilot_power = tf.constant(power.flat[0], self.rdtype)
        eig_time, pilot_time, grid_time = _covariance_basis(
            mask.shape[0], tuple(symbols), 2 * np.pi * doppler_hz * symbol_duration,
            "time",
        )
        eig_freq, pilot_freq, grid_freq = _covariance_basis(
            mask.shape[1], tuple(subcarriers), 2 * np.pi * delay_spread_s * resource_grid.subcarrier_spacing,
            "frequency",
        )
        values = eig_time[:, None] * eig_freq[None, :]
        self._mode_power = tf.constant(values, self.rdtype)
        self._mode_energy = tf.constant(np.sum(values**2), self.rdtype)
        self._pilot_time = tf.constant(pilot_time.conj().T, self.cdtype)
        self._pilot_freq = tf.constant(pilot_freq.conj(), self.cdtype)
        self._grid_time = tf.constant(grid_time, self.cdtype)
        self._grid_freq = tf.constant(grid_freq.T, self.cdtype)
        self._grid_time_power = tf.constant(np.abs(grid_time)**2, self.rdtype)
        self._grid_freq_power = tf.constant(np.abs(grid_freq.T)**2, self.rdtype)

    def call(self, y, no):
        y = self._remove_nulls(y)
        pilot_y = tf.gather(tf.gather(y, self._symbols, axis=-2), self._subcarriers, axis=-1)
        pilot_h = pilot_y * self._pilot_inverse
        modes = tf.matmul(tf.matmul(self._pilot_time, pilot_h), self._pilot_freq)
        pilot_noise = expand_to_rank(no, 5, -1) / self._pilot_power

        # Fit one channel-power scale per batch and receiver by matching the
        # projected pilot power, pooling antennas and emphasizing strong modes.
        mode_energy = tf.math.real(modes * tf.math.conj(modes))
        signal_energy = (tf.reduce_mean(mode_energy, axis=2, keepdims=True)
                         - tf.reduce_mean(pilot_noise, axis=2, keepdims=True))
        power = tf.reduce_sum(signal_energy * self._mode_power, axis=(-2, -1), keepdims=True)
        power = tf.maximum(power / self._mode_energy, 0)

        # The pilot covariance is diagonal in this basis. Apply its inverse
        # there, then use the cross covariance to predict the full grid.
        weight = tf.math.divide_no_nan(power, power * self._mode_power + pilot_noise)
        estimate = tf.matmul(tf.matmul(self._grid_time, modes * tf.cast(weight, self.cdtype)),
                             self._grid_freq)
        explained = tf.matmul(tf.matmul(self._grid_time_power, weight), self._grid_freq_power)
        error = power * tf.maximum(1 - explained, 0)
        return estimate[:, :, :, None, None], error[:, :, :, None, None]
