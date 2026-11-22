"""Unbiased Rank 1 LMMSE detection for the single-UE PUSCH link."""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.mapping import Demapper
from sionna.phy.ofdm import RemoveNulledSubcarriers
from sionna.phy.utils import expand_to_rank, flatten_last_dims


class RankOneDetector(Block):
    """Match Sionna's linear detector without forming receive covariance matrices."""

    def __init__(self, resource_grid, num_bits_per_symbol, *, precision):
        super().__init__(precision=precision)
        mask = resource_grid.pilot_pattern.mask[0, 0]
        self._data_indices = tf.where(tf.reshape(mask, [-1]) == 0)[:, 0]
        self._remove_nulls = RemoveNulledSubcarriers(resource_grid, precision=precision)
        self._demapper = Demapper("maxlog", "qam", num_bits_per_symbol, precision=precision)

    def call(self, y, h_hat, err_var, no):
        y = self._remove_nulls(y)
        h = h_hat[:, :, :, 0, 0]
        estimate_variance = expand_to_rank(err_var, 7, 0)[:, :, :, 0, 0]
        variance = expand_to_rank(no, 5, -1) + estimate_variance

        # A single desired stream and independent receive errors give diagonal
        # C. Debiasing the LMMSE estimate cancels its scalar shrinkage, leaving
        # x_hat = (h^H C^-1 y)/(h^H C^-1 h) and no_eff = 1/(h^H C^-1 h).
        weights = tf.math.conj(h) / tf.cast(variance, self.cdtype)
        gain = tf.reduce_sum(tf.math.real(weights * h), axis=2)
        numerator = tf.reduce_sum(weights * y, axis=2)
        symbols = tf.math.divide_no_nan(numerator, tf.cast(gain, self.cdtype))
        effective_variance = tf.math.reciprocal(gain)

        # [batch, 1, 1, data symbols], with the same RE ordering as PUSCH mapping.
        symbols = tf.gather(flatten_last_dims(symbols), self._data_indices, axis=-1)
        effective_variance = tf.gather(
            flatten_last_dims(effective_variance), self._data_indices, axis=-1,
        )
        return self._demapper(symbols[:, :, None], effective_variance[:, :, None])
