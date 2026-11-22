"""Frequency-domain NR SRS resources and least-squares channel estimation."""

from __future__ import annotations

from math import isfinite, isqrt, sqrt

import numpy as np

from .config import SRS_BANDWIDTHS_PRB


# 3GPP TS 38.211 Table 5.2.2.2-4, one row per sequence group u.
_SHORT_PHASE = np.array([
    [-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3],
    [-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3],
    [-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3],
    [1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3],
    [-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3],
    [-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3],
    [-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3],
    [-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3],
    [-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3],
    [1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3],
    [-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3],
    [-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3],
    [3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3],
    [-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3],
    [-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3],
    [-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3],
    [-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3],
    [3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3],
    [-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3],
    [-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3],
    [-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3],
    [3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3],
    [-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3],
    [-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3],
    [-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3],
    [-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3],
    [-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3],
    [3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3],
    [-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3],
    [3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3],
], dtype=np.int8)


def _sequence(length: int, group: int) -> np.ndarray:
    if length == 24:
        return np.exp((1j * np.pi / 4) * _SHORT_PHASE[group])

    # TS 38.211 §5.2.2.1, with v=0 for SRS without group or sequence hopping.
    prime = next(
        n for n in range(length - 1, 1, -1)
        if all(n % divisor for divisor in range(2, isqrt(n) + 1))
    )
    root = (2 * prime * (group + 1) + 31) // 62
    index = np.arange(length) % prime
    return np.exp((-1j * np.pi * root / prime) * index * (index + 1))


def _checked_noise_variance(value: float) -> float:
    variance = float(value)
    if not isfinite(variance) or variance < 0:
        raise ValueError("SRS noise_variance must be finite and nonnegative")
    return variance


class SrsEstimator:
    """Sound all physical TX ports with a single multi-port NR SRS resource.

    Per 3GPP TS 38.211 §6.4.1.4 and TS 38.214 §6.1.1.1, a single resource with
    N_ap = num_tx_ports ports (1000..1000+N_ap-1) occupies N_symb = N_ap
    consecutive repeated OFDM symbols with comb 2, comb offset 0, B_SRS=0, and
    no hopping. Ports share the comb subcarriers via orthogonal cyclic shifts
    n_SRS^cs,i = (8 * i / N_ap) mod 8 (0; 0/4; or 0/2/4/6) and split the
    configured total active-RE power equally (1 / N_ap per port).
    """

    def __init__(self, num_tx_ports: int, num_prbs: int, sequence_id: int, *,
                 start_symbol: int | None = None, power_per_active_re: float = 1.0):
        if type(num_tx_ports) is not int or num_tx_ports not in (1, 2, 4):
            raise ValueError("SRS supports 1, 2, or 4 TX ports")
        if type(num_prbs) is not int or not 4 <= num_prbs <= 275:
            raise ValueError("SRS requires a grid of 4 to 275 PRBs")
        if type(sequence_id) is not int or not 0 <= sequence_id <= 1023:
            raise ValueError("SRS sequence_id must be an integer in [0, 1023]")
        if not isfinite(power_per_active_re) or power_per_active_re <= 0:
            raise ValueError("SRS power_per_active_re must be finite and positive")
        num_symbols = num_tx_ports
        last_start = 14 - num_symbols
        if start_symbol is None:
            start_symbol = last_start
        elif type(start_symbol) is not int or not 0 <= start_symbol <= last_start:
            raise ValueError(f"SRS start_symbol must be an integer in [0, {last_start}]")

        self._num_subcarriers = 12 * num_prbs
        self._num_tx_ports = num_tx_ports
        self._num_symbols = num_symbols
        self.power_per_active_re = float(power_per_active_re)
        self.power_per_port_active_re = self.power_per_active_re / num_tx_ports
        self.num_sounded_prbs = max(width for width in SRS_BANDWIDTHS_PRB if width <= num_prbs)
        self.symbol_indices = tuple(range(start_symbol, start_symbol + num_symbols))
        self.cyclic_shifts = tuple(8 * port // num_tx_ports for port in range(num_tx_ports))
        self.subcarrier_indices = np.arange(0, 12 * self.num_sounded_prbs, 2)
        self.reference_sequence = _sequence(self.subcarrier_indices.size, sequence_id % 30)
        pilot_index = np.arange(self.subcarrier_indices.size)
        shifts = np.asarray(self.cyclic_shifts)[:, None]
        self.port_sequences = self.reference_sequence[None, :] * np.exp(
            (1j * 2 * np.pi / 8) * shifts * pilot_index[None, :]
        )
        self.subcarrier_indices.setflags(write=False)
        self.reference_sequence.setflags(write=False)
        self.port_sequences.setflags(write=False)

        # Orthogonal block OCC demodulator over N_ap consecutive comb pilots.
        # Since sounded PRBs are a multiple of 4, pilot count 6*N_PRB is divisible by 1, 2, 4.
        block_offset = np.arange(num_tx_ports)[:, None]
        self._cdm_demod = np.exp(
            (-1j * 2 * np.pi / 8) * block_offset * np.asarray(self.cyclic_shifts)[None, :]
        ) / num_tx_ports
        num_blocks = self.subcarrier_indices.size // num_tx_ports
        block_centers = np.arange(num_blocks) * (2 * num_tx_ports) + (num_tx_ports - 1)
        subcarriers = np.arange(self._num_subcarriers)
        left = np.clip(np.searchsorted(block_centers, subcarriers, side="right") - 1, 0, num_blocks - 2)
        self._interp_left = left
        self._interp_right = left + 1
        span = float(2 * num_tx_ports)
        self._interp_weight = np.clip((subcarriers - block_centers[left]) / span, 0.0, 1.0)
        self._interp_variance_scale = (1.0 - self._interp_weight) ** 2 + self._interp_weight ** 2

    def error_variance(self, noise_variance: float) -> np.ndarray:
        """Return per-subcarrier variance from AWGN on the SRS observations."""
        noise_variance = _checked_noise_variance(noise_variance)
        # Across N_symb symbols and N_ap pilots per OCC block, each port collects
        # N_symb * N_ap * (power_per_active_re / N_ap) = N_symb * power_per_active_re.
        block_variance = noise_variance / (self._num_symbols * self.power_per_active_re)
        return block_variance * self._interp_variance_scale

    def sample_pilots(self, channel) -> np.ndarray:
        """Extract [RX,TX,symbol,pilot] from an [RX,TX,14,subcarrier] channel."""
        channel = np.asarray(channel)
        if (channel.ndim != 4 or channel.shape[0] < 1
                or channel.shape[1:] != (self._num_tx_ports, 14, self._num_subcarriers)):
            raise ValueError("SRS channel must have shape [RX,TX,14,subcarrier]")
        pilot_channel = channel[:, :, self.symbol_indices, :]
        return pilot_channel[..., self.subcarrier_indices]

    def estimate(self, channel, noise_variance: float, rng: np.random.Generator) -> np.ndarray:
        """Return [subcarrier,RX,TX] from an [RX,TX,14,subcarrier] channel."""
        return self.estimate_from_pilots(self.sample_pilots(channel), noise_variance, rng)

    def estimate_from_pilots(
        self, pilot_channel, noise_variance: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Estimate [subcarrier,RX,TX] from the true channel on active pilot REs."""
        pilot_channel = np.asarray(pilot_channel)
        expected_tail = (self._num_tx_ports, self._num_symbols, self.subcarrier_indices.size)
        if pilot_channel.ndim != 4 or pilot_channel.shape[0] < 1 or pilot_channel.shape[1:] != expected_tail:
            raise ValueError("SRS pilot channel must have shape [RX,TX,symbol,pilot]")
        noise_variance = _checked_noise_variance(noise_variance)
        dtype = np.result_type(pilot_channel.dtype, np.complex64)
        port_amplitude = sqrt(self.power_per_port_active_re)
        tx_sequences = (port_amplitude * self.port_sequences).astype(dtype, copy=False)
        # All TX ports transmit simultaneously on every SRS symbol via cyclic-shift CDM.
        observed = np.einsum("rtmk,tk->rmk", pilot_channel, tx_sequences)
        if noise_variance > 0:
            real_dtype = observed.real.dtype
            noise = rng.standard_normal(observed.shape, dtype=real_dtype)
            noise = noise + 1j * rng.standard_normal(observed.shape, dtype=real_dtype)
            observed = observed + sqrt(noise_variance / 2) * noise
        base = self.reference_sequence.astype(dtype, copy=False)
        despread = (observed.mean(axis=1) * base.conj()) / port_amplitude
        blocks = despread.reshape(despread.shape[0], -1, self._num_tx_ports)
        demod = self._cdm_demod.astype(dtype, copy=False)
        ls = np.moveaxis(blocks @ demod, 1, 0)  # [block, RX, TX]
        weight = self._interp_weight.astype(observed.real.dtype)[:, None, None]
        return (1 - weight) * ls[self._interp_left] + weight * ls[self._interp_right]
