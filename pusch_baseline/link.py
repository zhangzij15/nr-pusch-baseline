"""Frequency-domain NR PUSCH link using the public Sionna 1.2.2 APIs."""

from __future__ import annotations

from copy import deepcopy
from math import radians, sqrt

import tensorflow as tf
from numpy import finfo, random
from scipy.constants import speed_of_light
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN, ApplyOFDMChannel, OFDMChannel
from sionna.phy.channel.tr38901 import CDL, TDL, PanelArray
from sionna.phy.nr import (
    PUSCHConfig, PUSCHLSChannelEstimator, PUSCHReceiver, PUSCHTransmitter, TBDecoder, TBEncoder,
)

from .config import SimulationConfig
from .dmrs_lmmse import DmrsLmmseEstimator
from .low_rate_tb import low_rate_plan, low_rate_tb
from .rank_one_detector import RankOneDetector


class _OrientationSampledCdl:
    """Choose one of the configured UE yaws for the whole CIR trajectory."""

    def __init__(self, models, seed):
        self._models = models
        self._seed = seed ^ 0x594157
        self.rewind()

    def rewind(self):
        # Keep orientation draws independent of the Sionna fading stream.
        self._rng = random.default_rng(self._seed)

    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        model = self._models[self._rng.integers(len(self._models))]
        return model(batch_size, num_time_steps, sampling_frequency)


class PairedCir:
    """Give an eager CIR model its own rewindable Sionna TensorFlow RNG stream.

    Custom models using other random generators or mutable state must manage
    their own replay. The caller's PUSCH random stream is restored after each
    invocation.
    """

    def __init__(self, model, seed: int):
        self._model = model
        self._rng = sionna_config.tf_rng
        channel_seed = seed ^ 0x9E3779B9
        self._initial_state = tf.random.Generator.from_seed(
            channel_seed, alg=self._rng.algorithm,
        ).state.read_value()
        self.rewind()

    def rewind(self) -> None:
        self._state = self._initial_state
        if isinstance(self._model, _OrientationSampledCdl):
            self._model.rewind()

    def __call__(self, batch_size, num_time_steps, sampling_frequency):
        caller_state = self._rng.state.read_value()
        self._rng.reset(self._state)
        try:
            response = self._model(batch_size, num_time_steps, sampling_frequency)
            self._state = self._rng.state.read_value()
            return response
        finally:
            self._rng.reset(caller_state)


def _pusch_config(cfg: SimulationConfig) -> PUSCHConfig:
    carrier = cfg.carrier
    grant = cfg.pusch
    dmrs = grant.dmrs
    pc = PUSCHConfig()
    pc.carrier.cyclic_prefix = "normal"
    pc.carrier.subcarrier_spacing = carrier.subcarrier_spacing_khz
    pc.carrier.n_size_grid = carrier.n_size_grid
    pc.carrier.n_cell_id = carrier.cell_id
    pc.carrier.slot_number = carrier.slot_number
    pc.num_antenna_ports = grant.num_antenna_ports
    pc.num_layers = grant.rank
    pc.precoding = grant.precoding
    pc.tpmi = grant.tpmi
    pc.n_rnti = grant.rnti
    pc.mapping_type = grant.mapping_type
    pc.symbol_allocation = list(grant.symbol_allocation)
    pc.transform_precoding = False
    pc.dmrs.config_type = dmrs.config_type
    pc.dmrs.length = dmrs.length
    pc.dmrs.additional_position = dmrs.additional_position
    pc.dmrs.num_cdm_groups_without_data = dmrs.num_cdm_groups_without_data
    pc.dmrs.dmrs_port_set = grant.dmrs_ports
    pc.dmrs.n_id = dmrs.n_id
    pc.dmrs.n_scid = dmrs.n_scid
    pc.dmrs.type_a_position = dmrs.type_a_position
    pc.tb.mcs_index = grant.mcs_index
    pc.tb.mcs_table = grant.mcs_table
    pc.tb.n_id = grant.data_scrambling_id
    pc.check_config()
    if pc.num_coded_bits <= 0:
        raise ValueError("The configured grant has no data resource elements.")
    return pc


def _panel(panel, carrier_frequency, precision):
    return PanelArray(
        num_rows_per_panel=panel.rows,
        num_cols_per_panel=panel.cols,
        polarization=panel.polarization,
        polarization_type=panel.polarization_type,
        antenna_pattern=panel.pattern,
        carrier_frequency=carrier_frequency,
        element_vertical_spacing=panel.vertical_spacing_lambda,
        element_horizontal_spacing=panel.horizontal_spacing_lambda,
        precision=precision,
    )


def build_cir_model(cfg, resource_grid, channel_model_factory):
    params = cfg.channel
    if params.kind == "custom":
        model = channel_model_factory(cfg, resource_grid)
        if not callable(model):
            raise TypeError("channel_model_factory must return a callable CIR model.")
        return model

    frequency = params.carrier_frequency_ghz * 1e9
    speed = params.ue_speed_kmh / 3.6
    model_args = dict(
        model=params.profile,
        delay_spread=params.delay_spread_ns * 1e-9,
        carrier_frequency=frequency,
        min_speed=speed,
        max_speed=speed,
        precision=cfg.precision,
    )
    if params.kind == "tdl":
        return TDL(
            **model_args,
            num_rx_ant=cfg.num_rx_ant,
            num_tx_ant=cfg.pusch.num_antenna_ports,
        )
    ut_array = _panel(params.ue_panel, frequency, cfg.precision)
    bs_array = _panel(params.bs_panel, frequency, cfg.precision)
    yaw, pitch, roll = params.ue_panel.orientation_deg
    bs_orientation = [radians(angle) for angle in params.bs_panel.orientation_deg]

    def cdl(yaw_offset):
        return CDL(
            **model_args,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            ut_orientation=[radians(yaw + yaw_offset), radians(pitch), radians(roll)],
            bs_orientation=bs_orientation,
        )

    if params.ue_yaw_offsets_deg:
        models = tuple(cdl(offset) for offset in params.ue_yaw_offsets_deg)
        return _OrientationSampledCdl(models, cfg.sweep.seed)
    return cdl(0.0)


def precoder_amplitude_scale(matrix, num_tx_ports, mode):
    """Set the data-RE power of the precoder to the configured power budget."""
    port_powers = (abs(matrix)**2).sum(axis=1) if matrix is not None else None
    if mode == "per_port":
        # Independent, unit-energy layers give each port the energy in its row.
        peak_port_power = float(port_powers.max()) if port_powers is not None else 1.0
        return 1 / sqrt(num_tx_ports * peak_port_power)
    total_power = float(port_powers.sum()) if port_powers is not None else num_tx_ports
    return 1 / sqrt(total_power)


def _build_channel(cfg, resource_grid, cir_model):
    params = cfg.channel
    perfect_csi = params.estimator == "perfect"
    precision = cfg.precision
    if params.kind == "awgn":
        noise = AWGN(precision=precision)

        def apply_awgn(x, no):
            # One TX port into identical unit-gain links, with independent RX noise.
            y = noise(tf.repeat(x, cfg.num_rx_ant, axis=2), no)
            if perfect_csi:
                h = tf.ones(
                    [tf.shape(x)[0], 1, cfg.num_rx_ant, 1, 1,
                     resource_grid.num_ofdm_symbols, resource_grid.fft_size],
                    dtype=x.dtype,
                )
                return y, h
            return y

        return apply_awgn

    return OFDMChannel(
        cir_model,
        resource_grid,
        normalize_channel=params.normalize,
        return_channel=perfect_csi,
        precision=precision,
    )


def _compiled_tb_decoder(pc, precision, cache):
    # DM-RS and the precoder depend on slot and TPMI. TB coding and scrambling
    # depend only on these parameters, so links can share the same TF graph.
    settings = {
        "target_tb_size": int(pc.tb_size),
        "num_coded_bits": int(pc.num_coded_bits),
        "target_coderate": float(pc.tb.target_coderate),
        "num_bits_per_symbol": int(pc.tb.num_bits_per_symbol),
        "num_layers": pc.num_layers,
        "n_rnti": pc.n_rnti,
        "n_id": pc.carrier.n_cell_id if pc.tb.n_id is None else pc.tb.n_id,
        "precision": precision,
    }
    key = tuple(settings.values())
    if key not in cache:
        encoder = TBEncoder(channel_type="PUSCH", **settings)
        decoder = TBDecoder(encoder, precision=precision)
        cache[key] = tf.function(decoder, autograph=False, reduce_retracing=True)
    return cache[key]


# Mid-rate placeholder MCS entries (mcs_table, mcs_index) per QAM modulation
# order, used only to initialize Sionna's waveform components before substituting
# the RV0 low-rate TB codec.
_PLACEHOLDER_MCS_BY_QAM = {
    2: (1, 5),   # QPSK, R = 379/1024
    4: (1, 14),  # 16-QAM, R = 553/1024
    6: (1, 22),  # 64-QAM, R = 719/1024
    8: (2, 21),  # 256-QAM, R = 711/1024
}


def _build_channel_estimator(cfg: SimulationConfig, pc: PUSCHConfig, resource_grid):
    estimator = cfg.channel.estimator
    if estimator == "perfect":
        return "perfect"
    if estimator == "lmmse":
        channel = cfg.channel
        delay_spread_s = 0.0
        doppler_hz = 0.0
        if channel.kind != "awgn":
            delay_spread_s = channel.delay_spread_ns * 1e-9
            doppler_hz = channel.ue_speed_kmh / 3.6 * channel.carrier_frequency_ghz * 1e9 / speed_of_light
        return DmrsLmmseEstimator(
            resource_grid, delay_spread_s, doppler_hz,
            precision=cfg.precision, symbol_duration_s=cfg.slot_duration_s / 14,
        )
    if pc.dmrs.length == 2:
        # Short grants can schedule fewer DM-RS pairs than the configured maximum.
        # Pass the actual pair count so Sionna averages the two symbols in each pair.
        num_dmrs_pairs = len(pc.dmrs_symbol_indices) // pc.dmrs.length
        return PUSCHLSChannelEstimator(
            resource_grid,
            dmrs_length=pc.dmrs.length,
            dmrs_additional_position=num_dmrs_pairs - 1,
            num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
            interpolation_type="lin",
            precision=cfg.precision,
        )
    return None


def _build_grant_metadata(pc: PUSCHConfig, tx_data_power: float, tx_amplitude_scale: float) -> dict:
    return {
        "waveform": "CP-OFDM",
        "tb_size_bits": int(pc.tb_size),
        "coded_bits": int(pc.num_coded_bits),
        "qam_bits_per_symbol": int(pc.tb.num_bits_per_symbol),
        "target_coderate": float(pc.tb.target_coderate),
        "allocated_prbs": int(pc.num_resource_blocks),
        "dmrs_slot_symbols": [int(index) for index in pc.dmrs_symbol_indices],
        "dmrs_ports": pc.dmrs.dmrs_port_set,
        "expected_tx_data_re_power": float(tx_data_power),
        "effective_tx_data_re_power": float(tx_data_power) * tx_amplitude_scale**2,
    }


class PuschLink:
    """Single-UE transmitter, channel and receiver for a fixed PUSCH grant."""

    def __init__(self, cfg: SimulationConfig, channel_model_factory=None, *,
                 external_channel=False, tb_decoder_cache=None, allow_low_rate=True):
        pc = _pusch_config(cfg)
        rate_plan = low_rate_plan(pc) if allow_low_rate else None
        tx_config = pc
        if rate_plan is not None:
            # The native transmitter constructs its own TB encoder. A grant with
            # the same RE grid and modulation order initializes its waveform
            # components before supplying the actual grant's RV0 codec.
            qam = int(pc.tb.num_bits_per_symbol)
            if qam not in _PLACEHOLDER_MCS_BY_QAM:
                raise ValueError(f"Unsupported modulation order for low-rate PUSCH: {qam}")
            tx_config = deepcopy(pc)
            tx_config.tb.mcs_table, tx_config.tb.mcs_index = _PLACEHOLDER_MCS_BY_QAM[qam]
        self._transmitter = PUSCHTransmitter(
            tx_config, return_bits=True, output_domain="freq", precision=cfg.precision
        )
        self.resource_grid = self._transmitter.resource_grid
        self._perfect_csi = cfg.channel.estimator == "perfect"
        channel_estimator = _build_channel_estimator(cfg, pc, self.resource_grid)
        detector = None
        if pc.num_layers == 1 and (cfg.channel.estimator == "lmmse" or (external_channel and cfg.num_rx_ant >= 8)):
            detector = RankOneDetector(
                self.resource_grid, int(pc.tb.num_bits_per_symbol), precision=cfg.precision,
            )
        if rate_plan is not None:
            codec, tb_decoder = low_rate_tb(pc, rate_plan, cfg.precision, tb_decoder_cache)
            self._transmitter._tb_encoder = codec.encode
            self._transmitter._tb_size = int(pc.tb_size)
        else:
            tb_decoder = None if tb_decoder_cache is None else _compiled_tb_decoder(
                pc, cfg.precision, tb_decoder_cache,
            )
        self._receiver = PUSCHReceiver(
            self._transmitter,
            channel_estimator=channel_estimator,
            mimo_detector=detector,
            tb_decoder=tb_decoder,
            return_tb_crc_status=True,
            input_domain="freq",
            precision=cfg.precision,
        )
        self._cir = None
        if external_channel:
            self._channel = None
            self._apply_ofdm = ApplyOFDMChannel(precision=cfg.precision)
        else:
            if cfg.channel.kind != "awgn":
                self._cir = PairedCir(
                    build_cir_model(cfg, self.resource_grid, channel_model_factory), cfg.sweep.seed,
                )
            self._channel = _build_channel(cfg, self.resource_grid, self._cir)
        self._real_dtype = tf.float64 if cfg.precision == "double" else tf.float32
        self._noise_limits = finfo(self._real_dtype.as_numpy_dtype)
        self.tb_size_bits = int(pc.tb_size)
        matrix = pc.precoding_matrix
        tx_data_power = pc.num_layers if matrix is None else (abs(matrix)**2).sum()
        complex_dtype = tf.complex128 if cfg.precision == "double" else tf.complex64
        tx_power_mode = cfg.adaptation.tx_power_mode if cfg.adaptation else "full_power"
        tx_amplitude_scale = (
            precoder_amplitude_scale(matrix, pc.num_antenna_ports, tx_power_mode)
            if external_channel else 1.0
        )
        self._tx_amplitude_scale = tf.constant(tx_amplitude_scale, dtype=complex_dtype)
        self.grant_metadata = _build_grant_metadata(pc, tx_data_power, tx_amplitude_scale)

    def rewind_channel(self) -> None:
        if self._cir is not None:
            self._cir.rewind()

    def noise_variance(self, snr_db: float):
        try:
            variance = 10.0 ** (-snr_db / 10.0)
        except OverflowError as exc:
            raise ValueError(f"SNR {snr_db} dB is outside the numeric range.") from exc
        if not self._noise_limits.tiny <= variance <= self._noise_limits.max:
            raise ValueError(f"SNR {snr_db} dB is outside the numeric range.")
        return tf.constant(variance, dtype=self._real_dtype)

    def count_block_errors(self, batch_size: int, no, *, channel_response=None) -> int:
        return int(self.count_block_errors_tensor(batch_size, no, channel_response=channel_response).numpy())

    def count_block_errors_tensor(self, batch_size: int, no, *, channel_response=None) -> tf.Tensor:
        """Return the TB error count without reading it back from TensorFlow."""
        x, bits = self._transmitter(batch_size)
        if self._channel is None:
            if channel_response is None:
                raise ValueError("A frequency response is required for the external channel")
            # Set the transmit power on the data grid and DM-RS before adding noise.
            y = self._apply_ofdm(x * self._tx_amplitude_scale, channel_response, no)
            # The perfect-CSI receiver applies the native precoder to this response.
            received = (y, channel_response * self._tx_amplitude_scale) if self._perfect_csi else y
        else:
            received = self._channel(x, no)
        if self._perfect_csi:
            y, h = received
            estimated_bits, crc_ok = self._receiver(y, no, h)
        else:
            estimated_bits, crc_ok = self._receiver(received, no)
        tf.debugging.assert_shapes([
            (bits, (batch_size, 1, self.tb_size_bits)),
            (estimated_bits, (batch_size, 1, self.tb_size_bits)),
            (crc_ok, (batch_size, 1)),
        ])
        payload_error = tf.reduce_any(tf.not_equal(bits, estimated_bits), axis=-1)
        failed = tf.logical_or(tf.logical_not(crc_ok), payload_error)
        return tf.math.count_nonzero(failed)
