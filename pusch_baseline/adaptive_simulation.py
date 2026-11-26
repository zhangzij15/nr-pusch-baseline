"""SRS-driven, single-UE uplink simulation on a slot timeline."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from itertools import count

import numpy as np
import tensorflow as tf
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.nr.utils import calculate_tb_size
from sionna.phy.ofdm import ResourceGrid

from .adaptation import select_grants, select_mcs_for_precoder
from .continuous_channel import ContinuousCir, _normal_cp_symbol_starts
from .link import PuschLink, _pusch_config, build_cir_model, precoder_amplitude_scale
from .mcs_stability import assess_mcs_stability
from .olla import McsIndexController, OllaController
from .srs import SrsEstimator
from .throughput_statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, add_throughput_statistics


def _ofdm_frequency_response(frequencies, path_gains, delays):
    if len(delays.shape) != 4:
        return cir_to_ofdm_channel(frequencies, path_gains, delays)

    # Common delays let all antenna pairs and OFDM symbols share one tap
    # Fourier matrix per link, avoiding a full path-by-subcarrier tensor.
    phase = (-2 * np.pi * frequencies) * delays[..., None]
    tap_response = tf.exp(tf.complex(tf.zeros_like(phase), phase))
    batch, rx, rx_ports, tx, tx_ports, num_paths, num_symbols = tf.unstack(tf.shape(path_gains))
    gains_by_link = tf.transpose(path_gains, (0, 1, 3, 2, 4, 6, 5))
    gains_by_link = tf.reshape(gains_by_link, (batch, rx, tx, rx_ports * tx_ports * num_symbols, num_paths))
    response_by_link = tf.matmul(gains_by_link, tap_response)
    response_by_link = tf.reshape(
        response_by_link, (batch, rx, tx, rx_ports, tx_ports, num_symbols, tf.size(frequencies)),
    )
    return tf.transpose(response_by_link, (0, 1, 3, 2, 4, 5, 6))


class SoundingChannel:
    """Sample CIR responses for a sounding slot and its following data slots.

    In ``per_period`` mode each SRS epoch draws an independent CIR trajectory.
    In ``continuous`` mode a single physical CIR realization evolves smoothly
    across all SRS periods using exact OFDM symbol timestamps.
    """

    def __init__(self, cfg, channel_model_factory, estimator, *,
                 full_sounding_grid: bool = False, total_slots: int | None = None):
        num_subcarriers = cfg.carrier.n_size_grid * 12
        subcarrier_spacing = cfg.carrier.subcarrier_spacing_khz * 1000
        self._frequencies = subcarrier_frequencies(
            num_subcarriers, subcarrier_spacing, precision=cfg.precision,
        )
        self._estimator = estimator
        self._full_sounding_grid = full_sounding_grid
        self._pilot_frequencies = tf.gather(self._frequencies, estimator.subcarrier_indices)
        self._normalize = cfg.channel.normalize
        self._continuous = (
            cfg.adaptation is not None and cfg.adaptation.channel_evolution == "continuous"
        )
        self._sampled_epoch = False
        self._drop_rms = None
        if cfg.channel.kind == "awgn":
            dtype = tf.complex128 if cfg.precision == "double" else tf.complex64
            self._awgn_response = tf.ones(
                (1, 1, cfg.num_rx_ant, 1, 1, 14, num_subcarriers), dtype=dtype,
            )
            self._model = None
        else:
            # The grid represents occupied tones with an integer CP. Physical
            # CIR timing is independent of FFT size and PRB allocation.
            average_cp = round(num_subcarriers * (cfg.slot_duration_s * subcarrier_spacing / 14 - 1))
            resource_grid = ResourceGrid(
                num_ofdm_symbols=14, fft_size=num_subcarriers,
                subcarrier_spacing=subcarrier_spacing, num_tx=1,
                cyclic_prefix_length=average_cp,
                num_streams_per_tx=cfg.pusch.num_antenna_ports, precision=cfg.precision,
            )
            self._sampling_frequency = 14 / cfg.slot_duration_s
            self._spacing_khz = cfg.carrier.subcarrier_spacing_khz
            self._first_carrier_symbol = 14 * cfg.carrier.slot_number
            cir = build_cir_model(cfg, resource_grid, channel_model_factory)
            total_symbols = total_slots * 14 if self._continuous and total_slots is not None else None
            self._model = ContinuousCir(cir, cfg.sweep.seed, total_symbols=total_symbols)

    def _ideal_sounding(self, channel_slot):
        symbols = tf.gather(channel_slot, self._estimator.symbol_indices, axis=2)
        return np.moveaxis(np.asarray(tf.reduce_mean(symbols, axis=2)), -1, 0)

    def slots(self, offsets, *, start_slot: int = 0, paired_sounding: bool = False):
        """Yield sounding channel info, then full responses on active data slots."""
        ideal_srs = self._full_sounding_grid or paired_sounding
        real_srs = not self._full_sounding_grid or paired_sounding
        if self._model is None:
            slot = self._awgn_response[0, 0, :, 0]
            ideal = self._ideal_sounding(slot) if ideal_srs else None
            real = self._estimator.sample_pilots(slot) if real_srs else None
            yield {"perfect": ideal, "real": real} if paired_sounding else (ideal if ideal_srs else real)
            for _ in offsets[1:]:
                yield self._awgn_response
            return

        if not self._continuous and self._sampled_epoch:
            self._model.next_drop()
            self._drop_rms = None
        self._sampled_epoch = True
        srs_abs_slot = start_slot + offsets[0]
        need_full_srs_slot = self._normalize and self._drop_rms is None
        srs_syms = (
            np.arange(14, dtype=np.int32) if need_full_srs_slot
            else np.asarray(self._estimator.symbol_indices, dtype=np.int32)
        )
        chunks = [srs_abs_slot * 14 + srs_syms]
        for off in offsets[1:]:
            chunks.append((start_slot + off) * 14 + np.arange(14, dtype=np.int32))
        all_indices = np.concatenate(chunks)
        origin = 0 if self._continuous else srs_abs_slot * 14
        times = _normal_cp_symbol_starts(all_indices + self._first_carrier_symbol, self._spacing_khz)
        times -= _normal_cp_symbol_starts(origin + self._first_carrier_symbol, self._spacing_khz)
        path_gains, delays = self._model.sample(
            all_indices - origin, self._sampling_frequency, sample_times=times,
        )

        n_srs = len(srs_syms)
        srs_gains = path_gains[..., :n_srs]
        if need_full_srs_slot:
            sounding = _ofdm_frequency_response(self._frequencies, srs_gains, delays)
            rms = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(sounding))))
            self._drop_rms = tf.cast(tf.where(rms > 0, rms, tf.ones_like(rms)), sounding.dtype)
            scaled = sounding[0, 0, :, 0] / self._drop_rms
            ideal = self._ideal_sounding(scaled) if ideal_srs else None
            real = self._estimator.sample_pilots(scaled) if real_srs else None
            del scaled, sounding
        else:
            ideal = real = None
            if ideal_srs:
                sounding = _ofdm_frequency_response(self._frequencies, srs_gains, delays)
                if self._drop_rms is not None:
                    sounding = sounding / self._drop_rms
                ideal = np.moveaxis(np.asarray(tf.reduce_mean(sounding[0, 0, :, 0], axis=2)), -1, 0)
                del sounding
            if real_srs:
                sounding = _ofdm_frequency_response(self._pilot_frequencies, srs_gains, delays)
                if self._drop_rms is not None:
                    sounding = sounding / self._drop_rms
                real = sounding[0, 0, :, 0]
                del sounding
        yield {"perfect": ideal, "real": real} if paired_sounding else (ideal if ideal_srs else real)

        cursor = n_srs
        for _ in offsets[1:]:
            response = _ofdm_frequency_response(
                self._frequencies, path_gains[..., cursor:cursor + 14], delays,
            )
            cursor += 14
            if self._drop_rms is not None:
                response = response / self._drop_rms
            yield response


def _sounding_epochs(settings, tb_limit, *, start_slot=0):
    """Yield an SRS slot followed by its PUSCH slots on the physical timeline.

    With a TDD pattern, SRS uses the first S slot, then each first S slot at
    least ``srs_period_slots`` after the previous SRS (the TS 38.211 periodicity
    T_SRS whenever it is a multiple of the S-slot spacing).
    """
    pattern = settings.tdd_pattern
    if pattern is None:
        data_per_epoch = settings.srs_period_slots - 1
        for sent in range(0, tb_limit, data_per_epoch):
            start = start_slot + sent + sent // data_per_epoch
            length = 1 + min(data_per_epoch, tb_limit - sent)
            yield range(start, start + length)
        return

    epoch = []
    sent = 0
    for slot in count(start_slot):
        kind = pattern[slot % len(pattern)]
        if kind == "S" and (not epoch or slot - epoch[0] >= settings.srs_period_slots):
            if epoch:
                yield epoch
            epoch = [slot]
        elif kind == "U" and epoch:
            epoch.append(slot)
            sent += 1
            if sent == tb_limit:
                yield epoch
                return


sounding_epochs = _sounding_epochs


def _grant_configurations(cfg, mcs_indices):
    options = {}
    candidates = {}
    sizes = {}
    for rank, tpmi in cfg.adaptation.rank_tpmi(cfg.pusch):
        grant = replace(cfg.pusch, rank=rank, tpmi=tpmi)
        pc = _pusch_config(replace(cfg, pusch=replace(grant, mcs_index=mcs_indices[0])))
        matrix = pc.precoding_matrix
        if matrix is None:
            matrix = np.eye(rank, dtype=np.complex64)
        matrix = matrix * precoder_amplitude_scale(
            matrix, cfg.pusch.num_antenna_ports, cfg.adaptation.tx_power_mode,
        )
        for mcs in mcs_indices:
            key = (rank, tpmi, mcs)
            if (rank, mcs) not in sizes:
                try:
                    pc.tb.mcs_index = mcs
                    tbs = int(pc.tb_size)
                    _, cb_size, _, _, _, cw_lengths = calculate_tb_size(
                        target_tb_size=tbs, num_coded_bits=pc.num_coded_bits,
                        target_coderate=pc.tb.target_coderate,
                        modulation_order=pc.tb.num_bits_per_symbol, num_layers=rank,
                    )
                    effective_rate = int(cb_size) / int(tf.reduce_max(cw_lengths))
                    if effective_rate > 0.95:
                        raise ValueError(f"Sionna cannot encode effective LDPC rate {effective_rate:.3f}")
                    sizes[rank, mcs] = tbs
                except (AssertionError, ValueError) as exc:
                    raise ValueError(f"Invalid adaptation candidate (rank={rank}, TPMI={tpmi}, "
                                     f"MCS={mcs}): {exc}") from exc
            candidates[key] = (matrix, sizes[rank, mcs])
            options[key] = replace(grant, mcs_index=mcs)
    return options, candidates


def _summarize_point(cfg, candidates, snr_db, num_slots, num_srs_slots, counts, srs_grants, stop_reason):
    num_tb = sum(count for count, _ in counts.values())
    tb_errors = sum(errors for _, errors in counts.values())
    attempted_bits = sum(count * candidates[key][1] for key, (count, _) in counts.items())
    delivered_bits = sum((count - errors) * candidates[key][1]
                         for key, (count, errors) in counts.items())
    slot_seconds = cfg.slot_duration_s
    net_thp = delivered_bits / (num_slots * slot_seconds * 1e6)
    point = {
        "snr_db": float(snr_db),
        "num_tb": num_tb,
        "tb_errors": tb_errors,
        "successful_tb": num_tb - tb_errors,
        "num_slots": num_slots,
        "num_srs_slots": num_srs_slots,
        "attempted_payload_bits": attempted_bits,
        "delivered_payload_bits": delivered_bits,
        "slot_seconds": slot_seconds,
        "ul_duty_cycle": cfg.sweep.ul_duty_cycle,
        "bler": tb_errors / num_tb,
        "thp_scheduled_mbps": delivered_bits / (num_tb * slot_seconds * 1e6),
        "thp_including_srs_mbps": net_thp,
        "thp_duty_cycle_mbps": net_thp * cfg.sweep.ul_duty_cycle,
        "thp_peak_mbps": attempted_bits / (num_tb * slot_seconds * 1e6),
        "stop_reason": stop_reason,
        "grant_counts": [
            {"rank": rank, "tpmi": tpmi, "mcs_index": mcs, "num_tb": count,
             "tb_errors": errors, "tb_size_bits": candidates[(rank, tpmi, mcs)][1]}
            for (rank, tpmi, mcs), (count, errors) in sorted(counts.items())
        ],
    }
    if cfg.adaptation.tdd_pattern is not None:
        point["num_idle_slots"] = num_slots - num_tb - num_srs_slots
        point["thp_wall_clock_mbps"] = net_thp
    if srs_grants:
        point["srs_grants"] = srs_grants
    return point


@dataclass
class _SweepPoint:
    noise_variance: tf.Tensor
    srs_error_variance: np.ndarray
    counts: dict
    srs_grants: dict
    mcs_trace: dict = field(default_factory=lambda: defaultdict(list))
    throughput_clusters: dict = field(default_factory=lambda: defaultdict(list))
    num_tb: int = 0
    num_slots: int = 0
    num_srs_slots: int = 0
    tb_errors: int = 0
    stop_reason: str = "max_tb"


def _slot_pusch_inputs(seed, absolute_slot, max_tb_size, noise_shape, real_dtype):
    """Pair the payload prefix and unit-variance complex noise by physical slot."""
    payload_seed = tf.constant([seed ^ 0x5041594C, absolute_slot], tf.int64)
    noise_seed = tf.constant([seed ^ 0x4157474E, absolute_slot], tf.int64)
    bits = tf.random.stateless_uniform(
        [1, 1, max_tb_size], payload_seed, minval=0, maxval=2, dtype=tf.int32, alg="philox",
    )
    components = tf.random.stateless_normal(
        (*noise_shape, 2), noise_seed, stddev=0.5**0.5, dtype=real_dtype, alg="philox",
    )
    return tf.cast(bits, real_dtype), tf.complex(components[..., 0], components[..., 1])


def _stability(trace, window_tb, olla):
    return assess_mcs_stability(
        [(sample["mcs_index"], sample["tb_error"]) for sample in trace],
        window_tb=window_tb, target_bler=olla.target_bler, bler_tolerance=olla.tolerance,
        max_mcs_spread=olla.max_mcs_spread,
    )


def _check_calibration(summary, trace, olla, lowest_mcs, highest_mcs):
    stats = _stability(trace, olla.window_tb, olla)
    recent = trace[-3 * olla.window_tb:]
    all_lowest = all(sample["mcs_index"] == lowest_mcs for sample in recent)
    all_highest = all(sample["mcs_index"] == highest_mcs for sample in recent)
    if stats["stable"]:
        status = "converged"
    elif stats["status"] != "bler_outside_window":
        status = stats["status"]
    elif stats["pooled_bler"] > olla.target_bler:
        status = "at_mcs_floor" if all_lowest else "above_target"
    else:
        status = "at_mcs_ceiling" if all_highest else "below_target"
    return {
        **summary, "calibration_bler": stats["pooled_bler"],
        "calibration_num_tb": len(recent), "calibration_status": status,
        "calibration_all_lowest_mcs": all_lowest, "calibration_all_highest_mcs": all_highest,
        "calibration_stability": stats,
    }


def _add_target_result(point, calibration, settings, highest_mcs, measurement_stability=None):
    bler = point["bler"]
    at_ceiling = all(grant["mcs_index"] == highest_mcs for grant in point["grant_counts"])
    if measurement_stability is not None and calibration["calibration_status"] == "at_mcs_floor":
        status = "at_mcs_floor"
    elif calibration["calibration_status"] not in {"converged", "at_mcs_ceiling"}:
        status = "calibration_not_converged"
    elif point["num_tb"] < settings.window_tb:
        status = "insufficient_samples"
    elif settings.in_target_window(bler):
        status = "at_target" if calibration["calibration_status"] == "converged" else "at_mcs_ceiling"
        if status == "at_target" and measurement_stability is not None and not measurement_stability["stable"]:
            status = "measurement_" + measurement_stability["status"]
    elif bler > settings.target_bler:
        status = "above_target"
    else:
        status = "at_mcs_ceiling" if at_ceiling else "below_target"

    point.update(calibration)
    point.update({
        "target_bler": settings.target_bler,
        "bler_tolerance": settings.tolerance,
        "target_status": status,
        "thp_at_target_mbps": point["thp_duty_cycle_mbps"] if status == "at_target" else None,
    })


def _record_cluster_start(clusters: list[dict], srs_slot: int, prev_slots: int) -> None:
    if clusters:
        # Charge idle slots up to the next sounding to the preceding period.
        clusters[-1]["num_slots"] += srs_slot - prev_slots
    clusters.append({
        "srs_slot_index": srs_slot,
        "num_slots": 1 if clusters else srs_slot + 1,
        "delivered_payload_bits": 0,
    })


def _schedule_grants(grants, controllers_at_snr, candidates, thresholds, margin_db, direct_mcs):
    if controllers_at_snr is None:
        return {name: key for name, (key, _) in grants.items()}
    if direct_mcs:
        return {name: (*key[:2], controllers_at_snr[name].mcs_index) for name, (key, _) in grants.items()}
    return {
        name: select_mcs_for_precoder(
            key[:2], sinr_db, candidates, thresholds, margin_db + controllers_at_snr[name].margin_offset_db,
        ) for name, (key, sinr_db) in grants.items()
    }


def _next_sounding_slot(settings, last_srs: int) -> int:
    """Find the next eligible sounding occasion on the same physical timeline."""
    earliest = last_srs + settings.srs_period_slots
    pattern = settings.tdd_pattern
    if pattern is None:
        return earliest
    return next(slot for slot in count(earliest) if pattern[slot % len(pattern)] == "S")


def _build_adaptive_metadata(
    cfg, metadata, estimator, *, direct_mcs, check_stability, mcs_indices, warmup_hash, measurement_hash,
):
    settings = cfg.adaptation
    olla = settings.olla
    tdd = settings.tdd_pattern is not None
    continuous = settings.channel_evolution == "continuous"
    metadata.update({
        "snr_reference": "nominal unit UE data-RE power budget before precoding, per-RX RE noise",
        "sampling": (
            "continuous CIR trajectory across all SRS periods"
            if continuous else
            "SRS-led channel trajectories with potentially correlated TBs per cycle"
        ),
        "throughput_bootstrap": {
            "unit": "SRS period",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "method": "percentile interval of resampled payload bits divided by resampled elapsed time",
            "scope": "observed measurement stage and controller trajectory; no warmup resampling",
        },
        "snr_pairing": "shared CIR, standardized SRS and PUSCH noise, and common payload prefixes per physical slot",
        "adaptation": {
            "srs_symbols": estimator.symbol_indices,
            "srs_cyclic_shifts": estimator.cyclic_shifts,
            "srs_sounded_prbs": estimator.num_sounded_prbs,
            "srs_estimator": settings.srs_estimator,
            "channel_evolution": settings.channel_evolution,
            "tx_power_mode": settings.tx_power_mode,
            "srs_power_per_active_re": estimator.power_per_active_re,
            "srs_power_per_port_active_re": estimator.power_per_port_active_re,
            "pusch_total_data_re_power": 1.0 if settings.tx_power_mode == "full_power" else None,
            "pusch_port_data_re_power_limit": (
                1 / cfg.pusch.num_antenna_ports if settings.tx_power_mode == "per_port" else None
            ),
            "ue_yaw_selection": (
                "uniform offset drawn once per continuous trajectory" if continuous and cfg.channel.ue_yaw_offsets_deg
                else "uniform offset drawn once per SRS period" if cfg.channel.ue_yaw_offsets_deg
                else "fixed"
            ),
            "policy": ("SRS selects rank and TPMI by sum mutual information across layers and ACK/NACK updates the MCS index"
                       if direct_mcs else "configured thresholds on MI-averaged post-LMMSE SINR"),
            "spatial_scoring": "12-tone PRB Gram averages for both SRS estimators, with noise debiasing for real SRS",
            "pusch_channel_estimator": cfg.channel.estimator,
            "tb_decoder": "shared TensorFlow graphs for native Sionna and low-rate RV0",
        },
    })
    if cfg.channel.kind != "awgn":
        metadata["adaptation"]["cir_symbol_clock"] = (
            "Built-in CDL/TDL use NR normal-CP symbol starts (TS 38.211 5.3.1). "
            "Custom callables receive a uniform symbol rate of 14 / slot_duration_s."
        )
    if olla is not None:
        metadata["adaptation"]["target_bler"] = olla.target_bler
        metadata["adaptation"]["feedback_control"] = olla.control
        metadata["adaptation"]["calibration"] = (
            "disjoint warmup and measurement samples; OLLA updates after every TB in both stages"
        )
    if direct_mcs:
        metadata["adaptation"]["mcs_axis"] = olla.mcs_axis
        metadata["adaptation"]["mcs_indices"] = mcs_indices
    if check_stability:
        metadata["adaptation"]["stability"] = {
            "windows": 3, "warmup_window_tb": olla.window_tb,
            "measurement_window_tb": cfg.sweep.num_tb_per_snr // 3,
            "max_mcs_spread": olla.max_mcs_spread,
        }
        metadata["channel_fingerprints_sha256"] = {"warmup": warmup_hash, "measurement": measurement_hash}
    if tdd:
        metadata["adaptation"]["tdd_pattern"] = settings.tdd_pattern
        metadata["adaptation"]["srs_trigger"] = (
            "first S slot, then the first S slot at least srs_period_slots after the previous SRS"
        )


def simulate_adaptive(cfg, metadata, channel_model_factory, on_point, *,
                      codebooks=None, receiver_variants=None, on_progress=None, phy_graph_cache=None):
    settings = cfg.adaptation
    sweep = cfg.sweep
    tdd = settings.tdd_pattern is not None
    thresholds = dict(settings.mcs_thresholds_db)
    olla = settings.olla
    direct_mcs = olla is not None and olla.control == "mcs_index"
    check_stability = direct_mcs and olla.max_mcs_spread is not None
    mcs_indices = tuple(sorted(settings.mcs_indices or thresholds)) if direct_mcs else tuple(thresholds)
    lowest_mcs, highest_mcs = min(mcs_indices), max(mcs_indices)
    options, candidates = _grant_configurations(cfg, mcs_indices)
    srs_candidates = tuple(key for key in candidates if not direct_mcs or key[2] == lowest_mcs)
    codebook_keys = {None: srs_candidates} if codebooks is None else {
        name: tuple(key for key in srs_candidates if key[1] in tpmi)
        for name, tpmi in codebooks.items()
    }
    variant_configs = {
        name: replace(cfg, channel=replace(cfg.channel, estimator=dmrs_mode),
                      adaptation=replace(settings, srs_estimator=srs_mode))
        for name, (srs_mode, dmrs_mode) in (receiver_variants or {}).items()
    }
    if variant_configs:
        codebook_keys = dict.fromkeys(variant_configs, srs_candidates)
    srs_groups = {}
    dmrs_modes = {}
    for name, keys in codebook_keys.items():
        variant = variant_configs.get(name, cfg)
        srs_groups.setdefault(variant.adaptation.srs_estimator, {})[name] = keys
        dmrs_modes[name] = variant.channel.estimator
    paired_sounding = len(srs_groups) > 1
    channel_by_dmrs = {variant.channel.estimator: variant.channel for variant in variant_configs.values()}
    channel_by_dmrs[cfg.channel.estimator] = cfg.channel
    estimator = SrsEstimator(
        cfg.pusch.num_antenna_ports, cfg.carrier.n_size_grid, settings.srs_sequence_id,
        start_symbol=settings.srs_start_symbol if tdd else None,
        power_per_active_re=1.0,
    )
    warmup_epochs = list(_sounding_epochs(settings, olla.warmup_tb)) if olla else []
    measurement_start = first_sounding = 0
    if warmup_epochs:
        first_sounding = _next_sounding_slot(settings, warmup_epochs[-1][0])
        # Preserve the initial lead-in before sounding when it fits after warmup.
        measurement_start = max(warmup_epochs[-1][-1] + 1, first_sounding - warmup_epochs[0][0])
    measurement_epochs = list(_sounding_epochs(settings, sweep.num_tb_per_snr, start_slot=first_sounding))
    total_slots = measurement_epochs[-1][-1] + 1
    channel = SoundingChannel(
        cfg, channel_model_factory, estimator, full_sounding_grid="perfect" in srs_groups, total_slots=total_slots,
    )
    slots_per_frame = 10 * cfg.carrier.subcarrier_spacing_khz // 15
    srs_noise = np.random.default_rng(sweep.seed)
    tb_decoder_cache = {}

    @lru_cache(maxsize=128 * len(set(dmrs_modes.values())))
    def link(dmrs_mode, key, slot_number):
        carrier = replace(cfg.carrier, slot_number=slot_number)
        return PuschLink(
            replace(cfg, carrier=carrier, pusch=options[key], channel=channel_by_dmrs[dmrs_mode]),
            external_channel=True, tb_decoder_cache=tb_decoder_cache, allow_low_rate=True,
            phy_graph_cache=phy_graph_cache,
        )

    reference_link = link(next(iter(dmrs_modes.values())), next(iter(candidates)), cfg.carrier.slot_number)
    max_tb_size = max(size for _, size in candidates.values())
    grid = reference_link.resource_grid
    noise_shape = (1, 1, cfg.num_rx_ant, grid.num_ofdm_symbols, grid.fft_size)
    noise_points = {snr_db: reference_link.noise_variance(snr_db) for snr_db in sweep.snr_db}
    controllers = None
    if olla is not None:
        controllers = {
            snr_db: {
                name: (McsIndexController(olla, cfg.pusch.mcs_index, mcs_indices)
                       if direct_mcs else OllaController(olla))
                for name in codebook_keys
            } for snr_db in noise_points
        }

    def run_stage(epochs, tb_limit, *, calibrating=False, slot_offset: int = 0):
        states = {
            snr_db: _SweepPoint(
                no, estimator.error_variance(float(no)),
                {name: defaultdict(lambda: [0, 0]) for name in codebook_keys},
                {name: [] for name in codebook_keys},
            ) for snr_db, no in noise_points.items()
        }
        channel_hashes = {mode: sha256() for mode in srs_groups} if check_stability else {}
        phase = "warmup" if calibrating else "measurement"
        progress_interval = max(1, tb_limit // 10)
        for epoch in epochs:
            active = {snr_db: state for snr_db, state in states.items() if state.stop_reason == "max_tb"}
            if not active:
                break
            srs_slot = epoch[0] - slot_offset
            responses = channel.slots(
                tuple(slot - epoch[0] for slot in epoch), start_slot=epoch[0], paired_sounding=paired_sounding,
            )
            sounding = next(responses)
            if not paired_sounding:
                sounding = {next(iter(srs_groups)): sounding}
            sounding = {mode: np.asarray(sample) for mode, sample in sounding.items()}
            for mode, digest in channel_hashes.items():
                digest.update(sounding[mode].tobytes())
            srs_seed = int(srs_noise.integers(0, 2**32))
            if "real" in sounding:
                observed, srs_standard_noise = estimator.prepare_pilots(sounding["real"], np.random.default_rng(srs_seed))
            decisions = {}
            for snr_db, state in active.items():
                if not calibrating:
                    for name in codebook_keys:
                        _record_cluster_start(state.throughput_clusters[name], srs_slot, state.num_slots)
                state.num_slots = srs_slot + 1
                state.num_srs_slots += 1
                noise = float(state.noise_variance)
                offsets = None if controllers is None or direct_mcs else {
                    name: controller.margin_offset_db for name, controller in controllers[snr_db].items()
                }
                grants = {}
                for mode, group in srs_groups.items():
                    ideal = mode == "perfect"
                    estimate = sounding[mode] if ideal else estimator.estimate_from_observation(
                        observed, noise, srs_standard_noise,
                    )
                    grants.update(select_grants(
                        estimate, noise, candidates, None if direct_mcs else thresholds,
                        settings.sinr_margin_db, group,
                        srs_error_variance=None if ideal else state.srs_error_variance,
                        margin_offsets_db=offsets,
                    ))
                decisions[snr_db] = grants
                if not calibrating and (tdd or codebooks is not None):
                    for name, ((rank, tpmi, mcs), sinr_db) in grants.items():
                        state.srs_grants[name].append({
                            "slot_index": srs_slot, "rank": rank, "tpmi": tpmi,
                            "mcs_index": controllers[snr_db][name].mcs_index if direct_mcs else mcs,
                            "effective_sinr_db": sinr_db,
                        })
            for absolute_data_slot, response in zip(epoch[1:], responses):
                data_slot = absolute_data_slot - slot_offset
                if channel_hashes:
                    channel_bytes = np.asarray(response[..., 0, 0]).tobytes()
                    for digest in channel_hashes.values():
                        digest.update(channel_bytes)
                # DM-RS and scrambling follow the physical slot of the shared CIR timeline.
                slot_number = (cfg.carrier.slot_number + absolute_data_slot) % slots_per_frame
                payload, standard_noise = _slot_pusch_inputs(
                    sweep.seed, absolute_data_slot, max_tb_size, noise_shape, response.dtype.real_dtype,
                )
                reuse_signal = len(decisions) > 1 or bool(variant_configs)
                # Within a slot the grant fixes the transmit grid and payload prefix.
                clean_signals = {}
                for snr_db, grants in decisions.items():
                    state = states[snr_db]
                    if state.stop_reason != "max_tb":
                        continue
                    scheduled = _schedule_grants(
                        grants, None if controllers is None else controllers[snr_db],
                        candidates, thresholds, settings.sinr_margin_db, direct_mcs,
                    )
                    errors = {}
                    decodes = {name: (dmrs_modes[name], key) for name, key in scheduled.items()}
                    for dmrs_mode, key in dict.fromkeys(decodes.values()):
                        pusch_link = link(dmrs_mode, key, slot_number)
                        symbols = pusch_link.resource_grid.num_ofdm_symbols
                        response_slice = response[..., :symbols, :]
                        payload_slice = payload[..., :pusch_link.tb_size_bits]
                        if reuse_signal and key not in clean_signals:
                            clean_signals[key] = pusch_link.noiseless_signal(payload_slice, response_slice)
                        errors[dmrs_mode, key] = pusch_link.count_block_errors(
                            1, state.noise_variance, channel_response=response_slice, payload=payload_slice,
                            standard_noise=standard_noise, noiseless_signal=clean_signals.get(key),
                        )
                    for name, key in scheduled.items():
                        failed = errors[decodes[name]]
                        if controllers is not None:
                            controller = controllers[snr_db][name]
                            before = controller.mcs_state if direct_mcs else None
                            controller.observe(
                                bool(failed), lowest_mcs=key[2] == lowest_mcs, highest_mcs=key[2] == highest_mcs,
                            )
                            if direct_mcs and (not calibrating or check_stability):
                                state.mcs_trace[name].append({
                                    "slot_index": data_slot, "mcs_index": key[2], "tb_error": bool(failed),
                                    "mcs_state_before": before, "mcs_state_after": controller.mcs_state,
                                })
                        if not calibrating:
                            totals = state.counts[name][key]
                            totals[0] += 1
                            totals[1] += failed
                            cluster = state.throughput_clusters[name][-1]
                            cluster["num_slots"] += data_slot + 1 - state.num_slots
                            if not failed:
                                cluster["delivered_payload_bits"] += candidates[key][1]
                    state.num_tb += 1
                    state.num_slots = data_slot + 1
                    if sweep.target_block_errors is not None:
                        state.tb_errors += errors[decodes[None]]
                        if state.num_tb >= sweep.min_tb_per_snr and state.tb_errors >= sweep.target_block_errors:
                            state.stop_reason = "target_block_errors"
                if on_progress is not None:
                    done = min(state.num_tb for state in states.values())
                    if done == tb_limit or done % progress_interval == 0:
                        on_progress(phase, done, tb_limit)
        return states, {mode: digest.hexdigest() for mode, digest in channel_hashes.items()}

    warmup_hash = {}
    if controllers is not None:
        warmup, warmup_hash = run_stage(warmup_epochs, olla.warmup_tb, calibrating=True)
        calibration = {
            snr_db: {name: controller.summary() for name, controller in group.items()}
            for snr_db, group in controllers.items()
        }
        if check_stability:
            for snr_db, group in calibration.items():
                for name, summary in group.items():
                    group[name] = _check_calibration(
                        summary, warmup[snr_db].mcs_trace[name], olla, lowest_mcs, highest_mcs,
                    )
    states, measurement_hash = run_stage(measurement_epochs, sweep.num_tb_per_snr, slot_offset=measurement_start)

    points = {name: [] for name in codebook_keys}
    for snr_db, state in states.items():
        for name in codebook_keys:
            point = _summarize_point(
                cfg, candidates, snr_db, state.num_slots, state.num_srs_slots,
                state.counts[name], state.srs_grants[name], state.stop_reason,
            )
            add_throughput_statistics(point, state.throughput_clusters[name])
            if controllers is not None:
                measured = _stability(state.mcs_trace[name], state.num_tb // 3, olla) if check_stability else None
                if measured is not None:
                    point["measurement_stability"] = measured
                _add_target_result(point, calibration[snr_db][name], olla, highest_mcs, measured)
                controller = controllers[snr_db][name]
                if direct_mcs:
                    point["measurement_mcs_state"] = controller.mcs_state
                    point["mcs_trace"] = state.mcs_trace[name]
                else:
                    point["measurement_margin_offset_db"] = controller.margin_offset_db
            points[name].append(point)
            if on_point is not None:
                on_point(deepcopy(point)) if name is None else on_point(name, deepcopy(point))

    def run_metadata(variant, mode):
        output = metadata.copy()
        _build_adaptive_metadata(
            variant, output, estimator, direct_mcs=direct_mcs, check_stability=check_stability,
            mcs_indices=mcs_indices, warmup_hash=warmup_hash.get(mode), measurement_hash=measurement_hash.get(mode),
        )
        output["adaptation"]["measurement_start_slot"] = measurement_start
        return output

    if variant_configs:
        return {
            name: {"config": variant.to_dict(), "metadata": run_metadata(variant, variant.adaptation.srs_estimator),
                   "points": points[name]}
            for name, variant in variant_configs.items()
        }
    metadata = run_metadata(cfg, settings.srs_estimator)
    if codebooks is None:
        return {"config": cfg.to_dict(), "metadata": metadata, "points": points[None]}

    runs = {}
    for name, tpmi in codebooks.items():
        variant_settings = replace(settings, rank_tpmi_candidates=tuple((cfg.pusch.rank, t) for t in tpmi))
        runs[name] = {
            "config": replace(cfg, adaptation=variant_settings).to_dict(),
            "metadata": {**metadata, "fixed_rank": cfg.pusch.rank, "codebook_tpmi": list(tpmi),
                         "pairing": ("shared channel, SRS, standardized PUSCH noise and payload prefixes "
                                     "with one decode per distinct grant")},
            "points": points[name],
        }
    return runs
