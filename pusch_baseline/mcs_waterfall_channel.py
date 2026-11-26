"""Reuse the same sampled PUSCH channel and sounded precoders for every MCS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from sionna.phy import config as sionna_config

from .adaptation import select_grants
from .adaptive_simulation import SoundingChannel, _sounding_epochs
from .link import _pusch_config, precoder_amplitude_scale
from .simulation import _CODEBOOK_SIZES
from .srs import SrsEstimator


def _variance_by_snr(snrs, dtype):
    real_dtype = np.empty((), dtype=dtype).real.dtype
    with np.errstate(over="ignore", under="ignore"):
        noise_variance = np.power(10.0, -np.asarray(snrs) / 10).astype(real_dtype)
    if not all(np.isfinite(noise_variance) & (noise_variance >= np.finfo(real_dtype).tiny)):
        raise ValueError("Noise variance is outside the numeric range")
    return noise_variance


def _srs_precoders(cfg):
    settings = cfg.adaptation
    grant = cfg.pusch
    estimator = SrsEstimator(
        grant.num_antenna_ports, cfg.carrier.n_size_grid, settings.srs_sequence_id,
        start_symbol=settings.srs_start_symbol if settings.tdd_pattern is not None else None,
        power_per_active_re=1.0,
    )
    if settings.rank_tpmi_candidates is None:
        tpmis = tuple(range(_CODEBOOK_SIZES[1][0]))
    else:
        if any(rank != 1 for rank, _ in settings.rank_tpmi_candidates):
            raise ValueError("The channel bank only supports Rank 1 adaptation.rank_tpmi_candidates")
        tpmis = tuple(tpmi for _, tpmi in settings.rank_tpmi_candidates)
    candidates = {}
    for tpmi in tpmis:
        pusch = _pusch_config(replace(cfg, pusch=replace(grant, tpmi=tpmi)))
        matrix = pusch.precoding_matrix
        scale = precoder_amplitude_scale(matrix, grant.num_antenna_ports, settings.tx_power_mode)
        candidates[(1, tpmi, grant.mcs_index)] = (matrix * scale, pusch.tb_size)
    return estimator, candidates


def _precoder_weights(estimator, candidates, observation, standard_noise, noise, srs_error):
    estimate = observation if standard_noise is None else estimator.estimate_from_observation(
        observation, noise, standard_noise,
    )
    key, _ = select_grants(
        estimate, noise, candidates, None, 0.0, {None: candidates}, srs_error_variance=srs_error,
    )[None]
    return candidates[key][0][:, 0]


class WaterfallWeights:
    """Read eager weights or share one atomically written file per measured SNR."""

    def __init__(self, cfg, directory, bank, *, locks=()):
        self._directory = Path(directory)
        self._locks = locks
        weights_path = self._directory / "weights.npy"
        self._eager = np.load(weights_path, mmap_mode="r") if weights_path.is_file() else None
        if self._eager is None:
            self._cfg = cfg
            self._srs_inputs = np.load(self._directory / "srs.npy", mmap_mode="r")
            self._epoch_stops = np.load(self._directory / "epoch_stops.npy")
            self._dtype = np.dtype(bank["dtype"])
            self._noise = _variance_by_snr(bank["snrs"], self._dtype)
            self._scoring = None
            self._srs_ideal = bank["srs_estimator"] == "perfect"
            self._shape = (bank["num_tb"], cfg.pusch.num_antenna_ports)

    def __getitem__(self, snr_index: int) -> np.ndarray:
        if self._eager is not None:
            return self._eager[snr_index]
        path = self._directory / f"weights_{snr_index}.npy"
        if path.is_file():
            return np.load(path)
        lock = self._locks[snr_index % len(self._locks)] if self._locks else nullcontext()
        with lock:
            if path.is_file():
                return np.load(path)
            if self._scoring is None:
                self._scoring = _srs_precoders(self._cfg)
            estimator, candidates = self._scoring
            noise = float(self._noise[snr_index])
            srs_error = None if self._srs_ideal else estimator.error_variance(noise)
            weights = np.empty(self._shape, dtype=self._dtype)
            start = 0
            for inputs, stop in zip(self._srs_inputs, self._epoch_stops, strict=True):
                observed, standard_noise = (inputs, None) if self._srs_ideal else inputs
                weights[start:stop] = _precoder_weights(
                    estimator, candidates, observed, standard_noise, noise, srs_error,
                )
                start = stop
            fd, staged = mkstemp(prefix=f"{path.stem}_", suffix=".tmp", dir=self._directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.save(stream, weights)
                try:
                    os.replace(staged, path)
                except PermissionError:
                    # Another caller without shared worker locks may have published it.
                    if not path.is_file():
                        raise
            finally:
                Path(staged).unlink(missing_ok=True)
            return weights


def write_channel_bank(cfg, snrs, num_tb, directory, *, on_progress=None, defer_weights=False):
    """Write physical channels, physical slot indices and the SRS inputs or weights."""
    cfg.validate()
    settings = cfg.adaptation
    grant = cfg.pusch
    if settings is None or (grant.num_antenna_ports, grant.rank, grant.precoding) != (4, 1, "codebook"):
        raise ValueError("The channel bank requires SRS and a four-port Rank 1 codebook configuration")
    if cfg.channel.kind == "custom":
        raise ValueError("The channel bank requires a built-in fading channel")
    if type(num_tb) is not int or num_tb < 1:
        raise ValueError("num_tb must be a positive integer")
    snrs = [float(snr) for snr in snrs]
    if not snrs or len(set(snrs)) != len(snrs) or not all(np.isfinite(snrs)):
        raise ValueError("snrs must contain distinct finite SNR values")

    dtype = np.dtype("complex128" if cfg.precision == "double" else "complex64")
    noise_variance = _variance_by_snr(snrs, dtype)
    estimator, candidates = _srs_precoders(cfg)
    srs_ideal = settings.srs_estimator == "perfect"
    # Per-port power metadata is an average over the entire candidate SNR grid.
    defer_weights = defer_weights and settings.tx_power_mode == "full_power"

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    channel_path = directory / "channel.npy"
    weight_path = directory / "weights.npy"
    srs_path = directory / "srs.npy"
    stops_path = directory / "epoch_stops.npy"
    slots_path = directory / "slots.npy"
    metadata_path = directory / "bank.json"
    bank_files = (channel_path, weight_path, srs_path, stops_path, slots_path, metadata_path)
    if any(path.exists() for path in bank_files) or any(directory.glob("weights_*.npy")):
        raise FileExistsError(f"Channel bank files already exist in {directory}")

    shape = (num_tb, cfg.num_rx_ant, grant.num_antenna_ports, 14, 12 * cfg.carrier.n_size_grid)
    channels = np.lib.format.open_memmap(channel_path, mode="w+", dtype=dtype, shape=shape)
    weights = None
    if not defer_weights:
        weights = np.empty((len(snrs), num_tb, grant.num_antenna_ports), dtype=dtype)
        selected = np.empty((len(snrs), grant.num_antenna_ports), dtype=dtype)
        srs_errors = [None if srs_ideal else estimator.error_variance(float(noise)) for noise in noise_variance]
    slots = np.empty(num_tb, dtype=np.int32)
    digest = sha256()
    sionna_config.seed = cfg.sweep.seed
    epochs = list(_sounding_epochs(settings, num_tb))
    total_slots = epochs[-1][-1] + 1
    channel = SoundingChannel(cfg, None, estimator, full_sounding_grid=srs_ideal, total_slots=total_slots)
    srs_noise = np.random.default_rng(cfg.sweep.seed)
    slots_per_frame = 10 * cfg.carrier.subcarrier_spacing_khz // 15

    index = 0
    progress_interval = max(1, num_tb // 10)
    srs_inputs = None
    epoch_stops = []
    for epoch_index, epoch in enumerate(epochs):
        pilot_slot = epoch[0]
        responses = channel.slots(tuple(slot - pilot_slot for slot in epoch), start_slot=pilot_slot)
        sounding = np.ascontiguousarray(next(responses))
        digest.update(memoryview(sounding).cast("B"))
        # Draw the SRS noise seed in both modes so real and ideal banks stay paired.
        srs_seed = int(srs_noise.integers(0, 2**32))
        observed, standard_noise = (sounding, None) if srs_ideal else estimator.prepare_pilots(
            sounding, np.random.default_rng(srs_seed),
        )
        if defer_weights:
            if srs_inputs is None:
                input_shape = observed.shape if srs_ideal else (2, *observed.shape)
                srs_inputs = np.lib.format.open_memmap(
                    srs_path, mode="w+", dtype=observed.dtype, shape=(len(epochs), *input_shape),
                )
            if srs_ideal:
                srs_inputs[epoch_index] = observed
            else:
                srs_inputs[epoch_index, 0] = observed
                srs_inputs[epoch_index, 1] = standard_noise
        else:
            for snr_index, (noise, srs_error) in enumerate(zip(noise_variance, srs_errors, strict=True)):
                selected[snr_index] = _precoder_weights(
                    estimator, candidates, observed, standard_noise, float(noise), srs_error,
                )

        for slot, response in zip(epoch[1:], responses, strict=True):
            channels[index] = np.asarray(response)[0, 0, :, 0]
            digest.update(memoryview(channels[index]).cast("B"))
            if weights is not None:
                weights[:, index] = selected
            slots[index] = (cfg.carrier.slot_number + slot) % slots_per_frame
            index += 1
            if on_progress is not None and (index % progress_interval == 0 or index == num_tb):
                on_progress(index, num_tb)
        if defer_weights:
            epoch_stops.append(index)

    if index != num_tb:
        raise RuntimeError(f"Expected {num_tb} PUSCH slots, obtained {index}")
    channels.flush()
    del channels
    if defer_weights:
        srs_inputs.flush()
        del srs_inputs
        np.save(stops_path, np.asarray(epoch_stops, dtype=np.intp))
    else:
        np.save(weight_path, weights)
    np.save(slots_path, slots)
    metadata = {
        "snrs": snrs,
        "num_tb": num_tb,
        "shape": list(shape),
        "dtype": dtype.name,
        "seed": cfg.sweep.seed,
        "num_tpmi_candidates": len(candidates),
        **({"tpmi_candidates": [key[1] for key in candidates]} if settings.rank_tpmi_candidates is not None else {}),
        "tx_power_mode": settings.tx_power_mode,
        "srs_estimator": settings.srs_estimator,
        "srs_power_per_active_re": estimator.power_per_active_re,
        "mean_data_re_tx_power": (1.0 if settings.tx_power_mode == "full_power" else
                                  float(np.mean(np.sum(np.abs(weights)**2, axis=-1)))),
        "ue_yaw_offsets_deg": list(cfg.channel.ue_yaw_offsets_deg),
        "physical_sha256": digest.hexdigest(),
        "physical_sha256_definition": (
            "C-order complex bytes of each epoch's SRS observation (raw pilot CFR, or the "
            "full-band CFR averaged over SRS symbols for an ideal SRS estimator) followed "
            "by full PUSCH slot CFRs, in physical-slot order"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
