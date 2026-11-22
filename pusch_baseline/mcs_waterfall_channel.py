"""Reuse the same sampled PUSCH channel and sounded precoders for every MCS."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sionna.phy import config as sionna_config

from .adaptation import select_grants
from .adaptive_simulation import SoundingChannel, _sounding_epochs
from .link import _pusch_config, precoder_amplitude_scale
from .simulation import _CODEBOOK_SIZES
from .srs import SrsEstimator


def write_channel_bank(cfg, snrs, num_tb, directory, *, on_progress=None):
    """Write physical channels, power-scaled weights per SNR, and data-slot indices."""
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
    real_dtype = np.empty((), dtype=dtype).real.dtype
    with np.errstate(over="ignore", under="ignore"):
        noise_variance = np.power(10.0, -np.asarray(snrs) / 10).astype(real_dtype)
    if not all(np.isfinite(noise_variance) & (noise_variance >= np.finfo(real_dtype).tiny)):
        raise ValueError("Noise variance is outside the numeric range")

    estimator = SrsEstimator(
        grant.num_antenna_ports, cfg.carrier.n_size_grid, settings.srs_sequence_id,
        start_symbol=settings.srs_start_symbol if settings.tdd_pattern is not None else None,
        power_per_active_re=1.0,
    )
    srs_error_variances = [estimator.error_variance(float(noise)) for noise in noise_variance]
    candidates = {}
    for tpmi in range(_CODEBOOK_SIZES[1][0]):
        pusch = _pusch_config(replace(cfg, pusch=replace(grant, tpmi=tpmi)))
        matrix = pusch.precoding_matrix
        scale = precoder_amplitude_scale(matrix, grant.num_antenna_ports, settings.tx_power_mode)
        candidates[(1, tpmi, grant.mcs_index)] = (matrix * scale, pusch.tb_size)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    channel_path = directory / "channel.npy"
    weight_path = directory / "weights.npy"
    slots_path = directory / "slots.npy"
    metadata_path = directory / "bank.json"
    if any(path.exists() for path in (channel_path, weight_path, slots_path, metadata_path)):
        raise FileExistsError(f"Channel bank files already exist in {directory}")

    shape = (num_tb, cfg.num_rx_ant, grant.num_antenna_ports, 14, 12 * cfg.carrier.n_size_grid)
    channels = np.lib.format.open_memmap(channel_path, mode="w+", dtype=dtype, shape=shape)
    weights = np.empty((len(snrs), num_tb, grant.num_antenna_ports), dtype=dtype)
    slots = np.empty(num_tb, dtype=np.int32)
    digest = sha256()
    sionna_config.seed = cfg.sweep.seed
    epochs = list(_sounding_epochs(settings, num_tb))
    total_slots = epochs[-1][-1] + 1
    channel = SoundingChannel(cfg, None, estimator, total_slots=total_slots)
    srs_noise = np.random.default_rng(cfg.sweep.seed)
    slots_per_frame = 10 * cfg.carrier.subcarrier_spacing_khz // 15

    index = 0
    progress_interval = max(1, num_tb // 10)
    for epoch in epochs:
        pilot_slot = epoch[0]
        responses = channel.slots(tuple(slot - pilot_slot for slot in epoch), start_slot=pilot_slot)
        sounding = np.ascontiguousarray(next(responses))
        digest.update(memoryview(sounding).cast("B"))
        srs_seed = int(srs_noise.integers(0, 2**32))
        selected = np.empty((len(snrs), grant.num_antenna_ports), dtype=dtype)
        for snr_index, noise in enumerate(noise_variance):
            estimate = estimator.estimate_from_pilots(sounding, float(noise), np.random.default_rng(srs_seed))
            key, _ = select_grants(
                estimate, float(noise), candidates, None, 0.0, {None: candidates},
                srs_error_variance=srs_error_variances[snr_index],
            )[None]
            selected[snr_index] = candidates[key][0][:, 0]

        for slot, response in zip(epoch[1:], responses, strict=True):
            channels[index] = np.asarray(response)[0, 0, :, 0]
            digest.update(memoryview(channels[index]).cast("B"))
            weights[:, index] = selected
            slots[index] = (cfg.carrier.slot_number + slot) % slots_per_frame
            index += 1
            if on_progress is not None and (index % progress_interval == 0 or index == num_tb):
                on_progress(index, num_tb)

    if index != num_tb:
        raise RuntimeError(f"Expected {num_tb} PUSCH slots, obtained {index}")
    channels.flush()
    del channels
    np.save(weight_path, weights)
    np.save(slots_path, slots)
    metadata = {
        "snrs": snrs,
        "num_tb": num_tb,
        "shape": list(shape),
        "dtype": dtype.name,
        "seed": cfg.sweep.seed,
        "num_tpmi_candidates": len(candidates),
        "tx_power_mode": settings.tx_power_mode,
        "srs_power_per_active_re": estimator.power_per_active_re,
        "mean_data_re_tx_power": (1.0 if settings.tx_power_mode == "full_power" else
                                  float(np.mean(np.sum(np.abs(weights)**2, axis=-1)))),
        "ue_yaw_offsets_deg": list(cfg.channel.ue_yaw_offsets_deg),
        "physical_sha256": digest.hexdigest(),
        "physical_sha256_definition": (
            "C-order complex bytes of each epoch's raw SRS pilot CFR followed "
            "by full PUSCH slot CFRs, in physical-slot order"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
