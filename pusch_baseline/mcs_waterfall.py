"""Fixed-MCS PUSCH batches on a replayable 4T64R channel bank."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

# Approximate 10% knees used only to choose the first measured SNRs.
_START_DB = (-26, -24, -23, -22, -21, -19, -18, -17, -16, -16,
             -14, -13, -12, -11, -10, -10, -9, -8, -8, -7,
             -7, -6, -6, -5, -4, -4, -3, -2, -1)
WATERFALL_SNRS = tuple(range(-36, 13))


def waterfall_snr_grid(step_db: float = 1.0) -> tuple[int | float, ...]:
    """Return the [-36, 12] dB waterfall SNR grid at a power-of-two sub-dB step."""
    if step_db == 1.0:
        return WATERFALL_SNRS
    if step_db <= 0:
        raise ValueError("step_db must be positive")
    ratio = 2.0 / float(step_db)
    steps_per_2db = round(ratio)
    if abs(ratio - steps_per_2db) > 1e-9 or steps_per_2db & (steps_per_2db - 1):
        raise ValueError("step_db must divide 2.0 by a power of two (e.g., 1, 0.5, 0.25, 0.125)")
    n_steps = round(48.0 / step_db)
    return tuple(round(-36.0 + i * step_db, 6) for i in range(n_steps + 1))


def _channel_batch(channels, indices):
    if len(indices) == 1:
        return channels[indices[0]:indices[0] + 1]
    step = indices[1] - indices[0]
    if np.all(np.diff(indices) == step):
        return channels[indices[0]:indices[-1] + 1:step]
    return channels[indices]


def simulate_mcs_curve(cfg, bank_path, mcs_index, batch_size, on_point=None, *,
                       snr_points=None, completed_points=(), min_snr_step: float = 1.0):
    """Measure one TS 38.214 Table 1 row with independently decoded TBs."""
    import tensorflow as tf
    from sionna.phy import config as sionna_config

    from .link import PuschLink

    bank_path = Path(bank_path)
    bank = json.loads((bank_path / "bank.json").read_text(encoding="utf-8"))
    channels = np.load(bank_path / "channel.npy", mmap_mode="r")
    weights = np.load(bank_path / "weights.npy", mmap_mode="r")
    frame_slots = np.load(bank_path / "slots.npy")
    slots = {
        int(slot): np.flatnonzero(frame_slots == slot)
        for slot in sorted(set(frame_slots))
    }
    count = len(frame_slots)
    table_snr_indices = {float(snr): i for i, snr in enumerate(bank["snrs"])}
    # The effective frequency response already includes the selected rank-one
    # precoder and its power scale. One logical port carries the data and DM-RS.
    grant = replace(
        cfg.pusch, num_antenna_ports=1, rank=1, precoding="non-codebook", tpmi=0,
        mcs_index=mcs_index,
    )
    sionna_config.seed = cfg.sweep.seed
    rng = sionna_config.tf_rng
    rng_state = rng.state.read_value()
    decoder_cache = {}
    links = {
        slot: PuschLink(
            replace(cfg, carrier=replace(cfg.carrier, slot_number=slot), pusch=grant),
            external_channel=True, tb_decoder_cache=decoder_cache, allow_low_rate=True,
        )
        for slot in slots
    }
    reference = next(iter(links.values()))
    execute = {
        slot: tf.function(link.count_block_errors_tensor, autograph=False, reduce_retracing=True)
        for slot, link in links.items()
    }
    points = {float(point["snr_db"]): point for point in completed_points}
    if any(
        snr not in table_snr_indices
        or point["mcs_index"] != mcs_index
        or point["num_tb"] != count
        or point["tb_size_bits"] != reference.tb_size_bits
        or point["tb_errors"] / count != point["bler"]
        for snr, point in points.items()
    ):
        raise ValueError("Measured checkpoint does not match the MCS, channel bank, or grant")

    def measure(snr):
        snr = float(snr)
        if snr in points:
            return points[snr]["bler"]
        no = reference.noise_variance(snr)
        snr_weights = weights[table_snr_indices[snr]]
        rng.reset(rng_state)
        errors = 0
        for slot, indices in slots.items():
            for start in range(0, len(indices), batch_size):
                group = indices[start:start + batch_size]
                effective = np.einsum(
                    "brtsf,bt->brsf", _channel_batch(channels, group), snr_weights[group], optimize=False,
                )
                h = tf.convert_to_tensor(effective[:, None, :, None, None])
                errors += int(execute[slot](len(group), no, channel_response=h).numpy())
        point = {
            "mcs_index": mcs_index,
            "qam_bits_per_symbol": reference.grant_metadata["qam_bits_per_symbol"],
            "target_coderate": reference.grant_metadata["target_coderate"],
            "tb_size_bits": reference.tb_size_bits,
            "snr_db": snr,
            "num_tb": count,
            "tb_errors": errors,
            "bler": errors / count,
        }
        points[snr] = point
        if on_point is not None:
            on_point(point)
        return point["bler"]

    if snr_points is not None:
        for snr in snr_points:
            measure(snr)
        return [points[float(snr)] for snr in snr_points]

    # The LMMSE receiver's transition lies below the original LS search hints.
    # The measured bounds still determine how far to extend each side.
    min_snr, max_snr = min(table_snr_indices), max(table_snr_indices)
    center = float(_START_DB[mcs_index] - (5 if cfg.channel.estimator == "lmmse" else 0))
    low, high = max(min_snr, center - 4.0), min(max_snr, center + 4.0)
    snr = low
    while snr <= high + 1e-9:
        measure(snr)
        snr += 2.0
    while measure(low) < 0.75 and low > min_snr:
        low = max(min_snr, low - 2.0)
    tail_target = 0.008 if (min_snr_step < 1.0 and count >= 256) else 0.03
    while measure(high) > tail_target and high < max_snr:
        high = min(max_snr, high + 2.0)
    measure(low)
    measure(high)

    # Iteratively refine brackets crossing the waterfall down to min_snr_step:
    # 0.5 dB across the full transition, and down to min_snr_step (e.g. 0.25 dB)
    # wherever BLER drops steeply or crosses the lower knee into zero errors.
    coarse_levels = (0.85, 0.6, 0.35, 0.15, 0.05, 0.015) if min_snr_step < 1.0 else (0.75, 0.4, 0.1, 0.03)
    while True:
        sampled = sorted(points)
        added = False
        for left, right in zip(sampled, sampled[1:]):
            gap = right - left
            if gap <= min_snr_step + 1e-9:
                continue
            mid = round((left + right) / 2.0, 6)
            if mid not in table_snr_indices:
                continue
            left_bler, right_bler = points[left]["bler"], points[right]["bler"]
            steep_drop = (
                min_snr_step < 1.0
                and left_bler >= 0.012
                and (right_bler == 0.0 or left_bler / right_bler >= 2.2)
            )
            if gap > 0.5 + 1e-9:
                refine = steep_drop or any(left_bler >= level > right_bler for level in coarse_levels)
            else:
                refine = steep_drop or (left_bler >= 0.12 and right_bler < 0.05)
            if refine:
                measure(mid)
                added = True
        if not added:
            break
    return [points[snr] for snr in sorted(points)]
