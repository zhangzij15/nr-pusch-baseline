"""Bootstrap measured throughput on the SRS period timeline."""

from __future__ import annotations

from itertools import combinations

import numpy as np


BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0x535253


def _ratio_ci95(payload_bits, slots, mbps_per_bit):
    if len(slots) < 2:
        return None
    bits = np.asarray(payload_bits, dtype=np.int64)
    slots = np.asarray(slots, dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rates = np.empty(BOOTSTRAP_REPLICATES)
    # Bound temporary memory for long sweeps.
    draw_batch = max(1, 200_000 // len(slots))
    for start in range(0, BOOTSTRAP_REPLICATES, draw_batch):
        stop = min(start + draw_batch, BOOTSTRAP_REPLICATES)
        draw = rng.integers(len(slots), size=(stop - start, len(slots)))
        rates[start:stop] = mbps_per_bit * bits[draw].sum(axis=1) / slots[draw].sum(axis=1)
    return np.quantile(rates, (0.025, 0.975)).tolist()


def _mbps_per_bit(point, field):
    duty_cycle = point["ul_duty_cycle"] if field == "thp_duty_cycle_mbps" else 1.0
    return duty_cycle / (point["slot_seconds"] * 1e6)


def add_throughput_statistics(point, clusters):
    """Record the observed periods and the CI for the wall-clock or duty-adjusted rate."""
    field = "thp_wall_clock_mbps" if "thp_wall_clock_mbps" in point else "thp_duty_cycle_mbps"
    point.update({
        "throughput_srs_clusters": clusters,
        "throughput_ci95_num_clusters": len(clusters),
        "throughput_ci95_mbps": _ratio_ci95(
            [cluster["delivered_payload_bits"] for cluster in clusters],
            [cluster["num_slots"] for cluster in clusters],
            _mbps_per_bit(point, field),
        ),
    })


def paired_throughput_differences(runs, field):
    """Compute A minus B using identical SRS-period draws for both codebooks."""
    by_codebook = {name: {point["snr_db"]: point for point in run["points"]}
                   for name, run in runs.items()}
    rows = []
    for (name_a, points_a), (name_b, points_b) in combinations(by_codebook.items(), 2):
        for snr_db in sorted(points_a.keys() & points_b.keys()):
            a, b = points_a[snr_db], points_b[snr_db]
            clusters_a = a.get("throughput_srs_clusters")
            clusters_b = b.get("throughput_srs_clusters")
            if not clusters_a or not clusters_b:
                continue
            timeline_a = [(cluster["srs_slot_index"], cluster["num_slots"]) for cluster in clusters_a]
            timeline_b = [(cluster["srs_slot_index"], cluster["num_slots"]) for cluster in clusters_b]
            scale = _mbps_per_bit(a, field)
            if timeline_a != timeline_b or scale != _mbps_per_bit(b, field):
                raise ValueError(f"Cannot pair {name_a} and {name_b} at {snr_db:g} dB: timelines differ")
            differences = [x["delivered_payload_bits"] - y["delivered_payload_bits"]
                           for x, y in zip(clusters_a, clusters_b)]
            rows.append({
                "codebook_a": name_a,
                "codebook_b": name_b,
                "snr_db": snr_db,
                "difference_mbps": a[field] - b[field],
                "difference_ci95_mbps": _ratio_ci95(
                    differences, [slots for _, slots in timeline_a], scale,
                ),
                "num_srs_clusters": len(clusters_a),
            })
    return rows
