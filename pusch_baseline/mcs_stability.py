"""Windowed MCS and BLER diagnostics for online link adaptation."""

from collections.abc import Sequence
from math import isclose


def assess_mcs_stability(
    samples: Sequence[tuple[int, bool]],
    *,
    window_tb: int,
    target_bler: float,
    bler_tolerance: float,
    max_mcs_spread: float,
) -> dict:
    """Assess the three most recent disjoint windows of (scheduled MCS, failed TB)."""
    if window_tb < 1:
        raise ValueError("window_tb must be positive")

    required_tb = 3 * window_tb
    if len(samples) < required_tb:
        return {
            "stable": False,
            "status": "insufficient_samples",
            "window_tb": window_tb,
            "windows": [],
            "mcs_spread": None,
            "pooled_bler": None,
        }

    recent = samples[-required_tb:]
    windows = []
    total_errors = 0
    for start in range(0, required_tb, window_tb):
        window = recent[start:start + window_tb]
        errors = sum(bool(failed) for _, failed in window)
        total_errors += errors
        windows.append({
            "mean_mcs": float(sum(mcs for mcs, _ in window) / window_tb),
            "bler": errors / window_tb,
        })

    means = [window["mean_mcs"] for window in windows]
    mcs_spread = max(means) - min(means)
    pooled_bler = total_errors / required_tb
    bler_deviation = abs(pooled_bler - target_bler)
    if mcs_spread > max_mcs_spread and not isclose(mcs_spread, max_mcs_spread):
        status = "mcs_drift"
    elif bler_deviation > bler_tolerance and not isclose(bler_deviation, bler_tolerance):
        status = "bler_outside_window"
    else:
        status = "stable"

    return {
        "stable": status == "stable",
        "status": status,
        "window_tb": window_tb,
        "windows": windows,
        "mcs_spread": mcs_spread,
        "pooled_bler": pooled_bler,
    }
