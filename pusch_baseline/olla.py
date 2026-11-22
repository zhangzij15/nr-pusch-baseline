"""Outer-loop link adaptation driven by PUSCH block-error feedback."""

from bisect import bisect_right
from collections import deque
from collections.abc import Iterable

from .config import OllaSettings


class _BlockErrorFeedback:
    def __init__(self, settings: OllaSettings):
        settings.validate()
        self._settings = settings
        self._window: deque[tuple[bool, bool, bool]] = deque(maxlen=settings.window_tb)
        self._total_tb = 0

    def observe(self, failed: bool, *, lowest_mcs: bool, highest_mcs: bool) -> None:
        self._total_tb += 1
        self._window.append((failed, lowest_mcs, highest_mcs))

    def summary(self) -> dict:
        num_tb = len(self._window)
        bler = sum(failed for failed, _, _ in self._window) / num_tb if num_tb else None
        all_lowest = bool(num_tb) and all(lowest for _, lowest, _ in self._window)
        all_highest = bool(num_tb) and all(highest for _, _, highest in self._window)

        if self._total_tb < self._settings.warmup_tb or bler is None:
            status = "insufficient_samples"
        elif self._settings.in_target_window(bler):
            status = "converged"
        elif bler > self._settings.target_bler:
            status = "above_target"
        elif all_highest:
            status = "at_mcs_ceiling"
        else:
            status = "below_target"

        return {
            "calibration_bler": bler,
            "calibration_num_tb": num_tb,
            "calibration_total_tb": self._total_tb,
            "calibration_status": status,
            "calibration_all_lowest_mcs": all_lowest,
            "calibration_all_highest_mcs": all_highest,
        }


class OllaController(_BlockErrorFeedback):
    """Track a bounded SINR threshold offset for legacy profiles."""

    def __init__(self, settings: OllaSettings):
        super().__init__(settings)
        self.margin_offset_db = 0.0

    def observe(self, failed: bool, *, lowest_mcs: bool, highest_mcs: bool) -> None:
        super().observe(failed, lowest_mcs=lowest_mcs, highest_mcs=highest_mcs)
        target = self._settings.target_bler
        delta = self._settings.step_db * (float(failed) - target)
        limit = self._settings.max_offset_db
        self.margin_offset_db = max(-limit, min(limit, self.margin_offset_db + delta))

    def summary(self) -> dict:
        return {**super().summary(), "margin_offset_db": self.margin_offset_db}


class McsIndexController(_BlockErrorFeedback):
    """Apply ACK/NACK updates directly to a continuous MCS state."""

    def __init__(self, settings: OllaSettings, initial_mcs: int, mcs_indices: Iterable[int]):
        super().__init__(settings)
        self._indices = tuple(sorted(mcs_indices))
        if not self._indices:
            raise ValueError("MCS feedback requires at least one supported index")
        if settings.mcs_axis == "ladder":
            self._axis = tuple(range(len(self._indices)))
            initial_mcs = bisect_right(self._indices, initial_mcs) - 1
        else:
            self._axis = self._indices
        self._state = self._clip(float(initial_mcs))

    def _clip(self, value: float) -> float:
        return float(max(self._axis[0], min(self._axis[-1], value)))

    @property
    def mcs_state(self) -> float:
        """Return the fractional state in MCS index units for telemetry."""
        if self._settings.mcs_axis == "index":
            return self._state
        lower = int(self._state)
        upper = min(lower + 1, len(self._indices) - 1)
        return self._indices[lower] + (self._state - lower) * (self._indices[upper] - self._indices[lower])

    @property
    def mcs_index(self) -> int:
        # A balanced ACK/NACK sequence can leave the state a few ulps below a boundary.
        index = bisect_right(self._axis, self._state + 1e-9) - 1
        return self._indices[index]

    def observe(self, failed: bool, *, lowest_mcs: bool, highest_mcs: bool) -> None:
        super().observe(failed, lowest_mcs=lowest_mcs, highest_mcs=highest_mcs)
        delta = self._settings.step_mcs * (self._settings.target_bler - float(failed))
        self._state = self._clip(self._state + delta)

    def summary(self) -> dict:
        return {**super().summary(), "mcs_state": self.mcs_state, "next_mcs_index": self.mcs_index}
