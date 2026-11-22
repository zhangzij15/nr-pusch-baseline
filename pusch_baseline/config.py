"""Typed settings for a single-UE NR PUSCH link and SNR sweep."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any


# Distinct m_SRS,0 values from 3GPP TS 38.211 Table 6.4.1.4.3-1.
SRS_BANDWIDTHS_PRB = (
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 52, 56, 60, 64, 72, 76, 80,
    88, 96, 104, 112, 120, 128, 132, 136, 144, 152, 160, 168, 176, 184, 192,
    208, 216, 224, 240, 256, 264, 272,
)


def _int_range(name: str, value: Any, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def _positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(name: str, value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _choice(name: str, value: Any, choices: tuple) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")


def _sequence(name: str, value: Any, length: int | None = None) -> None:
    if not isinstance(value, (list, tuple)) or (
        length is not None and len(value) != length
    ):
        suffix = f" of length {length}" if length is not None else ""
        raise ValueError(f"{name} must be an array{suffix}")


@dataclass(frozen=True)
class CarrierSettings:
    subcarrier_spacing_khz: int = 30
    n_size_grid: int = 24
    cell_id: int = 1
    slot_number: int = 0

    def validate(self) -> None:
        if type(self.subcarrier_spacing_khz) is not int:
            raise ValueError("carrier.subcarrier_spacing_khz must be an integer")
        _choice(
            "carrier.subcarrier_spacing_khz", self.subcarrier_spacing_khz,
            (15, 30, 60, 120),
        )
        _int_range("carrier.n_size_grid", self.n_size_grid, 1, 275)
        _int_range("carrier.cell_id", self.cell_id, 0, 1007)
        slots_per_frame = 10 * self.subcarrier_spacing_khz // 15
        _int_range("carrier.slot_number", self.slot_number, 0, slots_per_frame - 1)


@dataclass(frozen=True)
class DmrsSettings:
    config_type: int = 1
    length: int = 1
    additional_position: int = 1
    num_cdm_groups_without_data: int = 2
    type_a_position: int = 2
    port_set: tuple[int, ...] | None = None
    n_id: int | tuple[int, int] | None = None
    n_scid: int = 0

    def validate(self, rank: int) -> None:
        _int_range("dmrs.config_type", self.config_type, 1, 2)
        _int_range("dmrs.length", self.length, 1, 2)
        _int_range("dmrs.additional_position", self.additional_position, 0,
                   3 if self.length == 1 else 1)
        _int_range("dmrs.type_a_position", self.type_a_position, 2, 3)
        _int_range("dmrs.n_scid", self.n_scid, 0, 1)
        groups = self.config_type + 1
        _int_range("dmrs.num_cdm_groups_without_data",
                   self.num_cdm_groups_without_data, 1, groups)
        if type(self.n_id) is int:
            _int_range("dmrs.n_id", self.n_id, 0, 65535)
        elif self.n_id is not None:
            _sequence("dmrs.n_id", self.n_id, 2)
            for item in self.n_id:
                _int_range("dmrs.n_id", item, 0, 65535)

        ports = self.port_set if self.port_set is not None else tuple(range(rank))
        _sequence("dmrs.port_set", ports, rank)
        port_stride = 2 * groups
        for port in ports:
            _int_range("dmrs.port_set entry", port, 0, port_stride * self.length - 1)
            if port % port_stride // 2 >= self.num_cdm_groups_without_data:
                raise ValueError(f"DMRS port {port} uses a CDM group reserved for data")
        if len(set(ports)) != rank:
            raise ValueError("dmrs.port_set must contain distinct ports")


@dataclass(frozen=True)
class PuschSettings:
    num_antenna_ports: int = 1
    rank: int = 1
    precoding: str = "non-codebook"
    tpmi: int = 0
    mcs_table: int = 1
    mcs_index: int = 5
    rnti: int = 1
    data_scrambling_id: int | None = None
    mapping_type: str = "A"
    symbol_allocation: tuple[int, int] = (0, 14)
    dmrs: DmrsSettings = field(default_factory=DmrsSettings)

    @property
    def dmrs_ports(self) -> list[int]:
        if self.dmrs.port_set is None:
            return list(range(self.rank))
        return list(self.dmrs.port_set)

    def validate(self) -> None:
        _int_range("pusch.num_antenna_ports", self.num_antenna_ports, 1, 4)
        _choice("pusch.num_antenna_ports", self.num_antenna_ports, (1, 2, 4))
        _int_range("pusch.rank", self.rank, 1, self.num_antenna_ports)
        _choice("pusch.precoding", self.precoding, ("codebook", "non-codebook"))
        if self.precoding == "codebook":
            if self.num_antenna_ports == 1:
                raise ValueError("Codebook precoding requires at least two TX ports")
            max_tpmi = {(2, 1): 5, (2, 2): 2, (4, 1): 27,
                        (4, 2): 21, (4, 3): 6, (4, 4): 4}
            _int_range("pusch.tpmi", self.tpmi, 0,
                       max_tpmi[self.num_antenna_ports, self.rank])
        else:
            if self.rank != self.num_antenna_ports:
                raise ValueError("Non-codebook precoding requires rank equal to TX ports")
            _int_range("pusch.tpmi with non-codebook precoding", self.tpmi, 0, 0)

        _int_range("pusch.mcs_table", self.mcs_table, 1, 3)
        _int_range("pusch.mcs_index", self.mcs_index, 0, (28, 27, 28)[self.mcs_table - 1])
        _int_range("pusch.rnti", self.rnti, 1, 65519)
        if self.data_scrambling_id is not None:
            _int_range("pusch.data_scrambling_id", self.data_scrambling_id, 0, 1023)
        self.dmrs.validate(self.rank)

        _choice("pusch.mapping_type", self.mapping_type, ("A", "B"))
        _sequence("pusch.symbol_allocation", self.symbol_allocation, 2)
        start, length = self.symbol_allocation
        # Sionna 1.2.2 indexes the DM-RS sequence locally for mapping type B.
        _int_range("pusch.symbol_allocation start", start, 0, 0)
        if self.mapping_type == "A":
            _int_range("mapping type A symbol count", length, 4, 14)
            if self.dmrs.type_a_position == 3 and (
                length == 4 or self.dmrs.additional_position == 3
            ):
                raise ValueError("Mapping type A requires DMRS position 2 for a 4-symbol "
                                 "allocation or additional_position 3")
        else:
            min_length = 2 if self.dmrs.length == 1 else 5
            _int_range("mapping type B symbol count", length, min_length, 14)


@dataclass(frozen=True)
class PanelSettings:
    rows: int = 1
    cols: int = 1
    polarization: str = "single"
    polarization_type: str = "V"
    pattern: str = "omni"
    horizontal_spacing_lambda: float = 0.5
    vertical_spacing_lambda: float = 0.5
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def num_antennas(self) -> int:
        return self.rows * self.cols * (2 if self.polarization == "dual" else 1)

    def validate(self, name: str) -> None:
        _positive_int(f"{name}.rows", self.rows)
        _positive_int(f"{name}.cols", self.cols)
        _choice(f"{name}.polarization", self.polarization, ("single", "dual"))
        polarizations = ("V", "H") if self.polarization == "single" else ("VH", "cross")
        _choice(f"{name}.polarization_type", self.polarization_type, polarizations)
        _choice(f"{name}.pattern", self.pattern, ("omni", "38.901"))
        for spacing in ("horizontal_spacing_lambda", "vertical_spacing_lambda"):
            value = getattr(self, spacing)
            _finite(f"{name}.{spacing}", value)
            if value <= 0:
                raise ValueError(f"{name}.{spacing} must be positive")
        _sequence(f"{name}.orientation_deg", self.orientation_deg, 3)
        for angle in self.orientation_deg:
            _finite(f"{name}.orientation_deg entry", angle)


@dataclass(frozen=True)
class ChannelSettings:
    kind: str = "awgn"
    profile: str = "C"
    delay_spread_ns: float = 100.0
    carrier_frequency_ghz: float = 3.5
    ue_speed_kmh: float = 3.0
    normalize: bool = True
    num_rx_ant: int = 1
    estimator: str = "ls"
    ue_panel: PanelSettings = field(default_factory=PanelSettings)
    bs_panel: PanelSettings = field(default_factory=PanelSettings)
    ue_yaw_offsets_deg: tuple[float, ...] = ()

    def validate(self, pusch: PuschSettings) -> None:
        _choice("channel.kind", self.kind, ("awgn", "tdl", "cdl", "custom"))
        _choice("channel.estimator", self.estimator, ("perfect", "ls", "lmmse"))
        _sequence("channel.ue_yaw_offsets_deg", self.ue_yaw_offsets_deg)
        if self.ue_yaw_offsets_deg and self.kind != "cdl":
            raise ValueError("channel.ue_yaw_offsets_deg requires CDL")
        for angle in self.ue_yaw_offsets_deg:
            _finite("channel.ue_yaw_offsets_deg entry", angle)
        if self.estimator == "lmmse" and pusch.rank != 1:
            raise ValueError("channel.estimator=lmmse requires Rank 1")
        if self.kind in ("tdl", "cdl"):
            _choice("channel.profile", self.profile, ("A", "B", "C", "D", "E"))
        if self.kind in ("tdl", "cdl") or (self.kind == "custom" and self.estimator == "lmmse"):
            for name in ("delay_spread_ns", "carrier_frequency_ghz", "ue_speed_kmh"):
                _finite(f"channel.{name}", getattr(self, name))
            if self.delay_spread_ns <= 0 or self.carrier_frequency_ghz <= 0:
                raise ValueError("Channel delay spread and carrier frequency must be positive")
            if self.ue_speed_kmh < 0:
                raise ValueError("channel.ue_speed_kmh must be nonnegative")
        if type(self.normalize) is not bool:
            raise ValueError("channel.normalize must be true or false")
        if self.kind == "cdl":
            self.ue_panel.validate("channel.ue_panel")
            self.bs_panel.validate("channel.bs_panel")
            if self.ue_panel.num_antennas != pusch.num_antenna_ports:
                raise ValueError("CDL UE panel antenna count must equal PUSCH TX ports")
            rx_count = self.bs_panel.num_antennas
        else:
            _positive_int("channel.num_rx_ant", self.num_rx_ant)
            if self.kind == "awgn" and pusch.num_antenna_ports != 1:
                raise ValueError("AWGN supports one TX port; use TDL or CDL for MIMO")
            rx_count = self.num_rx_ant
        if rx_count < pusch.rank:
            raise ValueError("RX antenna count must be at least the PUSCH rank")


@dataclass(frozen=True)
class SweepSettings:
    snr_db: tuple[float, ...] = (-6, -4, -2, 0, 2, 4, 6)
    num_tb_per_snr: int = 100
    batch_size: int = 8
    seed: int = 20261112
    target_block_errors: int | None = None
    min_tb_per_snr: int = 20
    ul_duty_cycle: float = 1.0

    def validate(self) -> None:
        _sequence("sweep.snr_db", self.snr_db)
        if not self.snr_db:
            raise ValueError("sweep.snr_db must contain at least one SNR point")
        for snr in self.snr_db:
            _finite("sweep.snr_db entry", snr)
        if len(set(self.snr_db)) != len(self.snr_db):
            raise ValueError("sweep.snr_db must contain distinct points")
        _positive_int("sweep.num_tb_per_snr", self.num_tb_per_snr)
        _positive_int("sweep.batch_size", self.batch_size)
        _positive_int("sweep.min_tb_per_snr", self.min_tb_per_snr)
        _int_range("sweep.seed", self.seed, 0, 2**32 - 1)
        if self.target_block_errors is not None:
            _int_range("sweep.target_block_errors", self.target_block_errors, 1,
                       self.num_tb_per_snr)
            _int_range("sweep.min_tb_per_snr", self.min_tb_per_snr, 1,
                       self.num_tb_per_snr)
        _finite("sweep.ul_duty_cycle", self.ul_duty_cycle)
        if not 0 <= self.ul_duty_cycle <= 1:
            raise ValueError("sweep.ul_duty_cycle must be in [0, 1]")


@dataclass(frozen=True)
class OllaSettings:
    target_bler: float = 0.1
    warmup_tb: int = 100
    window_tb: int = 40
    tolerance: float = 0.05
    step_db: float = 1.0
    max_offset_db: float = 20.0
    control: str = "mcs_index"
    step_mcs: float = 1.0
    max_mcs_spread: float | None = None
    mcs_axis: str = "index"

    def validate(self) -> None:
        _choice("adaptation.olla.control", self.control, ("mcs_index", "sinr_margin"))
        _choice("adaptation.olla.mcs_axis", self.mcs_axis, ("index", "ladder"))
        if self.mcs_axis == "ladder" and self.control != "mcs_index":
            raise ValueError("adaptation.olla.mcs_axis=ladder requires mcs_index control")
        _finite("adaptation.olla.target_bler", self.target_bler)
        if not 0 < self.target_bler < 1:
            raise ValueError("adaptation.olla.target_bler must be between 0 and 1 exclusive")
        _positive_int("adaptation.olla.warmup_tb", self.warmup_tb)
        _int_range("adaptation.olla.window_tb", self.window_tb, 1, self.warmup_tb)
        _finite("adaptation.olla.tolerance", self.tolerance)
        if not 0 <= self.tolerance < 1:
            raise ValueError("adaptation.olla.tolerance must be in [0, 1)")
        if self.max_mcs_spread is not None:
            _finite("adaptation.olla.max_mcs_spread", self.max_mcs_spread)
            if self.control != "mcs_index" or self.max_mcs_spread <= 0:
                raise ValueError("adaptation.olla.max_mcs_spread requires mcs_index control and a positive spread")
            if self.warmup_tb < 3 * self.window_tb:
                raise ValueError("MCS stability requires at least three warmup windows")
        for field_name in ("step_db", "max_offset_db", "step_mcs"):
            name = f"adaptation.olla.{field_name}"
            value = getattr(self, field_name)
            _finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def in_target_window(self, bler: float) -> bool:
        deviation = abs(bler - self.target_bler)
        return deviation <= self.tolerance or math.isclose(deviation, self.tolerance)

    def fit_budget(self, num_tb_per_snr: int, warmup_tb: int | None = None) -> OllaSettings:
        """Override the warmup and shrink window_tb so the TB budgets stay valid.

        A stability gate needs three windows in both stages; otherwise the
        window only has to fit within the warmup.
        """
        warmup = self.warmup_tb if warmup_tb is None else warmup_tb
        max_window = warmup
        if self.max_mcs_spread is not None:
            max_window = min(warmup // 3, num_tb_per_snr // 3)
        return replace(self, warmup_tb=warmup, window_tb=max(1, min(self.window_tb, max_window)))


@dataclass(frozen=True)
class AdaptationSettings:
    srs_period_slots: int = 4
    srs_sequence_id: int = 0
    rank_tpmi_candidates: tuple[tuple[int, int], ...] | None = None
    mcs_thresholds_db: tuple[tuple[int, float], ...] = ((5, 2.0), (9, 6.0), (14, 10.0))
    sinr_margin_db: float = 1.0
    tdd_pattern: str | None = None
    srs_start_symbol: int = 10
    olla: OllaSettings | None = None
    mcs_indices: tuple[int, ...] | None = None
    tx_power_mode: str = "full_power"
    srs_estimator: str = "real"
    channel_evolution: str = "per_period"

    def rank_tpmi(self, pusch: PuschSettings) -> tuple[tuple[int, int], ...]:
        if self.rank_tpmi_candidates is None:
            return ((pusch.rank, pusch.tpmi),)
        return self.rank_tpmi_candidates

    def validate(self, cfg: SimulationConfig) -> None:
        _choice("adaptation.tx_power_mode", self.tx_power_mode, ("full_power", "per_port"))
        _choice("adaptation.srs_estimator", self.srs_estimator, ("real", "perfect"))
        _choice("adaptation.channel_evolution", self.channel_evolution, ("per_period", "continuous"))
        if self.olla is not None:
            self.olla.validate()
            if cfg.sweep.target_block_errors is not None:
                raise ValueError("adaptation.olla requires fixed TB counts; remove sweep.target_block_errors")
            if self.olla.max_mcs_spread is not None and cfg.sweep.num_tb_per_snr < 3 * self.olla.window_tb:
                raise ValueError("MCS stability requires at least three evaluation windows")
        _int_range("adaptation.srs_start_symbol", self.srs_start_symbol, 0, 10)
        if self.tdd_pattern is None:
            _int_range("adaptation.srs_period_slots", self.srs_period_slots, 2, 20)
            _choice("adaptation.srs_period_slots", self.srs_period_slots, (2, 4, 5, 8, 10, 16, 20))
            if self.srs_start_symbol != 10:
                raise ValueError("adaptation.srs_start_symbol requires adaptation.tdd_pattern")
        else:
            self._validate_tdd(cfg)
        _int_range("adaptation.srs_sequence_id", self.srs_sequence_id, 0, 1023)
        _finite("adaptation.sinr_margin_db", self.sinr_margin_db)
        if self.sinr_margin_db < 0:
            raise ValueError("adaptation.sinr_margin_db must be nonnegative")
        if cfg.carrier.n_size_grid < 4:
            raise ValueError("SRS requires a grid with at least 4 PRBs")

        max_mcs_index = (28, 27, 28)[cfg.pusch.mcs_table - 1]
        if self.mcs_indices is not None:
            name = "adaptation.mcs_indices"
            if self.olla is None or self.olla.control != "mcs_index":
                raise ValueError(f"{name} requires adaptation.olla.control=mcs_index")
            _sequence(name, self.mcs_indices)
            if not self.mcs_indices:
                raise ValueError(f"{name} must contain at least one MCS")
            for mcs in self.mcs_indices:
                _int_range(f"{name} entry", mcs, 0, max_mcs_index)
            if len(set(self.mcs_indices)) != len(self.mcs_indices):
                raise ValueError(f"{name} contains a duplicate MCS")

        thresholds = self.mcs_thresholds_db
        _sequence("adaptation.mcs_thresholds_db", thresholds)
        if not thresholds and self.mcs_indices is None:
            raise ValueError("adaptation.mcs_thresholds_db must contain at least one MCS")
        mcs_indices = set()
        for entry in thresholds:
            _sequence("adaptation.mcs_thresholds_db entry", entry, 2)
            mcs, threshold = entry
            _int_range("adaptation MCS index", mcs, 0, max_mcs_index)
            _finite("adaptation MCS threshold", threshold)
            if mcs in mcs_indices:
                raise ValueError("adaptation.mcs_thresholds_db contains a duplicate MCS")
            mcs_indices.add(mcs)

        pairs = self.rank_tpmi(cfg.pusch)
        _sequence("adaptation.rank_tpmi_candidates", pairs)
        if not pairs:
            raise ValueError("adaptation.rank_tpmi_candidates must contain at least one pair")
        valid_pairs = set()
        for entry in pairs:
            _sequence("adaptation.rank_tpmi_candidates entry", entry, 2)
            rank, tpmi = entry
            _int_range("adaptation rank", rank, 1, min(cfg.num_rx_ant, cfg.pusch.num_antenna_ports))
            if cfg.channel.estimator == "lmmse" and rank != 1:
                raise ValueError("channel.estimator=lmmse requires Rank 1 adaptation candidates")
            replace(cfg.pusch, rank=rank, tpmi=tpmi).validate()
            if (rank, tpmi) in valid_pairs:
                raise ValueError("adaptation.rank_tpmi_candidates contains a duplicate pair")
            valid_pairs.add((rank, tpmi))

    def _validate_tdd(self, cfg: SimulationConfig) -> None:
        name = "adaptation.tdd_pattern"
        if (not isinstance(self.tdd_pattern, str) or set(self.tdd_pattern) - {"D", "S", "U"}
                or "S" not in self.tdd_pattern or "U" not in self.tdd_pattern):
            raise ValueError(f"{name} must contain S and U and use only D, S, and U")
        _positive_int("adaptation.srs_period_slots", self.srs_period_slots)
        if cfg.pusch.num_antenna_ports != 4 or cfg.pusch.precoding != "codebook":
            raise ValueError("TDD sounding requires four TX ports and codebook PUSCH precoding")
        if cfg.carrier.n_size_grid not in SRS_BANDWIDTHS_PRB:
            raise ValueError("TDD sounding requires a standard B_SRS=0 bandwidth that covers the full grid")
        if cfg.sweep.ul_duty_cycle != 1.0:
            raise ValueError("TDD goodput already counts the pattern; sweep.ul_duty_cycle must be 1")


@dataclass(frozen=True)
class SimulationConfig:
    carrier: CarrierSettings = field(default_factory=CarrierSettings)
    pusch: PuschSettings = field(default_factory=PuschSettings)
    channel: ChannelSettings = field(default_factory=ChannelSettings)
    sweep: SweepSettings = field(default_factory=SweepSettings)
    precision: str = "single"
    adaptation: AdaptationSettings | None = None

    @property
    def slot_duration_s(self) -> float:
        return 0.001 * 15 / self.carrier.subcarrier_spacing_khz

    @property
    def occupied_bandwidth_hz(self) -> int:
        return 12_000 * self.carrier.subcarrier_spacing_khz * self.carrier.n_size_grid

    @property
    def num_rx_ant(self) -> int:
        if self.channel.kind == "cdl":
            return self.channel.bs_panel.num_antennas
        return self.channel.num_rx_ant

    def validate(self) -> None:
        _choice("precision", self.precision, ("single", "double"))
        self.carrier.validate()
        self.pusch.validate()
        self.channel.validate(self.pusch)
        self.sweep.validate()
        if self.adaptation is not None:
            self.adaptation.validate(self)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.adaptation is None:
            del values["adaptation"]
        else:
            if self.adaptation.olla is None:
                del values["adaptation"]["olla"]
            if self.adaptation.srs_estimator == "real":
                del values["adaptation"]["srs_estimator"]
            if self.adaptation.channel_evolution == "per_period":
                del values["adaptation"]["channel_evolution"]
        return values


_SECTIONS = {
    SimulationConfig: {
        "carrier": CarrierSettings,
        "pusch": PuschSettings,
        "channel": ChannelSettings,
        "sweep": SweepSettings,
        "adaptation": AdaptationSettings,
    },
    PuschSettings: {"dmrs": DmrsSettings},
    ChannelSettings: {"ue_panel": PanelSettings, "bs_panel": PanelSettings},
    AdaptationSettings: {"olla": OllaSettings},
}
_ARRAY_FIELDS = {
    PuschSettings: ("symbol_allocation",),
    DmrsSettings: ("port_set", "n_id"),
    PanelSettings: ("orientation_deg",),
    ChannelSettings: ("ue_yaw_offsets_deg",),
    SweepSettings: ("snr_db",),
    AdaptationSettings: ("rank_tpmi_candidates", "mcs_thresholds_db", "mcs_indices"),
}


def _settings(data: Any, cls, name: str):
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = data.keys() - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {name} field(s): {', '.join(sorted(unknown))}")
    values = dict(data)
    if cls is AdaptationSettings:
        olla = values.get("olla")
        if isinstance(olla, dict) and "control" not in olla:
            uses_mcs_state = "step_mcs" in olla or values.get("mcs_indices") is not None
            values["olla"] = {**olla, "control": "mcs_index" if uses_mcs_state else "sinr_margin"}
    for key, child_class in _SECTIONS.get(cls, {}).items():
        if key in values:
            if child_class in (AdaptationSettings, OllaSettings) and values[key] is None:
                continue
            values[key] = _settings(values[key], child_class, f"{name}.{key}")
    for key in _ARRAY_FIELDS.get(cls, ()):
        if isinstance(values.get(key), list):
            values[key] = tuple(values[key])
            if cls is AdaptationSettings:
                values[key] = tuple(tuple(entry) if isinstance(entry, list) else entry
                                    for entry in values[key])
    return cls(**values)


def load_config(path: str | Path | None = None) -> SimulationConfig:
    """Load strictly named settings from JSON, filling omitted fields with defaults."""
    if path is None:
        config = SimulationConfig()
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        config = _settings(data, SimulationConfig, "configuration")
    config.validate()
    return config
