"""Monte Carlo sweep for fixed-grant or SRS-adaptive single-UE PUSCH."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import os
from typing import Any

from .config import SimulationConfig
from .outputs import summarize_point


# Four-port CP-OFDM Type-I PUSCH TPMI counts per fixed rank. Each smaller
# coherence subset uses a prefix of the corresponding 38.212 DCI table.
_CODEBOOK_SIZES = {1: (28, 12, 4), 2: (22, 14, 6), 3: (7, 3, 1), 4: (5, 3, 1)}
_CODEBOOK_NAMES = ("full", "partial_noncoherent", "noncoherent")


def configure_worker_env(
    workers: int = 1,
    *,
    intra_threads: int | None = None,
    inter_threads: int | None = None,
) -> None:
    """Populate runtime logging and thread-count environment defaults before TF init."""
    if workers < 1:
        raise ValueError("workers must be positive")
    default_intra = min(os.cpu_count() or 1, 2 if workers > 1 else 4)
    default_inter = 1 if workers > 1 else 2
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(default_intra if intra_threads is None else intra_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(default_inter if inter_threads is None else inter_threads))
    if workers > 1:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")


def apply_tensorflow_threads() -> None:
    """Apply TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS if set."""
    intra = os.environ.get("TF_NUM_INTRAOP_THREADS")
    inter = os.environ.get("TF_NUM_INTEROP_THREADS")
    if intra is None and inter is None:
        return
    intra_val = int(intra) if intra is not None else None
    inter_val = int(inter) if inter is not None else None
    if (intra_val is not None and intra_val < 1) or (inter_val is not None and inter_val < 1):
        raise ValueError("TF_NUM_INTRAOP_THREADS and TF_NUM_INTEROP_THREADS must be positive integers")
    import tensorflow as tf

    try:
        if intra_val is not None:
            tf.config.threading.set_intra_op_parallelism_threads(intra_val)
        if inter_val is not None:
            tf.config.threading.set_inter_op_parallelism_threads(inter_val)
    except RuntimeError:
        # TensorFlow's execution context was already initialized in this process.
        pass


def simulate(
    cfg: SimulationConfig,
    *,
    channel_model_factory: Callable[[SimulationConfig, Any], Any] | None = None,
    on_point: Callable[[dict], None] | None = None,
) -> dict:
    """Return the configuration, link metadata and one result per SNR.

    For channel.kind="custom", the factory receives (cfg, resource_grid).
    It returns a Sionna CIR model callable with arguments (batch_size,
    num_time_steps, sampling_frequency) and return value (a, tau).
    """
    return _simulate(cfg, channel_model_factory, on_point)


def simulate_codebook_comparison(
    cfg: SimulationConfig,
    *,
    channel_model_factory: Callable[[SimulationConfig, Any], Any] | None = None,
    on_point: Callable[[str, dict], None] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Sweep three standard four-port subsets at cfg.pusch.rank on paired slots."""
    cfg.validate()
    if cfg.adaptation is None or cfg.pusch.num_antenna_ports != 4 or cfg.pusch.precoding != "codebook":
        raise ValueError("Codebook comparison requires adaptive four-port codebook PUSCH")
    if cfg.sweep.target_block_errors is not None:
        raise ValueError("Codebook comparison requires a fixed TB count for all subsets")
    rank = cfg.pusch.rank
    codebooks = {name: range(size) for name, size in zip(_CODEBOOK_NAMES, _CODEBOOK_SIZES[rank])}
    full_candidates = tuple((rank, tpmi) for tpmi in codebooks["full"])
    cfg = replace(cfg, adaptation=replace(cfg.adaptation, rank_tpmi_candidates=full_candidates))
    return _simulate(cfg, channel_model_factory, on_point, codebooks=codebooks, on_progress=on_progress)


def _simulate(cfg, channel_model_factory, on_point, *, codebooks=None, on_progress=None):
    cfg.validate()
    if (cfg.channel.kind == "custom") != (channel_model_factory is not None):
        raise ValueError('Provide channel_model_factory exactly when channel.kind is "custom".')

    try:
        import tensorflow as tf
        apply_tensorflow_threads()
        import sionna
        import sionna.phy
    except ImportError as exc:
        raise RuntimeError("Sionna PHY and TensorFlow are required. See requirements.txt.") from exc
    if sionna.__version__ != "1.2.2":
        raise RuntimeError(f"Use Sionna 1.2.2 for this baseline. Found {sionna.__version__}.")

    from .link import PuschLink

    sweep = cfg.sweep
    tf.random.set_seed(sweep.seed)
    sionna.phy.config.seed = sweep.seed
    metadata = {
        "sionna_version": sionna.__version__,
        "tensorflow_version": tf.__version__,
        "domain": "frequency",
        "num_ue": 1,
        "num_rx_ant": cfg.num_rx_ant,
        "slot_duration_s": cfg.slot_duration_s,
        "occupied_bandwidth_hz": cfg.occupied_bandwidth_hz,
        "receiver": "LMMSE with max-log soft demapping",
        "noise_variance": "10^(-SNR_dB/10) per complex receive antenna RE",
        "tb_error_definition": "CRC failed or decoded payload differs from transmitted payload",
    }
    if cfg.adaptation is not None:
        from .adaptive_simulation import simulate_adaptive

        return simulate_adaptive(
            cfg, metadata, channel_model_factory, on_point, codebooks=codebooks, on_progress=on_progress,
        )

    link = PuschLink(cfg, channel_model_factory, allow_low_rate=True)
    points = []
    for snr_db in sweep.snr_db:
        link.rewind_channel()
        no = link.noise_variance(snr_db)
        num_tb = tb_errors = 0
        stop_reason = "max_tb"
        while num_tb < sweep.num_tb_per_snr:
            batch_size = min(sweep.batch_size, sweep.num_tb_per_snr - num_tb)
            tb_errors += link.count_block_errors(batch_size, no)
            num_tb += batch_size
            if (sweep.target_block_errors is not None
                    and num_tb >= sweep.min_tb_per_snr
                    and tb_errors >= sweep.target_block_errors):
                stop_reason = "target_block_errors"
                break
        point = summarize_point(
            snr_db, num_tb, tb_errors, link.tb_size_bits,
            cfg.slot_duration_s, sweep.ul_duty_cycle, stop_reason,
        )
        points.append(point)
        if on_point is not None:
            on_point(point.copy())
    metadata.update({
        "pusch": link.grant_metadata,
        "pairing": "same Sionna CIR random stream across SNR points",
        "snr_reference": "unit average QAM symbol energy per layer before precoding and channel",
        "sampling": "fixed TB count" if sweep.target_block_errors is None else
                    "TB and block-error limits checked at batch boundaries",
    })
    return {"config": cfg.to_dict(), "metadata": metadata, "points": points}
