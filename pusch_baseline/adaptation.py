"""Select a single-UE PUSCH grant using a measured uplink channel."""

from collections.abc import Iterable, Mapping

import numpy as np


GrantKey = tuple[int, int, int]  # Rank, TPMI, MCS index
GrantCandidate = tuple[np.ndarray, int]  # TX-port-by-layer precoder, TB size in bits


def select_mcs_for_precoder(
    spatial_key: tuple[int, int], sinr_db: float,
    candidates: Mapping[GrantKey, GrantCandidate],
    thresholds: Mapping[int, float], margin_db: float,
) -> GrantKey:
    """Update MCS with the latest feedback while retaining the SRS precoder."""
    return max(
        ((*spatial_key, mcs) for mcs, threshold in thresholds.items()
         if sinr_db >= threshold + margin_db),
        key=lambda key: (candidates[key][1], -key[2]),
        default=(*spatial_key, min(thresholds)),
    )


def _effective_sinr_metrics(
    gram: np.ndarray, noise_variance: float, tone_counts: np.ndarray | None
) -> tuple[float, float]:
    # The diagonal of (I + H_eff^H H_eff / noise)^-1 is the per-layer
    # MMSE error variance, equal to 1 / (1 + SINR) for white noise.
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    mode_error = noise_variance / (noise_variance + eigenvalues)
    layer_error = np.sum(np.abs(eigenvectors)**2 * mode_error[:, None, :], axis=-1)

    # A Gaussian-input abstraction: average log(1 + SINR) over frequency
    # and layers, then map to an equivalent single SINR. Also track the
    # multi-layer sum mutual information for thresholdless multi-rank choice.
    rank = gram.shape[-1]
    log_rate = -np.log(np.clip(layer_error, 1e-30, 1.0)).mean(axis=-1)
    log_rate = float(np.average(log_rate, weights=tone_counts))
    linear_sinr = max(np.expm1(log_rate), 1e-30)
    return float(10.0 * np.log10(linear_sinr)), rank * log_rate


def select_grant(
    channel: np.ndarray,
    noise_variance: float,
    candidates: Mapping[GrantKey, GrantCandidate],
    mcs_thresholds_db: Mapping[int, float],
    margin_db: float,
    *,
    srs_error_variance: float | np.ndarray | None = None,
) -> tuple[GrantKey, float]:
    """Return the selected (rank, TPMI, MCS) and its predicted effective SINR.

    The channel estimate has shape [subcarrier, RX port, TX port]. Noise
    variance is per receive antenna and complex RE. A rank/TPMI pair uses
    the same precoder across its candidates, including any TX power scaling.
    Thresholds are policy parameters to calibrate for the intended BLER,
    receiver, channel estimation and grant size.

    Both ideal and estimated SRS channels use Gram matrices averaged within
    each 12-tone PRB. An optional scalar or per-tone error variance removes
    the expected energy of independent RX/TX estimation errors before PRB
    averaging and clipping negative Gram eigenvalues.
    """
    return select_grants(
        channel, noise_variance, candidates, mcs_thresholds_db, margin_db,
        {"": candidates}, srs_error_variance=srs_error_variance,
    )[""]


def select_grants(
    channel: np.ndarray,
    noise_variance: float,
    candidates: Mapping[GrantKey, GrantCandidate],
    mcs_thresholds_db: Mapping[int, float] | None,
    margin_db: float,
    codebook_keys: Mapping[str | None, Iterable[GrantKey]],
    *,
    srs_error_variance: float | np.ndarray | None = None,
    margin_offsets_db: Mapping[str | None, float] | None = None,
) -> dict[str | None, tuple[GrantKey, float]]:
    """Apply the same grant policy to each subset of a shared candidate table.

    Score a rank and TPMI pair once on the common SRS estimate, including
    when the pair supports multiple MCS indices or belongs to several subsets.
    Without MCS thresholds, use the sum of the estimated rates over layers.
    """
    subsets = {name: tuple(keys) for name, keys in codebook_keys.items()}
    if not subsets or any(not keys for keys in subsets.values()):
        raise ValueError("Link adaptation needs at least one PUSCH grant candidate.")
    if not np.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("Link adaptation needs finite positive noise variance.")

    channel = np.asarray(channel, dtype=np.complex128)
    tx_gram = channel.conj().swapaxes(-1, -2) @ channel
    if srs_error_variance is not None:
        error_variance = np.asarray(srs_error_variance, dtype=float)
        if (error_variance.shape not in ((), tx_gram.shape[:1])
                or not np.all(np.isfinite(error_variance) & (error_variance >= 0.0))):
            raise ValueError("SRS error variance must be finite and nonnegative, scalar or per tone.")
        tx_gram -= channel.shape[1] * error_variance[..., None, None] * np.eye(channel.shape[2])
    prb_starts = np.arange(0, len(tx_gram), 12)
    tone_counts = np.diff(prb_starts, append=len(tx_gram))
    tx_gram = np.add.reduceat(tx_gram, prb_starts, axis=0) / tone_counts[:, None, None]
    sinr_by_precoder = {}
    sum_rate_by_precoder = {}
    for key in dict.fromkeys(key for keys in subsets.values() for key in keys):
        precoder, _ = candidates[key]
        spatial_key = key[:2]
        if spatial_key not in sinr_by_precoder:
            gram = precoder.conj().T @ tx_gram @ precoder
            sinr_by_precoder[spatial_key], sum_rate_by_precoder[spatial_key] = (
                _effective_sinr_metrics(gram, noise_variance, tone_counts)
            )

    def spatial_order(key: GrantKey):
        rank, tpmi, _ = key
        return sinr_by_precoder[rank, tpmi], -rank, -tpmi

    def sum_rate_order(key: GrantKey):
        rank, tpmi, _ = key
        return sum_rate_by_precoder[rank, tpmi], sinr_by_precoder[rank, tpmi], -rank, -tpmi

    decisions = {}
    for name, keys in subsets.items():
        if mcs_thresholds_db is None:
            selected = max(keys, key=sum_rate_order)
            decisions[name] = (selected, sinr_by_precoder[selected[:2]])
            continue
        margin = margin_db if margin_offsets_db is None else margin_db + margin_offsets_db[name]
        selected = max(
            (key for key in keys if sinr_by_precoder[key[:2]] >= mcs_thresholds_db[key[2]] + margin),
            key=lambda key: (candidates[key][1], *spatial_order(key), -key[2]),
            default=None,
        )
        if selected is None:
            selected = max(keys, key=lambda key: (-key[2], *spatial_order(key)))
        decisions[name] = (selected, sinr_by_precoder[selected[:2]])
    return decisions
