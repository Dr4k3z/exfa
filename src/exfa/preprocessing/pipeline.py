from collections.abc import Sequence
from typing import Protocol

import numpy as np


class PanelData:
    """
    Aligned NumPy data and positions in the original asset/time axes.

    Returns have shape (assets, times). Optional loadings have shape
    (assets, factors, times); static (assets, factors) loadings are broadcast
    and copied. Optional weights have shape (assets, times); static asset
    weights are also accepted. NaN and infinity are retained for cleaning.
    """

    def __init__(
        self,
        returns: np.ndarray,
        loadings: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        *,
        asset_indices: np.ndarray | None = None,
        time_indices: np.ndarray | None = None,
    ) -> None:
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2 or 0 in returns.shape:
            raise ValueError("returns must have nonempty shape (assets, times)")
        num_assets, num_times = returns.shape
        self.returns = returns.copy()
        self.loadings = None
        if loadings is not None:
            loadings = np.asarray(loadings, dtype=float)
            if loadings.ndim == 2:
                loadings = np.repeat(loadings[:, :, None], num_times, axis=2)
            if (
                loadings.ndim != 3
                or loadings.shape[0] != num_assets
                or loadings.shape[2] != num_times
                or loadings.shape[1] == 0
            ):
                raise ValueError("loadings must have shape (assets, factors[, times])")
            self.loadings = loadings.copy()
        self.weights = None
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape == (num_assets,):
                weights = np.repeat(weights[:, None], num_times, axis=1)
            if weights.shape != returns.shape:
                raise ValueError("weights must have shape (assets,) or (assets, times)")
            self.weights = weights.copy()
        self.asset_indices = self._validate_indices(asset_indices, num_assets, "asset_indices")
        self.time_indices = self._validate_indices(time_indices, num_times, "time_indices")

    @staticmethod
    def _validate_indices(indices: np.ndarray | None, size: int, name: str) -> np.ndarray:
        if indices is None:
            return np.arange(size)
        indices = np.asarray(indices)
        if (
            indices.shape != (size,)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.unique(indices).size != size
        ):
            raise ValueError(f"{name} must contain distinct non-negative integer positions")
        return indices.copy()

    def copy(self) -> PanelData:
        return PanelData(
            self.returns,
            self.loadings,
            self.weights,
            asset_indices=self.asset_indices,
            time_indices=self.time_indices,
        )

    def select(self, assets: np.ndarray, times: np.ndarray) -> PanelData:
        """
        Select aligned boolean masks without changing original index positions.
        """

        assets = np.asarray(assets)
        times = np.asarray(times)
        if (
            assets.dtype != bool
            or assets.shape != (self.returns.shape[0],)
            or times.dtype != bool
            or times.shape != (self.returns.shape[1],)
        ):
            raise ValueError("Selection masks must be boolean and match the panel axes")
        return PanelData(
            self.returns[assets][:, times],
            self.loadings[assets][:, :, times] if self.loadings is not None else None,
            self.weights[assets][:, times] if self.weights is not None else None,
            asset_indices=self.asset_indices[assets],
            time_indices=self.time_indices[times],
        )


class CleaningStep(Protocol):
    """
    A model-independent cleaning function, callable object, or partial.

    Return an aligned PanelData, preserving original index positions when
    filtering. User steps should use copy()/select() before modifying values.
    """

    def __call__(self, panel: PanelData, /) -> PanelData: ...


class PreprocessingPipeline:
    """
    Run explicitly configured steps in order, without implicit cleaning policies.
    """

    def __init__(self, steps: Sequence[CleaningStep] = ()) -> None:
        self.steps = tuple(steps)

    def run(self, panel: PanelData) -> PanelData:
        """
        Clean a copy, leaving the caller's input intact even if a step mutates it.
        """

        result = panel.copy()
        for step in self.steps:
            previous_assets = result.asset_indices.copy()
            previous_times = result.time_indices.copy()
            previous_num_factors = (
                result.loadings.shape[1] if result.loadings is not None else None
            )
            previous_has_weights = result.weights is not None
            result = step(result)
            if not isinstance(result, PanelData):
                raise TypeError("Cleaning steps must return PanelData")
            # Revalidate even when a custom step mutated a panel in-place.
            result = result.copy()
            num_factors = result.loadings.shape[1] if result.loadings is not None else None
            if (
                num_factors != previous_num_factors
                or (result.weights is not None) != previous_has_weights
            ):
                raise ValueError("Steps must preserve the factor axis and supplied fields")
            for previous, current in (
                (previous_assets, result.asset_indices),
                (previous_times, result.time_indices),
            ):
                expected = previous[np.isin(previous, current)]
                if not np.array_equal(expected, current):
                    raise ValueError(
                        "Steps must preserve original index order and cannot add positions"
                    )
        return result


def _select_columns(loadings: np.ndarray, columns: Sequence[int] | None) -> tuple[int, ...]:
    indices = tuple(range(loadings.shape[1])) if columns is None else tuple(columns)
    if len(set(indices)) != len(indices) or any(
        isinstance(i, bool)
        or not isinstance(i, (int, np.integer))
        or i < 0
        or i >= loadings.shape[1]
        for i in indices
    ):
        raise ValueError("columns must contain distinct valid loading column indices")
    return indices


def fill_missing_loadings(panel: PanelData, *, columns: Sequence[int]) -> PanelData:
    """
    Fill nonfinite loadings with the same-time cross-sectional median.

    Entirely missing cross-sections remain missing. Select continuous columns
    explicitly to avoid imputing categorical indicators. Returns are untouched;
    no information is borrowed from earlier or later dates.
    """

    result = panel.copy()
    if result.loadings is None:
        raise ValueError("Loading imputation requires loadings")
    for column in _select_columns(result.loadings, columns):
        for time in range(result.returns.shape[1]):
            values = result.loadings[:, column, time]
            finite = np.isfinite(values)
            if np.any(finite):
                values[~finite] = np.median(values[finite])
    return result


def winsorize_loadings(
    panel: PanelData,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    columns: Sequence[int],
) -> PanelData:
    """
    Clip selected loadings to finite same-time cross-sectional quantiles.

    Missing entries are preserved. No temporal information is used. Choose
    continuous columns explicitly to leave categorical indicators untouched.
    """

    if not (0 <= lower <= upper <= 1):
        raise ValueError("Expected 0 <= lower <= upper <= 1")
    result = panel.copy()
    if result.loadings is None:
        raise ValueError("Winsorization requires loadings")
    for column in _select_columns(result.loadings, columns):
        for time in range(result.returns.shape[1]):
            values = result.loadings[:, column, time]
            finite = np.isfinite(values)
            if np.any(finite):
                limits = np.quantile(values[finite], [lower, upper])
                values[finite] = np.clip(values[finite], limits[0], limits[1])
    return result


def _compute_valid_observations(panel: PanelData) -> np.ndarray:
    valid = np.isfinite(panel.returns)
    if panel.loadings is not None:
        valid &= np.isfinite(panel.loadings).all(axis=1)
    if panel.weights is not None:
        valid &= np.isfinite(panel.weights) & (panel.weights > 0)
    return valid


def drop_empty_times(panel: PanelData) -> PanelData:
    """
    Drop only times with no usable asset observations across the supplied fields.

    A single missing asset never causes an otherwise usable date to be dropped.
    """

    return panel.select(
        np.ones(panel.returns.shape[0], dtype=bool),
        _compute_valid_observations(panel).any(axis=0),
    )


def drop_incomplete_assets(panel: PanelData) -> PanelData:
    """
    Keep assets usable at every retained time, preserving every retained date.

    This explicit complete-case policy may remove many assets and selects using
    the entire supplied window. Run it only on the intended estimation window,
    never on a combined training/test window. It does not impute returns.
    """

    return panel.select(
        _compute_valid_observations(panel).all(axis=1),
        np.ones(panel.returns.shape[1], dtype=bool),
    )


def require_finite(panel: PanelData) -> PanelData:
    """
    Require a finite complete panel with at least one asset and two times.

    This final opt-in check matches current model input requirements, without
    importing or depending on either model.
    """

    if panel.returns.shape[0] < 1 or panel.returns.shape[1] < 2:
        raise ValueError("Cleaning must retain at least one asset and two times")
    if not _compute_valid_observations(panel).all():
        raise ValueError("Panel still contains missing/nonfinite values or invalid weights")
    return panel
