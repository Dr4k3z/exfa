import numpy as np


def clean_factor(
    values: np.ndarray,
    impute: bool = True,
    winsorize: bool = True,
    standardize: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> np.ndarray:
    """
    Clean continuous loadings cross-sectionally, without mutating input arrays.

    Entirely missing columns remain missing. No future observations are used.
    """

    if not 0 <= winsor_lower <= winsor_upper <= 1:
        raise ValueError("Invalid winsorization quantiles")
    if not (impute or winsorize or standardize):
        return values
    result = values.copy()
    for time in range(result.shape[1]):
        column = result[:, time]
        valid = np.isfinite(column)
        if not valid.any():
            continue
        if impute:
            column[~valid] = np.median(column[valid])
            valid = np.isfinite(column)
        if winsorize:
            lower, upper = np.quantile(column[valid], [winsor_lower, winsor_upper])
            column[valid] = np.clip(column[valid], lower, upper)
        if standardize:
            std = np.std(column[valid])
            column[valid] = (column[valid] - np.mean(column[valid])) / std if std > 0 else 0.0
    return result
