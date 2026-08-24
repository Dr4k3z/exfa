import numpy as np
import pandas as pd


def median_imputation(x: pd.Series) -> pd.Series:
    x = x.astype(float)

    median = x.median()
    x = x.fillna(median)

    if x.isna().all():
        return pd.Series(np.nan, index=x.index)
    return x


def winsorization(
    x: pd.Series, winsor_lower: float = 0.01, winsor_upper: float = 0.99
) -> pd.Series:
    lower = x.quantile(winsor_lower)
    upper = x.quantile(winsor_upper)

    x = x.clip(lower=lower, upper=upper)
    return x


def z_scoring(x: pd.Series) -> pd.Series:
    mean = x.mean()
    std = x.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=x.index)

    return (x - mean) / std


def clean_factor(
    x: pd.Series,
    impute: bool = True,
    winsorize: bool = True,
    standardize: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.Series:
    def clean_cross_section(s: pd.Series) -> pd.Series:
        if impute:
            s = median_imputation(s)

        if winsorize:
            s = winsorization(
                s,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
            )

        if standardize:
            s = z_scoring(s)

        return s

    return x.groupby(
        level="date",
        group_keys=False,
    ).apply(clean_cross_section)
