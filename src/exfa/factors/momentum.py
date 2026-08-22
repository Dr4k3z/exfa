from collections.abc import Sequence
from typing import override

import numpy as np
import polars as pl

from exfa.factors.base import Factor, FactorExposure, FactorKind


class MomentumFactor(Factor):
    """
    Trailing-price momentum factor based on exponentially weighted log returns.

    For asset i at time t, the raw momentum exposure is

        M_{i,t} = sum_k w_k log(1 + r_{i,t-lag-k})

    over a trailing lookback window.

    The most recent `lag` observations are excluded to reduce contamination
    from short-term reversal effects.
    """

    def __init__(
        self,
        returns_col: str = "returns",
        *,
        symbol_col: str = "symbol",
        date_col: str = "date",
        lookback: int = 252,
        lag: int = 21,
        half_life: float | None = 126.0,
        min_samples: int | None = None,
        name: str = "momentum",
    ) -> None:
        super().__init__(
            name=name,
            kind=FactorKind.CONTINUOUS,
        )

        self._validate_parameters(
            lookback=lookback,
            lag=lag,
            half_life=half_life,
            min_samples=min_samples,
        )

        self.returns_col = returns_col
        self.symbol_col = symbol_col
        self.date_col = date_col

        self.lookback = lookback
        self.lag = lag
        self.half_life = half_life
        self.min_samples = lookback if min_samples is None else min_samples

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (
            self.date_col,
            self.symbol_col,
            self.returns_col,
        )

    @property
    def weights(self) -> list[float] | None:
        """
        Exponential rolling weights ordered from oldest to newest observation.

        The newest observation receives weight 1.0. If `half_life` is None,
        equal weighting is used.
        """
        if self.half_life is None:
            return None

        decay = np.log(2.0) / self.half_life

        ages = np.arange(
            self.lookback - 1,
            -1,
            -1,
            dtype=np.float64,
        )

        weights = np.exp(-decay * ages)

        return weights.tolist()

    @override
    def compute(self, data: pl.LazyFrame) -> Sequence[FactorExposure]:
        self.validate_data(data)

        log_returns = pl.col(self.returns_col).cast(pl.Float64).log1p()
        valid_return = log_returns.is_not_null() & log_returns.is_finite()

        # Polars does not support weighted rolling sums over arrays containing
        # nulls. Replace invalid observations only for the aggregation, and use
        # a parallel validity count to enforce the requested minimum sample
        # count without treating those replacements as real observations.
        rolling_momentum = (
            pl.when(valid_return)
            .then(log_returns)
            .otherwise(0.0)
            .rolling_sum(
                window_size=self.lookback,
                weights=self.weights,
                min_samples=1,
            )
        )
        valid_samples = valid_return.cast(pl.UInt32).rolling_sum(
            window_size=self.lookback,
            min_samples=1,
        )

        expression = (
            pl.when(valid_samples >= self.min_samples)
            .then(rolling_momentum)
            .otherwise(None)
            .shift(self.lag)
            .over(
                self.symbol_col,
                order_by=self.date_col,
            )
            .alias(self.name)
        )

        return (
            FactorExposure(
                name=self.name,
                expression=expression,
            ),
        )

    @staticmethod
    def _validate_parameters(
        *,
        lookback: int,
        lag: int,
        half_life: float | None,
        min_samples: int | None,
    ) -> None:
        if lookback <= 0:
            raise ValueError(f"`lookback` must be strictly positive, got {lookback}.")

        if lag < 0:
            raise ValueError(f"`lag` must be non-negative, got {lag}.")

        if half_life is not None and half_life <= 0:
            raise ValueError(
                f"`half_life` must be strictly positive or None, got {half_life}."
            )

        if min_samples is not None:
            if min_samples <= 0:
                raise ValueError(
                    f"`min_samples` must be strictly positive, got {min_samples}."
                )

            if min_samples > lookback:
                raise ValueError(
                    f"`min_samples` cannot exceed `lookback`: {min_samples} > {lookback}."
                )

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"lookback={self.lookback}, "
            f"lag={self.lag}, "
            f"half_life={self.half_life}, "
            f"min_samples={self.min_samples}"
            ")"
        )
