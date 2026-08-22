from collections.abc import Sequence
from typing import override

import polars as pl

from exfa.factors.base import Factor, FactorExposure, FactorKind


class MarketFactor(Factor):
    """
    Market factor represented by a unit exposure for every asset.

    The estimated return associated with this exposure represents the
    cross-sectional market return of the model.
    """

    def __init__(self, name: str = "market") -> None:
        super().__init__(
            name=name,
            kind=FactorKind.INTERCEPT,
        )

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return ()

    @override
    def compute(
        self,
        data: pl.LazyFrame,
    ) -> Sequence[FactorExposure]:
        return (
            FactorExposure(
                name=self.name,
                expression=pl.lit(1.0).alias(self.name),
            ),
        )
