from collections.abc import Sequence
from typing import override

import pandas as pd

from .base import ContinuousFactor


class MomentumFactor(ContinuousFactor):
    def __init__(
        self,
        name: str,
        window: int,
        price_col: str = "price",
        impute: bool = True,
        winsorize: bool = True,
        standardize: bool = True,
    ) -> None:
        self.window = window
        self.price_col = price_col

        super().__init__(name, impute, winsorize, standardize)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.price_col,)

    @override
    def _compute(self, data) -> pd.Series:
        prices = data[self.price_col]
        lagged_prices = prices.groupby(level="asset").shift(self.window)

        return prices / lagged_prices - 1
