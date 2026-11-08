from collections.abc import Sequence
from typing import override

import numpy as np

from .base import ContinuousFactor, MarketData


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
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.price_col = price_col
        super().__init__(name, impute, winsorize, standardize)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.price_col,)

    @override
    def _compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        prices = np.asarray(data[self.price_col], dtype=float)
        if prices.ndim != 2 or prices.shape[1] < shape[1] + self.window:
            raise ValueError(
                f"Momentum requires at least {self.window} leading price observations before the fitted returns"
            )
        current = prices[:, -shape[1] :]
        lagged = prices[:, -shape[1] - self.window : -self.window]
        result = np.full(shape, np.nan)
        np.divide(current, lagged, out=result, where=lagged != 0)
        return result - 1
