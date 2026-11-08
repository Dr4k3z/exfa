from collections.abc import Sequence
from typing import override

import numpy as np

from .base import ContinuousFactor, MarketData


class SizeFactor(ContinuousFactor):
    def __init__(
        self,
        name: str,
        size_col: str = "volume",
        impute: bool = True,
        winsorize: bool = True,
        standardize: bool = True,
    ) -> None:
        self.size_col = size_col
        super().__init__(name, impute, winsorize, standardize)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.size_col,)

    @override
    def _compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError("SizeFactor computation has not been implemented")
