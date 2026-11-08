from collections.abc import Sequence
from typing import override

import numpy as np

from .base import ContinuousFactor, MarketData


class InterceptFactor(ContinuousFactor):
    def __init__(self) -> None:
        super().__init__("intercept", False, False, False)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return ()

    @property
    @override
    def include_in_vif(self) -> bool:
        return False

    @override
    def _compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        return np.ones(shape)
