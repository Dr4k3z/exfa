from collections.abc import Sequence
from typing import override

import pandas as pd

from exfa.models.fundamental_model.factors.base import ContinuousFactor


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
    def _compute(self, data) -> pd.Series:
        # TODO
        return pd.Series([])
