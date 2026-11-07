from collections.abc import Sequence
from typing import override

import pandas as pd

from exfa.models.fundamental_model.factors.base import CategoricalFactor


class SectorFactor(CategoricalFactor):
    """
    Categorical sector factor represented by one-hot asset loadings.
    """

    def __init__(self, sector_col: str = "sector"):
        super().__init__(name="sector")
        self.sector_col = sector_col

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.sector_col,)

    @override
    def _compute(self, data) -> pd.DataFrame:
        return pd.get_dummies(
            data[self.sector_col],
            prefix=self.name,
            dtype=float,
        )
