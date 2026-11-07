from collections.abc import Sequence
from typing import override

import pandas as pd

from exfa.models.fundamental_model.factors.base import CategoricalFactor


class CountryFactor(CategoricalFactor):
    def __init__(
        self,
        column: str = "country",
    ) -> None:
        super().__init__(name="country")
        self.column = column

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.column,)

    @override
    def _compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            data[self.column],
            prefix=self.name,
            dtype=float,
        )
