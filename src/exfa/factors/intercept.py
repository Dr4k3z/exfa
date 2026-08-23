from collections.abc import Sequence
from typing import override

import pandas as pd

from exfa.factors.base import ContinuousFactor


class InterceptFactor(ContinuousFactor):
    def __init__(self) -> None:
        super().__init__(name="intercept", impute=False, winsorize=False, standardize=False)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return ()

    @override
    def _compute(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            1.0,
            index=data.index,
            name=self.name,
        )
