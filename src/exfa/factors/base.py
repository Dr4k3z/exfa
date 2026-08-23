from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override

import pandas as pd

from .clean import clean_factor


class Factor(ABC):
    def __init__(self, name: str, impute: bool, winsorize: bool, standardize: bool) -> None:
        self.name = name

        # Cleaning configuration
        self.impute = impute
        self.winsorize = winsorize
        self.standardize = standardize

    @property
    @abstractmethod
    def required_features(self) -> Sequence[str]:
        """Columns required to compute the raw exposure."""
        ...

    @property
    @abstractmethod
    def requires_constraint(self) -> bool:
        """In the regression"""
        ...

    def validate_data(self, data: pd.DataFrame) -> None:
        missing = [feature for feature in self.required_features if feature not in data.columns]

        if missing:
            raise ValueError(f"{type(self).__name__} requires missing features: {missing}")

        # NOTE: are we sure about this?
        if not isinstance(data.index, pd.MultiIndex):
            raise TypeError("Factor data must use a MultiIndex with levels ('date', 'asset').")

        required_levels = {"date", "asset"}
        missing_levels = required_levels - set(data.index.names)

        if missing_levels:
            raise ValueError(f"Factor data is missing index levels: {sorted(missing_levels)}")

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame: ...

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ContinuousFactor(Factor):
    def __init__(self, name: str, impute: bool, winsorize: bool, standardize: bool) -> None:
        super().__init__(name, impute, winsorize, standardize)

    @abstractmethod
    def _compute(self, data: pd.DataFrame) -> pd.Series: ...

    @property
    @override
    def requires_constraint(self) -> bool:
        return False

    @override
    def compute(self, data: pd.DataFrame, clean: bool = True) -> pd.Series:
        self.validate_data(data)
        exposure = self._compute(data)
        if clean:
            exposure = clean_factor(exposure, self.impute, self.winsorize, self.standardize)
        return exposure.rename(self.name)


class CategoricalFactor(Factor):
    def __init__(self, name):
        super().__init__(name, False, False, False)

    @abstractmethod
    def _compute(self, data: pd.DataFrame) -> pd.DataFrame: ...

    @property
    @override
    def requires_constraint(self) -> bool:
        return True

    @override
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_data(data)
        exposure = self._compute(data)
        return exposure


"""
class Factor(ABC):
    def __init__(self, name: str, kind: FactorKind) -> None:
        self.name = name
        self.kind = kind

    @property
    @abstractmethod
    def required_features(self) -> Sequence[str]: ...

    @property
    def requires_fit(self) -> bool:
        return False

    def validate_data(self, data: pl.LazyFrame) -> None:
        Validate that required raw features are available
        schema = data.collect_schema()
        missing = [feature for feature in self.required_features if feature not in schema]

        if missing:
            raise ValueError(f"Factor {self.name!r} requires missing features: {missing}")

    def fit(self, data: pl.LazyFrame) -> None:
        if self.requires_fit:
            raise NotImplementedError(
                f"{type(self).__name__} requires fitting but does not implement fit()."
            )

    @abstractmethod
    def compute(self, data: pl.LazyFrame) -> Sequence[FactorExposure]: ...

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind.name})"
"""
