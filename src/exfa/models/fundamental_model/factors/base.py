from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import override

import numpy as np

from .preprocessing import clean_factor

MarketData = Mapping[str, np.ndarray]


def get_feature(data: MarketData, name: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Return a feature aligned to fitted times without copying static values.

    Longer histories end at the same time as returns; their trailing columns
    align with returns. Leading columns are available for rolling factors.
    """

    values = np.asarray(data[name])
    return np.broadcast_to(values[:, None], shape) if values.ndim == 1 else values[:, -shape[1] :]


class Factor(ABC):
    def __init__(self, name: str, impute: bool, winsorize: bool, standardize: bool) -> None:
        self.name = name
        self.impute = impute
        self.winsorize = winsorize
        self.standardize = standardize

    @property
    @abstractmethod
    def required_features(self) -> Sequence[str]: ...

    @property
    @abstractmethod
    def requires_constraint(self) -> bool: ...

    @property
    def include_in_vif(self) -> bool:
        return False

    def validate_data(self, data: MarketData, shape: tuple[int, int]) -> None:
        missing = [name for name in self.required_features if name not in data]
        if missing:
            raise ValueError(f"{type(self).__name__} requires missing features: {missing}")
        for name in self.required_features:
            values = np.asarray(data[name])
            if not (
                values.shape == (shape[0],)
                or (
                    values.ndim == 2 and values.shape[0] == shape[0] and values.shape[1] >= shape[1]
                )
            ):
                raise ValueError(
                    f"Feature {name!r} must have shape (assets,) or (assets, times), optionally with leading history"
                )

    @abstractmethod
    def compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray: ...

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ContinuousFactor(Factor):
    @property
    @override
    def requires_constraint(self) -> bool:
        return False

    @property
    @override
    def include_in_vif(self) -> bool:
        return True

    @abstractmethod
    def _compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray: ...

    @override
    def compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        self.validate_data(data, shape)
        values = np.asarray(self._compute(data, shape), dtype=float)
        if values.shape != shape:
            raise ValueError(f"Factor {self.name!r} must produce (assets, times)")
        return clean_factor(values, self.impute, self.winsorize, self.standardize)[:, None, :]


class CategoricalFactor(Factor):
    def __init__(self, name: str, column: str) -> None:
        super().__init__(name, False, False, False)
        self.column = column

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.column,)

    @property
    @override
    def requires_constraint(self) -> bool:
        return True

    @override
    def compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        self.validate_data(data, shape)
        values = get_feature(data, self.column, shape)
        if any(value is None or value != value for value in values.flat):
            raise ValueError(f"Categorical feature {self.column!r} contains missing values")
        categories = np.unique(values)
        return (values[:, None, :] == categories[None, :, None]).astype(float)
