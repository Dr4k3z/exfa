from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import override

import polars as pl


class FactorKind(Enum):
    INTERCEPT = auto()
    CONTINUOUS = auto()
    CATEGORICAL = auto()
    STATISTICAL = auto()


@dataclass(frozen=True)
class FactorExposure:
    name: str
    expression: pl.Expr


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
        """Validate that required raw features are available."""
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
