from collections.abc import Sequence
from typing import Literal, override

import polars as pl

from exfa.factors.base import Factor, FactorExposure, FactorKind


class SectorFactor(Factor):
    """
    Categorical sector factor represented by one-hot asset exposures.
    """

    def __init__(
        self,
        sector_col: str = "sector",
        *,
        categories: Sequence[str] | None = None,
        name: str = "sector",
        prefix: str | None = None,
        allow_null: bool = False,
        unknown_category: Literal["RAISE", "IGNORE"] = "RAISE",
    ) -> None:
        super().__init__(
            name=name,
            kind=FactorKind.CATEGORICAL,
        )

        if categories is not None and not categories:
            raise ValueError("`categories` cannot be empty.")

        if unknown_category not in {"raise", "ignore"}:
            raise ValueError(
                f"`unknown_category` must be either 'raise' or 'ignore', got {unknown_category!r}."
            )

        self.sector_col = sector_col
        self.prefix = name if prefix is None else prefix
        self.allow_null = allow_null
        self.unknown_category = unknown_category

        self._categories = self._validate_categories(categories) if categories is not None else None

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.sector_col,)

    @property
    @override
    def requires_fit(self) -> bool:
        return self._categories is None

    @property
    def categories(self) -> tuple[str, ...]:
        if self._categories is None:
            raise RuntimeError(
                f"Factor {self.name!r} has not been fitted yet. "
                "Call `fit()` before accessing its categories."
            )

        return self._categories

    @override
    def fit(self, data: pl.LazyFrame) -> None:
        self.validate_data(data)

        categories = (
            data.select(pl.col(self.sector_col)).unique().collect().get_column(self.sector_col)
        )

        if categories.null_count() and not self.allow_null:
            raise ValueError(f"Factor {self.name!r} found null values in {self.sector_col!r}.")

        values = [value for value in categories.to_list() if value is not None]

        if not values:
            raise ValueError(
                f"Factor {self.name!r} found no valid categories in {self.sector_col!r}."
            )

        self._categories = self._validate_categories(values)

    @override
    def compute(
        self,
        data: pl.LazyFrame,
    ) -> Sequence[FactorExposure]:
        self.validate_data(data)

        if self.requires_fit:
            raise RuntimeError(
                f"Factor {self.name!r} must be fitted before exposures can be computed."
            )

        if self.unknown_category == "raise":
            self._validate_observed_categories(data)

        exposures = tuple(
            FactorExposure(
                name=self._exposure_name(category),
                expression=(
                    pl.col(self.sector_col)
                    .eq(category)
                    .cast(pl.Float64)
                    .alias(self._exposure_name(category))
                ),
            )
            for category in self.categories
        )

        return exposures

    def _validate_observed_categories(
        self,
        data: pl.LazyFrame,
    ) -> None:
        observed = set(
            data.select(pl.col(self.sector_col))
            .unique()
            .collect()
            .get_column(self.sector_col)
            .drop_nulls()
            .to_list()
        )

        expected = set(self.categories)
        unknown = observed - expected

        if unknown:
            raise ValueError(
                f"Factor {self.name!r} encountered unknown categories: {sorted(unknown)}."
            )

    def _exposure_name(self, category: str) -> str:
        normalized = (
            category.strip().lower().replace("&", "and").replace(" ", "_").replace("-", "_")
        )

        return f"{self.prefix}_{normalized}"

    @staticmethod
    def _validate_categories(
        categories: Sequence[str],
    ) -> tuple[str, ...]:
        values = tuple(categories)

        if any(not isinstance(category, str) for category in values):
            raise TypeError("All sector categories must be strings.")

        if any(not category.strip() for category in values):
            raise ValueError("Sector categories cannot be empty strings.")

        if len(values) != len(set(values)):
            raise ValueError("Sector categories must be unique.")

        return tuple(sorted(values))

    @override
    def __repr__(self) -> str:
        categories = self._categories if self._categories is not None else "<unfitted>"

        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"sector_col={self.sector_col!r}, "
            f"categories={categories!r}"
            ")"
        )
