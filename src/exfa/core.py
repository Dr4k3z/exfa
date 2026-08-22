from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self, override

import polars as pl

from exfa.factors.base import Factor, FactorExposure, FactorKind


class FactorModel:
    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        date_col: str = "date",
        asset_col: str = "symbol",
        returns_col: str = "returns",
    ) -> None:
        self._data = data.lazy()

        self.date_col = date_col
        self.asset_col = asset_col
        self.returns_col = returns_col

        self._factors: dict[str, Factor] = {}
        self._is_fitted = False

    @property
    def data(self) -> pl.LazyFrame:
        return self._data

    @property
    def factors(self) -> tuple[Factor, ...]:
        return tuple(self._factors.values())

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(self._factors.keys())

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __contains__(self, name: str) -> bool:
        return name in self._factors

    def __len__(self) -> int:
        return len(self._factors)

    def __iter__(self) -> Iterator[Factor]:
        return iter(self._factors.values())

    @override
    def __repr__(self) -> str:
        factors = ", ".join(self._factors)
        return (
            f"{type(self).__name__}("
            f"n_factors={len(self)}, "
            f"factors=[{factors}], "
            f"is_fitted={self.is_fitted}"
            ")"
        )

    def add_factor(
        self,
        factor: Factor,
        *,
        replace: bool = False,
    ) -> Self:
        """
        Register a factor with the model.

        Parameters
        ----------
        factor:
            Factor instance to register.

        replace:
            Whether an existing factor with the same logical name should be
            replaced.

        Returns
        -------
        Self
            The model itself, allowing method chaining.
        """
        if factor.name in self._factors and not replace:
            raise ValueError(f"Factor {factor.name!r} is already registered.")

        self._factors[factor.name] = factor
        self._is_fitted = False

        return self

    def remove_factor(
        self,
        name: str,
        *,
        missing_ok: bool = False,
    ) -> Self:
        """
        Remove a registered factor by name.
        """
        if name not in self._factors:
            if missing_ok:
                return self

            raise KeyError(f"Unknown factor {name!r}.")

        del self._factors[name]
        self._is_fitted = False

        return self

    def get_factor(self, name: str) -> Factor:
        """
        Return a registered factor by logical name.
        """
        try:
            return self._factors[name]
        except KeyError:
            raise KeyError(f"Unknown factor {name!r}.") from None

    @property
    def intercept_factors(self) -> tuple[Factor, ...]:
        return tuple(
            factor for factor in self._factors.values() if factor.kind is FactorKind.INTERCEPT
        )

    def fit(self) -> Self:
        """
        Fit all stateful factors.

        Deterministic factors require no fitting and are simply validated.
        """
        if not self._factors:
            raise RuntimeError("Cannot fit a factor model with no registered factors.")

        self.validate()

        for factor in self._factors.values():
            if factor.requires_fit:
                factor.fit(self._data)

        self._is_fitted = True

        return self

    def validate(self) -> None:
        """
        Validate the model and all registered factors.
        """
        self._validate_core_schema()
        self._validate_factor_names()

        for factor in self._factors.values():
            factor.validate_data(self._data)

    def exposures(self) -> pl.LazyFrame:
        """
        Construct the complete asset-by-factor exposure panel.

        Returns
        -------
        pl.LazyFrame
            LazyFrame containing

                date | asset | exposure_1 | ... | exposure_n

        Notes
        -----
        Factors requiring fitting must be fitted before their exposures can
        be constructed.
        """
        if not self._factors:
            raise RuntimeError("Cannot construct exposures without registered factors.")

        self.validate()

        unfitted = [
            factor.name
            for factor in self._factors.values()
            if factor.requires_fit and not self._is_fitted
        ]

        if unfitted:
            raise RuntimeError(
                "The following factors require fitting before exposures "
                f"can be computed: {unfitted}."
            )

        exposures = self._compute_exposures()

        return self._data.select(
            self.date_col,
            self.asset_col,
            *(exposure.expression for exposure in exposures),
        )

    def exposure_names(self) -> tuple[str, ...]:
        """
        Return all regression-column names generated by registered factors.
        """
        exposures = self._compute_exposures()

        return tuple(exposure.name for exposure in exposures)

    def factor_exposure_map(
        self,
    ) -> dict[str, tuple[str, ...]]:
        """
        Map each logical factor to its generated exposure columns.

        Examples
        --------
        {
            "momentum": ("momentum",),
            "sector": (
                "sector_energy",
                "sector_financials",
                "sector_technology",
            ),
        }
        """
        result: dict[str, tuple[str, ...]] = {}

        for factor in self._factors.values():
            exposures = factor.compute(self._data)

            self._validate_factor_exposures(
                factor=factor,
                exposures=exposures,
            )

            result[factor.name] = tuple(exposure.name for exposure in exposures)

        return result

    def _compute_exposures(
        self,
    ) -> tuple[FactorExposure, ...]:
        exposures: list[FactorExposure] = []

        for factor in self._factors.values():
            factor_exposures = factor.compute(self._data)

            self._validate_factor_exposures(
                factor=factor,
                exposures=factor_exposures,
            )

            exposures.extend(factor_exposures)

        self._validate_global_exposure_names(exposures)

        return tuple(exposures)

    def _validate_core_schema(self) -> None:
        schema = self._data.collect_schema()

        required = (
            self.date_col,
            self.asset_col,
        )

        missing = [column for column in required if column not in schema]

        if missing:
            raise ValueError(f"FactorModel requires missing columns: {missing}.")

    def _validate_factor_names(self) -> None:
        if len(self._factors) != len(set(self._factors)):
            raise RuntimeError("Factor names must be unique.")

    @staticmethod
    def _validate_factor_exposures(
        *,
        factor: Factor,
        exposures: Sequence[FactorExposure],
    ) -> None:
        if not exposures:
            raise ValueError(f"Factor {factor.name!r} produced no exposures.")

        names = [exposure.name for exposure in exposures]

        if len(names) != len(set(names)):
            raise ValueError(f"Factor {factor.name!r} produced duplicate exposure names: {names}.")

    @staticmethod
    def _validate_global_exposure_names(
        exposures: Sequence[FactorExposure],
    ) -> None:
        names = [exposure.name for exposure in exposures]

        duplicates = {name for name in names if names.count(name) > 1}

        if duplicates:
            raise ValueError(
                f"Multiple factors produced the same exposure names: {sorted(duplicates)}."
            )
