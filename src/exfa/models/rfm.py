from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self, overload

import numpy as np
from numpy.typing import ArrayLike


class fitted_property[S, T]:
    def __init__(self, getter: Callable[[S], T | None]) -> None:
        self._getter = getter
        self._name = getattr(getter, "__name__", "fitted property")
        self.__doc__ = getter.__doc__

    @property
    def __isabstractmethod__(self) -> bool:
        return bool(getattr(self._getter, "__isabstractmethod__", False))

    def __set_name__(self, owner: type[S], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type[S] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: S, owner: type[S] | None = None) -> T: ...

    def __get__(self, instance: S | None, owner: type[S] | None = None) -> T | Self:
        if instance is None:
            return self
        value = self._getter(instance)
        if value is None:
            raise RuntimeError(f"{self._name} is unavailable: model has not been fitted")
        return value

    def __set__(self, instance: S, value: object) -> None:
        raise AttributeError(f"{self._name} is read-only")


class RiskFactorModel(ABC):
    """
    Shared fitted API: returns/residuals (assets, times), loadings
    (assets, factors), factor_returns (factors, times), covariance (assets, assets).

    residual_variance is scalar for isotropic noise or an asset-length vector.
    Dimensions and fitted properties are unavailable before a successful fit.
    """

    def __init__(self) -> None:
        self._returns: np.ndarray | None = None

    @fitted_property
    def returns(self) -> np.ndarray | None:
        return self._returns

    @fitted_property
    @abstractmethod
    def factor_returns(self) -> np.ndarray: ...

    @fitted_property
    def residuals(self) -> np.ndarray:
        return self.returns - self.loadings @ self.factor_returns

    @fitted_property
    def covariance(self) -> np.ndarray:
        factor_covariance = np.atleast_2d(np.cov(self.factor_returns, ddof=1))
        return self.loadings @ factor_covariance @ self.loadings.T + np.diag(
            np.broadcast_to(self.residual_variance, (self.num_assets,))
        )

    @fitted_property
    @abstractmethod
    def loadings(self) -> np.ndarray: ...

    @fitted_property
    def residual_variance(self) -> float | np.ndarray:
        return np.var(self.residuals, axis=1, ddof=1)

    @fitted_property
    def num_assets(self) -> int:
        return self.returns.shape[0]

    @fitted_property
    def num_times(self) -> int:
        return self.returns.shape[1]

    @fitted_property
    def num_factors(self) -> int:
        return self.loadings.shape[1]

    @staticmethod
    def _validate_returns(returns: ArrayLike) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2 or returns.shape[0] < 1 or returns.shape[1] < 2:
            raise ValueError(
                "returns must have shape (assets, times) with at least one asset and two times"
            )
        if not np.isfinite(returns).all():
            raise ValueError("returns must contain only finite values")
        return returns

    @abstractmethod
    def _fit(self, returns: np.ndarray, *args, **kwargs) -> Self:
        """
        Estimate using owned, normalized returns and commit state only on success.

        Implementations derive candidate dimensions from returns.shape and publish
        the returns used in their decomposition with the remaining fitted results.
        """

        ...

    def fit(self, returns: ArrayLike, *args, **kwargs) -> Self:
        returns = RiskFactorModel._validate_returns(returns).copy()
        return self._fit(returns, *args, **kwargs)
