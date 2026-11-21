from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class Distribution(ABC):
    """
    Distribution with standardized base innovations.

    By default:
        E[Z_t] = 0
        Cov(Z_t) = I_n

    An affine transformation produces a new instance of the
    same distribution, without modifying the original.
    """

    _root: np.ndarray | None = field(default=None, repr=False, compare=False)
    _mean: np.ndarray | None = field(default=None, repr=False, compare=False)

    @staticmethod
    def _validate_covariance(
        covariance: np.ndarray,
    ) -> np.ndarray:
        covariance = np.array(covariance, dtype=float, copy=True)

        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or covariance.size == 0
        ):
            raise ValueError("Covariance must be a nonempty square matrix")

        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance must be finite and symmetric")

        return (covariance + covariance.T) / 2

    @abstractmethod
    def _sample_standardized(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def sample(
        self,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        n, T = shape

        if n < 1 or T < 1:
            raise ValueError("Sample dimensions must be positive")

        Z = self._sample_standardized(shape, rng)

        if self._root is None:
            return Z

        if n != self._root.shape[0]:
            raise ValueError("Sample dimension does not match covariance")

        assert self._mean is not None
        return self._root @ Z + self._mean[:, None]

    def affine_transformation(
        self,
        covariance: np.ndarray,
        mean: np.ndarray | None = None,
    ) -> Self:
        """
        Return the distribution of A @ X + mean,
        where A is the symmetric square root of covariance.

        For a standardized distribution, the resulting
        covariance is exactly covariance.
        """

        covariance = self._validate_covariance(covariance)

        eigenvalues, U = np.linalg.eigh(covariance)

        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))

        if eigenvalues[0] < -tolerance:
            raise ValueError("Covariance must be positive semidefinite")

        A = (U * np.sqrt(np.maximum(eigenvalues, 0))) @ U.T

        n = A.shape[0]
        mu = np.zeros(n) if mean is None else np.array(mean, dtype=float, copy=True)

        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError("Mean must be a finite vector of length n")

        if self._root is None:
            root = A
            location = mu
        else:
            if self._root.shape != A.shape:
                raise ValueError("Incompatible transformation dimensions")

            assert self._mean is not None
            root = A @ self._root
            location = A @ self._mean + mu

        return replace(self, _root=root, _mean=location)

    def moments(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n < 1:
            raise ValueError("Dimension must be positive")

        if self._root is None:
            return np.zeros(n), np.eye(n)

        if self._root.shape != (n, n):
            raise ValueError("Incompatible distribution dimension")

        assert self._mean is not None
        return self._mean.copy(), self._root @ self._root.T


@dataclass(frozen=True, slots=True)
class GaussianDistribution(Distribution):
    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        return rng.standard_normal(shape)


@dataclass(frozen=True, slots=True)
class TStudentDistribution(Distribution):
    dof: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.dof) or self.dof <= 2:
            raise ValueError("dof must be finite and greater than 2")

    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        Z = rng.standard_t(self.dof, size=shape)
        return Z * np.sqrt((self.dof - 2) / self.dof)


@dataclass(frozen=True, slots=True)
class LaplaceDistribution(Distribution):
    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        return rng.laplace(0, 1 / np.sqrt(2), size=shape)


@dataclass(frozen=True, slots=True)
class UniformDistribution(Distribution):
    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=shape)


@dataclass(frozen=True, slots=True)
class RademacherDistribution(Distribution):
    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        return 2.0 * rng.integers(0, 2, size=shape) - 1.0


@dataclass(frozen=True, slots=True)
class GammaDistribution(Distribution):
    concentration: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.concentration) or self.concentration <= 0:
            raise ValueError("concentration must be positive")

    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        k = self.concentration
        Z = rng.gamma(shape=k, scale=1.0, size=shape)
        return (Z - k) / np.sqrt(k)


@dataclass(frozen=True, slots=True)
class GaussianMixtureDistribution(Distribution):
    tail_probability: float = 0.05
    tail_scale: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.tail_probability) or not (0 < self.tail_probability < 1):
            raise ValueError("tail_probability must be in (0, 1)")

        if not np.isfinite(self.tail_scale) or self.tail_scale <= 1:
            raise ValueError("tail_scale must be greater than 1")

    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        large = rng.random(shape) < self.tail_probability
        scales = np.where(large, self.tail_scale, 1.0)

        variance = 1 - self.tail_probability + self.tail_probability * self.tail_scale**2

        return rng.standard_normal(shape) * scales / np.sqrt(variance)


@dataclass(frozen=True, slots=True)
class EllipticalStudentTDistribution(Distribution):
    dof: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.dof) or self.dof <= 2:
            raise ValueError("dof must be finite and greater than 2")

    def _sample_standardized(
        self, shape: tuple[int, int], rng: np.random.Generator
    ) -> np.ndarray:
        _, T = shape

        scales = np.sqrt((self.dof - 2) / rng.chisquare(self.dof, size=T))

        return rng.standard_normal(shape) * scales[None, :]
