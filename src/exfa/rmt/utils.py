import numpy as np
from scipy.linalg import eigh


def top_sample_eigenvalues(R: np.ndarray, k: int) -> np.ndarray:
    n, T = R.shape

    if not 1 <= k <= n:
        raise ValueError("Expected 1 <= k <= n")

    G = (R @ R.T if n <= T else R.T @ R) / T
    q = G.shape[0]
    count = min(k, q)

    eigenvalues = eigh(
        G,
        subset_by_index=(q - count, q - 1),
        eigvals_only=True,
        check_finite=False,
    )[::-1]

    return np.pad(
        np.maximum(eigenvalues, 0.0),
        (0, k - count),
    )


def sample_eigenvalues(R: np.ndarray) -> np.ndarray:
    """
    Compute the complete spectrum of the sample covariance matrix.

    R has shape (n, T), with assets along rows and observations
    along columns. Returns n eigenvalues in descending order.

    When n > T, use the smaller Gram matrix and append the
    n - T zero eigenvalues of the sample covariance.
    """
    n, T = R.shape

    gram = (R @ R.T if n <= T else R.T @ R) / T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    if n > T:
        eigenvalues = np.pad(eigenvalues, (0, n - T))

    return eigenvalues
