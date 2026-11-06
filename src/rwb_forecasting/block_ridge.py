"""Ridge with an unpenalized intercept for dense and sparse feature blocks.

The squared-loss objective and alpha convention match sklearn.linear_model.Ridge.
No n-by-(numeric+categorical) dense design matrix is constructed.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class BlockRidgeModel:
    """Coefficients use feature-by-output order; intercept is output-length."""
    numeric_coef: np.ndarray
    categorical_coef: np.ndarray
    intercept: np.ndarray
    alpha: float

    @property
    def coef_(self) -> np.ndarray:
        """Return sklearn's output-by-feature coefficient convention."""
        return np.vstack([self.numeric_coef, self.categorical_coef]).T

    @property
    def intercept_(self) -> np.ndarray:
        return self.intercept

    def predict(self, numeric: np.ndarray, categorical: sparse.spmatrix) -> np.ndarray:
        numeric, categorical = _blocks(numeric, categorical)
        if numeric.shape[1] != len(self.numeric_coef) or categorical.shape[1] != len(self.categorical_coef):
            raise ValueError("Prediction feature counts differ from training")
        return numeric @ self.numeric_coef + categorical @ self.categorical_coef + self.intercept


def _blocks(numeric: np.ndarray, categorical: sparse.spmatrix):
    numeric = np.asarray(numeric, dtype=np.float64)
    if numeric.ndim != 2 or not sparse.issparse(categorical) or categorical.ndim != 2:
        raise ValueError("Expected a 2D dense numeric block and a 2D sparse categorical block")
    categorical = sparse.csr_matrix(categorical, dtype=np.float64)
    if numeric.shape[0] != categorical.shape[0]:
        raise ValueError("Numeric and categorical row counts differ")
    if not np.isfinite(numeric).all() or not np.isfinite(categorical.data).all():
        raise ValueError("Features must be finite")
    return numeric, categorical


def fit_block_ridge(
    numeric: np.ndarray,
    categorical: sparse.spmatrix,
    targets: np.ndarray,
    alpha: float = 4.0,
) -> BlockRidgeModel:
    """Solve sum((y - X beta - intercept)**2) + alpha * sum(beta**2).

    Numerical standardization, if wanted, must already be fitted on training
    data and applied to ``numeric``. Sparse categories are not standardized.
    The intercept is removed analytically by centering the Gram matrix and RHS.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    numeric, categorical = _blocks(numeric, categorical)
    targets = np.asarray(targets, dtype=np.float64)
    n_rows, n_numeric = numeric.shape
    n_categorical = categorical.shape[1]
    n_features = n_numeric + n_categorical
    if n_rows == 0 or n_features == 0:
        raise ValueError("At least one training row and feature are required")
    if targets.ndim != 2 or targets.shape[0] != n_rows or targets.shape[1] == 0:
        raise ValueError("Targets must be a nonempty row-aligned 2D array")
    if not np.isfinite(targets).all():
        raise ValueError("Targets must be finite")

    feature_sums = np.concatenate([
        numeric.sum(axis=0), np.asarray(categorical.sum(axis=0)).ravel(),
    ])
    feature_means = feature_sums / n_rows
    target_means = targets.mean(axis=0)

    gram = np.empty((n_features, n_features), dtype=np.float64)
    gram[:n_numeric, :n_numeric] = numeric.T @ numeric
    cross = categorical.T @ numeric
    gram[n_numeric:, :n_numeric] = cross
    gram[:n_numeric, n_numeric:] = cross.T
    gram[n_numeric:, n_numeric:] = (categorical.T @ categorical).toarray()
    gram -= np.outer(feature_sums, feature_means)
    gram.flat[:: n_features + 1] += alpha

    rhs = np.vstack([numeric.T @ targets, categorical.T @ targets])
    rhs -= np.outer(feature_sums, target_means)
    coefficients = cho_solve(
        cho_factor(gram, lower=True, overwrite_a=True, check_finite=False),
        rhs, overwrite_b=True, check_finite=False,
    )
    intercept = target_means - feature_means @ coefficients
    return BlockRidgeModel(
        coefficients[:n_numeric], coefficients[n_numeric:], intercept, float(alpha),
    )
