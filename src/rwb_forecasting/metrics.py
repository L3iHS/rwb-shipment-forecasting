import numpy as np


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Official global WAPE + abs(relative bias), with explicit input checks."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("Metric inputs must have the same nonempty shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")
    if (truth < 0).any() or (prediction < 0).any() or truth.sum() <= 0:
        raise ValueError("Volumes must be nonnegative and truth sum positive")
    wape = float(np.abs(prediction - truth).sum() / truth.sum())
    bias = float(prediction.sum() / truth.sum() - 1)
    return {"wape": wape, "signed_bias": bias, "abs_bias": abs(bias), "score": wape + abs(bias)}
