"""Evaluation helpers adhering to the :math:`y` (performance) / :math:`x` (condition) notation."""

from sklearn.metrics import r2_score, mean_absolute_error


def element_wise_r2(pred, target):
    """Per-dimension :math:`R^2` scores for predicted conditions :math:`x`."""
    return [r2_score(target[:, i], pred[:, i]) for i in range(0, pred.shape[1])]


def element_wise_mae(pred, target):
    """Per-dimension MAE for condition predictions :math:`x`."""
    return [mean_absolute_error(target[:, i], pred[:, i]) for i in range(0, pred.shape[1])]
