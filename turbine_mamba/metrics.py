import numpy as np

def compute_r2(predictions, ground_truth):
    """
    Compute the R² (coefficient of determination) score.

    Args:
        predictions (np.ndarray): Predicted values.
        ground_truth (np.ndarray): Ground truth values.

    Returns:
        float: R² score.
    """
    ss_total = np.sum((ground_truth - np.mean(ground_truth, axis=0)) ** 2)
    ss_residual = np.sum((ground_truth - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2
