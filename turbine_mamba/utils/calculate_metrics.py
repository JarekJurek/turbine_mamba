from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def ensure_directory_exists(path):
    """
    Ensure that the directory of the given path exists.

    Args:
        path (Path): Path whose parent directory should exist.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_data(file_path):
    """
    Load predictions and ground truths from a saved .npz file.

    Args:
        file_path (Path): Path to the .npz file.

    Returns:
        tuple: (predictions, ground_truth) arrays.
    """
    data = np.load(file_path)
    predictions = data["predictions"]
    ground_truth = data["ground_truth"]
    print(f"Loaded predictions and ground truth from {file_path}")
    print(f"Predictions shape: {predictions.shape}, Ground Truth shape: {ground_truth.shape}")
    return predictions, ground_truth


def calculate_metrics(predictions, ground_truth):
    """
    Calculate MSE, MAE, and R-squared for the given predictions and ground truth.

    Args:
        predictions (np.ndarray): Predictions (shape: [num_samples,]).
        ground_truth (np.ndarray): Ground truth values (shape: [num_samples,]).

    Returns:
        dict: Dictionary containing MSE, MAE, and R-squared.
    """
    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)

    return {
        "MSE": mse,
        "MAE": mae,
        "R-squared": r2
    }


def main():
    run_dir = "run_3"
    labels = ["Mz1", "Mz2", "Mz3"]
    num_points = 100

    # Paths setup
    project_dir = Path(__file__).parent.parent.parent
    data_path = project_dir / "predictions_ground_truth.npz"
    output_dir = project_dir

    # Load predictions and ground truths
    predictions, ground_truth = load_data(data_path)

    # Ensure residuals directory exists
    ensure_directory_exists(output_dir)

    # Generate plots and calculate metrics for each momentum
    for i, label in enumerate(labels):
        print(f"Calculating metrics for {label}...")

        # Select data for the current momentum
        pred = predictions[:, i]
        gt = ground_truth[:, i]

        # Calculate metrics
        metrics = calculate_metrics(pred, gt)
        print(f"Metrics for {label}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
