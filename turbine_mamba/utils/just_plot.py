"""Module for visualizing predictions, ground truths, and their residuals. Useful for validating models."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from turbine_mamba.plots import plot_predictions  # Assumes an existing `plot_predictions` function


def ensure_directory_exists(path):
    """
    Ensure that the directory of the given path exists.

    Args:
        path (Path): Path whose parent directory should exist.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


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


def plot_residuals(residuals, labels, save_path, num_points=None):
    """
    Plot residuals between predictions and ground truths.

    Args:
        residuals (np.ndarray): Residuals to plot (shape: [num_samples, num_labels]).
        labels (list): List of labels for the residuals.
        save_path (Path): Path to save the plot.
        num_points (int, optional): Number of points to plot. If None, plot all.
    """
    plt.figure(figsize=(10, 6))

    if num_points:
        residuals = residuals[:num_points]

    for i, label in enumerate(labels):
        plt.plot(residuals[:, i], label=f"Residual {label}")

    plt.title("Residuals: Ground Truth vs Predictions")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid()
    ensure_directory_exists(save_path)

    plt.show()


def compute_and_plot(data_path, plot_path, labels, plot_type="comparison", num_points=100):
    """
    Load data and generate either predictions vs. ground truths or residuals plots.

    Args:
        data_path (Path): Path to the .npz file containing predictions and ground truth.
        plot_path (Path): Path to save the plot.
        labels (list): List of labels for the data.
        plot_type (str): Type of plot ('comparison' or 'residuals').
        num_points (int): Number of points to include in the plot.
    """
    predictions, ground_truth = load_data(data_path)

    if plot_type == "comparison":
        print("Plotting predictions vs ground truths...")
        ensure_directory_exists(plot_path)
        plot_predictions(
            predictions,
            ground_truth,
            labels=labels,
            save_path=plot_path,
            num_points=num_points
        )
        print(f"Comparison plot saved to {plot_path}")
    elif plot_type == "residuals":
        print("Plotting residuals...")
        residuals = ground_truth - predictions
        print(f"Residuals computed with shape: {residuals.shape}")
        plot_residuals(
            residuals,
            labels=labels,
            save_path=plot_path,
            num_points=num_points
        )
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'comparison' or 'residuals'.")


def main():
    # Settings for the script
    run_dir = "run_3"
    labels = ["Mz1", "Mz2", "Mz3"]
    num_points = 100

    # Paths setup
    project_dir = Path(__file__).parent.parent.parent
    data_path = project_dir / "predictions_ground_truth.npz"

    comparison_plot_path = project_dir / "reports" / run_dir / "comparison_plot.png"
    compute_and_plot(
        data_path=data_path,
        plot_path=comparison_plot_path,
        labels=labels,
        plot_type="comparison",
        num_points=num_points
    )

    # Plot residuals
    residuals_plot_path = project_dir / "residuals_plot.png"
    compute_and_plot(
        data_path=data_path,
        plot_path=residuals_plot_path,
        labels=labels,
        plot_type="residuals",
        num_points=num_points
    )


if __name__ == "__main__":
    main()
