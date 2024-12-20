"""Module for visualizing predictions, ground truths, and their residuals in a combined plot."""

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


def plot_combined(predictions, ground_truth, residuals, label, save_path, num_points=None):
    """
    Plot predictions vs. ground truths and residuals in one figure with two subplots.

    Args:
        predictions (np.ndarray): Predictions to plot (shape: [num_samples,]).
        ground_truth (np.ndarray): Ground truths to plot (shape: [num_samples,]).
        residuals (np.ndarray): Residuals to plot (shape: [num_samples,]).
        label (str): Label for the momentum (e.g., "Mz1").
        save_path (Path): Path to save the plot.
        num_points (int, optional): Number of points to include in the plot.
    """
    if num_points:
        predictions = predictions[:num_points]
        ground_truth = ground_truth[:num_points]
        residuals = residuals[:num_points]

    # Create the figure and subplots
    plt.figure(figsize=(12, 8))

    # Top subplot: Predictions vs Ground Truths
    plt.subplot(2, 1, 1)
    plt.plot(ground_truth, label="Ground Truth", linestyle="-", alpha=0.8)
    plt.plot(predictions, label="Predictions", linestyle="--", alpha=0.8)
    plt.title(f"Predictions vs Ground Truth ({label})")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend(loc="upper right")  # Set legend to the top-right corner
    plt.grid()

    # Bottom subplot: Residuals
    plt.subplot(2, 1, 2)
    plt.plot(residuals, label=f"Residual ({label})", linestyle="-", alpha=0.8)
    plt.title(f"Residuals ({label})")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual Value")
    plt.legend(loc="upper right")  # Set legend to the top-right corner
    plt.grid()

    # Save the figure
    ensure_directory_exists(save_path)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Combined plot saved to {save_path}")


def plot_scatter_predictions_vs_ground_truth(predictions, ground_truth, label, save_path, alpha=0.6):
    """
    Plot predictions vs ground truths as scatter points, with an optional ideal diagonal line.

    Args:
        predictions (np.ndarray): Predictions to plot (shape: [num_samples,]).
        ground_truth (np.ndarray): Ground truths to plot (shape: [num_samples,]).
        label (str): Label for the momentum (e.g., "Mz1").
        save_path (Path): Path to save the plot.
        alpha (float): Transparency for scatter points (default: 0.6).
    """
    # Create the plot
    plt.figure(figsize=(8, 8))

    # Scatter plot of predictions vs ground truths
    plt.scatter(ground_truth, predictions, alpha=alpha, edgecolor='k', label="Data Points")

    # Plot ideal line (diagonal)
    min_val = min(min(ground_truth), min(predictions))
    max_val = max(max(ground_truth), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Line")

    # Add labels, title, legend, and grid
    plt.title(f"Scatter Plot: Predictions vs Ground Truth ({label})")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend(loc="upper right")  # Legend in the top-right corner
    plt.grid()

    # Save the figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def main():
    # Settings for the script
    run_dir = "run_3"
    labels = ["Mz1", "Mz2", "Mz3"]
    num_points = 100

    # Paths setup
    project_dir = Path(__file__).parent
    data_path = project_dir / "predictions_ground_truth.npz"
    output_dir = project_dir

    # Load predictions and ground truths
    predictions, ground_truth = load_data(data_path)

    # Ensure residuals directory exists
    ensure_directory_exists(output_dir)

    # Generate plots for each momentum
    for i, label in enumerate(labels):
        print(f"Generating plots for {label}...")

        # Select data for the current momentum
        pred = predictions[:, i]
        gt = ground_truth[:, i]
        residuals = abs(gt - pred)

        # Save path for the combined plot
        save_path = output_dir / f"{label}_combined_plot.png"

        # Generate the combined plot
        plot_combined(pred, gt, residuals, label, save_path, num_points)
        # plot_scatter_predictions_vs_ground_truth(pred, gt, label, save_path)


if __name__ == "__main__":
    main()
