from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_metrics(train_losses, val_losses, metric_values, metric_name="R² Score", save_path=None):
    """
    Plot training and validation loss, along with a selected metric, and optionally save the plot.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        metric_values (list): List of metric values for each epoch.
        metric_name (str): Name of the metric (e.g., "R² Score").
        save_path (str): Path to save the plot. If None, the plot will be displayed.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o')
    plt.plot(epochs, metric_values, label=metric_name, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training/Validation Loss and Metric")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_predictions(predictions, ground_truth, labels=["Mz1", "Mz2", "Mz3"], save_path: Path=None, num_points=100):
    """
    Plot ground truth and predictions as separate line plots for each output.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted values (batch_size, num_outputs).
        ground_truth (torch.Tensor or np.ndarray): Ground truth values (batch_size, num_outputs).
        labels (list): List of labels for the outputs.
        save_path (str): Path to save the plot. If None, the plots will be displayed.
        num_points (int): Number of data points to plot (X-axis size).
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Limit to specified number of data points
    if num_points is not None:
        print(f'Plotting for {num_points} points.')
        predictions = predictions[:num_points, :]
        ground_truth = ground_truth[:num_points, :]

    num_outputs = predictions.shape[1]
    plt.figure(figsize=(15, 5 * num_outputs))  # Adjust figure size to fit multiple plots

    for i in range(num_outputs):
        plt.subplot(num_outputs, 1, i + 1)
        plt.plot(ground_truth[:, i], label=f"{labels[i]} Ground Truth", linestyle="-", color="blue")
        plt.plot(predictions[:, i], label=f"{labels[i]} Predictions", linestyle="--", color="red")
        plt.xlabel("Data Point Index")
        plt.ylabel("Momentum Values")
        plt.title(f"{labels[i]}: Predictions vs Ground Truth")
        plt.legend()
        plt.grid()

    plt.tight_layout()

    plt.show()

    plt.close()
