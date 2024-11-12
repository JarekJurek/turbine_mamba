import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_metrics(train_losses, val_losses, metric_values, metric_name="R² Score"):
    """
    Plot training and validation loss, along with a selected metric.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        metric_values (list): List of metric values for each epoch.
        metric_name (str): Name of the metric (e.g., "R² Score").
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
    plt.show()

def plot_predictions(predictions, ground_truth, labels=["Mz1", "Mz2", "Mz3"]):
    """
    Plot predicted vs. ground truth values for each output.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted values (batch_size, num_outputs).
        ground_truth (torch.Tensor or np.ndarray): Ground truth values (batch_size, num_outputs).
        labels (list): List of labels for the outputs.
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    num_outputs = predictions.shape[1]
    plt.figure(figsize=(15, 5))

    for i in range(num_outputs):
        plt.subplot(1, num_outputs, i + 1)
        plt.scatter(ground_truth[:, i], predictions[:, i], alpha=0.6)
        plt.plot([ground_truth[:, i].min(), ground_truth[:, i].max()],
                 [ground_truth[:, i].min(), ground_truth[:, i].max()],
                 color="red", linestyle="--", label="Perfect Prediction")
        plt.xlabel(f"Ground Truth {labels[i]}")
        plt.ylabel(f"Predicted {labels[i]}")
        plt.title(f"{labels[i]}: Predicted vs. Ground Truth")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()
