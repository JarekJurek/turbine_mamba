import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from transformers import get_scheduler

from turbine_mamba import WindTurbineModel, get_dataloaders, test_model, train_one_epoch, validate_one_epoch
from turbine_mamba.metrics import compute_r2
from turbine_mamba.plots import plot_loss_and_metrics, plot_predictions


def train_model(epochs, model, train_loader, val_loader, optimizer, scheduler, criterion, device):
    train_losses = []
    val_losses = []
    metric_values = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, val_predictions, val_ground_truth = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Compute R² score
        r2_score = compute_r2(val_predictions, val_ground_truth)
        metric_values.append(r2_score)
        print(f"Validation R² Score: {r2_score:.4f}")

        # Step scheduler
        scheduler.step()

    return train_losses, val_losses, metric_values


def main():
    project_dir = Path(__file__).parent
    model_save_path = project_dir / "models" / "wind_turbine_model.pth"
    predictions_save_path = project_dir / "models" / "predictions_ground_truth.npz"
    files_dir = project_dir / "data" / "raw"
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    model_name = "state-spaces/mamba-130m-hf"
    batch_size = 32
    epochs = 30
    fc_learning_rate = 1e-3  # Higher learning rate for FC layers
    slice_size = 2000
    step = 4
    max_slices = 4

    # Initialize model
    model = WindTurbineModel(model_name).to(device)

    # Define optimizer with separate learning rates
    optimizer = AdamW([
        {"params": model.mamba.parameters(), "lr": 5e-5},  # Lower LR for pretrained layers
        {"params": model.regression_head.parameters(), "lr": fc_learning_rate}  # Higher LR for custom regression head
    ])

    # Scheduler: Linear schedule
    train_loader, val_loader, test_loader = get_dataloaders(
        files_dir, files, model_name, batch_size, slice_size, step, max_slices
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Loss function
    criterion = MSELoss()

    # Training loop
    train_losses, val_losses, metric_values = train_model(
        epochs, model, train_loader, val_loader, optimizer, scheduler, criterion, device
    )

    # Plot loss and metrics
    plot_loss_and_metrics(train_losses, val_losses, metric_values, metric_name="R² Score",
                          save_path="reports/figures/loss_and_metrics.png")

    # Testing
    predictions, ground_truth = test_model(model, test_loader, device)
    print(f"Test Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")
    plot_predictions(predictions, ground_truth, labels=["Mz1", "Mz2", "Mz3"],
                     save_path="reports/figures/predictions_vs_ground_truth.png")

    if not os.path.exists(model_save_path.parent):
        os.makedirs(model_save_path.parent)

    try:
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        if not predictions_save_path.parent.exists():
            predictions_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(predictions_save_path, predictions=predictions, ground_truth=ground_truth)
        print(f"Predictions and ground truth saved to {predictions_save_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")


if __name__ == "__main__":
    main()
