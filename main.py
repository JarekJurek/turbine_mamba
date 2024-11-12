from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from turbine_mamba.metrics import compute_r2  # Implemented compute_r2 in metrics.py

from turbine_mamba import WindTurbineModel, get_dataloaders, test_model, train_one_epoch, validate_one_epoch
from turbine_mamba.plots import plot_loss_and_metrics, plot_predictions


def train_model(epochs, model, train_loader, val_loader, optimizer, criterion, device):
    train_losses = []
    val_losses = []
    metric_values = []

    # Training and Validation
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Compute a metric (e.g., R² score for validation predictions)
        predictions, ground_truth = test_model(model, val_loader, device)
        r2_score = compute_r2(predictions, ground_truth)
        metric_values.append(r2_score)
        print(f"Validation R² Score: {r2_score:.4f}")

    return predictions, ground_truth, train_losses, val_losses, metric_values


def main():
    project_dir = Path(__file__).parent
    model_save_path = project_dir / "models" / "wind_turbine_model.pth"
    files_dir = project_dir / "data" / "raw"  # Replace with your directory path
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    model_name = "state-spaces/mamba-130m-hf"
    batch_size = 32
    epochs = 2
    learning_rate = 1e-3
    model = WindTurbineModel(model_name).to(device)
    criterion = MSELoss()
    optimizer = Adam(model.fc.parameters(), lr=learning_rate)  # Train only FC layers

    train_loader, val_loader, test_loader = get_dataloaders(files_dir, files, model_name, batch_size,
                                                            sample_fraction=0.01)

    predictions, ground_truth, train_losses, val_losses, metric_values = train_model(
        epochs, model, train_loader, val_loader, optimizer, criterion, device
    )

    # Plot loss and metrics
    plot_loss_and_metrics(train_losses, val_losses, metric_values, metric_name="R² Score", save_path="reports/figures/loss_and_metrics.png")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Testing
    predictions, ground_truth = test_model(model, test_loader, device)
    print(f"Test Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")
    plot_predictions(predictions, ground_truth, labels=["Mz1", "Mz2", "Mz3"], save_path="reports/figures/predictions_vs_ground_truth.png")


if __name__ == "__main__":
    main()
