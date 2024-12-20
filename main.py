
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from turbine_mamba import get_dataloaders, test_model, train_one_epoch, validate_one_epoch
from turbine_mamba.metrics import compute_r2
from turbine_mamba.plots import plot_loss_and_metrics, plot_predictions

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return predictions, ground_truth


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu().numpy())
            ground_truth.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    return total_loss / len(val_loader), predictions, ground_truth

def denormalize(predictions, target_mean, target_std):
    return predictions * target_std.values + target_mean.values

class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

def train_model(epochs, model, train_loader, val_loader, optimizer, criterion, device):
    train_losses = []
    val_losses = []
    metric_values = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, predictions, ground_truth = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Compute R^2 
        r2_score = compute_r2(predictions, ground_truth)
        metric_values.append(r2_score)

        print(f"TRAIN Loss: {train_loss:.4f}, VAL Loss: {val_loss:.4f}, R²: {r2_score:.4f}")

    return predictions, ground_truth, train_losses, val_losses, metric_values

def main():
    project_dir = Path(__file__).parent
    model_save_path = project_dir / "models" / "ffnn_wind_turbine_model.pth"
    predictions_save_path = project_dir / "models" / "predictions_ground_truth.npz"
    files_dir = project_dir / "data" / "raw"
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Features and Targets
    input_features = ['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']
    target_features = ['Mz1', 'Mz2', 'Mz3']

    # Hyperparameters
    input_size = len(input_features)
    hidden_size = 64
    output_size = len(target_features)
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    slice_size = 3000
    step = 4
    max_slices = 4

    model = FFNNModel(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader, test_loader, normalization_stats = get_dataloaders(files_dir, files, batch_size, slice_size, step, max_slices)

    predictions, ground_truth, train_losses, val_losses, metric_values = train_model(
        epochs, model, train_loader, val_loader, optimizer, criterion, device
    )

    plot_loss_and_metrics(train_losses, val_losses, metric_values, metric_name="R² Score",
                          save_path="reports/figures/loss_and_metrics.png")

    # Testing
    predictions, ground_truth = test_model(model, test_loader, device)

    target_mean, target_std = normalization_stats["target_mean"], normalization_stats["target_std"]
    predictions = denormalize(predictions, target_mean, target_std)
    ground_truth = denormalize(ground_truth, target_mean, target_std)

    print(f"Test Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")
    plot_predictions(predictions, ground_truth, labels=["Mz1", "Mz2", "Mz3"],
                     save_path="reports/figures/predictions_vs_ground_truth.png")

    if not os.path.exists(model_save_path.parent):
        os.makedirs(model_save_path.parent)

    try:
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
