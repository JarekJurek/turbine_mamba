from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from turbine_mamba import WindTurbineModel, get_dataloaders, test_model, train_one_epoch, validate_one_epoch


def train_model(epochs, model, train_loader, val_loader, optimizer, criterion, device):
    # Training and Validation
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

    return model


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

    train_loader, val_loader, test_loader = get_dataloaders(files_dir, files, model_name, batch_size, sample_fraction=0.01)

    train_model(epochs, model, train_loader, val_loader, optimizer, criterion, device)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Testing
    predictions, ground_truth = test_model(model, test_loader, device)
    print(f"Test Predictions: {predictions.shape}, Ground Truth: {ground_truth.shape}")


if __name__ == "__main__":
    main()
