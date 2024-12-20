import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer



class WindTurbineDataset(Dataset):
    """
    Custom Dataset for Wind Turbine Data
    """

    def __init__(self, data, input_mean, input_std, target_mean=None, target_std=None):
        """
        Args:
            data (pd.DataFrame): Data containing input features and targets.
            input_mean (pd.Series): Mean values of input features for normalization.
            input_std (pd.Series): Standard deviation of input features for normalization.
            target_mean (pd.Series): Mean values of targets for normalization (optional).
            target_std (pd.Series): Standard deviation of targets for normalization (optional).
        """
        self.inputs = (data[['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']] - input_mean) / input_std
        self.targets = data[['Mz1', 'Mz2', 'Mz3']]

        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / target_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs.iloc[idx].values, dtype=torch.float32)
        targets = torch.tensor(self.targets.iloc[idx].values, dtype=torch.float32)
        return inputs, targets

    
def get_dataloaders(file_dir, file_names, batch_size=32, slice_size=100, step=3, max_slices=None):
    file_paths = [os.path.join(file_dir, file) for file in file_names]
    data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        num_slices = len(df) // slice_size
        if max_slices:
            num_slices = min(num_slices, max_slices)
        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = start_idx + slice_size
            data.append(df.iloc[start_idx:end_idx].iloc[::step])

    full_data = pd.concat(data, ignore_index=True).dropna()

    # Split dataset
    dataset_size = len(full_data)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = torch.randperm(dataset_size)
    train_data = full_data.iloc[indices[:train_size]]
    val_data = full_data.iloc[indices[train_size:train_size + val_size]]
    test_data = full_data.iloc[indices[train_size + val_size:]]

    # Compute normalization stats
    input_features = ['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']
    target_features = ['Mz1', 'Mz2', 'Mz3']

    input_mean = train_data[input_features].mean()
    input_std = train_data[input_features].std()
    target_mean = train_data[target_features].mean()
    target_std = train_data[target_features].std()

    train_dataset = WindTurbineDataset(train_data, input_mean, input_std, target_mean, target_std)
    val_dataset = WindTurbineDataset(val_data, input_mean, input_std, target_mean, target_std)
    test_dataset = WindTurbineDataset(test_data, input_mean, input_std, target_mean, target_std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    normalization_stats = {
        "input_mean": input_mean, "input_std": input_std,
        "target_mean": target_mean, "target_std": target_std
    }

    return train_loader, val_loader, test_loader, normalization_stats




def test_dataloader():
    project_dir = Path(__file__).parent.parent

    files_dir = project_dir / "data" / "raw"  
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]
    batch_size = 32
    sample_fraction = 0.5  # Use 50% of the dataset for quicker testing

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(files_dir, files, batch_size, sample_fraction)

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}")
        print(f"Inputs: {inputs.shape}")
        print(f"Input: {inputs[0]}")
        print(f"Targets: {targets.shape}")
        break


if __name__ == "__main__":
    test_dataloader()
