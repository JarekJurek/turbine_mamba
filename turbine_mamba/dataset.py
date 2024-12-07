# dataset.py
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class WindTurbineDataset(Dataset):
    """
    Custom Dataset for Wind Turbine Data
    """

    def __init__(self, df, tokenizer_name):
        """
        Args:
            df (pd.DataFrame): DataFrame containing already normalized input and target columns.
            tokenizer_name (str): Name of the Hugging Face tokenizer to use.
        """
        self.data = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = row[['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']].values.astype(str)
        target = row[['Mz1', 'Mz2', 'Mz3']].values.astype(float)

        input_text = " ".join(inputs)
        tokenized = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding='max_length',     # Explicitly pad all sequences to max_length
            truncation=True,
            max_length=64
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return input_ids, attention_mask, torch.tensor(target, dtype=torch.float32)


def get_dataloaders(file_dir: Path, file_names, tokenizer_name, batch_size=32, slice_size=100, step=3, max_slices=None):
    """
    Creates train, validation, and test dataloaders.

    Args:
        file_dir (Path): Directory containing the CSV files.
        file_names (list of str): List of file names to load.
        tokenizer_name (str): Name of the Hugging Face tokenizer to use.
        batch_size (int): Batch size for the dataloaders.
        slice_size (int): Number of data points per slice.
        step (int): Step size for downsampling within each slice (e.g., keep every 3rd or 4th point).
        max_slices (int): Maximum number of slices to extract from each file.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Load and slice raw data
    all_slices = []
    for file in file_names:
        df = pd.read_csv(os.path.join(file_dir, file))
        num_slices = len(df) // slice_size
        if max_slices is not None:
            num_slices = min(num_slices, max_slices)
        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = start_idx + slice_size
            slice_df = df.iloc[start_idx:end_idx].iloc[::step]
            all_slices.append(slice_df)

    full_data = pd.concat(all_slices, ignore_index=True)
    full_data = full_data.dropna().reset_index(drop=True)

    # Split into train, val, test before normalization
    torch.manual_seed(42)
    dataset_size = len(full_data)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    indices = torch.randperm(dataset_size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    train_data = full_data.iloc[train_idx].reset_index(drop=True)
    val_data = full_data.iloc[val_idx].reset_index(drop=True)
    test_data = full_data.iloc[test_idx].reset_index(drop=True)

    # Compute normalization parameters only from training set
    feature_cols = ["beta1", "beta2", "beta3", "Theta", "omega_r", "Vwx", "Mz1", "Mz2", "Mz3"]
    train_mean = train_data[feature_cols].mean()
    train_std = train_data[feature_cols].std()

    # Normalize train, val, and test using train mean/std
    train_data[feature_cols] = (train_data[feature_cols] - train_mean) / train_std
    val_data[feature_cols] = (val_data[feature_cols] - train_mean) / train_std
    test_data[feature_cols] = (test_data[feature_cols] - train_mean) / train_std

    # Create datasets from normalized data
    train_dataset = WindTurbineDataset(train_data, tokenizer_name)
    val_dataset = WindTurbineDataset(val_data, tokenizer_name)
    test_dataset = WindTurbineDataset(test_data, tokenizer_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def test_dataloader():
    # Example Usage
    project_dir = Path(__file__).parent.parent
    files_dir = project_dir / "data" / "raw"
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]
    batch_size = 32
    sample_fraction = 0.5  # Use 50% of the dataset for quicker testing

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(files_dir, files, "state-spaces/mamba-130m-hf", batch_size, sample_fraction)

    for batch_idx, (input_ids, attention_mask, targets) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}")
        print(f"Input IDs: {input_ids.shape}")
        print(f"Attention Mask: {attention_mask.shape}")
        print(f"Targets: {targets.shape}")
        break


if __name__ == "__main__":
    test_dataloader()
