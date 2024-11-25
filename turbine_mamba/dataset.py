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

    def __init__(self, file_paths, tokenizer_name, slice_size=100, step=3, max_slices=None):
        """
        Args:
            file_paths (list of str): List of CSV file paths.
            tokenizer_name (str): Name of the Hugging Face tokenizer to use.
            slice_size (int): Number of data points per slice.
            step (int): Step size for down sampling within each slice (e.g., keep every 3rd or 4th point).
            max_slices (int): Maximum number of slices to extract from each file.
        """
        self.data = []
        for file_path in file_paths:
            # Load each CSV file
            df = pd.read_csv(file_path)

            # Calculate the total number of slices available
            num_slices = len(df) // slice_size

            # If max_slices is specified, limit the number of slices
            if max_slices is not None:
                num_slices = min(num_slices, max_slices)

            # Extract the slices
            for i in range(num_slices):
                start_idx = i * slice_size
                end_idx = start_idx + slice_size
                slice_df = df.iloc[start_idx:end_idx]

                # Downsample the slice
                downsampled_slice = slice_df.iloc[::step]
                self.data.append(downsampled_slice)

        # Combine all slices into a single DataFrame
        self.data = pd.concat(self.data, ignore_index=True)

        # Normalize data
        self.data = (self.data - self.data.mean()) / self.data.std()

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = row[['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']].values.astype(str)
        target = row[['Mz1', 'Mz2', 'Mz3']].values.astype(float)

        # Convert numerical input into a textual representation
        input_text = " ".join(inputs)

        return input_text, torch.tensor(target, dtype=torch.float32)


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
    file_paths = [os.path.join(file_dir, file) for file in file_names]

    # Create the full dataset
    full_dataset = WindTurbineDataset(file_paths, tokenizer_name, slice_size, step, max_slices)

    # Split dataset into train, validation, and test sets
    torch.manual_seed(42)  # Ensure reproducibility
    train_size = int(0.7 * len(full_dataset))  # Training is 70%
    val_size = int(0.2 * len(full_dataset))  # Validation is 20%
    test_size = len(full_dataset) - train_size - val_size  # Test is 10%

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def test_dataloader():
    # Example Usage
    project_dir = Path(__file__).parent.parent

    files_dir = project_dir / "data" / "raw"  # Replace with your directory path
    files = [
        "wind_speed_11_n.csv", "wind_speed_13_n.csv",
        "wind_speed_15_n.csv", "wind_speed_17_n.csv", "wind_speed_19_n.csv"
    ]
    batch_size = 32
    sample_fraction = 0.5  # Use 50% of the dataset for quicker testing

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(files_dir, files, batch_size, sample_fraction)

    # Print the first batch
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}")
        print(f"Inputs: {inputs.shape}")
        print(f"Input: {inputs[0]}")
        print(f"Targets: {targets.shape}")
        break


if __name__ == "__main__":
    test_dataloader()
