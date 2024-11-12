import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


from transformers import AutoTokenizer

class WindTurbineDataset(Dataset):
    """
    Custom Dataset for Wind Turbine Data
    """

    def __init__(self, file_paths, tokenizer_name, sample_fraction=1.0):
        """
        Args:
            file_paths (list of str): List of CSV file paths.
            tokenizer_name (str): Name of the Hugging Face tokenizer to use.
            sample_fraction (float): Fraction of the dataset to use (0.0 to 1.0).
        """
        self.data = []
        for file_path in tqdm(file_paths, desc="Loading dataset files", mininterval=0.1):
            # Load each CSV file and append to the data list
            df = pd.read_csv(file_path)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)

        # Optionally take a fraction of the dataset
        if sample_fraction < 1.0:
            self.data = self.data.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

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



def get_dataloaders(file_dir: Path, file_names, tokenizer_name, batch_size=32, sample_fraction=1.0):
    """
    Creates train, validation, and test dataloaders.

    Args:
        file_dir (Path): Directory containing the CSV files.
        file_names (list of str): List of file names to load.
        tokenizer_name (str): Name of the Hugging Face tokenizer to use.
        batch_size (int): Batch size for the dataloaders.
        sample_fraction (float): Fraction of the dataset to use (0.0 to 1.0).

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    file_paths = [os.path.join(file_dir, file) for file in file_names]

    # Create the full dataset
    full_dataset = WindTurbineDataset(file_paths, tokenizer_name, sample_fraction)

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
