# dataset.py
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def preprocess_and_save_data(file_dir, file_names, tokenizer_name, save_path, slice_size=100, step=3, max_slices=None):
    """
    Tokenizes and saves data for faster training.

    Args:
        file_dir (Path): Directory of raw CSV files.
        file_names (list): List of CSV filenames.
        tokenizer_name (str): Hugging Face tokenizer name.
        save_path (Path): Path to save the preprocessed data.
        slice_size (int): Number of data points per slice.
        step (int): Down-sampling step size.
        max_slices (int): Maximum number of slices per file.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = []

    for file_name in file_names:
        df = pd.read_csv(file_dir / file_name)
        num_slices = len(df) // slice_size
        if max_slices is not None:
            num_slices = min(num_slices, max_slices)

        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = start_idx + slice_size
            slice_df = df.iloc[start_idx:end_idx].iloc[::step].dropna()

            inputs = slice_df[["beta1", "beta2", "beta3", "Theta", "omega_r", "Vwx"]].astype(str)
            targets = slice_df[["Mz1", "Mz2", "Mz3"]].values.astype(float)

            # Tokenize inputs
            tokenized = tokenizer(
                [" ".join(row) for row in inputs.values],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )

            for j in range(len(slice_df)):
                data.append({
                    "input_ids": tokenized["input_ids"][j],
                    "attention_mask": tokenized["attention_mask"][j],
                    "targets": torch.tensor(targets[j], dtype=torch.float32)
                })

    # Save the tokenized data
    torch.save(data, save_path)
    print(f"Preprocessed data saved to {save_path}")


class WindTurbineDataset(Dataset):
    """
    Dataset that loads preprocessed and tokenized inputs for faster training.
    """

    def __init__(self, preprocessed_data_path):
        """
        Args:
            preprocessed_data_path (Path): Path to preprocessed tokenized data.
        """
        self.data = torch.load(preprocessed_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_ids"], item["attention_mask"], item["targets"]


def get_dataloaders(preprocessed_data_path, batch_size=32, train_ratio=0.7, val_ratio=0.2):
    """
    Create DataLoaders from preloaded tokenized data.

    Args:
        preprocessed_data_path (Path): Path to the saved preprocessed data.
        batch_size (int): Batch size for DataLoaders.
        train_ratio (float): Ratio of data for training.
        val_ratio (float): Ratio of data for validation.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    dataset = WindTurbineDataset(preprocessed_data_path)
    dataset_size = len(dataset)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
