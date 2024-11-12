import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to train on (CPU/GPU).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        targets = targets.to(device)  # Inputs are already tokenized text

        # Forward pass
        outputs = model(inputs)  # Model handles tokenization internally
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
