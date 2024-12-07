import torch
from tqdm import tqdm

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to validate on (CPU/GPU).

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            targets = targets.to(device)  # Inputs are already tokenized text

            # Forward pass
            outputs = model(inputs)  # Model handles tokenization internally
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu())
            ground_truth.append(targets.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()
    avg_loss = total_loss / len(dataloader)

    return avg_loss, predictions, ground_truth
