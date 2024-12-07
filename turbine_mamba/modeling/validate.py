import torch
from tqdm import tqdm


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch and return predictions and ground truth for metric computation.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to validate on (CPU/GPU).

    Returns:
        tuple: Average validation loss, predictions, and ground truth as NumPy arrays.
    """
    model.eval()
    total_loss = 0
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(dataloader, desc='Validating'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions.append(outputs.cpu())
            ground_truth.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    predictions = torch.cat(predictions, dim=0).numpy()
    ground_truth = torch.cat(ground_truth, dim=0).numpy()

    return avg_loss, predictions, ground_truth
