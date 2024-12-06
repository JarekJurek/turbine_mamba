import torch

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

    with torch.no_grad():
        print('Validating...')

        for input_ids, attention_mask, targets in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
