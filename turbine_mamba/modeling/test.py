import torch


def test_model(model, dataloader, device):
    """
    Test the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to test on (CPU/GPU).

    Returns:
        tuple: Predictions and ground truth as NumPy arrays for further analysis.
    """
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for input_ids, attention_mask, targets in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            predictions.append(outputs.cpu())
            ground_truth.append(targets.cpu())

    return torch.cat(predictions, dim=0).numpy(), torch.cat(ground_truth, dim=0).numpy()
