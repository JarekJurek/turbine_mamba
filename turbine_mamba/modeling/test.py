import torch
from tqdm import tqdm


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
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            targets = targets.to(device)  # Inputs are already tokenized text

            # Forward pass
            outputs = model(inputs)  # Model handles tokenization internally
            predictions.append(outputs.cpu())
            ground_truth.append(targets.cpu())

    return torch.cat(predictions, dim=0).numpy(), torch.cat(ground_truth, dim=0).numpy()
