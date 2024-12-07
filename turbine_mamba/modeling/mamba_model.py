from transformers import AutoModel
import torch.nn as nn


class WindTurbineModel(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=3):
        """
        Initializes the model with a pretrained Mamba model and a custom regression head.

        Args:
            pretrained_model_name (str): Name of the pretrained Mamba model.
            num_labels (int): Number of output labels (e.g., Mz1, Mz2, Mz3).
        """
        super(WindTurbineModel, self).__init__()
        print(f"Loading {pretrained_model_name}...")
        self.mamba = AutoModel.from_pretrained(pretrained_model_name)  # Load Mamba model
        print(f"Model loaded.")

        # Get the hidden size from the config
        input_dim = self.mamba.config.hidden_size

        # Add a regression head
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tokenized input ids.
            attention_mask (torch.Tensor): Attention masks.

        Returns:
            torch.Tensor: Model output.
        """
        outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        logits = self.regression_head(pooled_output)
        return logits
