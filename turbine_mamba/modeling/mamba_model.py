import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class WindTurbineModel(nn.Module):
    def __init__(self, pretrained_model_name):
        """
        Initializes the model with a pretrained Mamba model and custom FC layers.

        Args:
            pretrained_model_name (str): Name of the pretrained model (Hugging Face model hub).
        """
        self.pretrained_model_name = pretrained_model_name
        super(WindTurbineModel, self).__init__()

        # Load the pretrained Mamba model
        print(f"Loading {self.pretrained_model_name}...")
        self.mamba = AutoModel.from_pretrained(self.pretrained_model_name)
        print(f"Loading completed.")

        # Freeze the pretrained model's parameters
        for param in self.mamba.parameters():
            param.requires_grad = False

        # FC layers
        input_dim = self.mamba.config.hidden_size  # Hidden size of Mamba model
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # First FC layer
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(256, 128),  # Second FC layer
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(128, 3)  # Output layer (3 outputs: Mz1, Mz2, Mz3)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (list of str): Input sequences as text.

        Returns:
            torch.Tensor: Model output.
        """
        # Tokenize inputs
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(
            next(self.mamba.parameters()).device)

        # Pass through the pretrained Mamba model
        x = self.mamba(**inputs).last_hidden_state  # (batch_size, seq_len, hidden_size)
        x = x.mean(dim=1)  # Pooling: Mean over sequence length

        # Pass through fully connected layers
        x = self.fc(x)
        return x
