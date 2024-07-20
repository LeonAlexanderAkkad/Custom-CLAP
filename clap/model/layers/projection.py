import torch
from torch import nn


class Projection(nn.Module):
    """The final projection layer for both the text and audio encoder of CLAP."""

    def __init__(
            self,
            n_input_features: int,
            n_hidden_features: int,
            n_output_features: int,
            activation_function: nn.Module = nn.GELU(),
            dropout: float = 0.5
    ):
        super().__init__()

        self.fc1 = nn.Linear(n_input_features, n_hidden_features)
        self.fc2 = nn.Linear(n_hidden_features, n_output_features)

        self.act_fn = activation_function
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_output_features)

    def forward(self, x: torch.Tensor):
        out1 = self.fc1(x)
        out2 = self.dropout(self.fc2(self.act_fn(out1)))
        out = self.layer_norm(out1 + out2)

        return out
