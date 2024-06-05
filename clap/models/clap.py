import numpy as np

import torch
from torch import nn


class Clap(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = TextEncoder()

        self.audio_encoder = AudioEncoder()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        text_embeddings = self.text_encoder(text)
        audio_embeddings = self.audio_encoder(audio)

        return text_embeddings, audio_embeddings, self.logit_scale.exp()


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Load pretrained TextEncoder.

    def forward(self, text: torch.Tensor):
        # TODO: Implement forward pass for TextEncoder.
        pass


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Load pretrained audio AudioEncoder.

    def forward(self, text: torch.Tensor):
        # TODO: Implement forward pass for AudioEncoder.
        pass


class Projection(nn.Module):
    def __init__(
            self,
            n_input_features: int,
            n_hidden_features: int,
            n_output_features: int,
            activation_function: nn.Module = nn.ReLU(),
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
