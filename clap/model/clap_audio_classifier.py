import torch
from torch import nn

from .clap import Clap


class ClapAudioClassifier(nn.Module):
    def __init__(
        self,
        model: Clap,
        in_dim: int,
        out_dim: int,
        act_fn: nn.Module = None,
        freeze_text: bool = True,
        freeze_audio: bool = False
    ):
        super().__init__()

        self.clap = model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_fn = act_fn
        self.fc = nn.Linear(self.in_dim, self.out_dim)

        if freeze_text:
            self.clap.freeze_text_encoder()

        if freeze_audio:
            self.clap.freeze_audio_encoder()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        out_emb = self.clap.audio_encoder(x)
        out = self.fc(out_emb)

        if self.act_fn is not None:
            out = self.act_fn(out)

        return out

    @classmethod
    def from_ckpt(cls, version: str) -> 'ClapAudioClassifier':
        pass
        # TODO: implement loading from checkpoint
