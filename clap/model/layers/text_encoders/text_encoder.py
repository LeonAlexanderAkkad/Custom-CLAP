import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..projection import Projection
from ....utils import get_target_device


TEXT_ENCODERS = {"RoBERTa"}


class TextEncoder(nn.Module):
    """Defines and loads the text encoder for CLAP."""

    def __init__(self, text_cfg: dict, proj_cfg: dict):
        super().__init__()

        self.text_cfg = text_cfg
        self.proj_cfg = proj_cfg
        self.name = self.text_cfg["name"]

        self.text_encoder, self.tokenizer = self.load_text_encoder()

        self.projection = Projection(
            n_input_features=self.text_cfg["out_size"],
            n_hidden_features=self.proj_cfg["hidden_size"],
            n_output_features=self.proj_cfg["out_size"],
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, text: list[str]):
        tokenized_text = self.tokenize(text)
        # Get the last hidden state
        output = self.text_encoder(**tokenized_text)[0]
        # Extract CLS token
        output = output[:, 0, :]

        # Projects the embedding into the same space as the audio embedding.
        projected = self.projection(output)

        return projected

    def load_text_encoder(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loads respective pretrained text encoder model from Huggingface."""

        if not self.is_valid():
            raise NotImplementedError(
                f"Text encoder '{self.name}' not implemented.\nAvailable encoders: {list(TEXT_ENCODERS)}"
            )

        return AutoModel.from_pretrained(self.name), AutoTokenizer.from_pretrained(self.name)

    def is_valid(self) -> bool:
        """Checks if the text encoder is valid."""
        name = self.name.upper()
        for encoder in TEXT_ENCODERS:
            if encoder.upper() in name:
                return True

        return False

    def tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        if "ROBERTA" in self.name.upper():
            # Tokenize text into a dictionary with the shape:
            # {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor}
            tokenized_text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.text_cfg["max_len"],
                return_tensors="pt"
            )

        else:
            raise NotImplementedError(f"No forward method implemented for {self.name}")

        # Move tensors to device
        tokenized_text = {key: value.to(get_target_device()) for key, value in tokenized_text.items()}

        return tokenized_text
