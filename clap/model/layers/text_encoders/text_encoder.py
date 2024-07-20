import torch
from torch import nn

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..projection import Projection


TEXT_ENCODERS = {"roberta"}


class TextEncoder(nn.Module):
    """Defines and loads the text encoder for CLAP."""

    def __init__(self, config_text: dict, config_proj: dict):
        super().__init__()

        self.config_text = config_text
        self.config_proj = config_proj
        self.name = self.config_text["name"]

        self.text_encoder, self.tokenizer = self.load_text_encoder()

        self.projection = Projection(
            n_input_features=self.config_text["out_size"],
            n_hidden_features=self.config_proj["hidden_size"],
            n_output_features=self.config_proj["out_size"],
            activation_function=nn.GELU(),
            dropout=0.5
        )

    def forward(self, text: torch.Tensor):
        if "roberta" in self.name:
            # Tokenize text into a dictionary with the shape:
            # {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'token_type_ids': torch.Tensor}
            tokenized_text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config_text["max_length"],
                return_tensors="pt"
            )

            # Get the last hidden state
            output = self.text_encoder(**tokenized_text)[0]
            # Extract CLS token
            output = output[:, 0, :]
        else:
            raise NotImplementedError(f"No forward method implemented for {self.name}")

        # Projects the embedding into the same space as the audio embedding.
        projected = self.projection(output)

        return projected

    def load_text_encoder(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loads respective pretrained text encoder model from Huggingface."""

        for encoder in TEXT_ENCODERS:
            if encoder not in self.name:
                raise NotImplementedError(
                    f"Text encoder '{self.name}' not implemented.\nAvailable encoders: {list(TEXT_ENCODERS)}")

        return AutoModel.from_pretrained(self.name), AutoTokenizer.from_pretrained(self.name)
