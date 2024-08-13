import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..projection import Projection
from ....utils import get_target_device

TEXT_ENCODERS = {"ROBERTA", "GPT2"}


class TextEncoder(nn.Module):
    """Defines and loads the text encoder for CLAP."""

    def __init__(self, text_cfg: dict, proj_cfg: dict):
        super().__init__()

        self.text_cfg = text_cfg
        self.proj_cfg = proj_cfg
        self.name = self.text_cfg["name"].upper()

        self.base, self.tokenizer = self.load_text_encoder()

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
        output = self.base(**tokenized_text)[0]
        if "GPT2" in self.name:
            # Get the actual sequence length without padding (including EOS token)
            sequence_lengths = torch.ne(tokenized_text['input_ids'], self.tokenizer.pad_token_id).sum(-1) - 1
            output = output[torch.arange(len(text), device=output.device), sequence_lengths]
        else:
            # Extract CLS token
            output = output[:, 0, :]

        # Projects the embedding into the same space as the audio embedding.
        projected = self.projection(output)

        return F.normalize(projected, dim=-1)

    def load_text_encoder(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Loads respective pretrained text encoder model from Huggingface."""
        if not self.__is_valid():
            raise NotImplementedError(
                f"Text encoder '{self.name}' not implemented.\nAvailable encoders: {list(TEXT_ENCODERS)}"
            )
        model, tokenizer = AutoModel.from_pretrained(self.name), AutoTokenizer.from_pretrained(self.name)

        # We need to add a padding token for GPT2 explicitly, otherwise we cannot pad the input if needed
        if "GPT2" in self.name:
            # This token is never used in any caption
            tokenizer.add_special_tokens({"pad_token": "!"})

        return model, tokenizer

    def tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        if "GPT2" in self.name:
            # Manually append the end-of-text token
            text = [caption + " <|endoftext|>" for caption in text]

        # Tokenize text into a dictionary with the shape: {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor}
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            max_length=self.text_cfg["max_len"],
            return_tensors="pt"
        )

        # Move tensors to device
        tokenized_text = {key: value.to(get_target_device()) for key, value in tokenized_text.items()}

        return tokenized_text

    def __is_valid(self) -> bool:
        """Checks if the text encoder is valid."""
        for encoder in TEXT_ENCODERS:
            if encoder in self.name:
                return True

        return False
