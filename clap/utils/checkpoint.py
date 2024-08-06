import torch

from pathlib import Path


BASE_PATH = Path(__file__).parent.parent


def load_clap_ckpt(audio_encoder: str, text_encoder: str, version: str | int) -> dict:
    """Load checkpoint from ckpt file based on encoder names and the specified version."""
    ckpt_path = BASE_PATH / "checkpoints" / f"clap_{audio_encoder}_{text_encoder}_v{version}.ckpt"

    ckpt = torch.load(ckpt_path)

    return ckpt


def load_clf_ckpt(audio_encoder: str, text_encoder: str, version: str | int) -> dict:
    """Load checkpoint from ckpt file based on encoder names and the specified version."""
    ckpt_path = BASE_PATH / "checkpoints" / f"clf_{audio_encoder}_{text_encoder}_v{version}.ckpt"

    ckpt = torch.load(ckpt_path)

    return ckpt
