import torch

from pathlib import Path


def load_clap_ckpt(ckpt_path: str | Path) -> dict:
    """Load checkpoint from ckpt file."""
    ckpt = torch.load(ckpt_path)

    return ckpt


def load_clf_ckpt(ckpt_path: str | Path) -> dict:
    """Load checkpoint from ckpt file."""
    ckpt = torch.load(ckpt_path)

    return ckpt
