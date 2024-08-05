import os

import torch

from glob import glob

from pathlib import Path


BASE_PATH = Path(__file__).parent


def load_clap_ckpt(audio_encoder: str, text_encoder: str, version: str | int) -> dict:
    """Load checkpoint from ckpt file based on encoder names and the specified version."""
    ckpt = None
    ckpt_paths = glob(os.path.join(BASE_PATH, "*.ckpt"))
    for ckpt_path in ckpt_paths:
        if audio_encoder in ckpt_path and text_encoder in ckpt_path and "v" + str(version) in ckpt_path:
            ckpt = ckpt_path

    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint file not found for {audio_encoder} and {text_encoder} and v{version}."
                                f"Available checkpoints: {ckpt_paths}")

    ckpt = torch.load(ckpt)

    return ckpt
