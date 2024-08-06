from pathlib import Path

import time

import numpy as np

import random

import os

import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(audio_path: Path | str, target_sampling_rate, resample: bool = True):
    """Loads the given audio and resamples it if wanted"""
    audio, sampling_rate = torchaudio.load(audio_path)

    if resample and sampling_rate != target_sampling_rate:
        resampler = T.Resample(sampling_rate, target_sampling_rate)
        audio = resampler(audio)
        sampling_rate = target_sampling_rate

    return audio.reshape(-1), sampling_rate


def get_target_device():
    """Get the target device where training takes place."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_random_seed(seed: int | None = 42) -> int:
    """Decide if behaviour is random or set by a seed."""
    if seed is None:
        seed = time.time_ns() % (2 ** 32)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    return seed
