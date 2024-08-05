from pathlib import Path

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
