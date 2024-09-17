from abc import ABC, abstractmethod

import requests

from pathlib import Path

from tqdm import tqdm

from typing import Literal

from torch.utils.data import Dataset


class AudioDataset(Dataset, ABC):
    """Simple abstract audio dataset used for training."""

    def __init__(
            self,
            base_path: str | Path,
            kind: Literal["train", "val", "test"] = "train",
            download: bool = False
    ):
        """Initialize the audio dataset and download it if needed."""
        self.base_path = Path(base_path)
        self.kind = kind
        self.download = download
        self.audio, self.text = self.get_samples()

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.audio)

    @staticmethod
    def download_file(file_url: str, destination_path: str | Path, desc: str):
        if not Path(destination_path).exists():
            response = requests.get(file_url, stream=True)
            block_size = 1024

            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
            with open(destination_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

    @abstractmethod
    def get_samples(self):
        raise NotImplementedError
