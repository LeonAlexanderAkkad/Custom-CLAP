from abc import ABC, abstractmethod

from typing import Literal

from torch.utils.data import Dataset


class AudioDataset(Dataset, ABC):
    """Simple abstract audio dataset used for training."""

    def __init__(
            self,
            kind: Literal["train", "val", "test"] = "train",
            download: bool = False
    ):
        """Initialize the audio dataset and download it if needed."""
        self.kind = kind
        self.download = download
        self.data, self.captions = self.get_samples()

    def __getitem__(self, index: int):
        """Returns file given an index."""
        audio_path = self.data[index]
        caption = self.captions[index]

        return audio_path, caption

    def __len__(self):
        """Returns the number of files in the dataset."""
        return len(self.data)

    @abstractmethod
    def get_samples(self):
        raise NotImplementedError
