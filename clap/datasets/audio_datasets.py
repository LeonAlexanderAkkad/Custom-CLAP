from torch.utils.data import Dataset, Subset
from pathlib import Path
from abc import ABC, abstractmethod


class AudioDataset(Dataset, ABC):
    """Abstract audio dataset."""
    def __init__(self, directory: str, download: bool = True):
        self.root = Path(directory).absolute()
        if download:
            self.download()

    def split(self, train_idx, val_idx, test_idx):
        """Splits the dataset into three subsets given the indices."""
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def download(self):
        raise NotImplementedError
