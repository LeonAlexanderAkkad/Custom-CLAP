import os

import shutil

import py7zr

from glob import glob

import pandas as pd

import requests

from tqdm import tqdm

from .audio_dataset import AudioDataset


BASE_URL = "https://zenodo.org/record/4783391/files/"
DATASET_DIR_BASE = os.path.join("clap", "datasets", "clotho")


class Clotho(AudioDataset):
    def get_samples(self):
        metadata_path = os.path.join(DATASET_DIR_BASE, f"{self.kind}.csv")
        audiodata_dir = os.path.join(DATASET_DIR_BASE, f"{self.kind}_audio")
        # Download metadata and audios if necessary
        if self.download:
            # Download metadata, create directory if necessary and split the caption to create new samples
            os.makedirs(DATASET_DIR_BASE, exist_ok=True)
            self.__download_metadata(metadata_path)
            metadata_df = pd.read_csv(metadata_path)
            metadata_df = self.split_captions(metadata_df)
            metadata_df.to_csv(metadata_path, index=False)

            # Download audios and create directories if necessary
            os.makedirs(audiodata_dir, exist_ok=True)
            self.__download_audio(audiodata_dir)

        metadata_df = pd.read_csv(metadata_path)

        audio_paths = sorted(glob(os.path.join(audiodata_dir, "*.wav")))
        captions = sum([metadata_df[metadata_df["file_name"] == os.path.basename(audio_path)]["caption"].tolist() for audio_path in audio_paths], [])

        # Generate new lists so that each audio file really belongs to 5 captions
        audio_paths_expanded = []

        for audio_path in audio_paths:
            audio_paths_expanded.extend([audio_path] * 5)

        return audio_paths_expanded, captions

    def __download_audio(self, audiodata_dir: str):
        match self.kind:
            case "train":
                filename = "clotho_audio_development.7z"
            case "val":
                filename = "clotho_audio_validation.7z"
            case "test":
                filename = "clotho_audio_evaluation.7z"
            case _:
                raise ValueError(f"Unknown kind {self.kind}")

        zip_path = os.path.join(audiodata_dir, self.kind + ".7z")
        self.__download_file(BASE_URL + filename, zip_path, f"Downloading {self.kind} audio data")

        # Extract downloaded zip file
        py7zr.SevenZipFile(zip_path, 'r').extractall(audiodata_dir)

        # Delete zip file
        os.remove(zip_path)

        # Remove redundant directory
        for root, dirs, files in os.walk(audiodata_dir):
            if root == audiodata_dir:
                continue
            for file_name in files:
                source_file = os.path.join(root, file_name)
                destination_file = os.path.join(audiodata_dir, file_name)
                shutil.move(source_file, destination_file)
            # Remove the now empty nested directory
            shutil.rmtree(root)

        print(f"Downloaded {self.kind} audio data to {audiodata_dir}")

    def __download_metadata(self, metadata_path: str):
        match self.kind:
            case "train":
                filename = "clotho_captions_development.csv"
            case "val":
                filename = "clotho_captions_validation.csv"
            case "test":
                filename = "clotho_captions_evaluation.csv"
            case _:
                raise ValueError(f"Unknown kind {self.kind}")

        self.__download_file(BASE_URL + filename, metadata_path, f"Downloading {self.kind} audio metadata")
        print(f"Downloaded {self.kind} metadata to {metadata_path}")

    @staticmethod
    def split_captions(metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Split captions to create 5x the amount of data."""
        metadata_df_melted = metadata_df.melt(
            id_vars=["file_name"],
            value_vars=[
                "caption_1",
                "caption_2",
                "caption_3",
                "caption_4",
                "caption_5"
            ],
            var_name="caption_num",
            value_name="caption"
        )

        return metadata_df_melted

    @staticmethod
    def __download_file(file_url: str, destination_path: str, desc: str):
        response = requests.get(file_url, stream=True)
        block_size = 1024

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
        with open(destination_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
