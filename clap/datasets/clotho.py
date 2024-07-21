import os

import zipfile

from glob import glob

import pandas as pd

import requests

from .audio_dataset import AudioDataset


# TODO: Finish Clotho

class ClothoDataset(AudioDataset):
    def get_data(self, audiodata_dir: str, metadata_dir: str):

        metadata_path = os.path.join(metadata_dir, f"{self.kind}.csv")
        # Download metadata and audios if necessary
        if self.download:
            # Download metadata, create directory if necessary and split the caption to create new samples
            os.makedirs(metadata_dir, exist_ok=True)
            self.download_metadata(metadata_path)
            metadata_df = pd.read_csv(metadata_path)
            metadata_df = self.split_captions(metadata_df)
            metadata_df.to_csv(metadata_path, index=False)

            # Download audios and create directories if necessary
            os.makedirs(audiodata_dir, exist_ok=True)
            self.download_audio(audiodata_dir)

        metadata_df = pd.read_csv(metadata_path)

        # TODO: Finish Clotho and test it, then start training

        audio_paths = [os.path.abspath(os.path.join(audiodata_dir, youtube_id, ".wav")) for youtube_id in
                       sorted(glob(os.path.join(audiodata_dir, "*.wav")))]
        ids = [os.path.basename(path)[:-4] for path in audio_paths]
        captions = [metadata_df[metadata_df["audiocap_id"] == audiocap_id]["caption"] for audiocap_id in ids]

        return audio_paths, captions

    def download_audio(self, audiodata_dir: str):
        base_url = "https://zenodo.org/record/3490684/files/"

        match self.kind:
            case "train":
                filename = "clotho_audio_development.7z"
            case "val":
                filename = "clotho_audio_validation.7z"
            case "test":
                filename = "clotho_audio_evaluation.7z"
            case _:
                raise ValueError(f"Unknown kind {self.kind}")

        zip_path = os.path.join(audiodata_dir, self.kind + ".zip")

        response = requests.get(base_url + filename, stream=True)
        block_size = 1024

        with open(zip_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)

        # Extract downloaded zip file
        zipfile.ZipFile(zip_path, 'r').extractall(audiodata_dir)

        # Delete zip file
        os.remove(zip_path)

    def download_metadata(self, metadata_path: str):
        base_url = "https://zenodo.org/record/3490684/files/"

        match self.kind:
            case "train":
                filename = "clotho_captions_development.csv"
            case "val":
                filename = "clotho_captions_validation.csv"
            case "test":
                filename = "clotho_captions_evaluation.csv"
            case _:
                raise ValueError(f"Unknown kind {self.kind}")

        response = requests.get(base_url + filename, stream=True)
        block_size = 1024

        with open(metadata_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)

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
            value_name="caption"
        )

        return metadata_df_melted
