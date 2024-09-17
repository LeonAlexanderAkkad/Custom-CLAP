import os

from pathlib import Path

import subprocess

import torchaudio

from .audio_dataset import AudioDataset

import yt_dlp

import pandas as pd

from glob import glob


BASE_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/"


class AudioCaps(AudioDataset):
    def get_samples(self):
        metadata_path = os.path.join(self.base_path, f"{self.kind}.csv")
        audiodata_dir = os.path.join(self.base_path, f"{self.kind}_audio")

        # Download metadata and audios if necessary
        if self.download:
            self.__download_dataset(self.base_path / "train.csv", self.base_path / "train_audio")
            self.__download_dataset(self.base_path / "val.csv", self.base_path / "val_audio")
            self.__download_dataset(self.base_path / "test.csv", self.base_path / "test_audio")

        metadata_df = pd.read_csv(metadata_path)

        audio_paths = sorted(glob(os.path.join(audiodata_dir, "*.wav")))
        if self.kind == "train":
            # Each audio has 1 caption
            captions = [metadata_df[metadata_df["youtube_id"] == os.path.basename(audio_path)[:-4]]["caption"].item() for audio_path in audio_paths]
        else:
            # Each audio has 5 captions
            captions = sum(
                [metadata_df[metadata_df["youtube_id"] == os.path.basename(audio_path)[:-4]]["caption"].tolist() for
                 audio_path in audio_paths], [])

            # Generate new lists so that there is a path for each of the 5 captions
            audio_paths_expanded = []

            for audio_path in audio_paths:
                audio_paths_expanded.extend([audio_path] * 5)

            audio_paths = audio_paths_expanded

        return audio_paths, captions

    def __download_dataset(self, metadata_path: str | Path, audiodata_dir: str | Path):
        os.makedirs(self.base_path, exist_ok=True)
        if not os.path.exists(metadata_path):
            # Download metadata and create directory if necessary
            self.__download_metadata(metadata_path)

        corrupted_sample = None
        # Download audios and create directories if necessary
        os.makedirs(audiodata_dir, exist_ok=True)
        download_dir = os.path.join(self.base_path, "full_audios")
        os.makedirs(download_dir, exist_ok=True)
        metadata_df = pd.read_csv(metadata_path)
        # Copy metadata for removal of invalid samples
        metadata_df_new = metadata_df.copy()
        for youtube_id, start_time in zip(metadata_df["youtube_id"], metadata_df["start_time"]):
            if not os.path.exists(os.path.join(audiodata_dir, f'{youtube_id}.wav')):
                successful_download = True
                if not os.path.exists(os.path.join(download_dir, f'{youtube_id}.wav')):
                    successful_download = self.__download_audio(youtube_id, download_dir)
                if successful_download:
                    # Extract audio segment using the given start time in the metadata
                    self.__extract_audio_segment(
                        os.path.join(download_dir, f'{youtube_id}.wav'),
                        start_time,
                        os.path.join(audiodata_dir, f'{youtube_id}.wav')
                    )
                    try:
                        # Try to load wav file
                        torchaudio.load(os.path.join(audiodata_dir, f'{youtube_id}.wav'))
                    except RuntimeError:
                        corrupted_sample = os.path.join(audiodata_dir, f'{youtube_id}.wav')

                else:
                    # Remove unavailable samples from csv file
                    metadata_df_new = metadata_df_new[metadata_df_new["youtube_id"] != youtube_id]
                    metadata_df_new.to_csv(metadata_path, index=False)

            # Remove corrupted samples
            if corrupted_sample is not None:
                print(corrupted_sample)
                metadata_df_new = metadata_df_new[metadata_df_new["youtube_id"] != youtube_id]
                metadata_df_new.to_csv(metadata_path, index=False)
                print(f"Removing corrupted sample: {corrupted_sample}")
                os.remove(corrupted_sample)
                corrupted_sample = None

        print(f"Downloaded {self.kind} audio data to {audiodata_dir}")

    def __download_metadata(self, metadata_path: str):
        if not os.path.exists(metadata_path):
            filename = self.kind + ".csv"
            self.download_file(BASE_URL + filename, metadata_path, f"Downloading audio metadata")
            print(f"Downloaded {self.kind} metadata to {metadata_path}")

    @staticmethod
    def __download_audio(youtube_id: str, output_dir: str) -> bool:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, f'{youtube_id}.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={youtube_id}'])
            return True
        except yt_dlp.utils.DownloadError:
            return False

    @staticmethod
    def __extract_audio_segment(file_path: str, start_time: int, output_path: str):
        command = [
            'ffmpeg',
            '-i', file_path,
            '-ss', str(start_time),
            '-t', "10",
            "-ac", "1",
            output_path
        ]

        subprocess.run(command, check=True)
