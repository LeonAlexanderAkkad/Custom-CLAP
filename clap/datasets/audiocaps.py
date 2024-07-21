import os
import subprocess

import requests

from .audio_dataset import AudioDataset

import yt_dlp

import pandas as pd

from glob import glob


class AudioCaps(AudioDataset):
    def get_data(self, audiodata_dir: str, metadata_dir: str):

        metadata_path = os.path.join(metadata_dir, f"{self.kind}.csv")
        # Download metadata and audios if necessary
        if self.download:
            # Download metadata and create directory if necessary
            os.makedirs(metadata_dir, exist_ok=True)
            self.download_metadata(metadata_path)
            metadata_df = pd.read_csv(metadata_path)

            # Download audios and create directories if necessary
            os.makedirs(audiodata_dir, exist_ok=True)
            download_dir = os.path.join(audiodata_dir, "full_audios")
            os.makedirs(download_dir, exist_ok=True)
            for youtube_id, audiocap_id, start_time in zip(metadata_df["youtube_id"], metadata_df["audiocap_id"],
                                                           metadata_df["start_time"]):
                if not os.path.exists(os.path.join(audiodata_dir, f'{audiocap_id}.wav')):
                    success = True
                    if not os.path.exists(os.path.join(download_dir, f'{youtube_id}.wav')):
                        success = self.download_audio(youtube_id, download_dir)
                    if success:
                        # Extract audio segment using the given start time in the metadata
                        self.extract_audio_segment(
                            os.path.join(download_dir, f'{youtube_id}.wav'),
                            start_time,
                            os.path.join(audiodata_dir, f'{audiocap_id}.wav')
                        )
                    else:
                        # Remove unavailable samples
                        metadata = metadata_df[metadata_df["youtube_id"] != youtube_id]
                        metadata.to_csv(metadata_path, index=False)

        metadata = pd.read_csv(metadata_path)

        audio_paths = [os.path.abspath(os.path.join(audiodata_dir, youtube_id, ".wav")) for youtube_id in sorted(glob(os.path.join(audiodata_dir, "*.wav")))]
        ids = [os.path.basename(path)[:-4] for path in audio_paths]
        captions = [metadata[metadata["audiocap_id"] == audiocap_id]["caption"] for audiocap_id in ids]

        return audio_paths, captions

    def extract_audio_segment(self, file_path: str, start_time: int, output_path: str):
        command = [
            'ffmpeg',
            '-i', file_path,
            '-ss', str(start_time),
            '-t', str(self.duration),
            "-ac", "1",
            output_path
        ]

        subprocess.run(command, check=True)

    def download_audio(self, youtube_id: str, output_dir: str) -> bool:
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

    def download_metadata(self, metadata_path: str):
        base_url = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/"

        response = requests.get(base_url + self.kind + ".csv", stream=True)
        block_size = 1024

        with open(metadata_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
