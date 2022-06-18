import os
import pandas as pd
import pytube
from pytube import YouTube
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob


def download_video(video_id: str, output_path: str) -> str:
    if len(glob(os.path.join(output_path, f"{video_id}_*"))) == 0:
        try:
            return (
                YouTube(f"http://youtube.com/watch?v={video_id}")
                .streams.filter(only_audio=True)
                .order_by("abr")
                .desc()
                .first()
                .download(output_path, filename_prefix=f"{video_id}_")
            )
        except pytube.exceptions.VideoUnavailable:
            print(f"Video {video_id} is unavailable!")
            return None


def download_videos(index: pd.DataFrame, max_workers: int = 3):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    list(
        tqdm(
            executor.map(
                lambda idx_and_song: download_video(idx_and_song[1]["videoId"]),
                list(index.iterrows()),
            ),
            total=len(index),
        )
    )
