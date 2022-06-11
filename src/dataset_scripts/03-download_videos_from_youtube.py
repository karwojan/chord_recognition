import pandas as pd
import pytube
from pytube import YouTube
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob

index = pd.read_csv("./data/index.csv", sep=";")


def download_video(idx_and_song):
    _, song = idx_and_song
    video_id = song['videoId']
    if len(glob(f"./data/audio/{video_id}_*")) == 0:
        try:
            YouTube(
                f"http://youtube.com/watch?v={video_id}"
            ).streams.filter(only_audio=True).order_by(
                "abr"
            ).desc().first().download(
                "./data/audio/", filename_prefix=f"{video_id}_"
            )
        except pytube.exceptions.VideoUnavailable:
            print(f"Video {video_id} for song {song['song']} is unavailable!")


executor = ThreadPoolExecutor(max_workers=3)
paths = list(tqdm(executor.map(download_video, list(index.iterrows())), total=len(index)))
