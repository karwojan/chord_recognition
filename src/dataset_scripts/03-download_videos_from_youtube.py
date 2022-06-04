import pandas as pd
import pytube
from pytube import YouTube
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from glob import glob

index = pd.read_csv("./data/index.csv", sep=";")


def download_video(idx_and_song):
    idx, song = idx_and_song
    if len(glob(f"./data/audio/{idx}_*")) > 0:
        print(f"Video {song['videoId']} for song {song['song']} already downloaded!")
    else:
        try:
            YouTube(
                f"http://youtube.com/watch?v={song['videoId']}"
            ).streams.filter(only_audio=True).order_by(
                "abr"
            ).desc().first().download(
                "./data/audio/", filename_prefix=f"{idx}_"
            )
        except pytube.exceptions.VideoUnavailable:
            print(f"Video {song['videoId']} for song {song['song']} is unavailable!")


executor = ThreadPoolExecutor(max_workers=3)
paths = list(tqdm(executor.map(download_video, list(index.iterrows())), total=len(index)))
