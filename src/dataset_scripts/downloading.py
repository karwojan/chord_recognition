import os
import pandas as pd
import pytube
from pytube import YouTube
from tqdm import tqdm
from glob import glob


def download_video(video_id: str, output_path: str) -> str:
    paths = glob(os.path.join(output_path, f"{video_id}_*"))
    if len(paths) == 0:
        try:
            return (
                YouTube(f"http://youtube.com/watch?v={video_id}")
                .streams.filter(only_audio=True)
                .order_by("abr")
                .desc()
                .first()
                .download(output_path, filename_prefix=f"{video_id}_")
            )
        except Exception as e:
            print(f"Video {video_id} is unavailable:", e)
            return None
    else:
        return paths[0]
