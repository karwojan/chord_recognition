import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.dataset_scripts.downloading import download_video


def _download(idx_and_song):
    idx, song = idx_and_song
    if not pd.isna(song["purpose"]):
        return download_video(song["video_id"], "./data/audio")
    else:
        return None


index = pd.read_csv("./data/index.csv", sep=";")
executor = ThreadPoolExecutor()
audio_filepaths = list(
    tqdm(
        executor.map(_download, list(index.iterrows())),
        total=len(index),
    )
)

index["audio_filepath"] = audio_filepaths
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
