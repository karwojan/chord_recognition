import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.dataset_scripts.downloading import download_video

index = pd.read_csv("./data/index.csv", sep=";")

executor = ThreadPoolExecutor()

audio_filepaths = list(
    tqdm(
        executor.map(
            lambda idx_and_song: download_video(idx_and_song[1]["video_id"], "./data/audio/"),
            list(index.iterrows()),
        ),
        total=len(index),
    )
)

index["audio_filepath"] = audio_filepaths
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
