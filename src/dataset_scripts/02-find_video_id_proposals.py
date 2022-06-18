import re
import tempfile
import pandas as pd
from ytmusicapi import YTMusic
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.dataset_scripts.downloading import download_video
from src.dataset_scripts.evaluation import evaluate_video


def normalize_name(name: str):
    return re.sub(r"\W+", "", name.lower())


def equal_by_names(song, search_result):
    if isinstance(song["album"], str):
        equal_album = normalize_name(song["album"]) == normalize_name(
            search_result["album"]["name"]
        )
    else:
        equal_album = True
    equal_title = normalize_name(song["song"]) == normalize_name(search_result["title"])
    equal_artist = any(
        normalize_name(song["artist"]) == normalize_name(artist["name"])
        for artist in search_result["artists"]
    )
    return equal_album and equal_title and equal_artist


def find_song(idx_and_song):
    # find songs basing on title, artist and album (optional)
    _, song = idx_and_song
    query = song["song"] + " " + song["artist"]
    if isinstance(song["album"], str):
        query += " " + song["album"]
    search_results = YTMusic().search(query, filter="songs")[:10]

    # SELECTION STEP 1: select videos basing on names
    video_ids = [s["videoId"] for s in search_results if equal_by_names(song, s)]
    # print(f"Found {len(video_ids)} videos for song {song['song']} basing on names.")
    if len(video_ids) == 0:
        return None, None

    # SELECTION STEP 2: select videos basing on CSR
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_filepaths = [download_video(video_id, tmpdir) for video_id in video_ids]
        metrics = [
            evaluate_video(song["filepath"], audio_filepath)
            for audio_filepath in audio_filepaths
        ]
    best_metrics = max(metrics, key=lambda m: m[2] if m[2] is not None else 0)
    # print(f"Best matching video for song {song['song']} has CSR = {best_metrics[2]}")
    if best_metrics[2] is not None:
        return video_ids[metrics.index(best_metrics)], best_metrics
    else:
        return None, None


index = pd.read_csv("./data/index.csv", sep=";")
executor = ProcessPoolExecutor(max_workers=16)
video_ids_and_metrics = list(
    tqdm(executor.map(find_song, index.iterrows()), total=len(index))
)
video_ids, metrics = zip(*video_ids_and_metrics)
start_diffs, stop_diffs, csrs = zip(*metrics)
index["video_id"] = video_ids
index["start_diff"] = start_diffs
index["stop_diff"] = stop_diffs
index["csr"] = csrs
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
