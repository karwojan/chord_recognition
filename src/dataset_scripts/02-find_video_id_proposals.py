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
        equal_album = normalize_name(song["album"]) in normalize_name(
            search_result["album"]["name"]
        )
    else:
        equal_album = True
    equal_title = normalize_name(song["song"]) in normalize_name(search_result["title"])
    equal_artist = any(
        normalize_name(song["artist"]) in normalize_name(artist["name"])
        for artist in search_result["artists"]
    )
    return equal_album and equal_title and equal_artist


def find_song(idx_and_song):
    _, song = idx_and_song
    # check if video is already selected
    if "video_id" in song and isinstance(song["video_id"], str):
        return song["video_id"], (song["start_diff"], song["stop_diff"], song["csr"])
    # find songs basing on title, artist and album (optional)
    query = song["song"] + " " + song["artist"]
    if isinstance(song["album"], str):
        query += " " + song["album"]
    try:
        yt = YTMusic()
        search_results = yt.search(query, filter="songs")[:10]
    except Exception as e:
        print(e)
        return None, (None, None, None)
    # SELECTION STEP 1: select videos basing on names
    video_ids = [s["videoId"] for s in search_results if equal_by_names(song, s)]
    print(f"Found {len(video_ids)} videos for song \"{song['song']}\" basing on names.")
    if len(video_ids) == 0:
        video_ids = [s["videoId"] for s in search_results]
        video_ids += [s["videoId"] for s in yt.search(query, filter="videos")[:3]]
    # SELECTION STEP 2: select videos basing on CSR
    if len(video_ids) > 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_filepaths = [download_video(video_id, tmpdir) for video_id in video_ids]
            metrics = [
                evaluate_video(song["filepath"], audio_filepath)
                for audio_filepath in audio_filepaths
            ]
        best_metrics = min(metrics, key=lambda m: (abs(m[0]) + abs(m[1])) if m[2] is not None else 1000)
        print(f"Best matching video for song \"{song['song']}\" has metrics = {best_metrics}")
        if best_metrics[2] is not None:
            return video_ids[metrics.index(best_metrics)], best_metrics
    return None, (None, None, None)


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
