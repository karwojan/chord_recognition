import pandas as pd
from ytmusicapi import YTMusic
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

yt = YTMusic()


def find_song(index_and_row):
    index, row = index_and_row
    query = row["song"] + " " + row["artist"]
    if isinstance(row["album"], str):
        query += " " + row["album"]
    search_results = yt.search(query, filter="songs")
    if len(search_results) > 0:
        return search_results[0]["videoId"]
    else:
        return ""


metadata = pd.read_csv("./data/index.csv", sep=";")
executor = ThreadPoolExecutor(max_workers=2)
videoIds = list(tqdm(executor.map(find_song, metadata.iterrows()), total=len(metadata)))
metadata["videoId"] = videoIds
metadata.to_csv("./data/index.csv", sep=";", header=True, index=False)
