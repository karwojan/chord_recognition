import numpy as np
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
    return yt.search(query, filter="songs")[0]["videoId"]
    
metadata = pd.read_csv("../chordlab/metadata.csv", sep=";")
executor = ThreadPoolExecutor(max_workers=3)
videoIds = list(tqdm(executor.map(find_song, metadata.iterrows()), total=len(metadata)))
metadata["videoId"] = videoIds
metadata.to_csv("
