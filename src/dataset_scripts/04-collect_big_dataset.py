import pandas as pd
import billboard
import datetime
import random
from threading import Lock
from dataclasses import dataclass
from tqdm import tqdm
from ytmusicapi import YTMusic
from concurrent.futures import ThreadPoolExecutor

from src.dataset_scripts.downloading import download_video


@dataclass(frozen=True)
class BillboardItem:
    title: str
    artist: str


def billboard_hot100_item_generator():
    # prepare shuffled list of all possible dates
    min_date = datetime.date(1958, 8, 3)
    max_date = datetime.date.today()
    dates = [
        min_date + datetime.timedelta(days=i)
        for i in range(0, (max_date - min_date).days, 7)
    ]
    random.shuffle(dates)
    # prepare set of returned items
    returned_items = set()
    for date in dates:
        new_items = {
            BillboardItem(item.title, item.artist)
            for item in billboard.ChartData("hot-100", date=str(date))
        } - returned_items
        for new_item in new_items:
            yield new_item
        returned_items |= new_items


def find_next_hot_song(
    lock, billboard_items_generator, found_video_ids, progress_bar, index_df
):
    try:
        # find new video
        search_results = []
        while len(search_results) == 0:
            with lock:
                item = next(billboard_items_generator)
            search_results = YTMusic().search(
                query=f"{item.artist} {item.title}", filter="songs"
            )[:3]
            with lock:
                search_results = [
                    s for s in search_results if s["videoId"] not in found_video_ids
                ]
                # mark videos as found
                found_video_ids |= {s["videoId"] for s in search_results}
        # download one of new videos
        audio_filepath = download_video(
            search_results[0]["videoId"], "./data/audio"
        )
        # add to index if download succeded
        if audio_filepath is not None:
            with lock:
                index_df.loc[len(index_df)] = pd.Series({
                    "song": search_results[0]["title"],
                    "artist": search_results[0]["artists"][0]["name"],
                    "video_id": search_results[0]["videoId"],
                    "audio_filepath": audio_filepath,
                    "purpose": "pretrain"
                })
                index_df.to_csv("./data/index.csv", sep=";", header=True, index=False)
                progress_bar.update(1)
    except Exception as e:
        print(e)


# read index
index_df = pd.read_csv("./data/index.csv", sep=";")

# launch searching and downloading threads
lock = Lock()
billboard_items_generator = billboard_hot100_item_generator()
found_video_ids = set(index_df.video_id)
progress_bar = tqdm(total=10000)
executor = ThreadPoolExecutor(max_workers=5)
for _ in range(10000):
    executor.submit(
        find_next_hot_song,
        lock,
        billboard_items_generator,
        found_video_ids,
        progress_bar,
        index_df,
    )
