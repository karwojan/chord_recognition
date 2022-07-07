import pandas as pd
import billboard
import datetime
from dataclasses import dataclass
from tqdm import tqdm


@dataclass(frozen=True)
class Song:
    title: str
    artist: str


def get_last_billboard_hot_100_songs(n_last):
    week = datetime.timedelta(weeks=1)
    now = datetime.date.today()
    songs = set()
    with tqdm(total=n_last) as progres_bar:
        while len(songs) < n_last:
            new_songs = {
                Song(song.title, song.artist)
                for song in billboard.ChartData("hot-100", date=str(now))
            } - songs
            songs |= new_songs
            now -= week
            progres_bar.update(len(new_songs))
    print("Back to", now)
    return list(songs)[:n_last]


songs = get_last_billboard_hot_100_songs(10000)
index = pd.DataFrame()
index["song"] = [song.title for song in songs]
index["artist"] = [song.artist for song in songs]
index.to_csv("./data/index_big.csv", sep=";", header=True, index=False)
