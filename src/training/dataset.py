import os
import warnings
import pandas as pd
import numpy as np
import librosa
from typing import List
from datetime import timedelta
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.training.preprocessing import Preprocessing
from src.annotation_parser import parse_annotation_file
from src.annotation_parser.chord_model import Chord


class SongDataset(Dataset):
    def __init__(
        self,
        purposes: List[str],
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        frames_per_item: int,
        items_per_song_factor: float,
        audio_preprocessing: Preprocessing,
        standardize_audio: bool = True,
        pitch_shift_augment: bool = False,
        labels_vocabulary: str = "root_only",
        subsets: List[str] = None,
    ):
        super().__init__()

        # store parameters
        self.purposes = purposes
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.frames_per_item = frames_per_item
        self.items_per_song_factor = items_per_song_factor
        self.audio_preprocessing = audio_preprocessing
        self.standardize_audio = standardize_audio
        self.pitch_shift_augment = pitch_shift_augment
        self.labels_vocabulary = labels_vocabulary
        self.subsets = subsets
        self.cache_path = "./data/cache"

        def _time_to_frame_index(t):
            t = int(t * sample_rate)
            return int(round(t // hop_size + t % hop_size / frame_size))

        def _load_item(idx_and_song_metadata):
            song_metadata = idx_and_song_metadata[1]
            cached_item_path = os.path.join(
                self.cache_path, f"{song_metadata.name}.npz"
            )

            if os.path.exists(cached_item_path):
                item = np.load(cached_item_path)
                audio = item["audio"]
                n_frames = audio.shape[0]
            else:
                # load audio file (supress librosa warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio = librosa.load(
                        path=song_metadata.audio_filepath, sr=self.sample_rate
                    )[0]

                # preprocess - split into frames
                audio = self.audio_preprocessing.preprocess(
                    audio,
                    self.sample_rate,
                    self.frame_size,
                    self.hop_size,
                )
                n_frames = audio.shape[0]

                # assign annotations (labels) to frames
                labels = np.zeros(shape=(n_frames,), dtype=int)
                for chord in parse_annotation_file(song_metadata.filepath):
                    labels[
                        _time_to_frame_index(chord.start): _time_to_frame_index(
                            chord.stop
                        )
                    ] = chord.to_label_occurence(labels_vocabulary).label

                # store in cache
                np.savez(cached_item_path, audio=audio, labels=labels)

            # prepare items
            items = [cached_item_path]
            if self.frames_per_item > 0:
                items = items * int(self.items_per_song_factor * n_frames)

            return items, n_frames, np.mean(audio), np.mean(audio**2)

        # load index
        songs_metadata = pd.read_csv("./data/index.csv", sep=";")
        songs_metadata = songs_metadata.query(
            " or ".join([f"purpose == '{purpose}'" for purpose in purposes])
        )
        if subsets is not None:
            songs_metadata = songs_metadata.query(
                " or ".join([f"subset == '{subset}'" for subset in subsets])
            )

        # load items
        items_per_song, n_frames_per_song, mean_per_song, mean_2_per_song = zip(
            *tqdm(
                ThreadPoolExecutor().map(_load_item, songs_metadata.iterrows()),
                total=songs_metadata.shape[0],
            )
        )
        self.items = [item for items in items_per_song for item in items]
        self.mean = np.mean(mean_per_song)
        self.std = np.sqrt(np.mean(mean_2_per_song) - self.mean**2)
        print(
            f"Loaded {len(items_per_song)} songs ({timedelta(seconds=int(sum(n_frames_per_song) * self.frame_size / self.sample_rate))})."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # load whole preprocessed audio file
        item = np.load(self.items[index])
        audio = item["audio"]
        labels = item["labels"]

        # select random item from this file if item size is defined
        if self.frames_per_item > 0:
            start_frame_index = np.random.randint(audio.shape[0] - self.frames_per_item)
            audio = item["audio"][
                start_frame_index: start_frame_index + self.frames_per_item
            ]
            labels = item["labels"][
                start_frame_index: start_frame_index + self.frames_per_item
            ]

        # optionally standardize frames values
        if self.standardize_audio:
            audio = (audio - self.mean) / self.std

        # optionally augment pitch
        if self.pitch_shift_augment:
            shift = np.random.randint(10) - 5
            audio = self.audio_preprocessing.pitch_shift_augment(audio, shift)
            labels = np.array([
                Chord.from_label(label, self.labels_vocabulary)
                .shift(shift)
                .to_label(self.labels_vocabulary)
                for label in labels
            ])

        return audio, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.training.preprocessing import CQTPreprocessing
    # from src.training.dataset import SongDataset

    ds = SongDataset(
        ["validate"],
        sample_rate=44100,
        frame_size=4410,
        hop_size=4410,
        frames_per_item=0,
        items_per_song_factor=1.0,
        audio_preprocessing=CQTPreprocessing(),
        standardize_audio=True,
        pitch_shift_augment=True,
        subsets=["isophonics", "rs200"],
    )
    print(ds.mean, ds.std)
    n = 3
    for i in range(n):
        plt.subplot(1, n, i + 1)
        item = ds[i]
        print(item[0].shape, item[1].shape)
        plt.imshow(item[0].T)
    plt.show()
