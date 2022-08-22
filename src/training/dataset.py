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
from src.annotation_parser.chord_model import Chord, vocabularies


class SongDataset(Dataset):
    def __init__(
        self,
        purposes: List[str],
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        frames_per_item: int,
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
        self.audio_preprocessing = audio_preprocessing
        self.standardize_audio = standardize_audio
        self.pitch_shift_augment = pitch_shift_augment
        self.labels_vocabulary = labels_vocabulary
        self.subsets = subsets
        self.n_classes = 1 + max(len(vocabularies[labels_vocabulary]), 1) * 12
        self.cache_path = "./data/cache"

        def _time_to_frame_index(t):
            t = int(t * sample_rate)
            return int(round(t // hop_size + t % hop_size / frame_size))

        def _load_song(song_metadata):
            cached_path = os.path.join(self.cache_path, f"{song_metadata.Index}.npz")

            if os.path.exists(cached_path):
                song = np.load(cached_path)
                audio = song["audio"]
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
                np.savez(cached_path, audio=audio, labels=labels)

            return song_metadata.Index, n_frames, np.mean(audio), np.mean(audio**2)

        # load index (songs metadata)
        self.songs_metadata = pd.read_csv("./data/index.csv", sep=";")
        self.songs_metadata = self.songs_metadata.query(
            " or ".join([f"purpose == '{purpose}'" for purpose in purposes])
        )
        if subsets is not None:
            self.songs_metadata = self.songs_metadata.query(
                " or ".join([f"subset == '{subset}'" for subset in subsets])
            )

        # load songs
        id_per_song, n_frames_per_song, mean_per_song, mean_2_per_song = zip(
            *tqdm(
                ThreadPoolExecutor().map(_load_song, self.songs_metadata.itertuples()),
                total=self.songs_metadata.shape[0],
            )
        )

        # prepare list of items - multiple mappings to same song, proportionally to song length
        self.items = []
        for song_id, n_frames in zip(id_per_song, n_frames_per_song):
            n_items = n_frames // self.frames_per_item if self.frames_per_item > 0 else 1
            self.items += [song_id] * n_items

        # store mean and std of whole dataset
        self.mean = np.mean(mean_per_song)
        self.std = np.sqrt(np.mean(mean_2_per_song) - self.mean**2)

        print(
            f"Loaded {len(id_per_song)} songs ({timedelta(seconds=int(sum(n_frames_per_song) * self.frame_size / self.sample_rate))})."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # load whole preprocessed audio file
        song = np.load(os.path.join(self.cache_path, f"{self.items[index]}.npz"))
        audio = song["audio"]
        labels = song["labels"]

        # select random item from this file if item size is defined
        if self.frames_per_item > 0:
            start_frame_index = np.random.randint(audio.shape[0] - self.frames_per_item)
            audio = audio[start_frame_index: start_frame_index + self.frames_per_item]
            labels = labels[start_frame_index: start_frame_index + self.frames_per_item]

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

    def get_song_metadata(self, index):
        return self.songs_metadata.loc[self.items[index]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.training.preprocessing import CQTPreprocessing
    # from src.training.dataset import SongDataset

    ds = SongDataset(
        ["validate"],
        sample_rate=44100,
        frame_size=4410,
        hop_size=4410,
        frames_per_item=100,
        audio_preprocessing=CQTPreprocessing(),
        standardize_audio=True,
        pitch_shift_augment=True,
        subsets=["isophonics", "rs200"],
    )
    print(ds.mean, ds.std)
    print("len(ds):", len(ds))
    n = 3
    for i in range(n):
        plt.subplot(1, n, i + 1)
        item = ds[i]
        print(ds.get_song_metadata(i))
        print(item[0].shape, item[1].shape)
        plt.imshow(item[0].T)
    plt.show()
