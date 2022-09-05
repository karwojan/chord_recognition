import os
import argparse
import warnings
import pandas as pd
import numpy as np
import librosa
from typing import List, Optional
from datetime import timedelta
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm
import pyrubberband

from src.training.preprocessing import (
    Preprocessing,
    JustSplitPreprocessing,
    CQTPreprocessing,
)
from src.annotation_parser import parse_annotation_file
from src.annotation_parser.chord_model import Chord, vocabularies


@dataclass
class SongDatasetConfig:
    sample_rate: int
    frame_size: int
    hop_size: int
    frames_per_item: int
    audio_preprocessing: Preprocessing
    standardize_audio: bool
    pitch_shift_augment: bool
    labels_vocabulary: str
    subsets: Optional[List[str]]

    @staticmethod
    def add_to_argparser(parser: argparse.ArgumentParser):
        parser.add_argument("--sample_rate", type=int, required=True)
        parser.add_argument("--frame_size", type=int, required=True)
        parser.add_argument("--hop_size", type=int, required=True)
        parser.add_argument("--frames_per_item", type=int, required=True)
        parser.add_argument("--audio_preprocessing", type=str, required=True, choices=["cqt", "raw"])
        parser.add_argument("--standardize_audio", action="store_true")
        parser.add_argument("--pitch_shift_augment", action="store_true")
        parser.add_argument(
            "--labels_vocabulary",
            type=str,
            required=True,
            choices=["maj_min", "root_only"],
        )
        parser.add_argument("--subsets", type=str, required=False, nargs="+")

    @staticmethod
    def create_from_args(args):
        if args.audio_preprocessing == "raw":
            audio_preprocessing = JustSplitPreprocessing()
        elif args.audio_preprocessing == "cqt":
            audio_preprocessing = CQTPreprocessing()
        else:
            raise ValueError(args.audio_preprocessing)
        return SongDatasetConfig(
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            frames_per_item=args.frames_per_item,
            audio_preprocessing=audio_preprocessing,
            standardize_audio=args.standardize_audio,
            pitch_shift_augment=args.pitch_shift_augment,
            labels_vocabulary=args.labels_vocabulary,
            subsets=args.subsets,
        )


class SongDataset(Dataset):
    def __init__(self, purposes: List[str], config: SongDatasetConfig):
        super().__init__()

        # store parameters
        self.purposes = purposes
        self.config = config
        self.n_classes = 1 + max(len(vocabularies[config.labels_vocabulary]), 1) * 12
        self.cache_path = "./data/cache"
        if not os.path.isdir(self.cache_path):
            os.mkdir(self.cache_path)

        def _time_to_frame_index(t):
            t = int(t * config.sample_rate)
            return int(round(t // config.hop_size + t % config.hop_size / config.frame_size))

        def _load_song(song_metadata):
            print(f"Loading {song_metadata.song}", flush=True)
            cached_data_no_shift_path = os.path.join(self.cache_path, f"{song_metadata.Index}_0.npz")

            if os.path.exists(cached_data_no_shift_path):
                song = np.load(cached_data_no_shift_path)
                audio = song["audio"]
            else:
                # load audio file (supress librosa warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio = librosa.load(
                        path=song_metadata.audio_filepath, sr=config.sample_rate, mono=True, res_type="kaiser_fast"
                    )[0]

                # load annotation file
                chords = parse_annotation_file(song_metadata.filepath)

                # optional pitch shift augmentation
                if config.pitch_shift_augment:
                    shifts = [0] + list(range(-5, 0)) + list(range(1, 7))
                    audio_list = [pyrubberband.pitch_shift(audio, config.sample_rate, shift) for shift in shifts]
                    chords_list = [[chord.shift_chord(shift) for chord in chords] for shift in shifts]
                else:
                    audio_list = [audio]
                    chords_list = [chords]

                # preprocess audio - split into frames
                audio_list = [
                    config.audio_preprocessing.preprocess(audio, config.sample_rate, config.frame_size, config.hop_size)
                    for audio in audio_list
                ]

                for i, (audio, chords) in enumerate(zip(audio_list, chords_list)):
                    # assign annotations (labels) to frames
                    labels = np.zeros(shape=audio.shape[:1], dtype=int)
                    for chord in chords:
                        labels[
                            _time_to_frame_index(chord.start): _time_to_frame_index(chord.stop)
                        ] = chord.to_label_occurence(config.labels_vocabulary).label

                    # store in cache
                    np.savez(
                        os.path.join(self.cache_path, f"{song_metadata.Index}_{i}.npz"),
                        audio=audio.astype(np.float32),
                        labels=labels.astype(np.int32),
                    )

                audio = audio_list[0]

            return song_metadata.Index, audio.shape, np.mean(audio), np.mean(audio ** 2)

        # load index (songs metadata)
        self.songs_metadata = pd.read_csv("./data/index.csv", sep=";")
        self.songs_metadata = self.songs_metadata.query(
            " or ".join([f"purpose == '{purpose}'" for purpose in purposes])
        )
        if config.subsets is not None:
            self.songs_metadata = self.songs_metadata.query(
                " or ".join([f"subset == '{subset}'" for subset in config.subsets])
            )

        # load songs
        id_per_song, shape_per_song, mean_per_song, mean_2_per_song = zip(
            *tqdm(
                ThreadPoolExecutor(max_workers=5).map(_load_song, self.songs_metadata.itertuples()),
                total=self.songs_metadata.shape[0],
                smoothing=0.0,
            )
        )

        # prepare list of items - multiple mappings to same song, proportionally to song length
        self.items = []
        for song_id, song_shape in zip(id_per_song, shape_per_song):
            n_items = song_shape[0] // config.frames_per_item if config.frames_per_item > 0 else 1
            self.items += [song_id] * n_items

        # store mean, std and dimensionality of dataset
        self.mean = np.mean(mean_per_song)
        self.std = np.sqrt(np.mean(mean_2_per_song) - self.mean**2)
        self.dim = shape_per_song[0][1]

        # print info about loaded songs
        dataset_duration = sum(shape[0] for shape in shape_per_song) * config.hop_size / config.sample_rate
        print(f"Loaded {len(id_per_song)} songs ({timedelta(seconds=int(dataset_duration))}).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # select random shift
        shift = np.random.randint(12) if self.config.pitch_shift_augment else 0

        # load whole preprocessed audio file
        song = np.load(os.path.join(self.cache_path, f"{self.items[index]}_{shift}.npz"))
        audio = song["audio"].astype(np.float32)
        labels = song["labels"].astype(np.int64)

        # select random item from this file if item size is defined
        if self.config.frames_per_item > 0:
            start_frame_index = np.random.randint(audio.shape[0] - self.config.frames_per_item)
            audio = audio[start_frame_index: start_frame_index + self.config.frames_per_item]
            labels = labels[start_frame_index: start_frame_index + self.config.frames_per_item]

        # optionally standardize frames values
        if self.config.standardize_audio:
            audio = (audio - self.mean) / self.std

        return audio, labels

    def get_song_metadata(self, index):
        return self.songs_metadata.loc[self.items[index]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # from src.training.dataset import SongDataset

    ds = SongDataset(
        purposes=["validate"],
        config=SongDatasetConfig(
            sample_rate=22050,
            frame_size=2048,
            hop_size=2048,
            frames_per_item=108,
            audio_preprocessing=JustSplitPreprocessing(),
            standardize_audio=True,
            pitch_shift_augment=True,
            labels_vocabulary="maj_min",
            subsets=["isophonics"],
        ),
    )
    print(ds.mean, ds.std)
    print("len(ds):", len(ds))
    n = 3
    for i in range(n):
        plt.subplot(1, n, i + 1)
        item = ds[i]
        print(ds.get_song_metadata(i))
        print(item[0].shape, item[1].shape)
        print(item[0].dtype, item[1].dtype)
        plt.imshow(item[0].T)
    plt.show()
