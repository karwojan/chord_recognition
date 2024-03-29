import os
import argparse
import warnings
import pandas as pd
import numpy as np
import librosa
import pyrubberband
import torch
from typing import List, Optional
from datetime import timedelta
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm
from retry.api import retry_call

from src.training.preprocessing import (
    Preprocessing,
    JustSplitPreprocessing,
    CQTPreprocessing,
)
from src.annotation_parser import parse_annotation_file
from src.annotation_parser.chord_model import Chord, vocabularies


def song_dataset_collate_fn(items):
    return (
        torch.concat([torch.from_numpy(item[0]) for item in items], dim=0),
        torch.concat([torch.from_numpy(item[1]) for item in items], dim=0),
    )


@dataclass
class SongDatasetConfig:
    sample_rate: int
    frame_size: int
    hop_size: int
    frames_per_item: int
    item_multiplier: int
    song_multiplier: int
    audio_preprocessing: Preprocessing
    standardize_audio: bool
    pitch_shift_augment: bool
    labels_vocabulary: str
    subsets: Optional[List[str]]
    fraction: Optional[float]
    use_ram_cache: bool

    def generate_cache_description(self) -> str:
        desc = "cache"
        desc += "_sr" + str(self.sample_rate)
        desc += "_fs" + str(self.frame_size)
        desc += "_hs" + str(self.hop_size)
        if isinstance(self.audio_preprocessing, CQTPreprocessing):
            desc += "_cqt"
        elif isinstance(self.audio_preprocessing, JustSplitPreprocessing):
            desc += "_raw"
        if self.standardize_audio:
            desc += "_norm"
        if self.pitch_shift_augment:
            desc += "_augm"
        desc += "_" + self.labels_vocabulary
        return desc

    @staticmethod
    def add_to_argparser(parser: argparse.ArgumentParser):
        parser.add_argument("--sample_rate", type=int, required=True)
        parser.add_argument("--frame_size", type=int, required=True)
        parser.add_argument("--hop_size", type=int, required=True)
        parser.add_argument("--frames_per_item", type=int, required=True)
        parser.add_argument("--item_multiplier", type=int, required=True)
        parser.add_argument("--song_multiplier", type=int, required=True)
        parser.add_argument(
            "--audio_preprocessing", type=str, required=True, choices=["cqt", "raw"]
        )
        parser.add_argument("--standardize_audio", action="store_true")
        parser.add_argument("--pitch_shift_augment", action="store_true")
        parser.add_argument(
            "--labels_vocabulary",
            type=str,
            required=True,
            choices=["maj_min", "root_only"],
        )
        parser.add_argument("--subsets", type=str, required=False, nargs="+")
        parser.add_argument("--dataset_fraction", type=float, required=False)
        parser.add_argument("--use_ram_cache", action="store_true")

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
            item_multiplier=args.item_multiplier,
            song_multiplier=args.song_multiplier,
            audio_preprocessing=audio_preprocessing,
            standardize_audio=args.standardize_audio,
            pitch_shift_augment=args.pitch_shift_augment,
            labels_vocabulary=args.labels_vocabulary,
            subsets=args.subsets,
            fraction=args.dataset_fraction,
            use_ram_cache=args.use_ram_cache
        )


class SongDataset(Dataset):
    def __init__(self, purposes: List[str], config: SongDatasetConfig):
        super().__init__()

        # store parameters
        self.purposes = purposes
        self.config = config
        self.n_classes = 1 + max(len(vocabularies[config.labels_vocabulary]), 1) * 12
        self.cache_path = f"./data/audio/cache/{config.generate_cache_description()}/"
        os.makedirs(self.cache_path, exist_ok=True)

        def _time_to_frame_index(t):
            t = int(t * config.sample_rate)
            return int(
                round(t // config.hop_size + t % config.hop_size / config.frame_size)
            )

        def _load_song(song_metadata):
            # check if song is already cached (with all augmentations if configured)
            if all(
                os.path.exists(
                    os.path.join(self.cache_path, f"{song_metadata.Index}_{i}.npz")
                )
                for i in range(0, 12 if config.pitch_shift_augment else 1)
            ):
                print(f"Loading {song_metadata.song} (from cache)", flush=True)
                audio_list = []
                labels_list = []
                for i in range(0, 12 if config.pitch_shift_augment else 1):
                    song = np.load(os.path.join(self.cache_path, f"{song_metadata.Index}_{i}.npz"))
                    audio_list.append(song["audio"])
                    labels_list.append(song["labels"])
            else:
                print(f"Loading {song_metadata.song}", flush=True)
                # load audio file (supress librosa warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio = retry_call(
                        librosa.load,
                        fkwargs={
                            "path": song_metadata.audio_filepath,
                            "sr": config.sample_rate,
                            "mono": True,
                            "res_type": "kaiser_fast",
                        },
                        tries=2,
                    )[0]

                # load annotation file
                if not pd.isna(song_metadata.filepath):
                    chords = parse_annotation_file(song_metadata.filepath)
                else:
                    chords = []

                # optional pitch shift augmentation
                if config.pitch_shift_augment:
                    shifts = [0] + list(range(-5, 0)) + list(range(1, 7))
                    audio_list = [
                        pyrubberband.pitch_shift(audio, config.sample_rate, shift)
                        for shift in shifts
                    ]
                    chords_list = [
                        [chord.shift_chord(shift) for chord in chords]
                        for shift in shifts
                    ]
                else:
                    audio_list = [audio]
                    chords_list = [chords]

                # preprocess audio - split into frames
                audio_list = [
                    config.audio_preprocessing.preprocess(
                        audio, config.sample_rate, config.frame_size, config.hop_size
                    )
                    for audio in audio_list
                ]

                # assign annotations (labels) to frames
                labels_list = []
                for audio, chords in zip(audio_list, chords_list):
                    labels = np.zeros(shape=audio.shape[:1], dtype=int)
                    for chord in chords:
                        labels[
                            _time_to_frame_index(chord.start): _time_to_frame_index(
                                chord.stop
                            )
                        ] = chord.to_label_occurence(config.labels_vocabulary).label
                    labels_list.append(labels)

                # store in disk cache
                for i, (audio, labels) in enumerate(zip(audio_list, labels_list)):
                    np.savez(
                        os.path.join(self.cache_path, f"{song_metadata.Index}_{i}.npz"),
                        audio=audio.astype(np.float32),
                        labels=labels.astype(np.int32),
                    )

            audio = audio_list[0]
            return (
                audio.shape,
                np.mean(audio),
                np.mean(audio**2),
                (audio_list, labels_list) if config.use_ram_cache else None
            )

        # load index (songs metadata)
        self.songs_metadata = pd.read_csv("./data/index.csv", sep=";")
        self.songs_metadata = self.songs_metadata.query(
            " or ".join([f"purpose == '{purpose}'" for purpose in purposes])
        )
        if config.subsets is not None:
            self.songs_metadata = self.songs_metadata.query(
                " or ".join([f"subset == '{subset}'" for subset in config.subsets])
            )
        if config.fraction is not None:
            self.songs_metadata = self.songs_metadata.sample(
                frac=config.fraction, random_state=47
            )

        # load songs
        shape_per_song, mean_per_song, mean_2_per_song, cache_per_song = zip(
            *tqdm(
                ThreadPoolExecutor(max_workers=5).map(
                    _load_song, self.songs_metadata.itertuples()
                ),
                total=self.songs_metadata.shape[0],
                smoothing=0.0,
            )
        )

        # store mean, std, dimensionality of dataset and cache
        self.mean = np.mean(mean_per_song)
        self.std = np.sqrt(np.mean(mean_2_per_song) - self.mean**2)
        self.dim = shape_per_song[0][1]
        self.ram_cache = cache_per_song

        # print info about loaded songs
        dataset_duration = (
            np.sum(shape_per_song, axis=0)[0] * config.hop_size / config.sample_rate
        )
        print(
            f"Loaded {len(self.songs_metadata)} songs ({timedelta(seconds=int(dataset_duration))})."
        )
        if config.frames_per_item > 0:
            print(
                f"Maximum length (in items): {int(np.max(shape_per_song, axis=0)[0]) // config.frames_per_item}"
            )
            print(
                f"Mean length (in items): {int(np.mean(shape_per_song, axis=0)[0]) // config.frames_per_item}"
            )

    def __len__(self):
        return len(self.songs_metadata) * self.config.song_multiplier

    def __getitem__(self, index):
        real_index = index % len(self.songs_metadata)
        # select random shift
        shift = np.random.randint(12) if self.config.pitch_shift_augment else 0

        # load whole preprocessed audio file
        if self.config.use_ram_cache:
            audio = self.ram_cache[real_index][0][shift]
            labels = self.ram_cache[real_index][1][shift]
        else:
            song = np.load(
                os.path.join(
                    self.cache_path,
                    f"{self.songs_metadata.iloc[real_index].name}_{shift}.npz",
                )
            )
            audio = song["audio"]
            labels = song["labels"]
        audio = audio.astype(np.float32)
        labels = labels.astype(np.int64)

        # select 'item_multiplier' random items if item size is defined
        if self.config.frames_per_item > 0:
            indices = np.array(
                [
                    np.arange(start, start + self.config.frames_per_item)
                    for start in np.random.randint(
                        audio.shape[0] - self.config.frames_per_item,
                        size=(self.config.item_multiplier,),
                    )
                ]
            )
            audio = audio[indices]
            labels = labels[indices]

        # optionally standardize frames values
        if self.config.standardize_audio:
            audio = (audio - self.mean) / self.std

        return audio, labels

    def get_song_metadata(self, index):
        return self.songs_metadata.iloc[index % len(self.songs_metadata)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # from src.training.dataset import SongDataset, SongDatasetConfig, CQTPreprocessing

    ds = SongDataset(
        purposes=["train", "test", "validate"],
        config=SongDatasetConfig(
            sample_rate=22050,
            frame_size=2048,
            hop_size=2048,
            frames_per_item=100,
            item_multiplier=1,
            song_multiplier=1,
            audio_preprocessing=CQTPreprocessing(),
            standardize_audio=True,
            pitch_shift_augment=True,
            labels_vocabulary="maj_min",
            subsets=["uspop"],
            fraction=0.01,
            use_ram_cache=True
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
        plt.imshow(item[0][0].T)
    plt.show()
