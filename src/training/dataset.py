import os
import warnings
import pandas as pd
import numpy as np
import librosa
from typing import Callable, Any, Dict, List
from datetime import timedelta
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

from src.annotation_parser import parse_annotation_file
from tqdm import tqdm


def preprocess_cqt(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    fmin=librosa.note_to_hz("C1"),
    n_bins=6 * 24,
    bins_per_octave=24,
) -> np.ndarray:
    return np.log(
        np.abs(
            librosa.cqt(
                audio,
                sr=sample_rate,
                hop_length=hop_size,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            ).T
        )
        + 10e-6
    )


def preprocess_just_split(
    audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int
) -> np.ndarray:
    # find number of whole frames in audio
    n_frames = len(audio) // hop_size
    while ((n_frames - 1) * hop_size + frame_size) > audio:
        n_frames = n_frames - 1

    # split into frames
    return audio[
        np.reshape(
            np.tile(np.arange(frame_size), n_frames)
            + np.repeat(np.arange(n_frames) * hop_size, frame_size),
            (n_frames, frame_size),
        )
    ]


class SongDataset(Dataset):
    def __init__(
        self,
        purposes: List[str],
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        frames_per_item: int,
        items_per_song_factor: float,
        audio_preprocessing: Callable[[np.ndarray, int, int, int, Any], np.ndarray],
        audio_preprocessing_kwargs: Dict[str, Any] = {},
        labels_vocabulary: str = "root_only",
        subsets: List[str] = None,
    ):
        super().__init__()

        # store parameters
        self.purposes = purposes
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration = frame_size / sample_rate
        self.hop_size = hop_size
        self.hop_duration = hop_size / sample_rate
        self.frames_per_item = frames_per_item
        self.item_size = frames_per_item * hop_size + (frame_size - hop_size)
        self.item_duration = self.item_size / sample_rate
        self.items_per_song_factor = items_per_song_factor
        self.audio_preprocessing = audio_preprocessing
        self.audio_preprocessing_kwargs = audio_preprocessing_kwargs
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
                n_frames = np.load(cached_item_path)["audio"].shape[0]
            else:
                # load audio file (supress librosa warnings)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio = librosa.load(
                        path=song_metadata.audio_filepath, sr=self.sample_rate
                    )[0]

                # preprocess - split into frames
                audio = self.audio_preprocessing(
                    audio,
                    self.sample_rate,
                    self.frame_size,
                    self.hop_size,
                    **self.audio_preprocessing_kwargs,
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
            items = [cached_item_path] * int(self.items_per_song_factor * n_frames)

            return items, n_frames

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
        items_per_song, n_frames_per_song = zip(
            *tqdm(
                ThreadPoolExecutor().map(_load_item, songs_metadata.iterrows()),
                total=songs_metadata.shape[0],
            )
        )
        self.items = [item for items in items_per_song for item in items]
        print(
            f"Loaded {len(items_per_song)} songs ({timedelta(seconds=int(sum(n_frames_per_song) * self.frame_duration))})."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = np.load(self.items[index])
        audio = item["audio"]
        labels = item["labels"]
        start_frame_index = np.random.randint(audio.shape[0] - self.frames_per_item)

        return (
            audio[start_frame_index: start_frame_index + self.frames_per_item],
            labels[start_frame_index: start_frame_index + self.frames_per_item],
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = SongDataset(
        ["validate"],
        sample_rate=44100,
        frame_size=4410,
        hop_size=4410,
        frames_per_item=100,
        items_per_song_factor=1.0,
        audio_preprocessing=preprocess_cqt,
        subsets=["isophonics", "rs200"]
    )
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        item = ds[i]
        print(item[0].shape, item[1].shape)
        plt.imshow(item[0])
    plt.show()
