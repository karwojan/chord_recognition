import warnings
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass

from src.annotation_parser import parse_annotation_file


@dataclass
class SongDatasetItem:
    path: str
    n_frames: int
    labels: np.ndarray


class SongDataset(Dataset):
    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        frames_per_item: int,
        items_per_song_factor: float,
        labels_vocabulary: str = "root_only",
        spectrogram_method: str = None,
    ):
        # store parameters
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration = frame_size / sample_rate
        self.hop_size = hop_size
        self.hop_duration = hop_size / sample_rate
        self.frames_per_item = frames_per_item
        self.item_size = frames_per_item * hop_size + (frame_size - hop_size)
        self.item_duration = self.item_size / sample_rate
        self.items_per_song_factor = items_per_song_factor
        self.spectrogram_method = spectrogram_method

        # prepare items
        self.items = []

        def _time_to_frame_index(t):
            t = int(t * sample_rate)
            return int(round(t // hop_size + t % hop_size / frame_size))

        for _, song_metadata in pd.read_csv("./data/index.csv", sep=";").iterrows():
            # get song duration in seconds and in frames
            duration = librosa.get_duration(filename=song_metadata.audio_filepath)
            n_frames = int(np.ceil(duration * sample_rate / hop_size))

            # assign annotations (labels) to frames
            labels_per_frames = np.zeros(shape=(n_frames,), dtype=int)
            for chord in parse_annotation_file(song_metadata.filepath):
                labels_per_frames[
                    _time_to_frame_index(chord.start): _time_to_frame_index(chord.stop)
                ] = chord.to_label_occurence(labels_vocabulary).label

            # store items (n_items references to the same song)
            n_items = int(self.items_per_song_factor * n_frames)
            self.items += [
                SongDatasetItem(
                    song_metadata.audio_filepath, n_frames, labels_per_frames
                )
                for _ in range(n_items)
            ]

        # prepare indices to gather frames from item
        self.item_indices = np.reshape(
            np.tile(np.arange(frame_size), frames_per_item)
            + np.repeat(np.arange(frames_per_item) * hop_size, frame_size),
            (frames_per_item, frame_size),
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        start_frame_index = np.random.randint(item.n_frames - self.frames_per_item)

        # load samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = librosa.load(
                path=item.path,
                sr=self.sample_rate,
                offset=start_frame_index * self.hop_duration,
                duration=self.item_duration,
            )[0]
        # pad to item size (just in case)
        audio = np.pad(audio, ((0, len(audio) - self.item_size),), mode="edge")

        # split into frames (with optional change to frequency domain)
        if self.spectrogram_method is None:
            audio = audio[self.item_indices]
        elif self.spectrogram_method == "cqt":
            audio = np.log(
                np.abs(
                    librosa.cqt(
                        audio,
                        sr=self.sample_rate,
                        hop_length=self.hop_size,
                        fmin=librosa.note_to_hz("C1"),
                        n_bins=6 * 24,
                        bins_per_octave=24,
                    )
                )
                + 10e-6
            )[:, : self.frames_per_item].T

        # get labels
        labels = self.items[index].labels[
            start_frame_index: start_frame_index + self.frames_per_item
        ]

        return audio, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = SongDataset(
        sample_rate=44100,
        frame_size=4410,
        hop_size=4410,
        frames_per_item=100,
        items_per_song_factor=1.0,
        spectrogram_method="cqt",
    )
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        item = ds[i]
        print(item[0].shape, item[1].shape)
        plt.imshow(item[0])
    plt.show()
