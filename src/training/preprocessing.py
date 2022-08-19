from abc import ABC, abstractmethod

import numpy as np
import librosa


class Preprocessing(ABC):
    @abstractmethod
    def preprocess(self, audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> np.ndarray:
        pass

    @abstractmethod
    def pitch_shift_augment(self, audio: np.ndarray, shift: int) -> np.ndarray:
        pass


class CQTPreprocessing(Preprocessing):
    def __init__(self, fmin=librosa.note_to_hz("C1"), n_bins=6 * 24, bins_per_octave=24):
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def preprocess(self, audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> np.ndarray:
        return np.log(
            np.abs(
                librosa.cqt(
                    audio,
                    sr=sample_rate,
                    hop_length=hop_size,
                    fmin=self.fmin,
                    n_bins=self.n_bins,
                    bins_per_octave=self.bins_per_octave,
                ).T
            )
            + 10e-6
        )

    def pitch_shift_augment(self, audio: np.ndarray, shift: int) -> np.ndarray:
        shift = shift * self.bins_per_octave // 12
        if shift > 0:
            audio = audio[:, :-shift]
            audio = np.pad(audio, ((0, 0), (shift, 0)), mode="minimum")
        elif shift < 0:
            audio = audio[:, -shift:]
            audio = np.pad(audio, ((0, 0), (0, -shift)), mode="minimum")
        return audio


class JustSplitPreprocessing(Preprocessing):

    def preprocess(self, audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> np.ndarray:
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

    def pitch_shift_augment(self, audio: np.ndarray, shift: int) -> np.ndarray:
        raise NotImplementedError()
