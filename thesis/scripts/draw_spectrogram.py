import numpy as np
import matplotlib.pyplot as plt
import librosa

wave = librosa.load(
    path="data/audio/A99sV18J0mk_Jailhouse Rock.webm",
    sr=22050,
    mono=True,
    res_type="kaiser_fast",
    offset=30,
    duration=10,
)[0]

spectrogram = np.flip(
    np.log(
        np.abs(
            librosa.cqt(
                wave,
                sr=22050,
                hop_length=2048,
                fmin=librosa.note_to_hz("C1"),
                n_bins=(6 * 24),
                bins_per_octave=24,
            )
        )
        + 10e-6
    ),
    axis=0,
)


plt.figure(figsize=(8.0, 2.0), layout="tight")
plt.axis("off")
plt.plot(wave)
plt.savefig("thesis/images/wave.png")

plt.figure(figsize=(8.0, 2.0), layout="tight")
plt.axis("off")
plt.tight_layout()
plt.imshow(spectrogram, aspect="auto", interpolation="nearest")
plt.savefig("thesis/images/spectrogram.png")

plt.figure(figsize=(8.0, 2.0))
n_frames = 10
for i in range(n_frames):
    plt.subplot(1, n_frames, i + 1)
    plt.axis("off")
    plt.imshow(
        spectrogram[:, i: i + 1],
        vmin=np.min(spectrogram),
        vmax=np.max(spectrogram),
        aspect=0.1,
        interpolation="nearest",
    )
plt.gcf().tight_layout(pad=1.0)
plt.savefig("thesis/images/spectrogram_splitted.png")
