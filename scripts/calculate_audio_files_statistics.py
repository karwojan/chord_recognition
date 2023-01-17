import subprocess
import pandas as pd
from tqdm import tqdm

# read index
index = pd.read_csv("data/index.csv", sep=";")


def get_sample_rate_and_fmt(filepath):
    sample_fmt, sample_rate = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-of",
                "default=nk=1:nw=1",
                "-show_entries",
                "stream=sample_rate,sample_fmt",
                filepath,
            ],
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\n")[:-1]
    )
    return int(sample_rate), sample_fmt


print("Labaled dataset")
subindex = index.query(
    "purpose == 'train' or purpose == 'test' or purpose == 'validate'"
)
sample_rates = [
    get_sample_rate_and_fmt(series.audio_filepath)
    for idx, series in tqdm(subindex.iterrows(), total=len(subindex))
]
print(pd.DataFrame(sample_rates, columns=["sample_rate", "sample_fmt"]).value_counts())

print("Unlabaled dataset")
subindex = index.query("purpose == 'pretrain'")
sample_rates = [
    get_sample_rate_and_fmt(series.audio_filepath)
    for idx, series in tqdm(subindex.iterrows(), total=len(subindex))
]
print(pd.DataFrame(sample_rates, columns=["sample_rate", "sample_fmt"]).value_counts())
