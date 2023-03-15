from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.annotation_parser import parse_annotation_file

# read index
index = pd.read_csv("data/index.csv", sep=";")
index = index.query(" or ".join(f"purpose == 'train_fold_{i}'" for i in range(5)))

# calculate statistics
statistics: Dict[int, float] = {}

for idx, series in tqdm(index.iterrows(), total=len(index)):
    for chord in parse_annotation_file(series.filepath):
        label = chord.to_label_occurence("maj_min")
        statistics[label.label] = statistics.get(label.label, 0) + (label.stop - label.start)

plt.xticks(np.arange(25), rotation=90, labels=[
    "N",
    "C:maj", "C#:maj", "D:maj", "D#:maj", "E:maj", "F:maj", "F#:maj", "G:maj", "G#:maj", "A:maj", "A#:maj", "B:maj",
    "C:min", "C#:min", "D:min", "D#:min", "E:min", "F:min", "F#:min", "G:min", "G#:min", "A:min", "A#:min", "B:min"
])
plt.xlabel("Nazwa akordu")
plt.ylabel("Łączny czas występowania [h]")
plt.bar(statistics.keys(), statistics.values())
plt.show()
