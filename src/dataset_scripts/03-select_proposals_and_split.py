import pandas as pd
import numpy as np

# load index
index = pd.read_csv("./data/index.csv", sep=";")

# select songs with "properly matched" videos
filtered_index = index.query("csr > 0.3 and abs(stop_diff) < 10")

# filter songs with repeated video_id
repeated_video_ids = filtered_index.groupby("video_id").count().query("filepath > 1").reset_index()["video_id"]
for repeated_video_id in repeated_video_ids:
    repeated_songs = filtered_index.query("video_id == @repeated_video_id")
    filtered_index = filtered_index.drop(index=repeated_songs.sort_values(by="csr").iloc[:-1].index)

# split to 5 folds and mark it in 'purpose' column (non empty purpose mark song
# as valid and selected for training - with properly matched, unique video)
k = 5
index["purpose"] = None
for subset in filtered_index.subset.unique():
    subset_indices = np.random.permutation(np.array(filtered_index.query("subset == @subset").index))
    fold_size = len(subset_indices) // k
    for i in range(k):
        index.loc[subset_indices[i * fold_size:], "purpose"] = f"train_fold_{i}"

# save new index
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
