import pandas as pd
import numpy as np

test_fraction = 0.1
validation_fraction = 0.1

# load index
index = pd.read_csv("./data/index.csv", sep=";")

# select songs with "properly matched" videos
filtered_index = index.query("csr > 0.3 and abs(stop_diff) < 10")

# filter songs with repeated video_id
repeated_video_ids = filtered_index.groupby("video_id").count().query("filepath > 1").reset_index()["video_id"]
for repeated_video_id in repeated_video_ids:
    repeated_songs = filtered_index.query("video_id == @repeated_video_id")
    filtered_index = filtered_index.drop(index=repeated_songs.sort_values(by="csr").iloc[:-1].index)

# split to train/test (not empty purpose mark song as selected - with properly matched, unique video)
index["purpose"] = None
for subset in filtered_index.subset.unique():
    subset_indices = np.random.permutation(np.array(filtered_index.query("subset == @subset").index))
    n_test_items = int(np.ceil(len(subset_indices) * test_fraction))
    n_validation_items = int(np.ceil(len(subset_indices) * validation_fraction))

    index.loc[subset_indices, "purpose"] = "train"
    index.loc[subset_indices[:n_test_items], "purpose"] = "test"
    index.loc[subset_indices[n_test_items:n_test_items + n_validation_items], "purpose"] = "validate"

# save new index
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
