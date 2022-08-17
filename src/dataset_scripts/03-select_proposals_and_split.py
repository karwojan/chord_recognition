import pandas as pd
import numpy as np

test_fraction = 0.1
validation_fraction = 0.05

# load index
index = pd.read_csv("./data/index.csv", sep=";")

# select songs with "properly matched" videos
filtered_index = index.query("csr > 0.3 and abs(start_diff) < 2 and abs(stop_diff) < 5")

# split to train/test
for subset in filtered_index.subset.unique():
    subset_indices = np.random.permutation(np.array(filtered_index.query("subset == @subset").index))
    n_test_items = int(np.ceil(len(subset_indices) * test_fraction))
    n_validation_items = int(np.ceil(len(subset_indices) * validation_fraction))

    index.loc[subset_indices, "purpose"] = "train"
    index.loc[subset_indices[:n_test_items], "purpose"] = "test"
    index.loc[subset_indices[n_test_items:n_test_items + n_validation_items], "purpose"] = "validate"

# save new index
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
