import sys
import os
import re
import tempfile
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

run_name = sys.argv[1]
client = mlflow.MlflowClient()

# get run
run = mlflow.search_runs(
    experiment_names=["karwomagisterka"],
    filter_string=f'tags.mlflow.runName = "{run_name}"',
).iloc[0]

# read template
with open("./thesis/scripts/supervised_experiment_results_template.tex") as f:
    template = f.read()

# write hiperparams to template
template = template.replace("!name!", run["tags.mlflow.runName"])
template = template.replace("!item_multiplier!", run["params.item_multiplier"])
template = template.replace("!song_multiplier!", run["params.song_multiplier"])
template = template.replace("!augment!", "TAK" if run["params.pitch_shift_augment"] else "NIE")
template = template.replace("!subsets!", re.sub("['\\[\\]]", "", run["params.subsets"]).replace("_", " "))
template = template.replace("!fraction!", "1.0" if run["params.dataset_fraction"] else run["params.dataset_fraction"])
template = template.replace("!model_dim!", run["params.model_dim"])
template = template.replace("!n_heads!", run["params.n_heads"])
template = template.replace("!n_blocks!", run["params.n_blocks"])
template = template.replace("!block_type!", run["params.block_type"])
template = template.replace("!dropout_p!", run["params.dropout_p"])
template = template.replace("!n_epochs!", run["params.n_epochs"])
template = template.replace("!batch_size!", run["params.batch_size"])
template = template.replace("!lr!", run["params.lr"])
template = template.replace("!early_stopping!", run["params.early_stopping"])

# download metrics
metrics = {}
artifacts = client.list_artifacts(run_id=run.run_id)
for fold in range(5):
    with tempfile.TemporaryDirectory() as tmp:
        root_evaluation_path = [
            a.path for a in artifacts if a.path.startswith(f"validate_ds_evaluation_{fold}")
        ][0]
        metrics_file = mlflow.artifacts.download_artifacts(
            run_id=run.run_id,
            artifact_path=os.path.join(root_evaluation_path, "global_metrics.json"),
        )
        with open(metrics_file) as f:
            metrics[fold] = json.load(f)

# write metrics to template
wcsr_list = []
best_epoch_list = []
for fold in range(5):
    wcsr_list.append(metrics[fold]["wcsr"])
    best_epoch_list.append(run[f"metrics.validate / epoch / best_epoch_{fold}"])
    template = template.replace(f"!wcsr_{fold}!", str(round(wcsr_list[-1], 3)))
    template = template.replace(f"!best_epoch_{fold}!", str(round(best_epoch_list[-1])))
template = template.replace("!wcsr_mean!", str(round(np.mean(wcsr_list), 3)) + r" \pm " + str(round(np.std(wcsr_list), 3)))
template = template.replace("!best_epoch_mean!", str(round(np.mean(best_epoch_list))) + r" \pm " + str(round(np.std(best_epoch_list))))

# print template
print(template)
