import sys
import os
import re
import tempfile
import json
import mlflow
from PIL import Image

run_name = sys.argv[1]

# get run
run = mlflow.search_runs(
    experiment_names=["karwomagisterka"],
    filter_string=f'tags.mlflow.runName = "{run_name}"',
).iloc[0]

# read template
with open("./thesis/scripts/supervised_experiment_results_template.tex") as f:
    template = f.read()


# write hiperparams to template
template = template.replace("!item_multiplier!", run["params.item_multiplier"])
template = template.replace("!song_multiplier!", run["params.song_multiplier"])
template = template.replace(
    "!augment!", "TAK" if run["params.pitch_shift_augment"] else "NIE"
)
template = template.replace(
    "!subsets!", re.sub("['\\[\\]]", "", run["params.subsets"]).replace("_", " ")
)
template = template.replace(
    "!fraction!",
    "1.0" if run["params.dataset_fraction"] else run["params.dataset_fraction"],
)
template = template.replace("!model_dim!", run["params.model_dim"])
template = template.replace("!n_heads!", run["params.n_heads"])
template = template.replace("!h_blocks!", run["params.n_blocks"])
template = template.replace("!dropout_p!", run["params.dropout_p"])
template = template.replace("!n_epochs!", run["params.n_epochs"])
template = template.replace("!batch_size!", run["params.batch_size"])
template = template.replace("!lr!", run["params.lr"])


# download metrics and confusion matrices
metrics = {}
confusion_matrices = {}
artifacts = mlflow.MlflowClient().list_artifacts(run_id=run.run_id)
for dataset in ["test", "train", "validate"]:
    with tempfile.TemporaryDirectory() as tmp:
        root_evaluation_path = [
            a.path for a in artifacts if a.path.startswith(f"{dataset}_ds_evaluation")
        ][0]
        metrics_file = mlflow.artifacts.download_artifacts(
            run_id=run.run_id,
            artifact_path=os.path.join(root_evaluation_path, "global_metrics.json"),
        )
        with open(metrics_file) as f:
            metrics[dataset] = json.load(f)
        confusion_matrix_file = mlflow.artifacts.download_artifacts(
            run_id=run.run_id,
            artifact_path=os.path.join(
                root_evaluation_path, "global_confusion_matrix.png"
            ),
        )
        confusion_matrices[dataset] = Image.open(confusion_matrix_file)


# write metrics to template
template = template.replace(
    "!duration!", str(round((run.end_time - run.start_time).seconds / 3600, 1))
)
template = template.replace("!test_wcsr!", str(round(metrics["test"]["wcsr"], 4)))
template = template.replace("!test_accuracy!", str(round(metrics["test"]["accuracy"], 4)))
template = template.replace("!train_wcsr!", str(round(metrics["train"]["wcsr"], 4)))
template = template.replace("!train_accuracy!", str(round(metrics["train"]["accuracy"], 4)))
template = template.replace("!validate_wcsr!", str(round(metrics["validate"]["wcsr"], 4)))
template = template.replace("!validate_accuracy!", str(round(metrics["validate"]["accuracy"], 4)))

for i in range(25):
    template = template.replace(f"!R{i}!", str(round(metrics["test"]["recall"][i], 4)))
    template = template.replace(f"!P{i}!", str(round(metrics["test"]["precision"][i], 4)))
    template = template.replace(f"!Q{i}!", str(round(metrics["test"]["quantity"][i], 4)))


# write image to template
confusion_matrices["test"].save(f"./thesis/results/confusion_matrix_{run['tags.mlflow.runName']}.png")
template = template.replace("!image_path!", f"./results/confusion_matrix_{run['tags.mlflow.runName']}.png")


# save template
with open(f"./thesis/results/results_{run['tags.mlflow.runName']}.tex", "w") as f:
    f.write(template)
