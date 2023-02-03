import os
import torch
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dataclasses import replace
import tempfile
import json
import mlflow
import matplotlib
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from einops import repeat

from src.training.dataset import SongDataset
from src.training.model import Transformer
from src.annotation_parser import parse_annotation_file
from src.annotation_parser.chord_model import Chord, LabelOccurence, ChordOccurence, csr
from src.annotation_parser.labfile_printer import print_labfile, print_chord


def partial_predict(
    model: torch.nn.Module, audio: torch.Tensor, frames_per_item: int, padding: int
):
    hop_size = frames_per_item - 2 * padding
    n_items = int(np.ceil(len(audio) / hop_size))
    n_missing_right = (n_items * hop_size) - len(audio)

    # add padding
    audio_with_padding = torch.cat(
        [
            repeat(audio[0], "f -> n f", n=padding),
            audio,
            repeat(audio[-1], "f -> n f", n=n_missing_right + padding),
        ],
        dim=0,
    )

    # split the padded sequence into items
    items = audio_with_padding[
        torch.stack(
            [
                torch.arange(i * hop_size, i * hop_size + frames_per_item)
                for i in range(n_items)
            ]
        )
    ]

    # predict
    with torch.no_grad():
        predictions = torch.argmax(model(items), dim=2)

    # remove paddings and flatten to create final predictions sequence
    return predictions[:, padding: frames_per_item - padding].flatten()[: len(audio)]


def evaluate(
    dataset: SongDataset,
    model: Transformer,
    output_dir_prefix: str,
    frames_per_item: int,
):
    assert dataset.config.frames_per_item <= 0, "dataset must return whole songs"
    model.eval()

    display_labels = [
        print_chord(Chord.from_label(label, dataset.config.labels_vocabulary))
        for label in np.arange(dataset.n_classes)
    ]
    recall_precision_kwargs = {
        "average": None,
        "labels": np.arange(dataset.n_classes),
        "zero_division": 0,
    }
    # matplotlib settings for confusion matrix
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.figsize": [12.0, 12.0],
            "figure.autolayout": True,
        }
    )

    output_dir = f"{output_dir_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    all_csrs = []
    all_n_frames = []
    all_predictions = []
    all_labels = []
    for item_index in tqdm(range(len(dataset)), total=len(dataset), unit="item"):
        audio, labels = dataset[item_index]
        audio, labels = torch.tensor(audio).cuda(), torch.tensor(labels).cuda()
        song_metadata = dataset.get_song_metadata(item_index)

        # get model predictions for song
        predictions = partial_predict(
            model, audio, frames_per_item, frames_per_item // 4
        )

        # convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # convert predictions to label occurences and merge repeated
        start_times = np.arange(len(predictions)) * (
            dataset.config.hop_size / dataset.config.sample_rate
        )
        stop_times = start_times + (
            dataset.config.frame_size / dataset.config.sample_rate
        )
        pred_label_occurences: List[LabelOccurence] = []
        for start, stop, label in zip(start_times, stop_times, predictions):
            if (
                len(pred_label_occurences) > 0
                and pred_label_occurences[-1].label == label
            ):
                pred_label_occurences[-1] = replace(
                    pred_label_occurences[-1], stop=stop
                )
            else:
                pred_label_occurences.append(LabelOccurence(start, stop, label))

        with tempfile.TemporaryDirectory() as tmp_dir:
            # create labfile
            with open(os.path.join(tmp_dir, "predictions.lab.txt"), "w") as f:
                f.write(
                    print_labfile(
                        [
                            label_occurence.to_chord_occurence(
                                dataset.config.labels_vocabulary
                            )
                            for label_occurence in pred_label_occurences
                        ]
                    ),
                )

            # calculate CSR and other classification ("framewise") metrics for this song
            metrics = {
                "csr": csr(
                    [
                        chord_occurence.to_label_occurence(
                            dataset.config.labels_vocabulary
                        )
                        for chord_occurence in parse_annotation_file(
                            song_metadata.filepath
                        )
                    ],
                    pred_label_occurences,
                ),
                "accuracy": accuracy_score(labels, predictions),
                "recall": list(
                    recall_score(labels, predictions, **recall_precision_kwargs)
                ),
                "precision": list(
                    precision_score(labels, predictions, **recall_precision_kwargs)
                ),
                "confusion_matrix": [
                    [int(j) for j in i]
                    for i in confusion_matrix(
                        labels,
                        predictions,
                        labels=np.arange(dataset.n_classes)
                    )],
                "quantity": [int((labels == label).sum()) for label in np.arange(dataset.n_classes)],
            }
            with open(os.path.join(tmp_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f)

            # plot confusion matrix
            ConfusionMatrixDisplay.from_predictions(
                y_true=labels,
                y_pred=predictions,
                labels=np.arange(dataset.n_classes),
                display_labels=display_labels,
                xticks_rotation="vertical",
                colorbar=False,
            )
            plt.savefig(os.path.join(tmp_dir, "confusion_matrix.png"))
            plt.close()

            # log to mlflow
            mlflow.log_artifacts(tmp_dir, f"{output_dir}/{song_metadata.name}")

        # store values for global metrics calculation
        all_csrs.append(metrics["csr"])
        all_n_frames.append(len(labels))
        all_labels.append(labels)
        all_predictions.append(predictions)

    # calculate WCSR and classification metrics for all songs
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    with tempfile.TemporaryDirectory() as tmp_dir:
        global_metrics = {
            "wcsr": np.average(all_csrs, weights=all_n_frames),
            "accuracy": accuracy_score(labels, predictions),
            "recall": list(
                recall_score(labels, predictions, **recall_precision_kwargs)
            ),
            "precision": list(
                precision_score(labels, predictions, **recall_precision_kwargs)
            ),
            "quantity": [int((labels == label).sum()) for label in np.arange(dataset.n_classes)],
            "worst_to_best_song_ids": [
                str(dataset.get_song_metadata(i).name) for i in np.argsort(all_csrs)
            ],
        }
        with open(os.path.join(tmp_dir, "global_metrics.json"), "w") as f:
            json.dump(global_metrics, f)
        ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=predictions,
            labels=np.arange(dataset.n_classes),
            display_labels=display_labels,
            xticks_rotation="vertical",
            colorbar=False,
            normalize="true",
            values_format=".2f",
        )
        plt.savefig(os.path.join(tmp_dir, "global_confusion_matrix.png"))
        plt.close()
        mlflow.log_artifacts(tmp_dir, output_dir)
