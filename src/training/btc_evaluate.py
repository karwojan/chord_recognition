import torch
import numpy as np
from tqdm import tqdm
from dataclasses import replace

from src.training.dataset import SongDataset
from src.training.btc_model import BTC_model
from src.annotation_parser.chord_model import LabelOccurence
from src.annotation_parser.labfile_printer import print_labfile


def evaluate(dataset: SongDataset, btc_model: BTC_model):
    assert dataset.frames_per_item <= 0, "dataset must return whole songs"
    frames_per_item = btc_model.timestep
    btc_model.eval()

    with torch.no_grad():
        for audio, labels in tqdm(dataset, total=len(dataset)):
            audio, labels = torch.tensor(audio).cuda(), torch.tensor(labels).cuda()
            predictions = torch.zeros_like(labels)

            # get model predictions for song
            for i in range(0, audio.shape[0], frames_per_item // 2):
                # correct to end of song for last item(s)
                if i + frames_per_item > audio.shape[0]:
                    i = audio.shape[0] - frames_per_item
                # extract frames of single item
                item_audio = audio[i: i + frames_per_item]
                item_labels = labels[i: i + frames_per_item]
                # predict labels
                prediction, _, _, _ = btc_model(
                    item_audio.unsqueeze(0), item_labels.unsqueeze(0)
                )
                # fill all predictions with new predictions
                if i > 0:
                    predictions[
                        i + (frames_per_item // 4): i + frames_per_item
                    ] = prediction[frames_per_item // 4:]
                else:
                    predictions[i: i + frames_per_item] = prediction

            # convert predictions to label occurences and merge repeated
            predictions = predictions.cpu().numpy()
            start_times = np.arange(len(predictions)) * (
                dataset.hop_size / dataset.sample_rate
            )
            stop_times = start_times + (dataset.frame_size / dataset.sample_rate)
            label_occurences = []
            for start, stop, label in zip(start_times, stop_times, predictions):
                if len(label_occurences) > 0 and label_occurences[-1].label == label:
                    label_occurences[-1] = replace(label_occurences[-1], stop=stop)
                else:
                    label_occurences.append(LabelOccurence(start, stop, label))

            # calculate CSR and other classification metrics and save
            # todo

            # create labfile and save
            # todo

    # calculate WCSR and save
