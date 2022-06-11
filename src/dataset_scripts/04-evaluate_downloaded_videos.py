import pandas as pd
import madmom
from math import isnan
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from ..annotation_parser import parse_annotation_file, parse_lab_annotation
from ..annotation_parser.chord_model import LabelOccurence, csr


chord_extractor = madmom.processors.SequentialProcessor(
    [
        madmom.audio.chroma.DeepChromaProcessor(),
        madmom.features.chords.DeepChromaChordRecognitionProcessor(),
    ]
)


def evaluate_video(idx_and_song):
    _, song = idx_and_song
    audio_files = glob(f"./data/audio/{song['videoId']}_*")
    if "csr" in song and not isnan(song["csr"]):
        return song["start_diff"], song["stop_diff"], song["csr"]
    if len(audio_files) > 0:
        real_occurences = [
            x.to_label_occurence("root_only")
            for x in parse_annotation_file(song["filepath"])
        ]
        try:
            pred_occurences = [
                LabelOccurence(x[0], x[1], parse_lab_annotation(x[2]).to_label("root_only"))
                for x in chord_extractor(audio_files[0])
            ]
        except madmom.io.audio.LoadAudioFileError:
            return (None, None, None)
        non_0_real_occurences = [o for o in real_occurences if o.label != 0]
        non_0_pred_occurences = [o for o in pred_occurences if o.label != 0]
        if len(non_0_real_occurences) > 0 and len(non_0_pred_occurences) > 0:
            start_diff = abs(non_0_real_occurences[0].start - non_0_pred_occurences[0].start)
            stop_diff = abs(non_0_real_occurences[-1].stop - non_0_pred_occurences[-1].stop)
            csr_value = csr(real_occurences, pred_occurences)
            return (round(start_diff, 2), round(stop_diff, 2), round(csr_value, 2))
    return (None, None, None)


index = pd.read_csv("./data/index.csv", sep=";")
executor = ProcessPoolExecutor()
evaluation = list(tqdm(executor.map(evaluate_video, list(index.iterrows())), total=len(index)))
index["start_diff"] = [x[0] for x in evaluation]
index["stop_diff"] = [x[1] for x in evaluation]
index["csr"] = [x[2] for x in evaluation]
index.to_csv("./data/index.csv", sep=";", header=True, index=False)
