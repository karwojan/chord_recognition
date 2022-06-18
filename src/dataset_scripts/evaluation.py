import madmom
from ..annotation_parser import parse_annotation_file, parse_lab_annotation
from ..annotation_parser.chord_model import LabelOccurence, csr


chord_extractor = madmom.processors.SequentialProcessor(
    [
        madmom.audio.chroma.DeepChromaProcessor(),
        madmom.features.chords.DeepChromaChordRecognitionProcessor(),
    ]
)


def evaluate_video(annotation_filepath, audio_filepath):
    real_occurences = [
        x.to_label_occurence("root_only")
        for x in parse_annotation_file(annotation_filepath)
    ]
    try:
        pred_occurences = [
            LabelOccurence(x[0], x[1], parse_lab_annotation(x[2]).to_label("root_only"))
            for x in chord_extractor(audio_filepath)
        ]
    except Exception:
        return (None, None, None)
    non_0_real_occurences = [o for o in real_occurences if o.label != 0]
    non_0_pred_occurences = [o for o in pred_occurences if o.label != 0]
    if len(non_0_real_occurences) > 0 and len(non_0_pred_occurences) > 0:
        start_diff = non_0_real_occurences[0].start - non_0_pred_occurences[0].start
        stop_diff = non_0_real_occurences[-1].stop - non_0_pred_occurences[-1].stop
        csr_value = csr(real_occurences, pred_occurences)
        return (round(start_diff, 2), round(stop_diff, 2), round(csr_value, 2))
    return (None, None, None)
