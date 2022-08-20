from src.annotation_parser.labfile_symbols import (
    natural_symbols,
    shorthand_symbols,
)
from src.annotation_parser.chord_model import Chord, Interval, ChordOccurence


def print_interval(interval: Interval) -> str:
    return (
        abs(interval.modifier) * ("b" if interval.modifier < 0 else "#")
        + str(interval.degree)
    )


def print_chord(chord: Chord) -> str:
    # check "no chord"
    if chord.root is None:
        return "N"

    # find pitchname
    natural_symbol = list(
        sorted(
            filter(lambda item: item[1] <= chord.root, natural_symbols.items()),
            key=lambda item: chord.root - item[1],
        )
    )[0]
    pitchname = natural_symbol[0] + "".join((chord.root - natural_symbol[1]) * ["#"])

    # find matching shorthand or just print components
    matching_shorthands = list(
        filter(lambda item: item[1] == chord.components, shorthand_symbols.items())
    )
    if len(matching_shorthands) == 1:
        components = ":" + matching_shorthands[0][0]
    else:
        components = (
            ":("
            + ",".join([print_interval(interval) for interval in chord.components])
            + ")"
        )

    # print bass
    if chord.bass != Interval(1, 0):
        bass = "/" + print_interval(chord.bass)
    else:
        bass = ""

    return pitchname + components + bass


def print_labfile(chord_occurences: ChordOccurence) -> str:
    return "\n".join(
        [
            f"{c.start:.3f}\t{c.stop:.3f}\t{print_chord(c.chord)}"
            for c in chord_occurences
        ]
    )
