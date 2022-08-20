from src.annotation_parser.chord_model import Chord, ChordOccurence, LabelOccurence, Interval
from src.annotation_parser.labfile_printer import print_interval, print_chord, print_labfile
from src.annotation_parser import parse_lab_annotation


def test_chord():
    # create example chords
    N = Chord(None, None, None)
    C = Chord(5, {Interval(1), Interval(3), Interval(5)}, Interval(1))
    c = Chord(5, {Interval(1), Interval(3, -1), Interval(5)}, Interval(1))

    # check shift
    assert N.shift(5) == Chord(None, None, None)
    assert C.shift(-2) == Chord(3, {Interval(1), Interval(3), Interval(5)}, Interval(1))
    assert C.shift(7) == Chord(0, {Interval(1), Interval(3), Interval(5)}, Interval(1))

    # check to_label
    assert N.to_label(vocabulary_name="maj_min") == 0
    assert C.to_label(vocabulary_name="maj_min") == 6
    assert c.to_label(vocabulary_name="maj_min") == 18

    # check from_label
    assert Chord.from_label(0, vocabulary_name="maj_min") == N
    assert Chord.from_label(6, vocabulary_name="maj_min") == C
    assert Chord.from_label(18, vocabulary_name="maj_min") == c


def test_labfile_printer():
    # test print interval
    assert print_interval(Interval(3, 1)) == "#3"
    assert print_interval(Interval(6, 2)) == "##6"
    assert print_interval(Interval(5, -2)) == "bb5"
    assert print_interval(Interval(7)) == "7"

    # test print chord
    assert print_chord(Chord(0, {Interval(1), Interval(3), Interval(5)}, Interval(1))) == "C:maj"
    assert print_chord(Chord(3, {Interval(1), Interval(3, -1), Interval(5)}, Interval(1))) == "D#:min"
    assert print_chord(Chord(11, {Interval(1), Interval(3, -1), Interval(5)}, Interval(3, -1))) == "B:min/b3"
    assert print_chord(Chord(11, {Interval(1), Interval(2, 1)}, Interval(3, -1))) == "B:(1,#2)/b3"

    # test print labfile
    chord_occurences = [
        ChordOccurence(0.0, 2.7, Chord(0, {Interval(1), Interval(3), Interval(5)}, Interval(1))),
        ChordOccurence(2.7, 3.93463, Chord(1, {Interval(1), Interval(3, -1), Interval(5)}, Interval(1))),
        ChordOccurence(3.93463, 8, Chord(2, {Interval(1), Interval(3), Interval(5)}, Interval(1))),
        ChordOccurence(8.0, 10.2, Chord(3, {Interval(1), Interval(3, -1), Interval(5)}, Interval(1))),
    ]
    assert print_labfile(chord_occurences) == (
        "0.000\t2.700\tC:maj\n2.700\t3.935\tC#:min\n3.935\t8.000\tD:maj\n8.000\t10.200\tD#:min"
    )


def test_parse_lab_annotation():
    assert parse_lab_annotation("C:maj") == Chord(0, {Interval(1), Interval(3), Interval(5)}, Interval(1))
