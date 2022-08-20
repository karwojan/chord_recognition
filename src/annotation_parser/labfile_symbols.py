from src.annotation_parser.chord_model import Interval

natural_symbols = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

modifier_symbols = {"b": -1, "#": +1}

shorthand_symbols = {
    "1": {Interval(1)},
    "5": {Interval(1), Interval(5)},
    "maj": {Interval(1), Interval(3), Interval(5)},
    "min": {Interval(1), Interval(3, -1), Interval(5)},
    "dim": {Interval(1), Interval(3, -1), Interval(5, -1)},
    "aug": {Interval(1), Interval(3), Interval(5, 1)},
    "maj7": {Interval(1), Interval(3), Interval(5), Interval(7)},
    "min7": {Interval(1), Interval(3, -1), Interval(5), Interval(7, -1)},
    "7": {Interval(1), Interval(3), Interval(5), Interval(7, -1)},
    "dim7": {Interval(1), Interval(3, -1), Interval(5, -1), Interval(7, -2)},
    "hdim7": {Interval(1), Interval(3, -1), Interval(5, -1), Interval(7, -1)},
    "minmaj7": {Interval(1), Interval(3, -1), Interval(5), Interval(7)},
    "maj6": {Interval(1), Interval(3), Interval(5), Interval(6)},
    "min6": {Interval(1), Interval(3, -1), Interval(5), Interval(6)},
    "9": {Interval(1), Interval(3), Interval(5), Interval(7, -1), Interval(9)},
    "maj9": {Interval(1), Interval(3), Interval(5), Interval(7), Interval(9)},
    "min9": {Interval(1), Interval(3, -1), Interval(5), Interval(7, -1), Interval(9)},
    "sus2": {Interval(1), Interval(2), Interval(5)},
    "sus4": {Interval(1), Interval(4), Interval(5)},
    "11": {Interval(1), Interval(3), Interval(5), Interval(7, -1), Interval(9), Interval(11)},
    "maj11": {Interval(1), Interval(3), Interval(5), Interval(7), Interval(9), Interval(11)},
    "min11": {Interval(1), Interval(3, -1), Interval(5), Interval(7, -1), Interval(9), Interval(11)},
    "13": {Interval(1), Interval(3), Interval(5), Interval(7, -1), Interval(9), Interval(11), Interval(13)},
    "maj13": {Interval(1), Interval(3), Interval(5), Interval(7), Interval(9), Interval(11), Interval(13)},
    "min13": {Interval(1), Interval(3, -1), Interval(5), Interval(7, -1), Interval(9), Interval(11), Interval(13)},
}
