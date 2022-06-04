import lark
from src.annotation_parser.chord_model import Interval, Chord, ChordOccurence


class LabFileTransformer(lark.Transformer):

    _naturals = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    _modifiers = {"b": -1, "#": +1}
    _shorthands = {
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

    def NATURAL(self, token):
        return self._naturals[token]

    def MODIFIER(self, token):
        return self._modifiers[token]

    def INTEGER(self, token):
        return int(token)

    def FLOAT(self, token):
        return float(token)

    def pitchname(self, children):
        return sum(children) % 12

    def interval(self, children):
        return Interval(children[-1], sum(children[:-1]))

    def components(self, children):
        return (
            {c.children[0] for c in children if c.data == "interval_to_add"},
            {c.children[0] for c in children if c.data == "interval_to_remove"},
        )

    def chord(self, children):
        if len(children) == 4:
            pitchname, shorthand, intervals_to_add_and_remove, bass = children
            components = set(self._shorthands[shorthand])
            if intervals_to_add_and_remove is not None:
                to_add, to_remove = intervals_to_add_and_remove
                components = (components | to_add) - to_remove
        elif len(children) == 3:
            pitchname, (to_add, to_remove), bass = children
            components = to_add - to_remove
        elif len(children) == 2:
            pitchname, bass = children
            components = set(self._shorthands["maj"])
        else:
            pitchname, components, bass = None, None, None
        if bass is None:
            bass = Interval(1, 0)
        return Chord(pitchname, components, bass)

    def line(self, children):
        return ChordOccurence(children[0], children[1], children[2])

    def labfile(self, children):
        return children
