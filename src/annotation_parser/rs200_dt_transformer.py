import lark
from src.annotation_parser.chord_model import Interval, Chord


class Rs200DtTransformer(lark.Transformer):

    def FLOAT(self, token):
        return float(token)

    def INTEGER(self, token):
        return int(token)

    def major_triad(self, children):
        return [Interval(1), Interval(3), Interval(5)], children[-1]

    def minor_triad(self, children):
        return [Interval(1), Interval(3, -1), Interval(5)], children[-1]

    def triad(self, children):
        intervals, rn = children[0]

        modifier = children[1]
        if modifier == "o" or modifier == "b5":
            intervals[2] = Interval(5, -1)
        elif modifier == "a":
            intervals[2] = Interval(5, +1)
        elif modifier == "s4":
            intervals[1] = Interval(4)
        elif modifier == "11":
            intervals += [Interval(7, -1), Interval(9), Interval(11)]

        inversion = children[2]
        if inversion == "6":
            bass = intervals[1]
        elif inversion == "64":
            bass = intervals[2]
        else:
            bass = intervals[0]

        return intervals, bass

    def seventh(self, children):
        intervals, rn = children[0]
        modifier = children[1]
        inversion = children[2]

        if intervals[1] == Interval(3):
            if modifier == "d" or rn == "V":
                intervals.append(Interval(7, -1))
            else:
                intervals.append(Interval(7))
        elif intervals[1] == Interval(3, -1):
            if modifier == "h":
                intervals[2] = Interval(5, -1)
                intervals.append(Interval(7, -1))
            elif modifier == "x":
                intervals[2] = Interval(5, -1)
                intervals.append(Interval(7, -2))
            else:
                intervals.append(Interval(7, -1))

        if inversion == "7":
            bass = intervals[0]
        elif inversion == "65":
            bass = intervals[1]
        elif inversion == "43":
            bass = intervals[2]
        elif inversion == "42":
            bass = intervals[3]

        return intervals, bass

    def chord(self, children):
        return children[0]

    def line(self, children):
        start_time = children[0]
        if len(children) == 7:
            intervals, bass = children[2]
            root = children[6]
            chord = Chord(root, intervals, bass)
            return start_time, chord
        else:
            return start_time, None

    def cltfile(self, children):
        return [(a[0], b[0], a[1]) for a, b in zip(children[:-1], children[1:])]
