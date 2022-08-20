import lark
from src.annotation_parser.chord_model import Interval, Chord, ChordOccurence
from src.annotation_parser.labfile_symbols import modifier_symbols, natural_symbols, shorthand_symbols


class LabFileTransformer(lark.Transformer):

    def NATURAL(self, token):
        return natural_symbols[token]

    def MODIFIER(self, token):
        return modifier_symbols[token]

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
            components = set(shorthand_symbols[shorthand])
            if intervals_to_add_and_remove is not None:
                to_add, to_remove = intervals_to_add_and_remove
                components = (components | to_add) - to_remove
        elif len(children) == 3:
            pitchname, (to_add, to_remove), bass = children
            components = to_add - to_remove
        elif len(children) == 2:
            pitchname, bass = children
            components = set(shorthand_symbols["maj"])
        else:
            pitchname, components, bass = None, None, None
        if bass is None:
            bass = Interval(1, 0)
        return Chord(pitchname, components, bass)

    def line(self, children):
        return ChordOccurence(children[0], children[1], children[2])

    def labfile(self, children):
        return children
