from __future__ import annotations
from typing import Set, List
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Interval:
    degree: int
    modifier: int = 0


@dataclass(frozen=True)
class Chord:
    root: int
    components: Set[Interval]
    bass: Interval

    def to_label(self, vocabulary_name: str) -> int:
        vocabulary = vocabularies[vocabulary_name]

        # 'no chord' is always labeled 0
        if self.root is None:
            return 0

        # if chord vocabulary is empty then consider root only
        if len(vocabulary) == 0:
            return 1 + self.root

        # match to chords with components, which are subsets of self components
        matching_chords = [c for c in vocabulary if c.components <= self.components]

        # consider bass notes only if there are inversions in chord_vocabulary
        if len({chord.bass for chord in vocabulary}) > 1:
            matching_chords = [c for c in matching_chords if c.bass == self.bass]

        # if there is no matches then return 'no chord'
        if len(matching_chords) == 0:
            return 0
        else:
            return (
                1
                + self.root
                + 12
                * vocabulary.index(
                    max(matching_chords, key=lambda chord: len(chord.components))
                )
            )

    @staticmethod
    def from_label(label: int, vocabulary_name) -> Chord:
        vocabulary = vocabularies[vocabulary_name]

        if label == 0:
            return Chord(None, None, None)

        root = (label - 1) % 12

        if len(vocabulary) == 0:
            return Chord(root, set(), Interval(1))

        pattern_chord = vocabulary[(label - 1) // 12]

        return Chord(
            root,
            {replace(interval) for interval in pattern_chord.components},
            replace(pattern_chord.bass),
        )

    def shift(self, shift: int) -> Chord:
        if self.root is None:
            return replace(self)
        return Chord(
            (self.root + shift) % 12,
            {replace(interval) for interval in self.components},
            replace(self.bass),
        )


@dataclass(frozen=True)
class ChordOccurence:
    start: float
    stop: float
    chord: Chord

    def to_label_occurence(self, vocabulary_name: str):
        return LabelOccurence(
            self.start, self.stop, self.chord.to_label(vocabulary_name)
        )


@dataclass(frozen=True)
class LabelOccurence:
    start: float
    stop: float
    label: int


vocabularies = {
    "root_only": [],
    "maj_min": [
        Chord(0, {Interval(1), Interval(3), Interval(5)}, Interval(1)),
        Chord(0, {Interval(1), Interval(3, -1), Interval(5)}, Interval(1)),
    ],
}


def csr(reals: List[LabelOccurence], preds: List[LabelOccurence]):
    reals = [("real", occurence) for occurence in reals]
    reals.append(("real", LabelOccurence(reals[-1][1].stop, np.inf, np.nan)))
    preds = [("pred", occurence) for occurence in preds]
    preds.append(("pred", LabelOccurence(preds[-1][1].stop, np.inf, np.nan)))
    occurences = list(sorted(reals + preds, key=lambda c: c[1].start))
    last_label = {"real": np.nan, "pred": np.nan}
    common_label_time = 0
    for (a_type, a), (b_type, b) in zip(occurences[:-1], occurences[1:]):
        last_label[a_type] = a.label
        if last_label["real"] == last_label["pred"]:
            common_label_time += b.start - a.start
    return common_label_time / (reals[-1][1].start - reals[0][1].start)
