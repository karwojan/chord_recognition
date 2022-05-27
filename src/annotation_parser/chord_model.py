from typing import Set, ClassVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Pitchname:
    naturals: ClassVar = ["A", "B", "C", "D", "E", "F", "G"]
    natural: str
    modifier: int = 0


@dataclass(frozen=True)
class Interval:
    degree: int
    modifier: int = 0


@dataclass(frozen=True)
class Chord:
    root: Pitchname
    components: Set[Interval]
    bass: Interval
