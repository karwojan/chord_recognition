from typing import Set, ClassVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    degree: int
    modifier: int = 0


@dataclass(frozen=True)
class Chord:
    root: int
    components: Set[Interval]
    bass: Interval
