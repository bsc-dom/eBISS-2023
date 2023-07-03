from typing import Counter

from dataclay import DataClayObject, activemethod


class WordCount(DataClayObject):
    counter: Counter[str]

    def __init__(self):
        self.counter = Counter()

    @activemethod
    def __getitem__(self, item: str) -> int:
        return self.counter[item]

    @activemethod
    def __setitem__(self, item: str, value: int):
        self.counter[item] = value

    @activemethod
    def most_common(self, n: int) -> list[tuple[str, int]]:
        return self.counter.most_common(n)

    @activemethod
    def total(self) -> int:
        return self.counter.total()

    @activemethod
    def __iadd__(self, other: "WordCount") -> "WordCount":
        """Perform in-place update of this WordCount object."""
        self.counter.update(other.counter)
        return self
    
    @activemethod
    def __add__(self, other: "WordCount") -> "WordCount":
        """Create a new WordCount object combining self and other."""
        wc = WordCount()
        wc.counter.update(other.counter)
        wc.counter.update(self.counter)
        return wc
