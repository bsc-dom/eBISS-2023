import string
import unicodedata

from dataclay import DataClayObject, activemethod
from .wordcount import WordCount


clean_chars = string.ascii_lowercase.encode("ascii")


def _cleanup_word(token):
    """Take some unicode token, and return a cleaned up lowercase version with no special characters."""
    ret = unicodedata.normalize("NFKD", token.lower()).encode("ascii", "ignore")
    ret = "".join(chr(b) for b in ret if b in clean_chars)
    return ret


class TextFile(DataClayObject):
    filename: str
    words: list[str]

    def __init__(self, filename: str):
        """Prepare a TextFile, but do not read it (yet)."""
        self.filename = filename
        self.words = list()

    @activemethod
    def read_file(self):
        with open(self.filename, "r") as f:
            for line in f:
                self.words.extend(
                    _cleanup_word(token) for token in line.split()
                )

    @activemethod
    def wordcount(self) -> WordCount:
        wc = WordCount()
        for word in self.words:
            wc[word] += 1

        return wc
