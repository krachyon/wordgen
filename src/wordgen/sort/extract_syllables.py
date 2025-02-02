import pyphen
from pathlib import Path
import re
import multiprocessing as mp
import itertools
import functools

allowed = re.compile("^[a-zA-ZäöüÄÖÜ]+$")
vowels = re.compile("[aeiouäöüy]")


dicts = [pyphen.Pyphen(lang="de_de"), pyphen.Pyphen(lang="en_en") , pyphen.Pyphen(filename="../data/hyph_fi_FI.dic")]
words = [Path(p).read_text().splitlines() for p in ["german.txt", "english.txt", "finnish.txt"]]

def get_syllables(wordlist: list[str], splitter: pyphen.Pyphen) -> set[str]:
    syllables = set()
    for word in wordlist:
        if not re.match(allowed, word):
            continue
        for syllable in splitter.inserted(word).split("-"):
            syllable = syllable.lower()
            if len(syllable) < 6 and re.search(vowels, syllable):
                syllables.add(syllable)
    return syllables

syllables = set()
with mp.Pool() as pool:
    for splitter, words in zip(dicts, words):
        sets = pool.map(functools.partial(get_syllables, splitter=splitter), itertools.batched(words, 10))
        for new_syllables in sets:
            syllables |= new_syllables

Path("../../../syllables.txt").write_text("\n".join(list(syllables)))

