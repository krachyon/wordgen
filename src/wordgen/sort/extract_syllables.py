import pyphen
from pathlib import Path
import re
import multiprocessing as mp
import itertools
import functools

from wordgen.data import fin_wiki, eng_wik, deu_mixed, finnish_hyp

allowed = re.compile("^[a-zA-ZäöüÄÖÜ]+$")
vowels = re.compile("[aeiouäöüy]")


dicts = [pyphen.Pyphen(lang="de_de"), pyphen.Pyphen(lang="en_en") , pyphen.Pyphen(filename=finnish_hyp)]
words = [Path(p).read_text().splitlines() for p in [fin_wiki, eng_wik, deu_mixed]]

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
    for splitter, wordslist in zip(dicts, words):
        sets = pool.map(functools.partial(get_syllables, splitter=splitter), itertools.batched(wordslist, 10))
        for new_syllables in sets:
            syllables |= new_syllables

Path("syllables.txt").write_text("\n".join(list(syllables)))

