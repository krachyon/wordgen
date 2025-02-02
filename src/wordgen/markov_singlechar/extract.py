from collections import defaultdict
from pathlib import Path
import random
from tqdm.auto import trange
import re

def get_corpora_string(*paths, alphabet: list[str]) -> str:
    corpora = [Path(p).read_text().splitlines() for p in paths]
    for corpus in corpora:
        random.shuffle(corpus)

    not_alphabet = re.compile(fr"[^{"".join(alphabet)}]")

    min_len = min(len(corpus) for corpus in corpora)
    words = [word for corpus in corpora for word in corpus[:min_len]]
    hugeass_string = " ".join(words).lower()

    hugeass_string = re.sub(not_alphabet, "", hugeass_string)
    return hugeass_string

TransitionsT = dict[str, dict[str, int]]

def get_transitions(corpus: str, alphabet: list[str], max_window: int) -> TransitionsT:
    def factory():
        return dict.fromkeys(alphabet, 0)

    transitions: TransitionsT = defaultdict(factory)

    for index in trange(max_window,len(corpus)):
        chunk = corpus[index-max_window:index]
        current_char = corpus[index]
        for size in range(0,max_window):
            transitions[chunk[size:]][current_char] += 1

    return transitions


