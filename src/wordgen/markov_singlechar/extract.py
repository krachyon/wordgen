from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from tqdm.auto import trange, tqdm
import re
import multiprocessing as mp

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


TransitionsT = dict[str, np.ndarray]


def get_transition_distributions(corpus: str, alphabet: list[str], max_window: int, position:int = 0) -> dict[str, np.ndarray]:
    distributions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(alphabet), dtype=np.uint32))

    index_lut = dict(zip(alphabet, range(len(alphabet))))

    for index in trange(max_window, len(corpus), position=position):
        chunk = corpus[index-max_window:index]
        current_char = corpus[index]
        for size in range(0,max_window):
            distributions[chunk[size:]][index_lut[current_char]] += 1

    return {key: dist.astype(np.float32)/np.float32(np.sum(dist)) for key, dist in distributions.items()}

def get_transition_distributions_multi(corpus: str, alphabet: list[str], max_window: int) -> TransitionsT:

    batch_length =  len(corpus)//mp.cpu_count()//2
    text_batches = [corpus[step:batch_length+step] for step in range(0, len(corpus), batch_length)]
    with mp.Pool() as pool:
        futures = [pool.apply_async(get_transition_distributions, args=(batch, alphabet, max_window, position))
                   for position, batch in enumerate(text_batches)]
        results = [future.get() for future in futures]

    distributions = dict()
    all_states = set(key for partial_transitions in results for key in partial_transitions.keys())
    for state in tqdm(all_states):
        distributions[state] = np.mean(
            [partial_transitions[state] for partial_transitions in results if state in partial_transitions],
            axis=0)

    return distributions