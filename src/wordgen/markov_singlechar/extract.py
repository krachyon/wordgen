import typing
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from tqdm.auto import trange, tqdm
import re
import multiprocessing as mp


def get_corpora_string(*paths: Path, alphabet: list[str]) -> str:
    """
    Given a bunch of paths, read the text from each and delete everything not in `alphabet`
    All text files are randomized and then truncated such that each has the same length as the shortest one.
    This should ensure that all inputs contribute roughly equally.

    :return: concatenated string of `alphabet`
    """
    corpora = [Path(p).read_text().splitlines() for p in paths]
    for corpus in corpora:
        random.shuffle(corpus)

    not_alphabet = re.compile(fr"[^{"".join(alphabet)}]")

    min_len = min(len(corpus) for corpus in corpora)
    words = [word for corpus in corpora for word in corpus[:min_len]]
    hugeass_string = " ".join(words).lower()

    hugeass_string = re.sub(not_alphabet, "", hugeass_string)
    return hugeass_string


DistributionT = np.ndarray[tuple[int,], np.dtype[np.float32]]
TransitionsT = dict[str, DistributionT]


def get_transition_distributions(corpus: str, alphabet: list[str], max_window: int, position: int = 0) -> TransitionsT:
    """
    Extract the probability of each letter in `alphabet` following the preceding letters from a long string.
    :param corpus: the input string from which to build model
    :param alphabet: set of allowed letters
    :param max_window: how many letters to consider at most.
    :param position: used for displaying multiple progress bar if called concurrently
    :return: a mapping of preceding letters to the probability distribution for the next letter.
             The distribution is expressed as a float-array that corresponds to alphabet.
    """
    distributions: TransitionsT = defaultdict(lambda: np.zeros(len(alphabet), dtype=np.uint32))

    index_lut = dict(zip(alphabet, range(len(alphabet))))

    for index in trange(max_window, len(corpus), position=position):
        chunk = corpus[index - max_window:index]
        current_char = corpus[index]
        for size in range(0, max_window):
            distributions[chunk[size:]][index_lut[current_char]] += 1

    return {key: typing.cast(DistributionT, (dist / np.sum(dist)).astype(np.float32))
            for key, dist in list(distributions.items())}


def aggregate_distributions(partial_distributions: list[TransitionsT], state: str) -> DistributionT:
    """combine multiple distributions from a transition dictionary for key `state`"""
    return np.mean(
        [distribution[state] for distribution in partial_distributions if state in distribution],
        axis=0)


def get_transition_distributions_multi(corpus: str, alphabet: list[str], max_window: int) -> TransitionsT:
    """
    Parallelized version of get_transition_distributions
    """
    batch_length = len(corpus) // mp.cpu_count() // 2
    text_batches = [corpus[step:batch_length + step] for step in range(0, len(corpus), batch_length)]
    with mp.Pool() as pool:
        futures = [pool.apply_async(get_transition_distributions, args=(batch, alphabet, max_window, position))
                   for position, batch in enumerate(text_batches)]
        results = [future.get() for future in futures]

    distributions = dict()
    all_states = set(key for partial_transitions in results for key in partial_transitions.keys())

    for state in tqdm(all_states):
        distributions[state] = aggregate_distributions(results, state)

    return distributions
