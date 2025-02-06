import textwrap
from pathlib import Path
import random
from diskcache import Cache
import numpy as np

from wordgen import CACHE_DIR
from wordgen.data import deu_mixed, eng_wik, fin_wiki
from wordgen.markov_singlechar.extract import get_transition_distributions_multi, get_corpora_string, TransitionsT
from wordgen.markov_singlechar.predict import PredictionParams, predict_n

# constants to tweak
MAX_WINDOW = 5  # predict next character from (up to) the previous N characters
ALPHABET = list("abcdefghijklmnopqrstuvwxyzäöüß '")  # everything not mentioned will be deleted before consuming text
TEXT_FILES = [deu_mixed, eng_wik, fin_wiki]  #
rng = random.SystemRandom()  # "secure" or fixed seed?
distribution_weights = np.array([0.2, 0.5, 1., 0.5, 0.]) # How much weight to give the preceding 1, 2, 3,... characters
N = 5000  # how many characters to generate

cache = Cache(CACHE_DIR)


@cache.memoize(name=None)
def get_transitions(dictionaries: list[Path], alphabet: list[str], max_window: int) -> TransitionsT:
    corpus = get_corpora_string(*dictionaries, alphabet=ALPHABET)
    return get_transition_distributions_multi(corpus, alphabet, max_window)


if __name__ == "__main__":
    transitions = get_transitions(TEXT_FILES, ALPHABET, MAX_WINDOW)
    initial = rng.choices(ALPHABET, k=MAX_WINDOW)
    # initial = list("hello")

    filled = predict_n(initial, n=N, params=PredictionParams(
        rng=rng,
        transitions=transitions,
        alphabet=ALPHABET,
        max_window=MAX_WINDOW,
        distribution_weights=distribution_weights
    ))

    print("\n".join(textwrap.wrap("".join(filled), width=100)))
