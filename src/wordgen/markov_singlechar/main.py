import pickle
import textwrap
from pathlib import Path
import random

import numpy as np

from wordgen.data import deu_mixed,nld_mixed,eng_wik, fin_wiki
from wordgen.markov_singlechar.extract import get_transition_distributions_multi, get_corpora_string
from wordgen.markov_singlechar.predict import PredictionParams, predict_n

# constants to tweak
MAX_WINDOW = 5
ALPHABET = list("abcdefghijklmnopqrstuvwxyzäöüß ") # everything not mentioned will be deleted
TRANSITIONS_CACHE = Path("transition_dict.pkl")
DICTIONARIES = [deu_mixed,nld_mixed,eng_wik,fin_wiki]
rng = random.SystemRandom()
# How much weight to give the preceding 1, 2 ,3, ... characters
distribution_weights = np.array([0.1,0.4,1.,0.8,0.3
                                 ])
N = 5000

if __name__ == "__main__":
    if TRANSITIONS_CACHE.exists():
        with TRANSITIONS_CACHE.open("rb") as f:
            transitions = pickle.load(f)
    else:
        corpus = get_corpora_string(*DICTIONARIES, alphabet=ALPHABET)
        transitions = get_transition_distributions_multi(corpus, ALPHABET, MAX_WINDOW)
        with TRANSITIONS_CACHE.open("wb") as f:
            pickle.dump(dict(transitions), f)

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