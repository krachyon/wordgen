import pickle
from pathlib import Path
import random

from wordgen.data import german, english, finnish
from wordgen.markov_singlechar.extract import get_transitions, get_corpora_string
from wordgen.markov_singlechar.predict import predict_n, PredictionParams

MAX_WINDOW = 5
ALPHABET = list("abcdefghijklmnopqrstuvwxyzäöüß ")
TRANSITIONS_CACHE = Path("transition_dict.pkl")
rng = random.SystemRandom()

if __name__ == "__main__":
    if TRANSITIONS_CACHE.exists():
        with TRANSITIONS_CACHE.open("rb") as f:
            transitions = pickle.load(f)
    else:
        corpus = get_corpora_string(german, english, finnish, alphabet=ALPHABET)
        transitions = get_transitions(corpus, ALPHABET, MAX_WINDOW)
        with TRANSITIONS_CACHE.open("wb") as f:
            pickle.dump(dict(transitions), f)

    initial = rng.choices(ALPHABET, k=MAX_WINDOW)
    # initial = list("hello")

    filled = predict_n(initial, n=5000, params=PredictionParams(
        rng=rng, transitions=transitions, alphabet=ALPHABET, max_window=MAX_WINDOW)
                       )

    print("".join(filled).replace(" ", "\n"))