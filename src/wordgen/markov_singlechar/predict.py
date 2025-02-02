import random
from tqdm.auto import trange
import numpy as np
import dataclasses

from wordgen.markov_singlechar.extract import TransitionsT

@dataclasses.dataclass
class PredictionParams:
    transitions: TransitionsT
    max_window: int
    alphabet: list[str]
    rng: random.Random
    distribution_weights: np.ndarray = dataclasses.field(default_factory= lambda: np.array([0.,0.1,0.3,0.8, 1.]))
    def __post_init__(self):
        if len(self.distribution_weights) != self.max_window:
            raise ValueError(f"Max weights must have {self.max_window=} elements")

def predict_n(chars: list[str], n: int, params: PredictionParams) -> list[str]:
    key_order = sorted(params.alphabet)

    for _iter in trange(n):
        distributions = []
        for idx in range(-1, -1-params.max_window, -1):
            chunk = "".join(chars[idx:])
            if chunk in params.transitions:
                transition = params.transitions[chunk]
                distribution = np.asarray([transition[key] for key in key_order], dtype=np.float64)
                #distribution = np.asarray(list(transitions[chunk].values()), dtype=np.float64)
                distribution /= distribution.sum()
                distributions.append(distribution)
            else:
                distributions.append(np.zeros(len(params.alphabet), dtype=np.float64))

        distribution = np.average(distributions, axis=0, weights=params.distribution_weights)
        if np.sum(distribution) == 0:
            distribution = np.ones(len(params.alphabet), dtype=np.float64)
        chars.append(params.rng.choices(key_order, weights=distribution, k=1)[0])
    return chars

