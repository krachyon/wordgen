from random import SystemRandom

import numpy as np

N_WINDOW = 3

transition_matrix = np.load(f"transition_matrix_{N_WINDOW}.npy")
states= np.load(f"states_{N_WINDOW}.npy").tolist()

TARGET = 250

rng = SystemRandom()

def gen() -> str:
    out = rng.choice(states)
    current_chunk = out
    while len(out) < TARGET:
        next_chunk = rng.choices(states, weights=transition_matrix[states.index(current_chunk)])[0]
        out += next_chunk
        current_chunk = next_chunk
    return out

for i in range(10):
    print(gen())
