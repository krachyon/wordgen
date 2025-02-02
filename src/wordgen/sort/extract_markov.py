import itertools
import random
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm

from wordgen.data import german, english, finnish

N = 3

corpora = [Path(p).read_text().splitlines() for p in [german, english, finnish]]
for corpus in corpora:
    random.shuffle(corpus)

min_len = min(len(corpus) for corpus in corpora)
words = [word for corpus in corpora for word in corpus[:min_len]]
hugeass_string = " ".join(words).lower().replace("-","")


state_set: set[str] = set()

def sliding_window(iterable_size: int, chunk_size: int):
    return zip(range(0,iterable_size-chunk_size), range(chunk_size,iterable_size))


for low, high in tqdm(list(sliding_window(len(hugeass_string), N))):
    state_set.add(hugeass_string[low:high])

states = list(sorted(state_set))

indices = dict(zip(states, range(len(states))))
print(len(states))


def partial_matrix(index_tuples):
    transition_matrix = np.zeros((len(states),len(states)), dtype=np.float64)

    for low, high in index_tuples:
        previous = hugeass_string[low:(high-N)]
        current = hugeass_string[(high-N):high]
        transition_matrix[indices[previous], indices[current]] += 1
    return transition_matrix

index_pairs = list(sliding_window(len(hugeass_string), 2 * N))
index_batches = list(itertools.batched(index_pairs, len(index_pairs)//mp.cpu_count()*2))

assert len(index_batches) < 20
with mp.Pool() as pool:
    matrices = pool.map(partial_matrix, index_batches)

transition_matrix = np.sum(matrices, axis=0)

print(transition_matrix)
np.save(f"states_{N}.npy", states)
np.save(f"transition_matrix_{N}.npy", transition_matrix)
