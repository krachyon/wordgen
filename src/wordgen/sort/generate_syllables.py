from pathlib import Path
import random

rng = random.SystemRandom()
syllables = Path("../../../syllables.txt").read_text().splitlines()

TARGET = 10
for i in range(100):
    passwd = ""

    while len(passwd) < TARGET:
        passwd += rng.choice(syllables)

    print(passwd)

