# Get random words
- install [uv](https://docs.astral.sh/uv/getting-started/installation/#pypi)
- edit constants in python src/wordgen/markov_singlechar/main.py   
  (Delete cache .pkl if you change extraction parameters to re-extract)
```shell
- uv sync
```
```shell
uv run python src/wordgen/markov_singlechar/main.py   
```