# Get random words
- install [uv](https://docs.astral.sh/uv/getting-started/installation/#pypi)
- edit constants in python src/wordgen/markov_singlechar/main.py
```shell
- uv sync
```
```shell
uv run python src/wordgen/markov_singlechar/main.py   
```

The output of extraction from text will be stored in src/wordgen/cache. In case you change anything
about the extraction process that isn't picked up due to memoization, you may have to delete the cache folder