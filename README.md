## Running the exercises with Poetry

- Install Poetry:

```
curl -sSL https://install.python-poetry.org/ | python3 -
```

- Uninstall Poetry:

```
curl -sSL https://install.python-poetry.org/ | python3 - --uninstall
```

- Install dependencies:

```
poetry install
```

- Add a new package:

```
poetry add <packageName>
```

- Run script:

```
poetry run python exs/$(ex)/app/main.py
```

- Run notebooks:
```
poetry run python -m ipykernel install --user --name=ml
poetry run jupyter lab --NotebookApp.token=ml
```
