# Hackathon code example

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then in this folder:

```
uv sync                # create .venv and install dependencies
uv run jupyter lab     # open the notebooks in jupyter
uv add scikit-learn    # add a package
```

In VS Code, open a notebook and select the `.venv` kernel.

## Positions format

`evaluation.py` takes Polars frames in wide format: an `AsOfDate` column plus one column per
instrument.

```
AsOfDate    diamonds  butter  cars  cucumber  bananas
2000-01-03  0.5       -1.0    0.0   0.2       1.3
```

You can build positions in NumPy or pandas and convert at the end.

From pandas, with dates in an index named `AsOfDate`:

```python
positions = pl.from_pandas(df.reset_index())
```

From NumPy, with a `(dates, instruments)` array:

```python
positions = pl.DataFrame(array, schema=instruments).with_columns(AsOfDate=dates)
```
