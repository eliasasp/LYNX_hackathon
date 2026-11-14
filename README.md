# Hackathon code example

## Vår modell: merged + pairs (vikter 0.6/0.4)

```
.venv/bin/python main.py                 # insample → output/submission.csv + nyckeltal
.venv/bin/python main.py --plot          # + dashboard, positions- och riskplottar i fönster
.venv/bin/python main.py --save-plots    # + plottarna som PNG i output/
.venv/bin/python main.py --prices full_prices.csv --macro full_macro.csv   # inlämning på hela datan
```

Test för framåtblickande data och export till en fristående inlämningsmapp:

```
.venv/bin/python leakage_test.py          # positionerna får inte ändras när senare data läggs till
.venv/bin/python package_submission.py    # → lynx_submission/ och output/lynx_submission.zip
```

`output/submission.csv` har samma format som exemplets `results.csv`: `AsOfDate` + en kolumn per instrument.

| Fil | Roll |
|---|---|
| `main.py` | kör strategierna, kombinerar dem, skriver inlämningsfilen och plottar |
| `strategy/` | merged: Linns trend + makro (`submission.py` m.fl.) |
| `ma_crossover/final_model.py` | merged: Davids MA10/250-trend |
| `pairs_trading/pairs_final.py` | pairs trading (kan även köras fristående för egna plottar) |
| `evaluation.py` | Lynx nyckeltal och dashboard |

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
