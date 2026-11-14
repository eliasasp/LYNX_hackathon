"""Plottar alla serier i close_prices_insample.csv och macro_data_insample.csv:
   1) nivåer, en panel per grupp   2) korrelationsmatris för dagliga förändringar.

Kör:  python plot_all.py            (visar figurerna)
      python plot_all.py plot.png   (sparar plot.png och plot_corr.png)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

D = Path(__file__).parent
rd = lambda f: pd.read_csv(D / f, index_col="AsOfDate", parse_dates=True)
P, M = rd("close_prices_insample.csv"), rd("macro_data_insample.csv")

# grupp = suffix för priser (bonds/stocks/titan_fx, resten = råvaror), prefix före "|" för makro
grp = lambda c: next((s for s in ("bonds", "stocks", "titan_fx") if c.endswith("_" + s)), "commodities")
groups = {f"{g} (pris)": P.loc[:, [grp(c) == g for c in P]] for g in ("bonds", "stocks", "titan_fx", "commodities")}
groups |= {m.upper(): M.filter(like=m + "|") for m in ("gdp", "cpi")}


def finish(fig, suffix=""):
    fig.tight_layout()
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        fig.savefig(out.with_name(out.stem + suffix + out.suffix), dpi=130)


# 1) nivåer (log-axel: serierna startar på ~1 men spänner över flera storleksordningar)
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
for ax, (name, df) in zip(axes.flat, groups.items()):
    df.plot(ax=ax, logy=True, lw=1, title=name, xlabel="", legend=False)
    ax.legend(fontsize=7, ncol=2 if len(df.columns) > 6 else 1, loc="upper left", framealpha=0.6)
    ax.yaxis.set_major_formatter("{x:g}")
    ax.yaxis.set_minor_formatter("{x:g}" if df.max().max() / df.min().min() < 10 else "")  # få etiketter när spannet är stort
finish(fig)

# 2) korrelationer mellan dagliga förändringar (nivåkorrelation mellan trendande serier är missvisande)
X = pd.concat([g.reindex(P.index) for g in groups.values()], axis=1).diff().dropna()   # makro (kalenderdagar) läggs på prisernas handelsdagar
C = X.corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C)), C.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(C)), C.columns, fontsize=7)
for b in pd.Series([len(g.columns) for g in groups.values()]).cumsum()[:-1]:   # linjer mellan grupperna
    ax.axhline(b - 0.5, c="k", lw=0.8), ax.axvline(b - 0.5, c="k", lw=0.8)
fig.colorbar(im, ax=ax, shrink=0.8, label="korrelation (dagliga förändringar)")
ax.set_title(f"Korrelationer, {X.index[0]:%Y-%m-%d} – {X.index[-1]:%Y-%m-%d}")
finish(fig, "_corr")

if len(sys.argv) == 1:
    plt.show()
