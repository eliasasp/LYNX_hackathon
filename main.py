"""Kombinerad modell: tre strategier körs på samma data och vägs ihop med lika risk.

    david  ma_crossover/final_model.py   MA10/250-trend på 18 instrument
    linn   strategy/submission.py        trend (3 tidsskalor) + makro (BNP/KPI), lika risk per tillgångsslag
    pairs  pairs_trading/pairs_final.py  pairs trading med kointegration
    merged Linns modell där trenddelen = Linns hysteres-momentum + Davids MA10/250 (DAVID_TREND_SHARE);
           Linns egen MA-crossover är borttagen

Gemensam tidshorisont för alla strategier:
    träning   < VALIDATION_START   ingen handel (alla positioner 0)
    validering ≥ VALIDATION_START  handel; alla nyckeltal räknas bara här

David och Linn är walk-forward (varje dag används bara data t.o.m. den dagen), så de räknas på hela
historiken och nollas före VALIDATION_START. Pairs skattar urval, alpha/beta och half-life på de sista
PAIRS_TRAIN_YEARS åren före VALIDATION_START och handlar därefter.

Positionerna läggs på hela universumet (0 där en strategi inte handlar) och kombineras med Linns
combine_positions: varje strategi skalas walk-forward till samma P&L-vol, bara med historisk P&L, och
vägs enligt WEIGHTS. Därefter portföljöverlägg på hela portföljen (portfolio_scale): villkorad volbroms
(VOL_BRAKE) och volmål (TARGET_VOL). Ingen strategi skiftas här; evaluation.py lägger på t+2-laggen.

Kör från repo-roten:
    .venv/bin/python main.py                       # insample-data, skriver output/main_positions.csv + nyckeltal
    .venv/bin/python main.py --plot                # visar dashboard + positions- och riskplottar i fönster
    .venv/bin/python main.py --save-plots          # sparar samma plottar som PNG i stället
    .venv/bin/python main.py --prices full.csv --macro full_macro.csv --output positions.csv

Allt skrivs till mappen output/ (ett --output med egen mapp eller absolut sökväg används som det är).
--plot visar och --save-plots sparar, bredvid --output (för main_positions.csv):
    main_positions.png             evaluation.py:s dashboard för den kombinerade modellen
    main_positions_tillgangar.png  riskposition per tillgång över tiden, uppdelad per strategi + kombinerad
    main_positions_exponering.png  brutto riskexponering per strategi och tillgångsslag, snitt per tillgång
    main_positions_risk.png        portföljvol (realiserad och ex ante) och riskbidrag per strategi/tillgångsslag
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "strategy"))  # Linns moduler importerar varandra platt (from combine import ...)

import evaluation  # noqa: E402
import submission as linn  # noqa: E402
from combine import combine_positions, risk_scale, strategy_pnl  # noqa: E402
from overlays import conditional_vol_target  # noqa: E402
from lead_lag import lead_lag_positions, load_prices  # noqa: E402
from macro import asset_class, load_macro  # noqa: E402
from macro_joint import macro_joint_positions  # noqa: E402
from ma_crossover import final_model as david  # noqa: E402
from pairs_trading import pairs_final as pairs  # noqa: E402

DATE = "AsOfDate"
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)  # uppvärmningsdagar i Linns trend/makro

OUTPUT_DIR = ROOT / "output"  # alla filer som main.py skriver (positioner, plottar)
VALIDATION_START = date(2010, 1, 1)  # träning före (ingen handel), validering från och med detta datum
PAIRS_TRAIN_YEARS = 2                # pairs skattas på [VALIDATION_START - så många år, dagen före VALIDATION_START]

# Andel av risken per strategi (normaliseras till summa 1) för de strategier som körs.
#   linn     = Linns hela submission (trend + makro, gruppvikter, volbroms)
#   leadlag  = bara Linns lead-lag-komponent (LL_CONFIG i strategy/submission.py)
#   macro    = bara Linns makrokomponent (gemensam ridge, samma inställningar som i submission.py)
#   merged   = en modell: Linns submission där trenden är hennes hysteres + Davids MA10/250 (ingen Linn-MA)
#   standard (merged, pairs): 80 / 20 % av risken
WEIGHTS = {"merged": 0.8, "david": 0.4, "linn": 0.4, "leadlag": 1.0, "macro": 1.0, "pairs": 0.2}
DEFAULT_STRATEGIES = ("merged", "pairs")  # körs utan --strategies
DAVID_TREND_SHARE = 0.7  # i merged: Davids MA-andel av trendrisken (resten Linns hysteres på 100/250/500 dagar)
# Portföljöverlägg på hela den kombinerade portföljen (alla strategier, efter viktningen)
VOL_BRAKE = True          # villkorad volbroms: vol 20 d (EWMA) > linn.VOL_THRESHOLD × vol 250 d → skala ner till lång nivå
TARGET_VOL = 10.0         # volmål: årlig P&L-vol i evaluation.py:s enhet (positioner × prisförändring); None = av
TARGET_VOL_WINDOW = 250   # dagar realiserad portfölj-P&L som volmålet skattas på (bara historik)
KEY_FIGURES = ["adjusted_sharpe", "sharpe", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias"]


# ============================== POSITIONER PER STRATEGI ==============================
def _on_universe(positions: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Samma datum och instrument som prices; saknade värden blir 0."""
    inst = [c for c in prices.columns if c != DATE]
    positions = positions.with_columns(pl.col(DATE).cast(pl.Date))
    out = prices.select(DATE).join(positions, on=DATE, how="left")
    return out.select(DATE, *[(pl.col(c) if c in out.columns else pl.lit(0.0)).cast(pl.Float64).fill_null(0.0)
                              .fill_nan(0.0).alias(c) for c in inst])


def _no_trading_before(positions: pl.DataFrame, start: date) -> pl.DataFrame:
    inst = [c for c in positions.columns if c != DATE]
    return positions.with_columns(pl.when(pl.col(DATE) >= start).then(pl.col(c)).otherwise(0.0).alias(c) for c in inst)


def david_positions(prices: pl.DataFrame) -> pl.DataFrame:
    return david.make_positions(prices)


def linn_positions(prices: pl.DataFrame, macro_path: str | None) -> pl.DataFrame:
    return linn.build_positions(prices, macro_path)


def merged_positions(prices: pl.DataFrame, macro_path: str | None) -> pl.DataFrame:
    """Linns modell där trenddelen är hennes hysteres-momentum (hennes MA-crossover avstängd) plus Davids
    MA10/250. Davids positioner läggs på hela universumet (0 utanför hans 18 instrument) och allt går sedan
    genom Linns gruppvikter, makro och volbroms."""
    return linn.build_positions(prices, macro_path, extra_trend=_on_universe(david_positions(prices), prices),
                                extra_trend_share=DAVID_TREND_SHARE, use_ma_cross=False,
                                vol_brake=not VOL_BRAKE)  # volbromsen ligger då på hela portföljen i stället


def leadlag_positions(prices: pl.DataFrame) -> pl.DataFrame:
    return lead_lag_positions(prices, linn.LL_CONFIG)


def macro_positions(prices: pl.DataFrame, macro_path: str | None) -> pl.DataFrame:
    """Linns makrokomponent med hennes inställningar. Tillgångsslag med makroandel 0 i submission.py
    (MACRO_SHARE_BY_GROUP, dvs obligationer) får inga makropositioner, precis som i hennes modell."""
    if macro_path is None:
        raise ValueError("makrostrategin kräver --macro")
    raw = load_macro(macro_path, prices, lag=linn.MACRO_LAG)
    pos = macro_joint_positions(prices, raw, windows=linn.MACRO_WINDOWS, train_window=linn.MACRO_TRAIN_WINDOW,
                                ridge=linn.MACRO_RIDGE, max_pos=linn.MACRO_MAX_POS)
    off = [c for c in pos.columns if c != DATE and linn.MACRO_SHARE_BY_GROUP.get(asset_class(c), 1) <= 0]
    return pos.with_columns(pl.lit(0.0).alias(c) for c in off)


def pairs_positions(prices: pl.DataFrame) -> pl.DataFrame:
    """pairs_final arbetar i pandas med datumindex; positionerna är antal enheter per tillgång.
    Träningsperioden sätts via modulens globala TRAIN_START/TRAIN_END, som alla dess funktioner läser."""
    train_start = VALIDATION_START.replace(year=VALIDATION_START.year - PAIRS_TRAIN_YEARS)
    pairs.TRAIN_START = train_start.isoformat()
    pairs.TRAIN_END = (VALIDATION_START - timedelta(days=1)).isoformat()
    results, positions, trades = pairs.run_strategy(prices.to_pandas().set_index(DATE))
    traded = [f"{v['a']}/{v['b']}" for v in results if v["traded"]]
    print(f"    pairs tränad {pairs.TRAIN_START} – {pairs.TRAIN_END}: {len(traded)} av {len(results)} kandidatpar "
          f"klarar screeningen, {len(trades)} affärer: {', '.join(traded) or '-'}")
    return pl.from_pandas(positions.rename_axis(DATE).reset_index())


def build_all(prices: pl.DataFrame, macro_path: str | None) -> dict[str, pl.DataFrame]:
    """Kör alla strategier med vikt > 0 och lägger dem på hela universumet (hela historiken, före nollning)."""
    runners = {
        "merged": lambda: merged_positions(prices, macro_path),
        "david": lambda: david_positions(prices),
        "linn": lambda: linn_positions(prices, macro_path),
        "leadlag": lambda: leadlag_positions(prices),
        "macro": lambda: macro_positions(prices, macro_path),
        "pairs": lambda: pairs_positions(prices),
    }
    out = {}
    for name, run in runners.items():
        if WEIGHTS.get(name, 0) > 0:
            print(f"  kör {name} ...", flush=True)
            out[name] = _on_universe(run(), prices)
    return out


def portfolio_scale(positions: pl.DataFrame, prices: pl.DataFrame) -> np.ndarray:
    """Faktor per dag för hela portföljen, bara från P&L känd vid dagen:
    1. volbroms (VOL_BRAKE): 1/kvot när vol 20 d / vol 250 d > linn.VOL_THRESHOLD, annars 1
    2. volmål (TARGET_VOL): TARGET_VOL / realiserad årlig vol (TARGET_VOL_WINDOW dagar) för den bromsade
       portföljen; 0 innan 100 dagars P&L finns."""
    scale = np.ones(positions.height)
    if VOL_BRAKE:
        positions, active, ratio = conditional_vol_target(positions, prices, threshold=linn.VOL_THRESHOLD, days=1,
                                                          mode="reduce", return_state=True)
        scale = np.where(active, 1.0 / ratio, 1.0)
    if TARGET_VOL:
        pnl = pl.Series(strategy_pnl(positions, prices)).replace(0.0, None)
        vol = pnl.rolling_std(TARGET_VOL_WINDOW, min_samples=100).forward_fill().to_numpy()
        vol = vol * np.sqrt(evaluation.TRADING_DAYS_PER_YEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = scale * np.nan_to_num(TARGET_VOL / vol, nan=0.0, posinf=0.0)
    return scale


def _weighted(strategies: dict[str, pl.DataFrame], prices: pl.DataFrame) -> pl.DataFrame:
    return combine_positions(strategies, prices, {n: WEIGHTS[n] for n in strategies})


def combine(strategies: dict[str, pl.DataFrame], prices: pl.DataFrame) -> pl.DataFrame:
    """Riskskalningen använder strategiernas kausala positioner även under träningen (simulerad historisk P&L,
    känd vid varje dag), så att vikterna är på plats från första valideringsdagen. Sedan portföljöverlägg."""
    weighted = _weighted(strategies, prices)
    inst = [c for c in weighted.columns if c != DATE]
    scaled = weighted.select(inst).to_numpy() * portfolio_scale(weighted, prices)[:, None]
    return pl.DataFrame(scaled, schema=inst).insert_column(0, weighted[DATE])


def strategy_parts(strategies: dict[str, pl.DataFrame], prices: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Varje strategis del av den kombinerade positionen: samma formel som combine_positions
    (riskandel × walk-forward-skala × portföljöverlägg × position), så summan över strategierna är den kombinerade positionen."""
    total = sum(WEIGHTS[n] for n in strategies)
    overlay = portfolio_scale(_weighted(strategies, prices), prices)
    out = {}
    for n, pos in strategies.items():
        inst = [c for c in pos.columns if c != DATE]
        scale = (WEIGHTS[n] / total) * risk_scale(pos, prices) * overlay
        out[n] = pl.DataFrame(pos.select(inst).to_numpy() * scale[:, None], schema=inst).insert_column(0, pos[DATE])
    return out


# ================================== UTVÄRDERING ==================================
def _validation(positions: pl.DataFrame) -> pl.DataFrame:
    """Valideringsperioden, bara instrument som handlas (annars delar evaluation.py 0/0)."""
    positions = positions.filter(pl.col(DATE) >= VALIDATION_START)
    inst = [c for c in positions.columns if c != DATE]
    return positions.select(DATE, *[c for c in inst if positions[c].abs().sum() > 0])


def key_figure_table(strategies: dict[str, pl.DataFrame], prices: pl.DataFrame) -> None:
    print(f"{'':12s}" + "".join(f"{k:>17s}" for k in KEY_FIGURES))
    for name, pos in strategies.items():
        pos = _validation(pos)
        kf = evaluation.calc_key_figures(pos, prices.select(pos.columns), key_figures=KEY_FIGURES)
        print(f"{name:12s}" + "".join(f"{kf[k]:17.2f}" for k in KEY_FIGURES))


def pnl_correlation(strategies: dict[str, pl.DataFrame], prices: pl.DataFrame) -> None:
    pnl = {n: strategy_pnl(p, prices) for n, p in strategies.items()}
    mask = (prices[DATE] >= VALIDATION_START).to_numpy()
    names = list(pnl)
    corr = np.corrcoef(np.vstack([pnl[n][mask] for n in names]))
    print("\nKorrelation mellan strategiernas dagliga P&L (validering)")
    print(" " * 8 + "".join(f"{n:>8s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:8s}" + "".join(f"{corr[i, j]:8.2f}" for j in range(len(names))))


# ============================ PLOTTAR: POSITIONER OCH RISK ============================
# Positionsstorlek mäts som riskposition = antal × prisvolatilitet (årlig), dvs. den P&L-vol positionen
# skulle ge om den hölls ensam. Antal enheter är inte jämförbart mellan instrument; riskpositionen är det
# (evaluation.py normerar på samma sätt). All risk är i samma enhet som evaluation.py:s returns.
VOL_WINDOW = 100  # dagar: rullande prisvol per instrument och kovarians för ex ante-risk
ANN = float(np.sqrt(evaluation.TRADING_DAYS_PER_YEAR))
COMBINED = "KOMBINERAD"
CLASSES = ["bonds", "stocks", "fx", "råvaror"]  # macro.asset_class
COLORS = {"merged": "#d4a017", "david": "#2a78d6", "linn": "#eb6834", "leadlag": "#8e44c9", "macro": "#c2185b", "pairs": "#1baf7a",
          COMBINED: "#0b0b0b"}
CLASS_COLORS = {"bonds": "#eda100", "stocks": "#e87ba4", "fx": "#008300", "råvaror": "#6250d6"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def risk_data(parts: dict[str, pl.DataFrame], combined: pl.DataFrame, prices: pl.DataFrame) -> dict:
    """Riskpositioner, P&L och ex ante-risk för strategidelarna och den kombinerade modellen, bara validering.
    Ex ante-risken dag t använder kovariansen för prisförändringarna t.o.m. dag t och positionen daterad t."""
    inst = [c for c in combined.columns if c != DATE]
    mask = (prices[DATE] >= VALIDATION_START).to_numpy()
    diff = np.nan_to_num(prices.select(pl.col(inst).diff()).to_numpy())
    sigma = prices.select(pl.col(inst).diff().rolling_std(VOL_WINDOW, min_samples=10)).to_numpy() * ANN
    frames = parts | {COMBINED: combined}
    pos = {n: f.select(inst).to_numpy() for n, f in frames.items()}
    assert np.allclose(sum(pos[n] for n in parts), pos[COMBINED]), "strategidelarna summerar inte till kombinerad"

    cls = np.array([asset_class(c) for c in inst])
    idx = np.flatnonzero(mask)
    alone = {n: np.zeros(len(idx)) for n in frames}  # ex ante-varians om delen hölls ensam
    share = {n: np.zeros(len(idx)) for n in parts}   # Euler-bidrag till kombinerad varians
    cls_share = {g: np.zeros(len(idx)) for g in CLASSES}
    for k, t in enumerate(idx):
        cov = np.cov(diff[t - VOL_WINDOW + 1:t + 1], rowvar=False)
        p = pos[COMBINED][t]
        cp = cov @ p
        for n in frames:
            alone[n][k] = pos[n][t] @ cov @ pos[n][t]
        for n in parts:
            share[n][k] = pos[n][t] @ cp
        for g in CLASSES:
            cls_share[g][k] = (p * cp)[cls == g].sum()
    var = alone[COMBINED]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = {n: 100 * s / var for n, s in share.items()}
        cls_share = {g: 100 * s / var for g, s in cls_share.items()}

    return {
        "dates": prices[DATE].to_numpy()[mask],
        "inst": inst,
        "cls": cls,
        "risk": {n: p[mask] * sigma[mask] for n, p in pos.items()},
        "pnl": {n: strategy_pnl(f, prices)[mask] for n, f in frames.items()},
        "ex_ante": {n: np.sqrt(v) * ANN for n, v in alone.items()},
        "share": share,
        "cls_share": cls_share,
    }


def _rolling_vol(pnl: np.ndarray, window: int) -> np.ndarray:
    return pl.Series(pnl).rolling_std(window, min_samples=window // 2).to_numpy() * ANN


def risk_table(d: dict) -> None:
    names = list(d["risk"])
    rows = {
        "realiserad vol": [np.std(d["pnl"][n]) * ANN for n in names],
        "ex ante vol, snitt": [np.nanmean(d["ex_ante"][n]) for n in names],
        "andel av portföljvariansen %": [np.nanmean(d["share"][n]) for n in names if n in d["share"]] + [100.0],
        "brutto riskexponering, snitt": [np.abs(d["risk"][n]).sum(1).mean() for n in names],
    }
    print(f"\nRisk i valideringen (årlig, samma enhet som evaluation.py:s returns)")
    print(f"{'':30s}" + "".join(f"{n:>12s}" for n in names))
    for label, vals in rows.items():
        print(f"{label:30s}" + "".join(f"{v:12.2f}" for v in vals))
    print("andel av portföljvariansen per tillgångsslag: "
          + ", ".join(f"{g} {np.nanmean(s):.0f} %" for g, s in d["cls_share"].items()))


def _style(ax: Axes, title: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _line(ax: Axes, x, y, name: str, label: str | None = None) -> None:
    combined = name == COMBINED
    ax.plot(x, y, color=COLORS.get(name, MUTED), lw=1.4 if combined else 0.9, zorder=3 if combined else 2,
            label=label or name)


def plot_assets(d: dict) -> Figure:
    """En panel per tillgång (rader per tillgångsslag): riskposition per strategidel och kombinerad."""
    ncols = 5
    grid = []
    for g in CLASSES:
        names = [c for c in d["inst"] if asset_class(c) == g]
        grid += [names[i:i + ncols] for i in range(0, len(names), ncols)]
    col = {c: j for j, c in enumerate(d["inst"])}

    fig = plt.figure("Positioner per tillgång", figsize=(18, 2.0 * len(grid) + 1.2), facecolor=SURFACE)
    axes = fig.subplots(len(grid), ncols, sharex=True, sharey=True, squeeze=False)
    for r, names in enumerate(grid):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(names):
                ax.set_visible(False)
                continue
            _style(ax, names[c], "riskposition (årlig vol)" if c == 0 else None)
            ax.axhline(0, color=AXIS, lw=0.8)
            for n, risk in d["risk"].items():
                _line(ax, d["dates"], risk[:, col[names[c]]], n)
    for r in range(len(grid)):  # x-etiketter på varje kolumns nedersta synliga panel
        for c in range(ncols):
            if axes[r, c].get_visible() and (r == len(grid) - 1 or not axes[r + 1, c].get_visible()):
                axes[r, c].xaxis.set_tick_params(labelbottom=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Riskposition per tillgång (antal × årlig prisvol, {VOL_WINDOW} d), validering från {VALIDATION_START}. "
                 "Kombinerad = summan av strategidelarna efter riskskalning och vikter",
                 fontsize=12, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_exposure(d: dict) -> Figure:
    """Brutto riskexponering per strategi och tillgångsslag över tiden, och snitt per tillgång."""
    x, risk, parts = d["dates"], d["risk"], [n for n in d["risk"] if n != COMBINED]
    fig = plt.figure("Exponering", figsize=(18, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.3, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    for n in risk:
        _line(ax, x, np.abs(risk[n]).sum(1), n)
    _style(ax, "Brutto riskexponering per strategi (summa |riskposition| över tillgångar)", "årlig vol")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, ncol=len(risk))

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    gross = np.abs(risk[COMBINED])
    ax.stackplot(x, *[gross[:, d["cls"] == g].sum(1) for g in CLASSES], labels=CLASSES,
                 colors=[CLASS_COLORS[g] for g in CLASSES], edgecolor=SURFACE, linewidth=0.6)
    _style(ax, "Kombinerad modell: brutto riskexponering per tillgångsslag (staplad)", "årlig vol")
    ax.legend(frameon=False, fontsize=9, ncol=len(CLASSES), loc="upper left")

    # snitt per tillgång, grupperat per tillgångsslag och sorterat på kombinerad storlek
    ax = fig.add_subplot(gs[:, 1])
    mean_comb = gross.mean(0)
    order = [j for g in CLASSES for j in sorted(np.flatnonzero(d["cls"] == g), key=lambda j: -mean_comb[j])]
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for n in parts:
        m = np.abs(risk[n]).mean(0)[order]
        ax.barh(y, m, left=left, height=0.72, color=COLORS.get(n, MUTED), edgecolor=SURFACE, linewidth=1, label=n)
        left += m
    ax.scatter(mean_comb[order], y, marker="|", s=140, linewidths=2, color=INK, zorder=3,
               label=f"{COMBINED} (efter kvittning)")
    bounds = np.cumsum([np.sum(d["cls"] == g) for g in CLASSES])[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color=AXIS, lw=0.8)
    ax.set_yticks(y, [d["inst"][j] for j in order], fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Snitt |riskposition| per tillgång, validering")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("årlig vol (staplar: strategidelarna, streck: kombinerad netto)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(f"Positionsstorlek, validering från {VALIDATION_START}", fontsize=13, color=INK, x=0.01, ha="left")
    return fig


def plot_risk(d: dict) -> Figure:
    """Portföljens volatilitet (realiserad och ex ante) och riskbidrag per strategi och tillgångsslag."""
    x, pnl = d["dates"], d["pnl"]
    fig = plt.figure("Portföljrisk", figsize=(18, 10), facecolor=SURFACE)
    axes = fig.subplots(2, 2, sharex=True)

    ax = axes[0, 0]
    ax.plot(x, _rolling_vol(pnl[COMBINED], 60), color=MUTED, lw=0.8, label="realiserad, 60 d")
    ax.plot(x, _rolling_vol(pnl[COMBINED], 260), color=INK, lw=1.6, label="realiserad, 260 d")
    ax.plot(x, d["ex_ante"][COMBINED], color=INK, lw=1.0, ls="--", label=f"ex ante ({VOL_WINDOW} d kovarians)")
    _style(ax, "Kombinerad modell: portföljvolatilitet", "årlig vol (före kostnader)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")

    ax = axes[0, 1]
    for n in pnl:
        _line(ax, x, _rolling_vol(pnl[n], 260), n)
    parts = [n for n in pnl if n != COMBINED]
    div = sum(np.std(pnl[n]) for n in parts) / np.std(pnl[COMBINED])
    _style(ax, "Realiserad vol, 260 d, per strategidel och kombinerad", "årlig vol")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, ncol=len(pnl), loc="lower left")
    ax.text(0.99, 0.97, f"diversifieringskvot (summa strategivol / portföljvol): {div:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2)

    ax = axes[1, 0]
    ax.axhline(0, color=AXIS, lw=0.8)
    if len({WEIGHTS[n] for n in parts}) == 1:  # referenslinje bara när strategierna har lika vikt
        ax.axhline(100 / len(parts), color=MUTED, lw=0.8)
        ax.text(x[0], 100 / len(parts), f" lika risk ({100 / len(parts):.0f} %)", fontsize=8, color=INK2,
                va="bottom")
    for n in parts:
        _line(ax, x, d["share"][n], n)
    _style(ax, "Riskbidrag per strategi (andel av ex ante-varians)", "%")
    ax.legend(frameon=False, fontsize=9, ncol=len(parts), loc="upper right")

    ax = axes[1, 1]
    ax.axhline(0, color=AXIS, lw=0.8)
    for g in CLASSES:
        ax.plot(x, d["cls_share"][g], color=CLASS_COLORS[g], lw=0.9, label=g)
    _style(ax, "Riskbidrag per tillgångsslag (andel av ex ante-varians)", "%")
    ax.legend(frameon=False, fontsize=9, ncol=len(CLASSES), loc="upper right")

    fig.suptitle(f"Portföljrisk, validering från {VALIDATION_START}", fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ===================================== MAIN =====================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", default=str(ROOT / "close_prices_insample.csv"))
    parser.add_argument("--macro", default=str(ROOT / "macro_data_insample.csv"),
                        help="makrofil till Linns strategi; ange 'none' för att köra utan makro")
    parser.add_argument("--output", default="main_positions.csv",
                        help=f"positionsfil; ett filnamn utan mapp hamnar i {OUTPUT_DIR.name}/")
    parser.add_argument("--no-eval", action="store_true", help="hoppa över nyckeltal")
    parser.add_argument("--plot", action="store_true",
                        help="visa dashboard, positions- och riskplottar i fönster på datorn")
    parser.add_argument("--save-plots", action="store_true", help="spara även plottarna som PNG bredvid --output")
    parser.add_argument("--strategies", nargs="+", choices=list(WEIGHTS),
                        help=f"vilka strategier som körs (standard: {' '.join(DEFAULT_STRATEGIES)}), "
                             "t.ex. --strategies david leadlag macro pairs")
    args = parser.parse_args()
    macro_path = None if args.macro.lower() == "none" else args.macro
    output = Path(args.output)
    if not output.is_absolute() and output.parent == Path("."):
        output = OUTPUT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    selected = args.strategies or DEFAULT_STRATEGIES
    for n in WEIGHTS:
        if n not in selected:
            WEIGHTS[n] = 0.0

    prices = load_prices(args.prices)
    print(f"Priser: {prices.height} dagar, {prices.width - 1} instrument, {prices[DATE][0]} – {prices[DATE][-1]}")
    print(f"Träning (ingen handel) < {VALIDATION_START} ≤ validering\n")
    raw = build_all(prices, macro_path)  # hela historiken; riskskalningen behöver P&L även före validering
    combined = _no_trading_before(combine(raw, prices), VALIDATION_START)
    strategies = {n: _no_trading_before(p, VALIDATION_START) for n, p in raw.items()}
    combined.write_csv(args.output)
    print(f"Kombinerade positioner sparade: {args.output}\n")

    if not args.no_eval:
        print(f"Nyckeltal (med kostnader), validering {VALIDATION_START} – {prices[DATE][-1]}")
        key_figure_table(strategies | {"KOMBINERAD": combined}, prices)
        pnl_correlation(strategies, prices)

    if args.plot or args.save_plots:
        # evaluation.py bygger dashboarden med matplotlib.figure.Figure, som inte kan visas i fönster;
        # med plt.figure blir den en vanlig pyplot-figur (samma API)
        evaluation.Figure = plt.figure
        figs = {}
        fig = evaluation.plot_key_figures(_validation(combined), prices)
        fig.suptitle(f"Kombinerad modell, validering från {VALIDATION_START}: "
                     + " + ".join(f"{n} ({WEIGHTS[n]:g})" for n in strategies))
        fig.canvas.manager.set_window_title("Dashboard")
        figs[""] = fig

        parts = {n: _no_trading_before(p, VALIDATION_START) for n, p in strategy_parts(raw, prices).items()}
        d = risk_data(parts, combined, prices)
        risk_table(d)
        for suffix, plot in (("_tillgangar", plot_assets), ("_exponering", plot_exposure), ("_risk", plot_risk)):
            figs[suffix] = plot(d)

        if args.save_plots:
            out = Path(args.output).with_suffix(".png")
            for suffix, fig in figs.items():
                path = out.with_name(f"{out.stem}{suffix}.png")
                fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
                print(f"Plot sparad: {path}")
        if args.plot:
            print("\nVisar plottarna; stäng fönstren för att avsluta.")
            plt.show()

if __name__ == "__main__":
    main()
