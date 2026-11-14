"""Dynamisk allokering mellan strategier, per instrument eller för hela portföljen. Walk-forward.

Varje strategi skalas först till samma risk (som i combine.py). Sedan får strategi i vikten

    w_i[t, instrument]  ∝  base_i * exp(strength * score_i[t, instrument])

normaliserat så att vikterna summerar till 1. strength = 0 ger de fasta basvikterna.

method="adjusted_sharpe": score = strategins justerade Sharpe de senaste `window` dagarna
method="performance":     score = strategins Sharpe de senaste `window` dagarna
                          (per instrument, eller för hela portföljen om per_instrument=False)
method="variance_ratio":  score = (VR - 1) för strategin som heter `trend_name`, 0 för övriga.
                          VR = var(q-dagarsrörelse) / (q * var(dagsrörelse)) över `window` dagar;
                          VR > 1 betyder att instrumentet har trendat (positiv autokorrelation).

All information är känd vid dag t: P&L dag t bygger på positionen från t-2 och priset t.o.m. t.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from combine import risk_scale
from lead_lag import DATE, POSITION_LAG, _instruments


def _instrument_pnl(pos: np.ndarray, diff: np.ndarray) -> np.ndarray:
    lagged = np.zeros_like(pos)
    lagged[POSITION_LAG:] = pos[:-POSITION_LAG]
    return np.nan_to_num(lagged * diff)


def _rolling_sharpe(pnl: np.ndarray, window: int) -> np.ndarray:
    df = pl.DataFrame(pnl)
    mean = df.select(pl.all().rolling_mean(window, min_samples=window // 2)).to_numpy()
    std = df.select(pl.all().rolling_std(window, min_samples=window // 2)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(mean / std * np.sqrt(260), nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_adjusted_sharpe(pnl: np.ndarray, window: int) -> np.ndarray:
    """Samma formel som evaluation.py, på rullande fönster (bara historik t.o.m. dag t):
    S * (1 + S/6 * skevhet - S^2/24 * överskottskurtos), S årlig. Klipps till [-3, 3]."""
    df = pl.DataFrame(pnl)
    mp = window // 2
    m1, m2, m3, m4 = (df.select((pl.all() ** k).rolling_mean(window, min_samples=mp)).to_numpy() for k in (1, 2, 3, 4))
    var = m2 - m1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(var)
        s = m1 / sd * np.sqrt(260)
        skew = (m3 - 3 * m1 * m2 + 2 * m1**3) / sd**3
        exkurt = (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / var**2 - 3
        adj = s * (1 + s / 6 * skew - s**2 / 24 * exkurt)
    return np.clip(np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0), -3, 3)


def variance_ratio(prices: pl.DataFrame, q: int = 50, window: int = 1000) -> np.ndarray:
    inst = _instruments(prices)
    diff = prices.select(pl.col(inst).diff())
    z = diff / diff.select(pl.all().rolling_std(100, min_samples=20).shift(1))
    var1 = z.select(pl.all().rolling_var(window, min_samples=window // 2)).to_numpy()
    varq = z.select(pl.all().rolling_sum(q).rolling_var(window, min_samples=window // 2)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(varq / (q * var1), nan=1.0)


def dynamic_allocation(
    strategies: dict[str, pl.DataFrame],
    prices: pl.DataFrame,
    base_weights: dict[str, float],
    method: str = "performance",
    strength: float = 1.0,
    window: int = 500,
    per_instrument: bool = True,
    trend_name: str = "trend",
    vr_q: int = 50,
    return_weights: bool = False,
):
    inst = _instruments(prices)
    names = list(strategies)
    diff = prices.select(pl.col(inst).diff()).to_numpy()
    scaled = {k: strategies[k].select(inst).to_numpy() * risk_scale(strategies[k], prices)[:, None] for k in names}

    scores = {}
    if method in ("performance", "adjusted_sharpe"):
        score_fn = _rolling_sharpe if method == "performance" else _rolling_adjusted_sharpe
        for k in names:
            pnl = _instrument_pnl(scaled[k], diff)
            if not per_instrument:
                pnl = np.repeat(pnl.sum(axis=1, keepdims=True), len(inst), axis=1)
            scores[k] = score_fn(pnl, window)
    else:
        vr = variance_ratio(prices, vr_q, window) - 1.0
        scores = {k: (vr if k == trend_name else np.zeros_like(vr)) for k in names}

    raw = {k: base_weights[k] * np.exp(strength * scores[k]) for k in names}
    total = sum(raw.values())
    weights = {k: raw[k] / total for k in names}
    pos = sum(weights[k] * scaled[k] for k in names)
    out = pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))
    return (out, weights) if return_weights else out


def instrument_tilt(positions: pl.DataFrame, prices: pl.DataFrame, strength: float = 0.5, window: int = 500,
                    return_weights: bool = False):
    """Skala varje instruments position med exp(strength * rullande justerad Sharpe för portföljens
    P&L i just det instrumentet), normaliserat så att snittet över instrumenten är 1 varje dag."""
    inst = _instruments(prices)
    pos = positions.select(inst).to_numpy()
    diff = prices.select(pl.col(inst).diff()).to_numpy()
    score = _rolling_adjusted_sharpe(_instrument_pnl(pos, diff), window)
    w = np.exp(strength * score)
    w = w / w.mean(axis=1, keepdims=True)
    out = pl.DataFrame(pos * w, schema=inst).insert_column(0, positions.get_column(DATE))
    return (out, w) if return_weights else out
