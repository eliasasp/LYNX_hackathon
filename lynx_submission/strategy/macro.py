"""Makrosignaler: BNP- och KPI-index per region som förklarande variabler för framtida avkastning.

Egenskaper (features) per instrument, alla kända vid dag t (makrovärdet för dag t, ev. fördröjt `lag` dagar):
    tillväxt_N  = log(BNP_t / BNP_{t-N})      inflation_N = log(KPI_t / KPI_{t-N})
  för instrumentets region. Valutor (<region>_titan_fx) får regionens värde minus titans.
  Råvaror får snittet över alla regioner. Varje egenskap z-normaliseras mot sitt eget rullande
  medel och std (500 handelsdagar), så att den mäter "högre eller lägre än nyligen".

Strategin är samma walk-forward-regression som lead-lag: för varje instrument och egenskap skattas
    kommande 20 dagars avkastning ≈ beta * egenskap
på par vars utfall är kända, prognoserna snittas och görs om till positioner.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lead_lag import DATE, LeadLagConfig, _forecast_to_positions, _instruments, _walk_forward_beta, build_signal_and_target

REGIONS = ("ganymede", "io", "titan", "callisto", "europa")
BASE = "titan"


def load_macro(path: str, prices: pl.DataFrame, lag: int = 0) -> dict[str, np.ndarray]:
    """Makroserier på prisernas datum: senaste värdet t.o.m. respektive dag, sedan `lag` handelsdagar fördröjt."""
    m = pl.read_csv(path, try_parse_dates=True, infer_schema_length=None)
    first = m.columns[0]
    m = m.rename({first: DATE}).with_columns(pl.col(DATE).cast(pl.Date)).sort(DATE)
    aligned = prices.select(DATE).join_asof(m, on=DATE, strategy="backward")
    return {c: aligned[c].shift(lag).to_numpy() for c in aligned.columns if c != DATE}


def asset_class(name: str) -> str:
    for g in ("bonds", "stocks", "fx"):
        if name.endswith(g):
            return g
    return "råvaror"


def _zscore(a: np.ndarray, window: int = 500) -> np.ndarray:
    s = pl.Series(a).fill_nan(None)
    mean = s.rolling_mean(window, min_samples=120)
    std = s.rolling_std(window, min_samples=120)
    return ((s - mean) / std).to_numpy()


def macro_features(prices: pl.DataFrame, macro: dict[str, np.ndarray],
                   windows: tuple[int, ...] = (20, 120)) -> dict[str, dict[str, np.ndarray]]:
    """{instrument: {egenskapsnamn: serie}}."""
    inst = _instruments(prices)

    def change(kind, region, n):
        x = np.log(macro[f"{kind}|{region}"])
        out = np.full_like(x, np.nan)
        out[n:] = x[n:] - x[:-n]
        return out

    feats: dict[str, dict[str, np.ndarray]] = {}
    for c in inst:
        region = c.split("_")[0]
        cls = asset_class(c)
        f = {}
        for kind, label in (("gdp", "tillväxt"), ("cpi", "inflation")):
            for n in windows:
                if cls == "råvaror":
                    raw = np.mean([change(kind, r, n) for r in REGIONS], axis=0)
                elif cls == "fx":
                    raw = change(kind, region, n) - change(kind, BASE, n)
                else:
                    raw = change(kind, region, n)
                f[f"{label}_{n}"] = _zscore(raw)
        feats[c] = f
    return feats


def macro_positions(prices: pl.DataFrame, macro: dict[str, np.ndarray], horizon: int = 20,
                    windows: tuple[int, ...] = (20, 120), train_window: int | None = None,
                    min_train: int = 500, use: tuple[str, ...] | None = None,
                    return_betas: bool = False, max_pos: float = 3.0,
                    halflife: float | None = None):
    inst = _instruments(prices)
    _, y, vol = build_signal_and_target(prices, horizon, horizon)
    feats = macro_features(prices, macro, windows)
    # train_window = rullande fönster i dagar (None = all historik); halflife = exponentiell viktning
    # (en observation väger hälften efter så många dagar) och tar över om den är satt.
    cfg = LeadLagConfig(lookback=horizon, horizon=horizon, train_window=train_window,
                        min_train=min_train, halflife=halflife)
    T = len(prices)
    forecast = np.zeros((T, len(inst)))
    n_used = np.zeros((T, len(inst)))
    betas = {}
    for j, c in enumerate(inst):
        for name, x in feats[c].items():
            if use is not None and not any(name.startswith(u) for u in use):
                continue
            beta, _ = _walk_forward_beta(x, y[:, j], horizon, cfg)
            pred = beta * x
            ok = np.isfinite(pred)
            forecast[:, j] += np.where(ok, pred, 0.0)
            n_used[:, j] += ok
            betas[(c, name)] = beta
    with np.errstate(invalid="ignore", divide="ignore"):
        forecast = np.where(n_used > 0, forecast / n_used, 0.0)
    pos = _forecast_to_positions(prices, forecast, vol, horizon, max_pos, True)  # max_pos: tak för |signal|
    return (pos, betas) if return_betas else pos
