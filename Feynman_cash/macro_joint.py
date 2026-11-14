"""Makro med en GEMENSAM regression per instrument i stället för medelvärdet av enkelregressioner.

Nuvarande macro.py skattar ett beta per makrovariabel för sig och snittar prognoserna. Här skattas
alla variabler samtidigt:

    y_t = beta_1 x1_t + beta_2 x2_t + ... + beta_K xK_t + eps        (ingen intercept)

med minsta kvadrat på de par (x_s, y_s) som var kända vid t (s <= t - 1 - horizon), inom ett rullande
fönster. Lösningen är beta = (X'X + lambda I)^-1 X'y, där lambda = ridge * medel(diag(X'X)) så att
samma ridge-tal betyder samma dämpning oavsett skala. Ridge behövs eftersom tillväxt_20 och
tillväxt_120 (och KPI-motsvarigheterna) är starkt korrelerade: utan dämpning blir betan stora och
motsatta i tecken.

X'X och X'y byggs som kumulativa summor och skiftas `known` rader, precis som i lead_lag.py, så att
varje dag bara ser data vars utfall redan var realiserat.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lead_lag import POSITION_LAG, _forecast_to_positions, _instruments, _shift, build_signal_and_target
from macro import macro_features


def _rolling_known(a: np.ndarray, window: int | None, known: int) -> np.ndarray:
    """Löpande summa av a (längs tid) över de senaste `window` dagarna, skiftad så att rad t
    bara innehåller det som var känt vid t."""
    c = np.cumsum(a, axis=0)
    if window is not None and window < len(c):
        c = c - np.concatenate([np.zeros((window,) + c.shape[1:]), c[:-window]], axis=0)
    flat = c.reshape(len(c), -1)
    return _shift(flat, known).reshape(c.shape)


def macro_joint_positions(prices: pl.DataFrame, macro: dict[str, np.ndarray], horizon: int = 20,
                          windows: tuple[int, ...] = (20, 120), train_window: int | None = 2500,
                          min_train: int = 500, ridge: float = 0.1, max_pos: float = 2.0,
                          use: tuple[str, ...] | None = None, return_betas: bool = False):
    inst = _instruments(prices)
    _, y_all, vol = build_signal_and_target(prices, horizon, horizon)
    feats = macro_features(prices, macro, windows)
    known = POSITION_LAG - 1 + horizon
    T = len(prices)
    forecast = np.zeros((T, len(inst)))
    betas = {}

    for j, c in enumerate(inst):
        names = [n for n in feats[c] if use is None or any(n.startswith(u) for u in use)]
        X = np.column_stack([feats[c][n] for n in names])  # [T x K]
        y = y_all[:, j]
        K = X.shape[1]
        m = np.isfinite(X).all(axis=1) & np.isfinite(y)
        Xm = np.where(m[:, None], np.nan_to_num(X), 0.0)
        ym = np.where(m, np.nan_to_num(y), 0.0)

        # X'X per dag (T x K x K) och X'y per dag (T x K), bara känd data
        xx = _rolling_known(Xm[:, :, None] * Xm[:, None, :], train_window, known)
        xy = _rolling_known(Xm * ym[:, None], train_window, known)
        n = _shift(np.cumsum(m.astype(float))[:, None], known)[:, 0]

        lam = ridge * np.nanmean(np.diagonal(xx, axis1=1, axis2=2), axis=1)  # skalat per dag
        A = xx + (np.nan_to_num(lam)[:, None, None] + 1e-12) * np.eye(K)
        ok = np.isfinite(n) & (n >= min_train) & np.isfinite(A).all(axis=(1, 2)) & np.isfinite(xy).all(axis=1)
        beta = np.full((T, K), np.nan)
        if ok.any():
            beta[ok] = np.linalg.solve(A[ok], xy[ok][:, :, None])[:, :, 0]
        # Prognosen behöver bara X och beta, INTE y: kräver vi finit y nollas de sista
        # horizon+1 dagarna (där utfallet ännu inte finns), alltså de färskaste dagarna.
        pred = np.where(np.isfinite(X).all(axis=1), (beta * np.nan_to_num(X)).sum(axis=1), 0.0)
        forecast[:, j] = np.nan_to_num(pred)
        for k, nm in enumerate(names):
            betas[(c, nm)] = beta[:, k]

    pos = _forecast_to_positions(prices, forecast, vol, horizon, max_pos, True)
    return (pos, betas) if return_betas else pos
