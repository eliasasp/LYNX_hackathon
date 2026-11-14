"""Kombinera flera strategiers positioner så att var och en bidrar med önskad andel risk.

Varje strategi skalas walk-forward till samma P&L-volatilitet (med enbart historisk P&L),
sedan viktas och summeras positionerna. Fungerar för alla strategier som ger
positioner i evaluation.py-format (AsOfDate + en kolumn per instrument).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lead_lag import DATE, POSITION_LAG, _instruments


def strategy_pnl(positions: pl.DataFrame, prices: pl.DataFrame) -> np.ndarray:
    """Daglig P&L (före slippage) med evaluation.py:s tidskonvention: pos[t-2] * diff[t]."""
    inst = _instruments(positions)
    diff = prices.select(pl.col(inst).diff()).to_numpy()
    pos = positions.select(inst).to_numpy()
    lagged = np.full_like(pos, np.nan)
    lagged[POSITION_LAG:] = pos[:-POSITION_LAG]
    return np.nansum(lagged * diff, axis=1)


def risk_scale(positions: pl.DataFrame, prices: pl.DataFrame, window: int = 500) -> np.ndarray:
    """Faktor per dag som ger strategin P&L-std ≈ 1, baserat bara på P&L känd vid dag t."""
    pnl = pl.Series(strategy_pnl(positions, prices)).replace(0.0, None)
    std = pnl.rolling_std(window, min_samples=100).forward_fill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = 1.0 / std
    return np.nan_to_num(scale, nan=0.0, posinf=0.0)


def combine_positions(
    strategies: dict[str, pl.DataFrame],
    prices: pl.DataFrame,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Riskviktad kombination. `weights` = andel av risken per strategi (normaliseras till summa 1)."""
    names = list(strategies)
    if weights is None:
        weights = {n: 1.0 for n in names}
    total = sum(weights[n] for n in names)
    inst = _instruments(next(iter(strategies.values())))

    combo = np.zeros((len(prices), len(inst)))
    for n in names:
        pos = strategies[n].select(inst).to_numpy()
        combo += (weights[n] / total) * risk_scale(strategies[n], prices)[:, None] * pos
    return pl.DataFrame(combo, schema=inst).insert_column(0, prices.get_column(DATE))
