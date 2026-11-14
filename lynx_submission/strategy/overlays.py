"""Lager ovanpå färdiga positioner (evaluation.py-format). Båda använder bara data känd vid dag t.

vol_target:  skalar hela portföljen så att dess förväntade dagliga P&L-volatilitet blir konstant.
             Skattas med EWMA-std av portföljens egen historiska P&L (före slippage).
buffer:      handlar bara när önskad position lämnar ett band runt nuvarande position, och då
             bara fram till bandets kant. Bandets halvbredd = `width` * instrumentets typiska
             |position| (rullande medel av |önskad position| senaste året).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lead_lag import DATE, POSITION_LAG, _instruments


def _portfolio_pnl(pos: np.ndarray, diff: np.ndarray) -> np.ndarray:
    lagged = np.full_like(pos, np.nan)
    lagged[POSITION_LAG:] = pos[:-POSITION_LAG]
    out = np.nansum(lagged * diff, axis=1)
    out[np.isnan(lagged * diff).all(axis=1)] = np.nan
    return out


def vol_target(
    positions: pl.DataFrame,
    prices: pl.DataFrame,
    span: int = 60,
    min_obs: int = 100,
    max_scale: float = 3.0,
    return_scale: bool = False,
):
    """Skalfaktor_t = 1 / EWMA-std(portfölj-P&L t.o.m. t), normaliserad mot sitt långsiktiga
    (expanderande) medianvärde och klippt till [1/max_scale, max_scale] så att hävstången inte skenar.

    Obs: portföljens P&L beror på tidigare skalfaktorer; vi skattar på den OSKALADE portföljen,
    vilket ger en ren skattning av strategins egen risk.
    """
    inst = _instruments(positions)
    pos = positions.select(inst).to_numpy()
    diff = prices.select(pl.col(inst).diff()).to_numpy()
    pnl = pl.Series(_portfolio_pnl(pos, diff)).fill_nan(None)
    live = pnl.fill_null(0.0) != 0
    pnl = pl.Series(np.where(live.to_numpy(), pnl.to_numpy(), np.nan)).fill_nan(None)
    ew_std = pnl.ewm_std(span=span, min_samples=min_obs, ignore_nulls=True).forward_fill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = 1.0 / ew_std
    # normalisera mot expanderande median (bara historik) så att nivån är jämförbar över tid
    s = pl.Series(raw).fill_nan(None)
    ref = s.cum_sum() / s.is_not_null().cum_sum()
    scale = (s / ref).to_numpy()
    scale = np.clip(np.nan_to_num(scale, nan=1.0), 1 / max_scale, max_scale)
    out = pl.DataFrame(pos * scale[:, None], schema=inst).insert_column(0, positions.get_column(DATE))
    return (out, scale) if return_scale else out


def buffer(positions: pl.DataFrame, width: float = 0.2, window: int = 260) -> pl.DataFrame:
    inst = _instruments(positions)
    desired = positions.select(inst).to_numpy()
    typical = (
        pl.DataFrame(np.abs(desired), schema=inst)
        .select(pl.all().replace(0.0, None).rolling_mean(window, min_samples=20).forward_fill())
        .to_numpy()
    )
    half = np.nan_to_num(width * typical)
    held = np.zeros_like(desired)
    cur = np.zeros(desired.shape[1])
    for t in range(desired.shape[0]):
        lo, hi = desired[t] - half[t], desired[t] + half[t]
        cur = np.clip(cur, lo, hi)  # flytta bara om vi är utanför bandet, och då till kanten
        held[t] = cur
    return pl.DataFrame(held, schema=inst).insert_column(0, positions.get_column(DATE))


def conditional_vol_target(
    positions: pl.DataFrame,
    prices: pl.DataFrame,
    short: int = 20,
    long: int = 250,
    threshold: float = 1.25,
    days: int = 10,
    mode: str = "reduce",
    max_scale: float = 2.0,
    return_state: bool = False,
):
    """Skala bara när volatiliteten varit förhöjd länge.

    ratio_t = EWMA-std(P&L, short) / EWMA-std(P&L, long), portföljens egen oskalade P&L t.o.m. t.
    Aktiv när ratio > threshold har gällt `days` dagar i följd; inaktiv så fort ratio < threshold.
      mode="reduce": när aktiv, skala med 1/ratio (minskar risken mot långsiktig nivå)
      mode="lean_in": när aktiv, skala med ratio (ökar risken), klippt till max_scale
    """
    inst = _instruments(positions)
    pos = positions.select(inst).to_numpy()
    diff = prices.select(pl.col(inst).diff()).to_numpy()
    raw = _portfolio_pnl(pos, diff)
    pnl = pl.Series(np.where(np.nan_to_num(raw) != 0, raw, np.nan)).fill_nan(None)
    s = pnl.ewm_std(span=short, min_samples=short, ignore_nulls=True).forward_fill().to_numpy()
    lg = pnl.ewm_std(span=long, min_samples=100, ignore_nulls=True).forward_fill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nan_to_num(s / lg, nan=1.0)

    above = ratio > threshold
    run = np.zeros(len(ratio), dtype=int)
    for t in range(len(ratio)):
        run[t] = run[t - 1] + 1 if (above[t] and t > 0) else int(above[t])
    active = run >= days

    if mode == "reduce":
        scale = np.where(active, 1.0 / ratio, 1.0)
    else:
        scale = np.where(active, np.minimum(ratio, max_scale), 1.0)
    out = pl.DataFrame(pos * scale[:, None], schema=inst).insert_column(0, positions.get_column(DATE))
    return (out, active, ratio) if return_state else out


def instrument_vol_brake(
    positions: pl.DataFrame,
    prices: pl.DataFrame,
    short: int = 20,
    base_window: int = 100,
    power: float = 1.0,
) -> pl.DataFrame:
    """Asymmetrisk volatilitetsbroms per instrument.

    Strategierna storleksbestämmer med 100-dagars vol. När instrumentets korta vol (EWMA, `short`)
    är högre än den, skalas positionen ner med (vol_100 / vol_kort)^power. Den skalas aldrig upp.
    Bara prisförändringar t.o.m. dag t används.
    """
    inst = _instruments(positions)
    diff = prices.select(pl.col(inst).diff())
    base = diff.select(pl.all().rolling_std(base_window, min_samples=20)).to_numpy()
    fast = diff.select(pl.all().ewm_std(span=short, min_samples=short)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.minimum(1.0, (base / fast) ** power)
    scale = np.nan_to_num(scale, nan=1.0)
    out = positions.select(inst).to_numpy() * scale
    return pl.DataFrame(out, schema=inst).insert_column(0, positions.get_column(DATE))


def compress_risk(positions: pl.DataFrame, prices: pl.DataFrame, gamma: float = 0.5,
                  window: int = 260) -> pl.DataFrame:
    """Jämna ut risken per instrument: risk r = pos * vol_100 normaliseras med sin typiska nivå
    (rullande medel av |r|, bara historik) och komprimeras: r' = sign(r) * |r|^gamma.
    gamma=1 lämnar positionerna orörda, gamma=0.5 krymper stora positioner och lyfter små,
    gamma=0 ger konstant risk (bara riktningen behålls)."""
    inst = _instruments(positions)
    vol = prices.select(pl.col(inst).diff().rolling_std(100, min_samples=20)).to_numpy()
    pos = positions.select(inst).to_numpy()
    r = pos * vol
    typ = pl.DataFrame(np.abs(r), schema=inst).select(
        pl.all().fill_nan(None).replace(0.0, None).rolling_mean(window, min_samples=20).shift(1).forward_fill()
    ).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = r / typ
        out = np.sign(rn) * np.abs(rn) ** gamma * typ / vol
    return pl.DataFrame(np.nan_to_num(out), schema=inst).insert_column(0, positions.get_column(DATE))
