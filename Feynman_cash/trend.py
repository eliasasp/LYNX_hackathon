"""Multi-horisont-trend: snitt av flera EWMA-crossovers (snabb/långsam) med olika hastighet.

För varje hastighet (snabb, långsam = 4 * snabb):
    råsignal = (EWMA_snabb(pris) - EWMA_långsam(pris)) / vol
    skalas till ~std 1 med walk-forward-std och klipps vid ±max_pos
Signalerna snittas över hastigheterna och positionen blir signal / vol (lika risk per instrument).
Allt använder bara data känd vid dag t.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from lead_lag import DATE, VOL_WINDOW, _instruments


def trend_positions(
    prices: pl.DataFrame,
    fast_spans: tuple[int, ...] = (8, 16, 32, 64),
    slow_mult: int = 4,
    max_pos: float = 3.0,
) -> pl.DataFrame:
    inst = _instruments(prices)
    p = prices.select(inst)
    vol = p.select(pl.all().diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()

    scores = []
    for fast in fast_spans:
        slow = fast * slow_mult
        raw = (
            p.select(pl.all().ewm_mean(span=fast, adjust=False)).to_numpy()
            - p.select(pl.all().ewm_mean(span=slow, adjust=False)).to_numpy()
        ) / vol
        raw_df = pl.DataFrame(raw, schema=inst).select(pl.all().fill_nan(None))
        # expanderande medel|x| (bara historik), ignorerar de första dagarna utan vol
        scale = raw_df.select(pl.all().abs().cum_sum() / pl.all().is_not_null().cum_sum()).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.clip(raw / (scale * 1.25), -max_pos, max_pos)  # medel|x| * 1.25 ≈ std för normalfördelning
        s[: slow] = np.nan  # vänta tills långsamma EWMA:n har landat
        scores.append(s)

    score = np.nanmean(np.stack(scores), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def momentum_signals(prices: pl.DataFrame, windows: tuple[int, ...] = (20, 50, 100)) -> dict[int, np.ndarray]:
    """z-värde per fönster N: (p_t - p_{t-N}) / (vol_t * sqrt(N)). Känt vid dag t."""
    inst = _instruments(prices)
    p = prices.select(inst)
    vol = p.select(pl.all().diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    out = {}
    for n in windows:
        change = p.select(pl.all().diff(n)).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[n] = change / (vol * np.sqrt(n))
    return out


def _hysteresis(z: np.ndarray, enter: float, exit: float) -> np.ndarray:
    """Tillståndsmaskin per instrument (kolumn):
        flat  -> +1 om z >  enter,  -1 om z < -enter
        +1    -> flat om z <  exit  (och direkt -1 om z < -enter)
        -1    -> flat om z > -exit  (och direkt +1 om z >  enter)
    exit=0 betyder att positionen hålls tills z byter tecken.
    """
    out = np.full_like(z, np.nan)
    state = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        zt = z[t]
        ok = np.isfinite(zt)
        new = state.copy()
        new[(state == 1) & (zt < exit)] = 0
        new[(state == -1) & (zt > -exit)] = 0
        new[zt > enter] = 1
        new[zt < -enter] = -1
        state = np.where(ok, new, 0.0)
        out[t] = np.where(ok, state, np.nan)
    return out


def hysteresis_trend_positions(
    prices: pl.DataFrame,
    windows: tuple[int, ...] = (20, 50, 100),
    enter: float = 0.5,
    exit: float = 0.0,
) -> pl.DataFrame:
    """Som discrete_trend_positions men med separata in- och utgångströsklar (minskar fram-och-tillbaka-handel)."""
    inst = _instruments(prices)
    vol = prices.select(pl.col(inst).diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    classes = [_hysteresis(z, enter, exit) for z in momentum_signals(prices, windows).values()]
    score = np.nanmean(np.stack(classes), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def discrete_trend_positions(
    prices: pl.DataFrame,
    windows: tuple[int, ...] = (20, 50, 100),
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Varje fönsters signal klassas som -1 / 0 / +1:
        +1 om z >  threshold,  -1 om z < -threshold,  annars 0.
    Klassningarna snittas över fönstren (värden i steg om 1/len(windows)) och delas med vol.
    """
    inst = _instruments(prices)
    vol = prices.select(pl.col(inst).diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    classes = []
    for z in momentum_signals(prices, windows).values():
        c = np.where(z > threshold, 1.0, np.where(z < -threshold, -1.0, 0.0))
        c[~np.isfinite(z)] = np.nan
        classes.append(c)
    score = np.nanmean(np.stack(classes), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def hysteresis_classes(
    prices: pl.DataFrame, windows: tuple[int, ...] = (50, 100), enter: float = 0.5, exit: float = 0.0
) -> list[np.ndarray]:
    """-1/0/+1-klassning med hysteres per fönster (en matris [dagar x instrument] per fönster)."""
    return [_hysteresis(z, enter, exit) for z in momentum_signals(prices, windows).values()]


def trend_from_classes(
    prices: pl.DataFrame,
    classes: list[np.ndarray],
    weights: list[np.ndarray] | None = None,
    vol: np.ndarray | None = None,
) -> pl.DataFrame:
    """Viktat snitt av fönstrens klassningar delat med vol.

    weights: en matris per fönster (samma form som klassningarna) eller None för lika vikt.
             Vikterna normaliseras så att de summerar till 1 per dag och instrument.
    vol:     egen volatilitetsskattning (t.ex. GARCH); default 100-dagars rullande std.
    """
    inst = _instruments(prices)
    if vol is None:
        vol = prices.select(pl.col(inst).diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    c = np.stack(classes)
    w = np.ones_like(c) if weights is None else np.stack(weights)
    w = np.where(np.isfinite(c), w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.nansum(np.nan_to_num(c) * w, axis=0) / w.sum(axis=0)
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def breakout_positions(
    prices: pl.DataFrame,
    windows: tuple[int, ...] = (50, 100),
    smooth_frac: float = 0.25,
) -> pl.DataFrame:
    """Donchian-breakout: var ligger priset i sitt N-dagars högsta–lägsta-intervall?

        score_N = (p - mitt) / (halva intervallet)   ∈ [-1, 1]
        +1 = vid N-dagars högsta, -1 = vid N-dagars lägsta.
    Varje fönsters score jämnas ut med EWMA (span = smooth_frac * N) för att minska handeln,
    snittas över fönstren och delas med vol. Bara data t.o.m. dag t används.
    """
    inst = _instruments(prices)
    p = prices.select(inst)
    vol = p.select(pl.all().diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    scores = []
    for n in windows:
        hi = p.select(pl.all().rolling_max(n)).to_numpy()
        lo = p.select(pl.all().rolling_min(n)).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (p.to_numpy() - (hi + lo) / 2) / ((hi - lo) / 2)
        span = max(2, int(smooth_frac * n))
        s = pl.DataFrame(s, schema=inst).select(pl.all().fill_nan(None).ewm_mean(span=span, adjust=False)).to_numpy()
        scores.append(s)
    score = np.nanmean(np.stack(scores), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def _ma_cross_scale(fast: int, slow: int) -> float:
    """Std för (SMA_fast - SMA_slow) om priset är en slumpvandring med daglig std 1.

    SMA_n = p_t - sum_k (k/n) * diff_{t-k+1}..., dvs vikten på prisförändringen k dagar bakåt är
    (1 - k/n) för k < n. Skillnaden mellan två SMA är en viktad summa av gamla prisförändringar,
    och dess varians är summan av de kvadrerade vikterna.
    """
    k = np.arange(slow)
    w = np.where(k < fast, 1 - k / fast, 0.0) - (1 - k / slow)
    return float(np.sqrt(np.sum(w**2)))


def ma_cross_z(prices: pl.DataFrame, fast: int, slow: int) -> np.ndarray:
    """z = (SMA_fast - SMA_slow) / (vol * skalfaktor), ~N(0,1) för en slumpvandring. Känt vid dag t."""
    inst = _instruments(prices)
    p = prices.select(inst)
    vol = p.select(pl.all().diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    gap = p.select(pl.all().rolling_mean(fast)).to_numpy() - p.select(pl.all().rolling_mean(slow)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return gap / (vol * _ma_cross_scale(fast, slow))


def ma_cross_tanh_positions(
    prices: pl.DataFrame,
    pairs: tuple[tuple[int, int], ...] = ((12, 50), (25, 100)),
    k: float = 1.0,
) -> pl.DataFrame:
    """Moving average crossover med kontinuerlig signal: signal = tanh(k * z) per (snabb, långsam),
    snittat över paren och delat med vol. tanh är ~linjär för svaga trender och mättas mot ±1."""
    inst = _instruments(prices)
    vol = prices.select(pl.col(inst).diff().rolling_std(VOL_WINDOW, min_samples=20)).to_numpy()
    signals = [np.tanh(k * ma_cross_z(prices, f, s)) for f, s in pairs]
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.nanmean(np.stack(signals), axis=0)
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))
