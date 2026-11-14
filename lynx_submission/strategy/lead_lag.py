"""Lead-lag-strategi: ta position idag baserat på vad historiken säger om de kommande `horizon` dagarna.

Idé
----
För varje målinstrument B och varje "ledare" A (A kan vara B själv = momentum/mean reversion):

    signal  x_t = A:s normaliserade avkastning de senaste `lookback` dagarna   (känd vid dag t)
    mål     y_t = B:s normaliserade avkastning över de `horizon` dagar som positionen faktiskt tjänar på

Vi skattar hur mycket x förutsäger y (beta) **walk-forward**: vid dag t används bara par (x_s, y_s)
där y_s var helt känt vid t. Positionen blir prognosen delad med instrumentets volatilitet.

Tidskonvention (matchar evaluation.py)
--------------------------------------
evaluation.py gör `positions.shift(2) * prices.diff()`, dvs en position daterad t tjänar
diff-avkastningen på raderna t+2, t+3, ... Målet för en `horizon`-dagars hållning är därför
summan av diff på raderna t+2 .. t+1+horizon, och det är helt känt först vid dag t+1+horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

DATE = "AsOfDate"
POSITION_LAG = 2  # samma som i evaluation.py
VOL_WINDOW = 100


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_prices(path: str) -> pl.DataFrame:
    """Läser pris-CSV och döper om datumkolumnen till AsOfDate.

    Om kolumnerna heter "<instrument>|Open/High/Low/Close" behålls bara stängningspriserna,
    med instrumentnamnet som kolumnnamn.
    """
    df = pl.read_csv(path, try_parse_dates=True, infer_schema_length=None)
    first = df.columns[0]
    if first != DATE:
        df = df.rename({first: DATE})
    close = [c for c in df.columns if c.endswith("|Close")]
    if close:
        df = df.select(DATE, *[pl.col(c).alias(c[: -len("|Close")]) for c in close])
    return df.with_columns(pl.col(DATE).cast(pl.Date)).sort(DATE)


def _instruments(prices: pl.DataFrame) -> list[str]:
    return [c for c in prices.columns if c != DATE]


def _diff_and_vol(prices: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Dagliga prisdifferenser och rullande volatilitet (känd vid t, dvs utan dag t+1)."""
    inst = _instruments(prices)
    diff = prices.select(pl.col(inst).diff())
    vol = diff.select(pl.col(inst).rolling_std(VOL_WINDOW, min_samples=20))
    return diff.to_numpy(), vol.to_numpy()


def _rolling_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Summa över de senaste n raderna (inklusive aktuell rad). NaN tills n värden finns."""
    out = np.full_like(a, np.nan)
    c = np.nancumsum(np.nan_to_num(a), axis=0)
    out[n - 1 :] = c[n - 1 :] - np.vstack([np.zeros((1, a.shape[1])), c[:-n]])
    valid = np.isfinite(a).astype(float)
    cnt = np.cumsum(valid, axis=0)
    cnt_n = np.full_like(cnt, np.nan)
    cnt_n[n - 1 :] = cnt[n - 1 :] - np.vstack([np.zeros((1, a.shape[1])), cnt[:-n]])
    out[cnt_n < n] = np.nan
    return out


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Som pandas/polars shift: k>0 flyttar bakåt i tiden (värdet från t-k hamnar på t)."""
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def build_signal_and_target(
    prices: pl.DataFrame, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returnerar (x, y, vol) som matriser [dagar x instrument].

    x[t] : summan av de senaste `lookback` dagarnas avkastning, delat med vol[t]  -> känd vid t
    y[t] : summan av avkastningen på rad t+2 .. t+1+horizon, delat med vol[t]     -> FRAMTID
    """
    diff, vol = _diff_and_vol(prices)
    z = diff / vol  # volatilitetsnormaliserad avkastning
    x = _rolling_sum(diff, lookback) / vol
    fwd = _rolling_sum(diff, horizon)  # summa rad t-h+1..t
    y = _shift(fwd, -(POSITION_LAG - 1 + horizon)) / vol  # flytta så att rad t = summa t+2..t+1+h
    del z
    return x, y, vol


# --------------------------------------------------------------------------------------
# 1) Utforskning: vilka par leder vilka? (OBS: in-sample, bara för att hitta idéer)
# --------------------------------------------------------------------------------------
def lead_lag_table(
    prices: pl.DataFrame,
    lookbacks: tuple[int, ...] = (1, 5, 20),
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    end_date: str | None = None,
) -> pl.DataFrame:
    """Korrelation mellan ledarens senaste `lookback`-dagarsavkastning och målets
    framtida `horizon`-dagarsavkastning, för alla par.

    Sätt `end_date` för att bara utforska på träningsdata och spara resten som test.
    """
    if end_date is not None:
        prices = prices.filter(pl.col(DATE) <= pl.lit(end_date).str.to_date())
    inst = _instruments(prices)
    rows = []
    for L in lookbacks:
        for H in horizons:
            x, y, _ = build_signal_and_target(prices, L, H)
            for i, leader in enumerate(inst):
                for j, target in enumerate(inst):
                    m = np.isfinite(x[:, i]) & np.isfinite(y[:, j])
                    if m.sum() < 100:
                        continue
                    r = float(np.corrcoef(x[m, i], y[m, j])[0, 1])
                    # Överlappande fönster -> ungefär n / max(L, H) oberoende observationer
                    n_eff = m.sum() / max(L, H)
                    rows.append(
                        dict(leader=leader, target=target, lookback=L, horizon=H,
                             corr=r, t_stat=r * np.sqrt(n_eff), n=int(m.sum()))
                    )
    return pl.DataFrame(rows).sort(pl.col("t_stat").abs(), descending=True)


# --------------------------------------------------------------------------------------
# 2) Walk-forward-strategi
# --------------------------------------------------------------------------------------
@dataclass
class LeadLagConfig:
    lookback: int = 5          # hur många dagars historik som utgör signalen
    horizon: int = 5           # hur många dagar framåt vi förutsäger / håller
    leaders: str = "all"       # "self" (bara eget instrument), "others" eller "all"
    pairs: list[tuple[str, str]] | None = None  # explicit [(ledare, mål), ...] skriver över `leaders`
    train_window: int | None = 1000  # None = expanderande fönster (ignoreras om halflife är satt)
    halflife: float | None = None    # exponentiell viktning: en observation väger hälften efter så många dagar
    min_train: int = 500       # minsta antal träningsobservationer innan vi handlar
    min_t_stat: float = 0.0    # använd bara ledare vars walk-forward t-stat överstiger detta
    smooth: bool = True        # snitta positionen över `horizon` dagar (överlappande portföljer)
    max_pos: float = 3.0       # tak på prognosen i "std-enheter" innan division med vol


def _walk_forward_beta(x: np.ndarray, y: np.ndarray, horizon: int, cfg: LeadLagConfig):
    """Beta och t-stat för y ~ beta * x, skattat med enbart data känd vid varje t.

    Paret (x_s, y_s) blir känt vid s + 1 + horizon, så vid t får vi använda s <= t - 1 - horizon.
    """
    known = POSITION_LAG - 1 + horizon
    m = np.isfinite(x) & np.isfinite(y)
    xy = np.where(m, x * y, 0.0)
    xx = np.where(m, x * x, 0.0)
    yy = np.where(m, y * y, 0.0)
    cnt = m.astype(float)

    def known_at_t(c):
        return _shift(c[:, None], known)[:, 0]  # bara det som var känt vid t

    def ew_sum(a, decay):
        # Exponentiellt viktad löpande summa: S_t = decay * S_{t-1} + a_t
        c = np.empty_like(a)
        s = 0.0
        for k, v in enumerate(a):
            s = decay * s + v
            c[k] = s
        return c

    def cum(a):
        if cfg.halflife is not None:
            return known_at_t(ew_sum(a, 0.5 ** (1.0 / cfg.halflife)))
        c = np.cumsum(a)
        if cfg.train_window is not None and cfg.train_window < len(c):  # kortare data: fönstret täcker allt
            w = cfg.train_window
            c = c - np.concatenate([np.zeros(w), c[:-w]])
        return known_at_t(c)

    sxy, sxx, syy = cum(xy), cum(xx), cum(yy)
    n_raw = known_at_t(np.cumsum(cnt))  # antal faktiska observationer, för min_train
    if cfg.halflife is not None:
        # Effektivt antal observationer med vikter w: (sum w)^2 / sum w^2
        decay = 0.5 ** (1.0 / cfg.halflife)
        sw, sw2 = known_at_t(ew_sum(cnt, decay)), known_at_t(ew_sum(cnt, decay**2))
        with np.errstate(invalid="ignore", divide="ignore"):
            n_eff = sw**2 / sw2
    else:
        n_eff = cum(cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = sxy / sxx
        corr = sxy / np.sqrt(sxx * syy)
        t_stat = corr * np.sqrt(n_eff / horizon)  # grov korrigering för överlapp
    beta[(n_raw < cfg.min_train) | ~np.isfinite(beta)] = np.nan
    return beta, t_stat


def lead_lag_positions(prices: pl.DataFrame, cfg: LeadLagConfig = LeadLagConfig()) -> pl.DataFrame:
    """Bygger positioner (AsOfDate + en kolumn per instrument) som kan matas in i evaluation.py."""
    inst = _instruments(prices)
    idx = {c: k for k, c in enumerate(inst)}
    x, y, vol = build_signal_and_target(prices, cfg.lookback, cfg.horizon)
    T = len(prices)

    if cfg.pairs is not None:
        pairs = [(idx[a], idx[b]) for a, b in cfg.pairs]
    else:
        pairs = [
            (i, j) for i in range(len(inst)) for j in range(len(inst))
            if cfg.leaders == "all" or (cfg.leaders == "self") == (i == j)
        ]

    forecast = np.zeros((T, len(inst)))
    n_used = np.zeros((T, len(inst)))
    for i, j in pairs:
        beta, t_stat = _walk_forward_beta(x[:, i], y[:, j], cfg.horizon, cfg)
        pred = beta * x[:, i]
        use = np.isfinite(pred) & (np.abs(t_stat) >= cfg.min_t_stat)
        forecast[:, j] += np.where(use, pred, 0.0)
        n_used[:, j] += use

    with np.errstate(invalid="ignore", divide="ignore"):
        forecast = np.where(n_used > 0, forecast / n_used, 0.0)

    # Skala prognosen till "std-enheter" med en walk-forward-std, klipp, och dela med vol
    # så att varje instrument bidrar med ungefär lika mycket risk.
    f = pl.DataFrame(forecast, schema=inst)
    f_std = f.select(pl.all().replace(0.0, None).rolling_std(500, min_samples=100).shift(1)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.clip(forecast / f_std, -cfg.max_pos, cfg.max_pos)
    score = np.nan_to_num(score)

    if cfg.smooth and cfg.horizon > 1:
        # Varje dags signal "äger" en h-dagars hållning; snittet av de h senaste = överlappande portföljer.
        score = pl.DataFrame(score, schema=inst).select(
            pl.all().rolling_mean(cfg.horizon, min_samples=1)
        ).to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)

    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


# --------------------------------------------------------------------------------------
# 3) Multivariat lead-lag: alla ledare i en ridge-regression per mål (walk-forward)
# --------------------------------------------------------------------------------------
def _forecast_to_positions(prices: pl.DataFrame, forecast: np.ndarray, vol: np.ndarray,
                           horizon: int, max_pos: float, smooth: bool) -> pl.DataFrame:
    """Samma skalning som lead_lag_positions: walk-forward-std -> klipp -> utjämning -> / vol."""
    inst = _instruments(prices)
    f_std = pl.DataFrame(forecast, schema=inst).select(
        pl.all().replace(0.0, None).rolling_std(500, min_samples=100).shift(1)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.nan_to_num(np.clip(forecast / f_std, -max_pos, max_pos))
    if smooth and horizon > 1:
        score = pl.DataFrame(score, schema=inst).select(pl.all().rolling_mean(horizon, min_samples=1)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.nan_to_num(score / vol)
    return pl.DataFrame(pos, schema=inst).insert_column(0, prices.get_column(DATE))


def ridge_lead_lag_positions(
    prices: pl.DataFrame,
    lookback: int = 20,
    horizon: int = 20,
    ridge: float = 1.0,
    train_window: int | None = 1000,
    min_train: int = 500,
    max_pos: float = 3.0,
    smooth: bool = True,
) -> pl.DataFrame:
    """För varje mål B:  y_B ~ X @ beta_B, där X = alla instruments signaler (inkl. B själv).

    beta_B,t = (X'X + λ I)^-1 X'y, skattat på par (x_s, y_s) som var helt kända vid t
    (s <= t - 1 - horizon), inom ett rullande fönster. λ = ridge * medel(diag(X'X)), så att
    ridge är skalfri. Gemensam skattning gör att ledare som bär samma information inte
    dubbelräknas, till skillnad från snittet av parvisa regressioner.
    """
    x, y, vol = build_signal_and_target(prices, lookback, horizon)
    T, n = x.shape
    known = POSITION_LAG - 1 + horizon
    xm = np.isfinite(x).all(axis=1)

    def windowed_cumsum(a):
        c = np.cumsum(a, axis=0)
        if train_window is not None and train_window < len(c):
            w = train_window
            c[w:] = c[w:] - c[:-w].copy()
        out = np.full_like(c, np.nan)
        out[known:] = c[:-known]  # bara det som var känt vid t
        return out

    x0 = np.where(xm[:, None], x, 0.0)
    xx = windowed_cumsum(np.einsum("ti,tj->tij", x0, x0))
    cnt = windowed_cumsum(xm.astype(float))
    forecast = np.zeros((T, n))
    for j in range(n):
        m = xm & np.isfinite(y[:, j])
        xj = np.where(m[:, None], x, 0.0)
        yj = np.where(m, y[:, j], 0.0)
        xx_j = windowed_cumsum(np.einsum("ti,tj->tij", xj, xj))
        xy_j = windowed_cumsum(xj * yj[:, None])
        n_j = windowed_cumsum(m.astype(float))
        for t in range(T):
            if not (n_j[t] >= min_train) or not xm[t]:
                continue
            A = xx_j[t]
            lam = ridge * np.trace(A) / n
            beta = np.linalg.solve(A + lam * np.eye(n), xy_j[t])
            forecast[t, j] = x[t] @ beta
    del xx, cnt
    return _forecast_to_positions(prices, forecast, vol, horizon, max_pos, smooth)
