"""Inlämningsstrategi: trend (MA-crossover + hysteres) + makro (BNP och KPI per region),
med lika risk per tillgångsslag (valutor, obligationer, aktier, råvaror) och villkorad volatilitetsminskning.
Lead-lag och dynamisk viktning finns kvar som avstängda val (USE_LEAD_LAG, USE_DYNAMIC_WEIGHTS).

Användning:
    python submission.py <priser.csv> [positions_ut.csv] [--macro <makro.csv>] [--eval]

    --macro: makrofil med kolumner "gdp|<region>" och "cpi|<region>". Utan den körs bara trend.

    priser.csv: första kolumnen datum, övriga en kolumn per instrument (som example_prices.csv).
    Skriver positioner i evaluation.py-format (AsOfDate + en kolumn per instrument).
    --eval: skriver ut nyckeltal för hela perioden, träning (1995-2011) och validering (2012-2016, VALIDATION_START)
            och sparar dashboards (kräver evaluation.py).

Strategi (alla steg använder bara data känd vid varje dag):
    1. Trend på tre tidsskalor, s = 100, 250 och 500 dagar (TREND_SPEEDS), kombinerade med lika risk.
       För varje s används två mått, också med lika risk:
         a) Moving average crossover, s/4 mot s dagar. Skillnaden normaliseras till ett z-värde
            (delas med vol och sin teoretiska std) och signalen är tanh(2z), kontinuerlig.
         b) Momentum-z över s/2 och s dagar, klassat -1/0/+1 med hysteres
            (in vid |z| > 0.5, ut när z byter tecken).
       (USE_MA_CROSS / USE_HYSTERESIS / USE_BREAKOUT slår av och på trendmåtten.)
    2. Lead-lag (om USE_LEAD_LAG): varje instruments senaste 20-dagarsavkastning förutsäger varje instruments
                 kommande 20 dagar, parvisa regressioner skattade walk-forward (fönster 1000 dagar).
    3. Kombination: basvikter 67 % av risken till trend, 33 % till lead-lag (riskviktat på historisk
       P&L-vol). Med USE_DYNAMIC_WEIGHTS justeras vikterna varje dag och per instrument:
           vikt ∝ basvikt * exp(DYN_STRENGTH * strategins justerade Sharpe i instrumentet,
                                senaste DYN_WINDOW dagarna)
       normaliserat till summa 1. Den strategi som nyligen fungerat bäst i ett instrument får mer vikt där.
    3b. Makro (om --macro): för varje instrument förutsägs kommande 20 dagars avkastning av regionens
       BNP- och KPI-förändring över 20 och 120 dagar (valutor: regionen minus titan, råvaror: snitt
       över regionerna). Alla fyra variablerna skattas i EN gemensam ridge-regression per instrument
       (USE_MACRO_JOINT), walk-forward på ett rullande fönster om MACRO_TRAIN_WINDOW = 2500 dagar
       (ca 10 år) i stället för all historik, så att sambandet får ändras över tiden. Ridge behövs
       eftersom tillväxt_20 och tillväxt_120 är korrelerade ~0.9. Signalen klipps vid
       ±MACRO_MAX_POS = 2 std, vilket sänker kurtosen. MACRO_RISK_SHARE av risken.
    3c. Tillgångsslag (om USE_GROUPS): instrumenten delas i valutor (*_fx), obligationer (*_bonds),
       aktier (*_stocks) och råvaror (övriga). Trend och makro vägs ihop inom varje grupp (makroandel
       MACRO_RISK_SHARE, eller MACRO_SHARE_BY_GROUP för grupper som avviker: obligationer får 0 %,
       där makro inte tjänar något), och grupperna får sedan lika risk. Utan detta får varje instrument lika risk, och råvarorna (13 av 27) dominerar.
    4. Positionsstorlek efter signalstyrka: positionen är proportionell mot signalen (gamma 1).
       gamma < 1 krymper starka signaler (lägre kurtos), gamma > 1 förstärker dem.
    5. Villkorad volatilitetsminskning: när portföljens korta vol (20d) överstiger 1.25 x lång vol
       (250d) skalas risken ner till den långa nivån. Aldrig uppskalning.

Kräver: lead_lag.py, trend.py, combine.py, overlays.py, dynamic_allocation.py, macro.py,
macro_joint.py (polars, numpy).
"""

from __future__ import annotations

import sys

import polars as pl

from datetime import date

from combine import combine_positions
from dynamic_allocation import dynamic_allocation
from lead_lag import LeadLagConfig, _instruments, lead_lag_positions, load_prices
from macro import asset_class, load_macro, macro_positions
from macro_joint import macro_joint_positions
from overlays import compress_risk, conditional_vol_target
from trend import breakout_positions, hysteresis_trend_positions, ma_cross_tanh_positions

TREND_SPEEDS = (100, 250, 500)  # trendens tidsskalor i dagar; (100,) ger den tidigare enkla trenden
TREND_WINDOWS = (50, 100)  # bara för breakout och äldre testskript
MA_PAIRS = ((25, 100),)  # används bara av äldre testskript (motsvarar s = 100)
USE_MA_CROSS = True
USE_HYSTERESIS = True
TANH_K = 2.0
USE_BREAKOUT = False  # Donchian-breakout som ytterligare trendmått
HYST_ENTER, HYST_EXIT = 0.5, 0.0
LL_CONFIG = LeadLagConfig(lookback=20, horizon=20, leaders="all", train_window=1000)
USE_LEAD_LAG = False  # avstängd: lead-lag över alla instrument tjänar inget på prices_insample.csv
TREND_RISK_SHARE = 0.67
USE_DYNAMIC_WEIGHTS = False  # True = vikterna mellan trend och lead-lag följer senaste resultat per instrument
DYN_WINDOW = 250     # dagar som justerad Sharpe räknas över
DYN_STRENGTH = 0.5   # 0 = fasta vikter; högre = vikterna följer senaste resultat mer
MACRO_RISK_SHARE = 0.33  # andel av risken till makrostrategin när en makrofil ges
MACRO_WINDOWS = (20, 120)  # dagar som BNP- och KPI-förändringen mäts över
MACRO_TRAIN_WINDOW = 2500  # rullande fönster (dagar) som makrobeta skattas på; None = all historik
MACRO_HALFLIFE = None  # alternativ till fönstret: exponentiell viktning, halv vikt efter så många dagar
MACRO_MAX_POS = 2.0  # tak på makrosignalen i std-enheter innan division med vol
MACRO_SHARE_BY_GROUP = {"bonds": 0.0}  # avvikande makroandel per tillgångsslag; övriga får MACRO_RISK_SHARE
USE_MACRO_JOINT = True  # True = en gemensam ridge-regression på alla makrovariabler; False = snitt av enkelregressioner
MACRO_RIDGE = 0.3  # ridge-styrka i den gemensamma regressionen (andel av medel(diag(X'X)))
USE_GROUPS = True  # lika risk per tillgångsslag i stället för per instrument
MACRO_LAG = 0  # handelsdagar som makrodatan fördröjs (höj om den publiceras med eftersläpning)
VOL_THRESHOLD = 1.25
VALIDATION_START = date(2012, 1, 1)  # validering = från detta datum (2012-2016); None ger andelen TRAIN_FRAC i stället
TRAIN_FRAC = 0.7  # används bara om VALIDATION_START är None: andel av dagarna som är träning
COMPRESS_GAMMA = 1.0  # positionsstorlek efter signalstyrka: 1 = proportionell, <1 krymper starka, >1 förstärker


def build_positions(prices: pl.DataFrame, macro_path: str | None = None,
                    extra_trend: pl.DataFrame | None = None, extra_trend_share: float = 0.0,
                    use_ma_cross: bool = USE_MA_CROSS, vol_brake: bool = True) -> pl.DataFrame:
    """use_ma_cross=False: Linns trend består bara av hysteres-momentum.
    vol_brake=False: hoppa över den villkorade volbromsen (när den läggs på en större portfölj i stället).
    extra_trend: ytterligare trendpositioner (t.ex. Davids MA10/250, på hela universumet) som vägs in i
    trenddelen med andelen extra_trend_share av trendrisken, före gruppvikter, makro och volbroms."""
    speeds = {}
    for sp in TREND_SPEEDS:
        parts = {}
        if use_ma_cross:
            parts["ma_cross"] = ma_cross_tanh_positions(prices, pairs=((sp // 4, sp),), k=TANH_K)
        if USE_HYSTERESIS:
            parts["hysteresis"] = hysteresis_trend_positions(prices, windows=(sp // 2, sp), enter=HYST_ENTER, exit=HYST_EXIT)
        speeds[f"s{sp}"] = next(iter(parts.values())) if len(parts) == 1 else combine_positions(parts, prices)
    if USE_BREAKOUT:
        speeds["breakout"] = breakout_positions(prices, windows=TREND_WINDOWS)
    trend = next(iter(speeds.values())) if len(speeds) == 1 else combine_positions(speeds, prices)
    if extra_trend is not None and extra_trend_share > 0:
        trend = combine_positions({"own": trend, "extra": extra_trend}, prices,
                                  {"own": 1 - extra_trend_share, "extra": extra_trend_share})
    if USE_LEAD_LAG:
        strategies = {"trend": trend, "lead_lag": lead_lag_positions(prices, LL_CONFIG)}
        base = {"trend": TREND_RISK_SHARE, "lead_lag": 1 - TREND_RISK_SHARE}
        if USE_DYNAMIC_WEIGHTS:
            combined = dynamic_allocation(strategies, prices, base, method="adjusted_sharpe",
                                          strength=DYN_STRENGTH, window=DYN_WINDOW, per_instrument=True)
        else:
            combined = combine_positions(strategies, prices, base)
    else:
        combined = trend
    macro = None
    if macro_path is not None:
        macro_raw = load_macro(macro_path, prices, lag=MACRO_LAG)
        if USE_MACRO_JOINT:
            macro = macro_joint_positions(prices, macro_raw, windows=MACRO_WINDOWS,
                                          train_window=MACRO_TRAIN_WINDOW, ridge=MACRO_RIDGE,
                                          max_pos=MACRO_MAX_POS)
        else:
            macro = macro_positions(prices, macro_raw, windows=MACRO_WINDOWS,
                                    train_window=MACRO_TRAIN_WINDOW, halflife=MACRO_HALFLIFE,
                                    max_pos=MACRO_MAX_POS)
    inst = _instruments(prices)
    groups = sorted({asset_class(c) for c in inst})
    if USE_GROUPS and len(groups) > 1:
        def only(pos, g):
            return pos.with_columns([pl.lit(0.0).alias(c) for c in inst if asset_class(c) != g])

        subs = {}
        for g in groups:
            share = MACRO_SHARE_BY_GROUP.get(g, MACRO_RISK_SHARE) if macro is not None else 0.0
            if share <= 0:
                subs[g] = only(combined, g)
            else:
                subs[g] = combine_positions({"core": only(combined, g), "macro": only(macro, g)}, prices,
                                            {"core": 1 - share, "macro": share})
        combined = combine_positions(subs, prices)
    elif macro is not None:
        combined = combine_positions({"core": combined, "macro": macro}, prices,
                                     {"core": 1 - MACRO_RISK_SHARE, "macro": MACRO_RISK_SHARE})
    combined = compress_risk(combined, prices, gamma=COMPRESS_GAMMA)
    positions = (conditional_vol_target(combined, prices, threshold=VOL_THRESHOLD, days=1, mode="reduce")
                 if vol_brake else combined)

    # Skydd: evaluation.py kraschar om ett instrument bara har nollor
    empty = [c for c in _instruments(positions) if positions[c].abs().sum() == 0]
    if empty:
        print(f"VARNING: inga positioner i {empty} (för kort historik?)", file=sys.stderr)
    return positions


def split_dates(prices: pl.DataFrame, train_frac: float = TRAIN_FRAC):
    """Kronologisk uppdelning: träning före VALIDATION_START, validering därefter
    (eller de första train_frac av dagarna som träning om VALIDATION_START är None)."""
    dates = prices["AsOfDate"]
    if VALIDATION_START is not None and train_frac == TRAIN_FRAC:
        cut = int((dates < VALIDATION_START).sum())
    else:
        cut = int(len(dates) * train_frac)
    return dates[0], dates[cut - 1], dates[cut], dates[-1]


def evaluate(positions: pl.DataFrame, prices: pl.DataFrame) -> None:
    from evaluation import calc_key_figures, plot_key_figures

    kf = ["adjusted_sharpe", "sharpe", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias"]
    t0, t1, v0, v1 = split_dates(prices)
    periods = {"hela": (t0, v1), "träning": (t0, t1), "validering": (v0, v1)}
    for name, (a, b) in periods.items():
        pos = positions.filter(pl.col("AsOfDate").is_between(a, b))
        r = calc_key_figures(pos, prices, key_figures=kf)
        print(f"{name:15s} {a} – {b}  " + "  ".join(f"{k}={r[k]:.2f}" for k in kf))
    plot_key_figures(positions, prices).savefig("submission_dashboard.png", dpi=90, bbox_inches="tight")
    val = positions.filter(pl.col("AsOfDate") >= v0)
    plot_key_figures(val, prices).savefig("submission_dashboard_validation.png", dpi=90, bbox_inches="tight")
    print("Dashboards sparade: submission_dashboard.png (hela), submission_dashboard_validation.png")


def main() -> None:
    argv = sys.argv[1:]
    macro_path = None
    if "--macro" in argv:
        k = argv.index("--macro")
        macro_path = argv[k + 1]
        del argv[k : k + 2]
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    prices = load_prices(args[0])
    positions = build_positions(prices, macro_path)
    out = args[1] if len(args) > 1 else "positions.csv"
    positions.write_csv(out)
    print(f"Positioner sparade: {out} ({len(positions)} dagar, {len(_instruments(positions))} instrument)")
    if "--eval" in sys.argv:
        evaluate(positions, prices)


if __name__ == "__main__":
    main()
