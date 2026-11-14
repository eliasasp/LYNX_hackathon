"""Pairs trading med kointegration.

Upplägg:
  0. Dataanalys: priser, korrelationer och de mest korrelerade paren.
  1. URVAL:          de N mest korrelerade paren på träningsperioden (minst MIN_CORR) blir kandidater.
  2. FAS 1 – skatta: för varje kandidat skattas jämvikten log A = alpha + beta * log B + brus på träningsdata.
                     Screening av bruset (residualen): ADF-t, Hurst-exponent, half-life och nollgenomgångar per år.
                     Med REQUIRE_COINT handlas bara par som klarar alla fyra kraven.
  3. FAS 2 – handel: dag för dag efter träningen räknas en z-score för bruset med EWMA (minne = parets half-life).
                     Stor avvikelse → öppna position mot jämvikten. Stäng när bruset är tillbaka vid medel,
                     vid stop-loss eller när failsafe-nivån nås (failsafe är avstängd som standard).
  4. Utvärdering: affärslista + Lynx nyckeltal (evaluation.py).

Kör filen direkt; vad som skrivs ut och plottas styrs av inställningarna nedan.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# ================================ INSTÄLLNINGAR ================================
# ---------- Dataanalys: True = visa, False = hoppa över (hela datan, bara för att titta) ----------
PLOT_PRICES      = False  # stängningspriser, en panel per tillgång
PLOT_MACRO       = False  # makrodata (GDP, CPI)
PLOT_CORR        = False  # korrelationsmatris mellan tillgångarna
PLOT_ROLL_CORR   = False  # rullande EWMA-korrelation för topp-paren
PLOT_PAIR_PRICES = False  # priser för topp-paren
PLOT_PAIR_LOG    = False  # log-priser för topp-paren

TOP_PAIRS    = 15     # antal par i dataanalysen
PRINT_PAIRS  = False  # skriv ut topp-paren (hela datan) i terminalen
EWM_HALFLIFE = 63     # halveringstid i handelsdagar för den rullande korrelationen (63 ≈ ett kvartal)

# ---------- Strategi ----------
RUN_STRATEGY   = True
PRINT_SCREENING = True  # skriv ut screeningtabellen: de fyra kraven per kandidatpar och beslutet (handlas / handlas ej)
PLOT_SCREENING = True   # scorecard: en panel per krav, en stapel per kandidatpar mot gränsvärdet
SAVE_SCREENING = True   # spara screeningtabellen till pairs_trading/pairs_screening.csv
PLOT_RESIDUALS = True   # översikt: brustermen för alla kandidatpar
PLOT_TRADES    = True   # en figur per handlat par: brusterm med affärer, z-score och resultat
SAVE_TRADES    = True   # spara affärslistan till pairs_trading/pairs_trades.csv
PRINT_DETAILS  = False  # True = skriv även ut alla affärer och stängningsskäl (annars bara sammanfattningen)

TRAIN_START = "2013-01-01"  # träningsperiod: urval, alpha, beta och half-life skattas på [TRAIN_START, TRAIN_END]
TRAIN_END   = "2014-12-31"  # handel från första handelsdagen efter TRAIN_END (out-of-sample)

N_TRADE_PAIRS = 15     # kandidater: de N mest korrelerade paren. Korrelationen mäts bara på träningsperioden,
                       #   alltså på det som var känt när handeln börjar (ingen lookahead i urvalet)
MIN_CORR      = 0.8    # ... och bara par med minst denna korrelation (dagliga prisförändringar).
                       #   0.4 släpper in alla 15 (det 15:e paret har 0.47 på 2013–14); screeningen nedan gör urvalet

# ---------- Löpande omscreening ----------
RESCREEN_DAYS   = 50    # None = urval och screening görs EN gång på träningsperioden och gäller hela handelsperioden.
                        #   Heltal = gör om urval + screening var RESCREEN_DAYS:e handelsdag (t.ex. 250 ≈ ett år) på de
                        #   senaste RESCREEN_WINDOW dagarna. Par som inte längre klarar kraven öppnar inga nya affärer
                        #   (en öppen affär sköts tills den stängs); nya par som klarar kraven börjar handlas.
APPROVAL_DAYS   = 250   # ett godkännande gäller så här många handelsdagar (250 ≈ ett år) från screeningen: paret får
                        #   öppna affärer så länge det har klarat minst en screening inom APPROVAL_DAYS. None = bara
                        #   fram till nästa screening
RESCREEN_WINDOW = 1250  # fönster i handelsdagar bakåt för varje omscreening (1250 ≈ 5 år; 3 år gav för många tillfälliga
                        #   par). None = lika långt som träningsperioden

# ---------- Screening: krav på brustermen i TRÄNINGSDATAN för att ett kandidatpar ska få handlas ----------
REQUIRE_COINT = True   # True = bara par som klarar ALLA fyra kraven nedan handlas. False = alla kandidater handlas
                       #   (kraven räknas och visas ändå, så att man kan se vad filtret hade gjort)
ADF_LAGS = 1           # antal laggade differenser i ADF-testet
ADF_CRIT = -2.57       # 1) ADF-t < ADF_CRIT: brustermen är stationär (ingen random walk). -2.57 = vanliga ADF-gränsen för 10 %
                       #    (5 %: -2.86).
                       #    Formellt Engle–Granger (alpha och beta är valda för att göra residualen så stationär som möjligt)
                       #    är striktare: 10 % -3.04, 5 % -3.34, 1 % -3.90. Kolumnen "EG-nivå" visar vilken av dem paret klarar.
HURST_MAX = 0.45       # 2) Hurst-exponent H < HURST_MAX: 0.5 = random walk, > 0.5 = trendar, < 0.5 = mean reversion
HURST_MAX_LAG = 100    #    H skattas på laggarna 2..HURST_MAX_LAG dagar
HL_MIN, HL_MAX = 2, 40  # 3) half-life i dagar: för snabb = brus som äts av avgifter, för långsam = affären hinner inte gå i mål
MIN_CROSS_PER_YEAR = 8   # 4) minst så många nollgenomgångar per år (brustermen korsar sitt medel): annars en långsam trend

# ---------- Spread och EWMA (hur z-scoren räknas) ----------
ROLLING_BETA = True     # True = alpha och beta skattas löpande med rullande OLS på de senaste BETA_WINDOW dagarna
                        #   (bara data t.o.m. dagen före). False = låsta från träningen → spreaden driver när relationen ändras
BETA_WINDOW  = 750      # fönster i handelsdagar för rullande OLS. 750 ≈ 3 år: med 250 anpassar sig alpha/beta så
                        #   snabbt att spreaden återgår till 0 av sig själv (utan vinst), 1250 hänger inte med
MEAN_MODE = "noll"      # vad bandet centreras kring:
                        #   "ewma" = rörligt EWMA-medel med span SPAN_MEAN_MULT × half-life (följer spreaden)
                        #   "noll" = fast 0. Spreaden är residualen från en regression med konstant, så dess jämvikt
                        #            är 0; bandet blir 0 ± Z·std, rör sig inte med dagsbruset och innehåller alltid 0
SPAN_MEAN_MULT = 1.5    # EWMA-span för MEDELVÄRDET = SPAN_MEAN_MULT × half-life (bara MEAN_MODE = "ewma").
                        #   1–1.5 = följer långsam drift i jämvikten (kräver FREEZE_AT_ENTRY = True),
                        #   5–10  = trögt medel som tvingar fram hård mean reversion (träffar stop oftare om paret bryts)
STD_MODE = "expanding"  # hur bandbredden (std) räknas:
                        #   "ewma"      = EWMA med span SPAN_STD
                        #   "shrink"    = SHRINK_W · std från träningen + (1 − SHRINK_W) · EWMA-std  (förankrade band)
                        #   "expanding" = std på alla avvikelser från TRAIN_START t.o.m. dag t  (nästan konstanta band)
                        #   Alla tre mäter avvikelsen från EWMA-medlet, alltså samma sak som z-scoren.
SPAN_STD = 250          # EWMA-span i dagar för std, fast och oberoende av half-life (250–500 = långsamma band).
                        #   None = samma span som medelvärdet (det gamla beteendet)
SHRINK_W = 0.7          # vikt på träningsperiodens std när STD_MODE = "shrink"
BAND_SCALE = "avvikelse"  # vad banden förankras i (startvärde för std och ankaret i "shrink"):
                        #   "nivå"      = spreadens spridning i träningen, inkl. långsam drift → bredare, försiktigare band
                        #                 (1.4–1.8× den typiska avvikelsen från medlet, mest för par med lång half-life)
                        #   "avvikelse" = typisk avvikelse från EWMA-medlet i träningen = samma sak som z-scoren mäter,
                        #                 så z = 1 är exakt en typisk avvikelse → smalare band och fler affärer
REHEDGE = True          # True = hedga om positionen varje dag med dagens beta och priser (kapital i A, beta · kapital i B),
                        #   så att det vi håller alltid är spreaden som handlas; exit och failsafe mäts på den.
                        #   False = antal enheter och beta låses vid inträdet: hedgen glider när beta och priserna ändras
                        #   och spreaden för de låsta enheterna kan driva iväg så att affären aldrig stängs
FREEZE_AT_ENTRY = True  # True = lås medel och std när positionen öppnas; exit och stop-loss mäts mot de låsta värdena,
                        #   så en affär kan bara stängas "vid medel" om spreaden faktiskt har gått tillbaka

# ---------- Skydd mot trend / random walk i spreaden ----------
VOL_ADAPT     = False  # True = vidga ALLA gränser (in, ut, stop, failsafe) när spreaden svänger kraftigare än i träningen.
                       #   Svängningarna mäts över SWING_HORIZON dagar, alltså bortom det kortsiktiga dagsbruset. Aldrig smalare.
SWING_HORIZON = 40     # svängningar = förändring i spreaden över så här många dagar
SWING_SPAN    = 60     # EWMA-span i dagar för hur stora svängningarna är just nu
ENTRY_CONFIRM = False  # True = gå inte in medan spreaden fortfarande rör sig utåt: vänta tills z vänder tillbaka in över ±Z_ENTRY
TREND_FILTER  = False  # True = inga nya affärer när spreaden de senaste VR_WINDOW dagarna betett sig som trend/random walk
VR_WINDOW     = 120    # fönster i dagar för variance ratio VR = Var(SWING_HORIZON-dagars förändring) / (SWING_HORIZON · Var(dagsförändring))
VR_MAX        = 1.0    # ny affär bara om VR < VR_MAX  (VR < 1 = återgång mot medel, ≈ 1 = random walk, > 1 = trend)

# ---------- In- och utgångar ----------
ENTRY_FREQ = 0.2  # dynamisk inträdesgräns: per par och screening sätts gränsen så att spreaden låg utanför bandet
                  #   ENTRY_FREQ av dagarna i screeningfönstret (0.2 ≈ var 5:e dag), dvs. (1 − ENTRY_FREQ)-kvantilen av |z|.
                  #   Par med tjocka svansar / långa svängningar får högre gräns. None = fast Z_ENTRY för alla par
Z_ENTRY = 1.5   # fast inträdesgräns, används bara när ENTRY_FREQ = None: öppna när |z| > Z_ENTRY
Z_EXIT  = 0.0   # stäng när z har gått tillbaka till ±Z_EXIT (gapet Z_ENTRY − Z_EXIT = förväntad vinst per affär i std)
Z_STOP  = None  # stop-loss när |z| > Z_STOP (mot låst medel/std om FREEZE_AT_ENTRY, annars mot det rörliga). None = av.
                #   OBS: med FREEZE_AT_ENTRY ligger failsafe-nivån vid z ≈ Z_ENTRY + FAILSAFE_STD, så ett Z_STOP under
                #   det slår alltid till först och failsafe-nivån gör ingen skillnad.
FAILSAFE_STD = 3.0  # None = ingen failsafe. Tal = failsafe: fast spreadnivå som låses när positionen öppnas, FAILSAFE_STD standardavvikelser
                    #   längre bort än inträdet (inträde vid z = 1.5 → failsafe vid z ≈ 5.5 mot inträdets medel).
                    #   Når spreaden nivån stängs positionen direkt, oavsett var EWMA-medlet har tagit vägen.
CAPITAL_PER_PAIR = 1.0  # kapital i ben A; ben B får beta * kapital
# ===============================================================================

ROOT = Path(__file__).resolve().parent.parent  # csv-filerna och evaluation.py ligger i repots rot
sys.path.insert(0, str(ROOT))
import evaluation  # noqa: E402  (Lynx nyckeltal)

read = lambda f: pd.read_csv(ROOT / f, index_col="AsOfDate", parse_dates=True)
prices, macro = read("close_prices_insample.csv"), read("macro_data_insample.csv")


# =============================== 0. DATAANALYS ================================
def plot_corr(df, title="Korrelation, dagliga prisförändringar", annotate=False):
    """Heatmap över korrelationen mellan kolumnernas dagliga förändringar (inte nivåer – trender ger falsk korrelation)."""
    C = df.diff().corr()
    fig, ax = plt.subplots(figsize=(0.4 * len(C) + 3, 0.4 * len(C) + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C)), C.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(C)), C.columns, fontsize=8)
    if annotate:
        for (i, j), v in pd.DataFrame(C.values).stack().items():
            if abs(v) >= 0.05:  # nära noll lämnas tomt
                ax.text(j, i, f"{v:.1f}".replace("0.", ".").replace("1.0", "1"), ha="center", va="center", fontsize=6,
                        color="w" if abs(v) > 0.6 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8), ax.set_title(title), fig.tight_layout()
    return C


def top_pairs(df, n=15, by_abs=True, verbose=True):
    """Returnerar (och skriver ut) de n mest korrelerade paren (dagliga förändringar) som en serie med index (a, b).
    by_abs=True rankar på |korrelation| så att starkt negativa par också kommer med; False = bara högst positiva."""
    C = df.diff().corr()
    pairs = C.where(np.triu(np.ones(C.shape, bool), k=1)).stack()       # varje par en gång, utan diagonalen
    pairs = pairs.loc[(pairs.abs() if by_abs else pairs).sort_values(ascending=False).index[:n]]
    if verbose:
        out = pairs.rename_axis(["tillgång 1", "tillgång 2"]).rename("korrelation").round(3).reset_index()
        out.index += 1
        print(f"\nTopp {n} mest korrelerade par (dagliga prisförändringar):\n{out.to_string()}\n")
    return pairs


def ewm_corr(df, a, b, halflife=EWM_HALFLIFE):
    """Rullande EWMA-korrelation mellan dagliga förändringar i a och b. Äldre dagar vägs ner, vikten halveras var halflife:e dag."""
    r = df[[a, b]].diff()
    return r[a].ewm(halflife=halflife, min_periods=halflife).corr(r[b])


def _grid(n, ncols=5, row_height=2.6):
    """n paneler i ett rutnät med ncols kolumner, delad x-axel; överblivna rutor döljs."""
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, row_height * nrows), sharex=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


_as_list = lambda pairs: list(pairs.index) if isinstance(pairs, pd.Series) else list(pairs)  # serie från top_pairs eller [(a, b), ...]


def plot_rolling_corr(df, pairs, halflife=EWM_HALFLIFE):
    """En panel per par: EWMA-korrelation över tid. Streckad linje = korrelation över hela perioden."""
    pairs = _as_list(pairs)
    fig, axes = _grid(len(pairs))
    for ax, (a, b) in zip(axes, pairs):
        ewm_corr(df, a, b, halflife).plot(ax=ax, lw=0.9, xlabel="")
        ax.axhline(df[a].diff().corr(df[b].diff()), c="k", ls="--", lw=0.8)
        ax.axhline(0, c="grey", lw=0.5)
        ax.set_ylim(-1, 1), ax.set_title(f"{a} / {b}", fontsize=8)
    fig.suptitle(f"Rullande EWMA-korrelation, dagliga förändringar (halveringstid {halflife} dagar)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))


def plot_pair_prices(df, pairs, log=False):
    """En panel per par med båda tillgångarnas priser (log=True: log-priser, där en konstant kvot blir ett konstant avstånd)."""
    pairs, P = _as_list(pairs), np.log(df) if log else df
    fig, axes = _grid(len(pairs))
    for ax, (a, b) in zip(axes, pairs):
        P[[a, b]].plot(ax=ax, lw=0.9, xlabel="")
        ax.legend(fontsize=7), ax.set_title(f"{a} / {b}", fontsize=8)
    fig.suptitle("Log-priser" if log else "Priser")
    fig.tight_layout(rect=(0, 0, 1, 0.97))


# ================================= 1. URVAL ==================================
def select_pairs(prices, n=None, min_corr=None, start=None, end=None):
    """Kandidatpar: de n mest (positivt) korrelerade paren på [start, end] (standard: träningsperioden), och bara de
    med korrelation ≥ min_corr. Returnerar en serie (a, b) → korrelation. Utan argument används N_TRADE_PAIRS och MIN_CORR."""
    n, min_corr = n or N_TRADE_PAIRS, MIN_CORR if min_corr is None else min_corr
    corr = top_pairs(prices.loc[start or TRAIN_START:end or TRAIN_END], n, by_abs=False, verbose=False)
    return corr[corr >= min_corr]


# ========================= 2. FAS 1: SKATTA & VALIDERA =========================
def ols(y, X):
    """OLS med intercept: y = c0 + c1*x1 + c2*x2 + ... Returnerar koefficienterna [c0, c1, ...] och deras standardfel."""
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return coef, se


def adf_tstat(s, lags=ADF_LAGS):
    """ADF-test: regressera Δs_t på s_{t-1} (plus laggade Δs för att ta bort autokorrelation).
    t-statistikan för s_{t-1} mäter dragningen tillbaka mot medel: ju mer negativ, desto mer stationär serie."""
    d = np.diff(s)                                                       # d[k] = s[k+1] - s[k]
    y = d[lags:]                                                         # Δs_t
    X = np.column_stack([s[lags:-1]] + [d[lags - i:-i] for i in range(1, lags + 1)])  # s_{t-1}, Δs_{t-1}, ...
    coef, se = ols(y, X)
    return coef[1] / se[1]


EG_CRIT = {"1 %": -3.90, "5 %": -3.34, "10 %": -3.04}  # Engle–Granger, 2 serier med konstant (MacKinnon 2010)


def eg_level(adf_t):
    """Strängaste Engle–Granger-nivån som ADF-t klarar ("1 %", "5 %", "10 %") eller "–" om ingen."""
    return next((lvl for lvl, crit in EG_CRIT.items() if adf_t < crit), "–")


def half_life_days(s):
    """Återgångshastighet: AR(1) på bruset, Δs_t = c + gamma * s_{t-1}.
    gamma < 0 betyder att bruset dras tillbaka; half-life = antal dagar tills halva avvikelsen är borta."""
    (_, gamma), _ = ols(np.diff(s), s[:-1])
    return -np.log(2) / gamma if gamma < 0 else np.inf  # gamma >= 0: ingen återställande kraft


def hurst_exponent(s, max_lag=HURST_MAX_LAG):
    """Hurst-exponent med variansmetoden: spridningen i förändringen över lag dagar växer som lag^H.
    Random walk: avvikelserna växer som √lag → H = 0.5. Mean reversion: avvikelserna slutar växa → H < 0.5 (närmare 0 = starkare).
    Trend: H > 0.5."""
    lags = np.arange(2, min(max_lag, len(s) // 4) + 1)
    tau = [np.std(s[lag:] - s[:-lag]) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def zero_crossings_per_year(s):
    """Antal gånger bruset korsar sitt medelvärde, omräknat till per år (252 handelsdagar).
    Få korsningar = långsam trend snarare än en spread som pendlar fram och tillbaka."""
    sign = np.sign(s - s.mean())
    sign = sign[sign != 0]
    return np.count_nonzero(sign[1:] != sign[:-1]) / (len(s) / 252)


def spread_metrics(s):
    """De fyra screeningmåtten för en spread s (array eller serie; NaN tas bort)."""
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    if len(s) < 60:  # för kort serie (t.ex. facit efter en sen omscreening): måtten går inte att skatta
        return dict(adf_t=np.nan, hurst=np.nan, half_life=np.nan, cross_per_year=np.nan)
    return dict(adf_t=adf_tstat(s), hurst=hurst_exponent(s), half_life=half_life_days(s),
                cross_per_year=zero_crossings_per_year(s))


def check_criteria(m):
    """Vilka av de fyra screeningkraven måtten m (från spread_metrics) klarar: dict krav → True/False."""
    return dict(adf=m["adf_t"] < ADF_CRIT, hurst=m["hurst"] < HURST_MAX,
                hl=HL_MIN <= m["half_life"] <= HL_MAX, cross=m["cross_per_year"] >= MIN_CROSS_PER_YEAR)


def validate_pair(prices, a, b, start=None, end=None):
    """FAS 1 för paret (a, b), bara på träningsdata. Returnerar en dict med alla mått.
    coint_ok = True om bruset klarar alla fyra screeningkraven (ADF-t, Hurst, half-life, nollgenomgångar).
    'facit' = samma mått på spreaden som handlas EFTER träningen – bara för utvärdering, används inte i beslutet."""
    start, end = start or TRAIN_START, end or TRAIN_END
    log_train = np.log(prices.loc[start:end, [a, b]])

    # 1. Jämviktsrelationen: log A = alpha + beta * log B + brus. beta = hedge-kvoten.
    (alpha, beta), _ = ols(log_train[a].values, log_train[b].values)

    # 2. Brustermen (residualen) = spreaden vi handlar på. alpha och beta är låsta från träningen,
    #    så bruset efter TRAIN_END är äkta out-of-sample.
    log_all = np.log(prices[[a, b]])
    spread = log_all[a] - (alpha + beta * log_all[b])
    s_train = spread.loc[start:end].values

    # 3. Screening av bruset på träningsdata: ADF-t (stationär?), Hurst (mean reverting?),
    #    half-life (hinner affären gå i mål?) och nollgenomgångar per år (tillräckligt många cykler?).
    metrics = spread_metrics(s_train)
    checks = check_criteria(metrics)
    coint_ok = all(checks.values())

    # 5. Spreaden som HANDLAS. Med ROLLING_BETA skattas alpha_t och beta_t om varje dag på de senaste BETA_WINDOW
    #    dagarna t.o.m. dagen före (shift(1) → ingen lookahead), så en långsamt ändrad relation inte ger en driftande spread.
    #    Utan ROLLING_BETA är det samma låsta spread som ovan.
    if ROLLING_BETA:
        la, lb = log_all[a], log_all[b]
        beta_t = la.rolling(BETA_WINDOW).cov(lb) / lb.rolling(BETA_WINDOW).var()
        alpha_t = la.rolling(BETA_WINDOW).mean() - beta_t * lb.rolling(BETA_WINDOW).mean()
        alpha_t, beta_t = alpha_t.shift(1), beta_t.shift(1)
    else:
        alpha_t, beta_t = pd.Series(alpha, index=spread.index), pd.Series(beta, index=spread.index)
    trade_spread = log_all[a] - (alpha_t + beta_t * log_all[b])

    # 6. Facit: samma fyra mått på spreaden som handlas efter träningen. Visar om paret behöll egenskaperna,
    #    men får INTE användas för att välja par (det vore lookahead).
    facit = spread_metrics(trade_spread[trade_spread.index > pd.Timestamp(end)])

    return dict(a=a, b=b, train_start=start, train_end=end, alpha=alpha, beta=beta, **metrics, checks=checks,
                coint_ok=coint_ok, facit=facit,
                spread=spread, alpha_t=alpha_t, beta_t=beta_t, trade_spread=trade_spread)


# ========================= 3. FAS 2: LÖPANDE HANDEL ===========================
def _band_params(v):
    """Parametrar för z-scoren från v:s träningsfönster (bara historik → ingen lookahead).
    Returnerar (fasta parametrar, startläge för de löpande EWMA-värdena)."""
    S = v["trade_spread"]  # spreaden vi handlar på (rullande eller låst beta, se validate_pair)
    start, end = v["train_start"], v["train_end"]

    # EWMA-minne för MEDELVÄRDET sätts av parets egen återgångstid: span = SPAN_MEAN_MULT × half-life, lambda = 2 / (span + 1).
    # STD får ett eget, fast span (SPAN_STD) så att banden inte krymper i lugna perioder eller blåser upp vid enstaka hopp.
    span_mean = SPAN_MEAN_MULT * v["half_life"]
    lam_mean = 0.0 if MEAN_MODE == "noll" else 2 / (span_mean + 1)  # lam 0 + startvärde 0 → medlet står kvar på 0
    lam_std = 2 / ((SPAN_STD or span_mean) + 1)

    # Startvärden från träningsfönstret, se BAND_SCALE:
    #   "avvikelse": EWMA-medlet körs genom träningen och std = typisk avvikelse från det (samma sak som z-scoren mäter)
    #   "nivå":      medel och std för spreadens nivå i träningen (inkl. långsam drift → bredare band)
    s_train = S.loc[start:end].dropna()
    if MEAN_MODE == "noll":
        mean_path = pd.Series(0.0, index=s_train.index)
    else:
        mean_path = s_train.ewm(alpha=lam_mean, adjust=False).mean()
    dev_train = s_train - mean_path
    if MEAN_MODE == "noll":  # avvikelsen från medlet är spreaden själv, oavsett BAND_SCALE
        mean, var = 0.0, (dev_train ** 2).mean()
    elif BAND_SCALE == "avvikelse":
        mean, var = mean_path.iloc[-1], (dev_train ** 2).mean()
    else:  # "nivå"
        mean, var = s_train.mean(), s_train.var()
    std_train = np.sqrt(var)  # ankaret i STD_MODE = "shrink"

    # Svängningar = förändring över SWING_HORIZON dagar. Typisk storlek i träningen + löpande EWMA (för VOL_ADAPT).
    swing = S - S.shift(SWING_HORIZON)
    swing_train = swing.loc[start:end].std()
    lam_swing = 2 / (SWING_SPAN + 1)
    # Variance ratio på de senaste VR_WINDOW dagarna, bara bakåt i tiden (för TREND_FILTER)
    vr = swing.rolling(VR_WINDOW).var() / (SWING_HORIZON * S.diff().rolling(VR_WINDOW).var())

    # Inträdesgräns: kvantil av |z| i fönstret, z mätt som avvikelse / typisk avvikelse (RMS) i fönstret
    if ENTRY_FREQ:
        z_train = dev_train / np.sqrt((dev_train ** 2).mean())
        z_entry = float(np.quantile(np.abs(z_train), 1 - ENTRY_FREQ))
    else:
        z_entry = Z_ENTRY

    params = (S, lam_mean, lam_std, std_train, swing, swing_train, lam_swing, vr, z_entry)
    running = (mean, var, len(dev_train), (dev_train ** 2).sum(), swing_train ** 2)  # mean, var, n_all, sumsq_all, swing_var
    return params, running


def trade_pair(prices, v, capital=CAPITAL_PER_PAIR, segments=None):
    """FAS 2 för ett par v (från validate_pair): går igenom handelsperioden dag för dag.
    Allt som räknas dag t använder bara data t.o.m. dag t, så ingen lookahead.
    (Lynx-konvention: en position daterad t handlas dag t+1 och ger avkastning från dag t+2; det sköter evaluation.py.)

    segments = [(från_datum, v eller None), ...] vid löpande omscreening: från varje datum gäller den screeningens
    parametrar (half-life, band); None = paret klarar inte kraven just då → inga nya affärer, en öppen affär sköts
    tills den stängs. Utan segments screenas paret en gång: v gäller från första dagen efter dess träningsfönster.

    Returnerar (days, trades):
      days   = DataFrame per dag: spread, EWMA-medel/-std, z-score, z_trade (z mot låsta värden under en affär),
               pos (+1/-1/0), antal enheter i A och B, och aktiv (= paret godkänt i senaste screeningen)
      trades = lista med en dict per affär (in/ut-datum, z, spread, failsafe-nivå, skäl till stängning)"""
    a, b = v["a"], v["b"]
    log_a, log_b = np.log(prices[a]), np.log(prices[b])
    if segments is None:
        segments = [(prices.index[prices.index > pd.Timestamp(v["train_end"])][0], v)]

    z_prev = 0.0
    pos, units_a, units_b, trade = 0, 0.0, 0.0, None
    blocked = False  # efter stop-loss/failsafe: vänta tills |z| < Z_ENTRY, annars öppnas samma förlustaffär igen direkt
    active, started, k, v_cur = False, False, 0, None
    days, trades = [], []
    for t in prices.index[prices.index >= segments[0][0]]:
        # 0. Ny screening från och med dag t: byt till dess parametrar. Med rullande beta är spreaden densamma, så de
        #    löpande EWMA-värdena fortsätter; med låst beta är det en ny spread och de startas om från fönstret.
        while k < len(segments) and t >= segments[k][0]:
            v_k = segments[k][1]
            k += 1
            same = v_k is not None and v_k is v_cur
            active, v_cur = v_k is not None, v_k
            if active and not same:
                (S, lam_mean, lam_std, std_train, swing, swing_train, lam_swing, vr, z_entry), running = _band_params(v_k)
                if not started or not ROLLING_BETA:
                    mean, var, n_all, sumsq_all, swing_var = running
                    started = True
        if not started:  # paret har ännu inte klarat någon screening
            days.append((t, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.0, 0.0, False, np.nan))
            continue
        s = S[t]
        # 1. Uppdatera rörligt medelvärde och EWMA-varians rekursivt med dagens spread
        mean = lam_mean * s + (1 - lam_mean) * mean
        dev = s - mean                                       # dagens avvikelse från medlet
        var = lam_std * dev ** 2 + (1 - lam_std) * var
        n_all, sumsq_all = n_all + 1, sumsq_all + dev ** 2

        # 2. Bandbredd (std) enligt STD_MODE
        if STD_MODE == "shrink":
            std = SHRINK_W * std_train + (1 - SHRINK_W) * np.sqrt(var)
        elif STD_MODE == "expanding":
            std = np.sqrt(sumsq_all / n_all)
        else:  # "ewma"
            std = np.sqrt(var)

        # 2b. VOL_ADAPT: svänger spreaden kraftigare än i träningen skalas bandbredden upp i samma proportion (aldrig ner).
        #     Då följer inträde, exit, stop-loss och failsafe med när random walk-delen blir starkare.
        swing_var = lam_swing * swing[t] ** 2 + (1 - lam_swing) * swing_var
        if VOL_ADAPT:
            std *= max(1.0, np.sqrt(swing_var) / swing_train)

        # 3. Z-score: hur många standardavvikelser spreaden ligger från sitt rörliga jämviktsläge (används för inträde)
        z = (s - mean) / std
        blocked = blocked and abs(z) >= z_entry
        z_trade = np.nan

        # 4a. Ingen position: öppna om avvikelsen är stor nog
        if pos == 0 and not blocked and active:
            if ENTRY_CONFIRM:  # vändning: igår utanför bandet, idag tillbaka innanför (men fortfarande långt från medel)
                go_short = z_prev > z_entry >= z > Z_EXIT
                go_long = z_prev < -z_entry <= z < -Z_EXIT
            else:
                go_short, go_long = z > z_entry, z < -z_entry
            if TREND_FILTER and not vr[t] < VR_MAX:  # spreaden beter sig som trend/random walk just nu → vänta
                go_short = go_long = False
            if go_short:         # spreaden hög: A dyr relativt B → short spread (blanka A, köp B)
                pos = -1
            elif go_long:        # spreaden låg: A billig relativt B → long spread (köp A, blanka B)
                pos = +1
            if pos != 0:
                # Lås dagens alpha och beta för affären: det är dem enheterna nedan bygger på.
                alpha_in, beta_in = v["alpha_t"][t], v["beta_t"][t]
                # Kapitalviktning med beta: B-benet får beta * kapital så att benens procentuella rörelser tar ut varandra.
                # Antalet enheter låses på dagens priser och hålls tills positionen stängs.
                units_a = pos * capital / prices.at[t, a]
                units_b = -pos * capital * beta_in / prices.at[t, b]
                # Lås medel och std vid inträdet (FREEZE_AT_ENTRY): exit/stop mäts mot dessa i stället för mot ett medel
                # som kan springa ikapp spreaden.
                mean_in, std_in = mean, std
                # Failsafe: fast spreadnivå längre bort i förlustriktningen (short: ovanför, long: nedanför).
                failsafe = s - pos * FAILSAFE_STD * std if FAILSAFE_STD is not None else np.nan
                trade = dict(a=a, b=b, riktning="long spread" if pos > 0 else "short spread", in_datum=t,
                             z_in=z, spread_in=s, beta_in=beta_in, mean_in=mean_in, std_in=std_in, failsafe=failsafe,
                             enheter_a=units_a, enheter_b=units_b)
                z_trade, s_trade = z, s

        # 4b. Öppen position: stäng vid återgång, stop-loss eller failsafe
        elif pos != 0:
            if REHEDGE:  # hedga om till dagens beta: det vi håller är spreaden S som handlas
                units_a = pos * capital / prices.at[t, a]
                units_b = -pos * capital * v["beta_t"][t] / prices.at[t, b]
                s_trade = s
            else:        # spreaden för de låsta enheterna (alpha/beta från inträdet) – det är den som ger PnL
                s_trade = log_a[t] - (alpha_in + beta_in * log_b[t])
            z_trade = (s_trade - mean_in) / std_in if FREEZE_AT_ENTRY else z
            if (pos == -1 and z_trade <= Z_EXIT) or (pos == +1 and z_trade >= -Z_EXIT):
                reason = "tillbaka vid medel"
            elif Z_STOP is not None and abs(z_trade) >= Z_STOP:
                reason = "stop-loss (z)"
            elif FAILSAFE_STD is not None and ((pos == -1 and s_trade >= trade["failsafe"])
                                               or (pos == +1 and s_trade <= trade["failsafe"])):
                reason = "failsafe (spreadnivå)"
            else:
                reason = None
            if reason:
                trades.append(trade | dict(ut_datum=t, z_ut=z_trade, spread_ut=s_trade, skäl=reason))
                pos, units_a, units_b, trade = 0, 0.0, 0.0, None
                blocked = reason != "tillbaka vid medel"

        days.append((t, s, mean, std, z, z_trade, pos, units_a, units_b, active, z_entry))
        z_prev = z

    if trade is not None:  # position som fortfarande är öppen när datan tar slut
        trades.append(trade | dict(ut_datum=t, z_ut=z_trade, spread_ut=s_trade, skäl="öppen vid periodens slut"))
    days = pd.DataFrame(days, columns=["AsOfDate", "spread", "mean", "std", "z", "z_trade", "pos", "units_a", "units_b",
                                       "aktiv", "z_entry"]
                        ).set_index("AsOfDate")
    return days, trades


def trade_pnl(prices, tr, days=None):
    """Resultat för en affär utan kostnader, med Lynx tidsförskjutning: signal dag t → handel på stängningskursen dag t+1.
    Med days (parets dagliga enheter) summeras resultatet dag för dag, så att omhedgning räknas med:
    positionen daterad d ger P(d+2) − P(d+1)."""
    if days is not None:
        legs = [tr["a"], tr["b"]]
        dp = prices[legs].diff().shift(-2).loc[tr["in_datum"]:tr["ut_datum"]].values
        units = days.loc[tr["in_datum"]:tr["ut_datum"], ["units_a", "units_b"]].values
        return float(np.nansum(units * dp))
    i = prices.index
    t_in = i[min(i.get_loc(tr["in_datum"]) + 1, len(i) - 1)]
    t_out = i[min(i.get_loc(tr["ut_datum"]) + 1, len(i) - 1)]
    return sum(tr[f"enheter_{leg}"] * (prices.at[t_out, tr[leg]] - prices.at[t_in, tr[leg]]) for leg in ("a", "b"))


def screening_windows(prices):
    """Screeningtillfällen: [(handel_från, fönster_start, fönster_slut), ...]. Utan RESCREEN_DAYS ett enda:
    träningsperioden. Annars var RESCREEN_DAYS:e handelsdag, på de RESCREEN_WINDOW handelsdagarna före."""
    idx = prices.index
    trade_idx = idx[idx > pd.Timestamp(TRAIN_END)]
    if not RESCREEN_DAYS:
        return [(trade_idx[0], TRAIN_START, TRAIN_END)]
    n_win = RESCREEN_WINDOW or ((idx >= pd.Timestamp(TRAIN_START)) & (idx <= pd.Timestamp(TRAIN_END))).sum()
    return [(t, idx[max(0, idx.get_loc(t) - n_win)], idx[idx.get_loc(t) - 1]) for t in trade_idx[::RESCREEN_DAYS]]


def run_strategy(prices):
    """Urval → FAS 1 → FAS 2 för varje kandidatpar, med omscreening var RESCREEN_DAYS:e dag om den är påslagen.
    Returnerar (results, positions, trades):
      results   = en dict per kandidatpar (mått från fas 1, plus 'days' och 'trades' för handlade par).
                  Vid omscreening: måtten från första screeningen där paret var kandidat, 'n_passed' = antal
                  screeningar paret klarade och 'n_screens' = antal screeningar
      positions = antal enheter per tillgång och dag, summerat över paren (Lynx-format)
      trades    = affärslista som DataFrame, en rad per affär"""
    windows = screening_windows(prices)
    screened = {}  # (a, b) → {screening nr: v}, bara screeningar där paret var kandidat
    for k, (_, w_start, w_end) in enumerate(windows):
        for (a, b), corr in select_pairs(prices, start=w_start, end=w_end).items():
            v = validate_pair(prices, a, b, w_start, w_end)
            v["corr"] = corr
            # Handla paret om bruset har någon återgång alls (ändlig half-life) och, med REQUIRE_COINT, klarar fas 1-kraven
            v["passed"] = bool(np.isfinite(v["half_life"]) and (v["coint_ok"] or not REQUIRE_COINT)
                               and v["trade_spread"].loc[w_start:w_end].notna().sum() >= 60)  # rullande β kräver historik
            screened.setdefault((a, b), {})[k] = v

    results = []
    positions = pd.DataFrame(0.0, index=prices.index[prices.index > TRAIN_END], columns=prices.columns)
    for (a, b), by_screen in screened.items():
        v = next(iter(by_screen.values())) | {}  # första screeningen där paret var kandidat (för tabeller och plottar)
        segments, last = [], None  # last = senaste godkända screeningen (nr, v)
        for k, (t, _, _) in enumerate(windows):
            if k in by_screen and by_screen[k]["passed"]:
                last = (k, by_screen[k])
            valid = last is not None and (last[0] == k or (APPROVAL_DAYS is not None and
                                          prices.index.get_loc(t) - prices.index.get_loc(windows[last[0]][0]) < APPROVAL_DAYS))
            segments.append((t, last[1] if valid else None))
        v["n_passed"] = sum(v_k["passed"] for v_k in by_screen.values())
        v["n_screens"] = len(windows)
        v["traded"] = any(seg is not None for _, seg in segments)
        if v["traded"]:
            v["days"], v["trades"] = trade_pair(prices, v, segments=segments)
            positions[a] += v["days"]["units_a"]  # en tillgång kan ingå i flera par
            positions[b] += v["days"]["units_b"]
        results.append(v)

    for v in results:
        for tr in v.get("trades", []):
            tr["resultat_%"] = 100 * trade_pnl(prices, tr, v["days"]) / CAPITAL_PER_PAIR
    trades = pd.DataFrame([tr for v in results for tr in v.get("trades", [])])
    if not trades.empty:
        trades["dagar"] = [prices.index.get_loc(o) - prices.index.get_loc(i) for i, o in zip(trades["in_datum"], trades["ut_datum"])]
        trades.index += 1
    return results, positions, trades


# ============================== 4. UTVÄRDERING ================================
to_pl = lambda df: pl.from_pandas(df.reset_index()).with_columns(pl.col("AsOfDate").cast(pl.Date))  # pandas → Lynx-format


def key_figures(pos):
    """Lynx nyckeltal med kostnader för positionerna pos: Sharpe, adjusted Sharpe och PnL i % av kapitalet per par."""
    pos = pos.loc[:, pos.abs().sum() > 0]  # tillgångar som aldrig handlas tas bort (annars 0/0 i evaluation.py)
    if pos.empty:
        return dict(sharpe=np.nan, adj_sharpe=np.nan, pnl_pct=0.0)
    kf = evaluation.calc_key_figures(to_pl(pos), to_pl(prices[pos.columns]), key_figures=["sharpe", "adjusted_sharpe", "returns"])
    return dict(sharpe=kf["sharpe"], adj_sharpe=kf["adjusted_sharpe"], pnl_pct=100 * kf["returns"]["returns"].sum() / CAPITAL_PER_PAIR)


def print_summary(results, positions, trades):
    """Kort sammanfattning: antal affärer, Sharpe, adjusted Sharpe och PnL per handlat par och totalt."""
    rows = {}
    for v in results:
        if v["traded"]:
            pair_pos = v["days"][["units_a", "units_b"]].set_axis([v["a"], v["b"]], axis=1)  # parets egna positioner
            rows[f"{v['a']} / {v['b']}"] = dict(affärer=len(v["trades"]), **key_figures(pair_pos))
    rows["TOTALT"] = dict(affärer=len(trades), **key_figures(positions))
    table = pd.DataFrame(rows).T.rename(columns={"adj_sharpe": "adj. sharpe", "pnl_pct": "PnL %"})
    table["affärer"] = table["affärer"].astype(int)
    print(f"\nRESULTAT {positions.index[0]:%Y-%m-%d} – {positions.index[-1]:%Y-%m-%d} "
          f"(Lynx nyckeltal med kostnader, PnL i % av kapitalet {CAPITAL_PER_PAIR} per par)")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"), "\n")


METRICS = ["adf_t", "hurst", "half_life", "cross_per_year"]                        # måtten, i samma ordning som kraven
CHECKS = dict(adf_t="adf", hurst="hurst", half_life="hl", cross_per_year="cross")  # mått → nyckel i check_criteria


def screening_table(results):
    """En rad per kandidatpar: måtten på träningsdatan (de som avgör), ok_* = om varje krav klaras, beslutet,
    och facit_* = samma mått efter träningen (bara utvärdering, används INTE i beslutet)."""
    rows = {}
    for v in results:
        facit_checks = check_criteria(v["facit"])
        rows[f"{v['a']} / {v['b']}"] = dict(
            korr=v["corr"], beta=v["beta"], **{m: v[m] for m in METRICS}, eg_nivå=eg_level(v["adf_t"]),
            **{f"ok_{k}": ok for k, ok in v["checks"].items()}, krav_klarade=sum(v["checks"].values()),
            klarar_alla=v["coint_ok"], handlas=v["traded"], godkänd=f"{v['n_passed']}/{v['n_screens']}",
            **{f"facit_{m}": v["facit"][m] for m in METRICS}, **{f"facit_ok_{k}": ok for k, ok in facit_checks.items()},
            facit_krav_klarade=sum(facit_checks.values()))
    return pd.DataFrame.from_dict(rows, orient="index")


def print_screening(table):
    """Skriver ut screeningtabellen: värde + ✓/✗ per krav, beslutet, och facit efter träningen."""
    mark = lambda ok: "✓" if ok else "✗"
    cell = lambda x, ok, fmt: f"{x:{fmt}} {mark(ok)}"
    fmts = dict(adf_t="6.2f", hurst="5.2f", half_life="6.1f", cross_per_year="5.1f")
    names = dict(adf_t="ADF-t", hurst="Hurst", half_life="HL dagar", cross_per_year="kors/år")
    pd.set_option("display.width", 250)

    show = pd.DataFrame(index=table.index)
    show["korr"], show["β"] = table["korr"].map("{:.2f}".format), table["beta"].map("{:.2f}".format)
    for m in METRICS:
        show[names[m]] = [cell(x, ok, fmts[m]) for x, ok in zip(table[m], table[f"ok_{CHECKS[m]}"])]
        if m == "adf_t":
            show["EG-nivå"] = table["eg_nivå"]
    show["krav"] = table["krav_klarade"].map("{}/4".format)
    show["beslut"] = table["handlas"].map({True: "HANDLAS", False: "handlas ej"})
    if RESCREEN_DAYS:
        show["godkänd"] = table["godkänd"]
    print(f"\nSCREENING på träningsdata {TRAIN_START} – {TRAIN_END}  (krav: ADF-t < {ADF_CRIT}, Hurst < {HURST_MAX}, "
          f"{HL_MIN} ≤ HL ≤ {HL_MAX} d, ≥ {MIN_CROSS_PER_YEAR} nollgenomgångar/år)")
    if not REQUIRE_COINT:
        print("OBS: REQUIRE_COINT = False → alla kandidater handlas oavsett krav")
    if RESCREEN_DAYS:
        print(f"OMSCREENING var {RESCREEN_DAYS}:e handelsdag, ett godkännande gäller {APPROVAL_DAYS} handelsdagar. Måtten nedan är från första screeningen där paret var "
              "kandidat; 'godkänd' = antal screeningar paret klarade av alla. Paret handlas (nya affärer) bara i "
              "perioderna efter en godkänd screening.")
    print(show.to_string(), "\n")

    facit = pd.DataFrame(index=table.index)
    for m in METRICS:
        facit[names[m]] = [cell(x, ok, fmts[m]) for x, ok in zip(table[f"facit_{m}"], table[f"facit_ok_{CHECKS[m]}"])]
    facit["krav"] = table["facit_krav_klarade"].map("{}/4".format)
    print("FACIT: samma mått på spreaden som handlas EFTER träningen (används inte i beslutet – visar om paret höll):")
    print(facit.to_string(), "\n")


def plot_screening(table):
    """Scorecard: en panel per krav och en stapel per kandidatpar (träningsdata).
    Grönt fält = godkänt område, streckad linje = gräns, grön/röd stapel = klarar/klarar ej.
    ◆ = samma mått efter träningen (facit – används inte i beslutet)."""
    hl_cap = 3 * HL_MAX  # längre half-life (eller ∞) ritas vid taket så att skalan inte förstörs
    panels = [  # mått, rubrik, godkänt intervall
        ("adf_t", f"ADF-t  (krav < {ADF_CRIT})", (-np.inf, ADF_CRIT)),
        ("hurst", f"Hurst H  (krav < {HURST_MAX}; 0.5 = random walk)", (-np.inf, HURST_MAX)),
        ("half_life", f"Half-life, dagar  (krav {HL_MIN}–{HL_MAX})", (HL_MIN, HL_MAX)),
        ("cross_per_year", f"Nollgenomgångar per år  (krav ≥ {MIN_CROSS_PER_YEAR})", (MIN_CROSS_PER_YEAR, np.inf)),
    ]
    n = len(table)
    y = np.arange(n)[::-1]  # första (mest korrelerade) paret överst
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 0.5 * n + 2.2), sharey=True)
    for ax, (m, title, (lo, hi)) in zip(axes, panels):
        x, fx = table[m].astype(float), table[f"facit_{m}"].astype(float)
        if m == "half_life":
            x, fx = x.clip(upper=hl_cap), fx.clip(upper=hl_cap)
        ok = table[f"ok_{CHECKS[m]}"]
        ax.barh(y, x, height=0.6, color=np.where(ok, "tab:green", "tab:red"), alpha=0.85)
        ax.scatter(fx, y, marker="D", s=30, c="k", edgecolors="w", linewidths=1, zorder=3)
        ax.axvline(0, c="k", lw=0.6)
        x0, x1 = ax.get_xlim()
        ax.axvspan(max(lo, x0), min(hi, x1), color="tab:green", alpha=0.08, zorder=0)
        for edge in (lo, hi):
            if np.isfinite(edge):
                ax.axvline(edge, c="k", ls="--", lw=0.9)
        # Värdet skrivs inne i stapeln vid dess ände (utanför om stapeln är för kort), så det inte krockar med ◆
        for yi, xi, raw in zip(y, x, table[m]):
            label = "∞" if np.isinf(raw) else f"{raw:.2f}" if abs(raw) < 10 else f"{raw:.0f}"
            inside = abs(xi) > 0.15 * (x1 - x0)
            toward_zero = "left" if xi < 0 else "right"
            ax.text(xi, yi, f" {label} ", va="center", fontsize=7, c="w" if inside else "k",
                    ha=toward_zero if inside else {"left": "right", "right": "left"}[toward_zero])
        ax.set_xlim(x0, x1), ax.set_title(title, fontsize=9), ax.grid(axis="x", alpha=0.3)
        if m == "half_life":
            ax.text(1, -0.08, f"stapel vid {hl_cap} = {hl_cap} d eller längre (∞ = ingen återgång)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, c="grey")
    axes[0].set_yticks(y, [f"{'✓' if ok else '✗'}  {name}" for name, ok in zip(table.index, table["klarar_alla"])], fontsize=8)
    for lbl, ok in zip(axes[0].get_yticklabels(), table["klarar_alla"]):
        lbl.set_color("tab:green" if ok else "tab:red")

    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:green", alpha=0.85), plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.85),
               plt.Line2D([], [], marker="D", ls="", c="k"), plt.Rectangle((0, 0), 1, 1, color="tab:green", alpha=0.08)]
    fig.legend(handles, ["klarar kravet (träning)", "klarar ej (träning)", "facit efter träning (används ej i beslutet)",
                         "godkänt område"], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Screening på träningsdata {TRAIN_START} – {TRAIN_END}: ett par handlas bara om alla fyra staplarna är gröna"
                 + ("" if REQUIRE_COINT else "  (REQUIRE_COINT = False: alla handlas ändå)"), fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))


def print_details(results, positions, trades):
    """Detaljer (PRINT_DETAILS = True): affärslistan, sammanfattning per par och fler nyckeltal."""
    pd.set_option("display.width", 250)
    if trades.empty:
        print("Inga affärer. Prova att sänka MIN_CORR, sätta REQUIRE_COINT = False eller ändra träningsperioden.")
        return
    cols = ["a", "b", "riktning", "in_datum", "ut_datum", "dagar", "z_in", "z_ut", "skäl", "resultat_%"]
    print(f"AFFÄRER ({len(trades)} st, resultat i % av kapitalet {CAPITAL_PER_PAIR} i A-benet, utan kostnader):")
    print(trades[cols].to_string(float_format=lambda x: f"{x:.2f}", formatters={
        "in_datum": "{:%Y-%m-%d}".format, "ut_datum": "{:%Y-%m-%d}".format}), "\n")

    summary = trades.groupby(["a", "b"]).agg(affärer=("resultat_%", "size"), vinstandel=("resultat_%", lambda r: (r > 0).mean()),
                                              snitt_dagar=("dagar", "mean"), resultat_tot_pct=("resultat_%", "sum"))
    print("PER PAR:\n" + summary.round(2).to_string())
    print("\nSTÄNGNINGSSKÄL:\n" + trades["skäl"].value_counts().to_string(), "\n")

    used = positions.loc[:, positions.abs().sum() > 0]  # tillgångar som aldrig handlas tas bort (annars 0/0 i evaluation.py)
    kf = evaluation.calc_key_figures(to_pl(used), to_pl(prices[used.columns]), key_figures=["sharpe_1d_lag", "av_holding"])
    print(f"FLER LYNX NYCKELTAL (med kostnader) {used.index[0]:%Y-%m-%d} – {used.index[-1]:%Y-%m-%d}:")
    for k, val in kf.items():
        print(f"  {k:<16} {val:.2f}")
    print()


def plot_residuals(results):
    """Brustermen ε_t = log A − (alpha + beta·log B) för varje kandidatpar. Grått fält = träningsperioden.
    Ett kointegrerat par har ett brus som pendlar kring 0; ett brus som driver iväg betyder att jämvikten saknas eller har brutits."""
    fig, axes = _grid(len(results), row_height=3.3)
    for ax, v in zip(axes, results):
        v["spread"].loc[TRAIN_START:].plot(ax=ax, lw=0.8, xlabel="", color="C0" if v["traded"] else "C7", label="låst β")
        if ROLLING_BETA:
            v["trade_spread"].loc[TRAIN_START:].plot(ax=ax, lw=0.8, xlabel="", color="C1", label=f"rullande β ({BETA_WINDOW} d)")
            ax.legend(fontsize=6, loc="lower left")
        ax.axvspan(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END), color="grey", alpha=0.15)
        ax.axhline(0, c="k", lw=0.6)
        c, mark = v["checks"], lambda ok: "✓" if ok else "✗"
        ax.set_title(f"{'HANDLAS' if v['traded'] else 'handlas ej'}: {v['a']} / {v['b']}\n"
                     f"korr={v['corr']:.2f}  β={v['beta']:.2f}  ({sum(c.values())}/4 krav på träningen)\n"
                     f"ADF-t={v['adf_t']:.2f}{mark(c['adf'])}  H={v['hurst']:.2f}{mark(c['hurst'])}  "
                     f"HL={v['half_life']:.0f}d{mark(c['hl'])}  kors={v['cross_per_year']:.0f}/år{mark(c['cross'])}", fontsize=7.5)
    fig.suptitle(f"Brusterm (residual) per kandidatpar – grått = träning ({TRAIN_START} – {TRAIN_END}), därefter handel"
                 + ("; orange = spreaden som handlas (rullande β)" if ROLLING_BETA else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def plot_trades(prices, v):
    """Tre paneler för ett handlat par:
    1) brustermen (träning + handel) med EWMA-band, in-/utgångar och failsafe-nivåer,
    2) z-score med trösklar och positioner, 3) ackumulerat resultat."""
    days, a, b = v["days"], v["a"], v["b"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3, 1]))

    # 1) Brustermen: träningsperioden (grått fält) och handelsperioden med EWMA-medel ± Z_ENTRY std
    ax1.plot(v["trade_spread"].loc[TRAIN_START:], lw=0.9, c="C0", label="spread som handlas")
    ax1.axvspan(pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END), color="grey", alpha=0.12, label="träning")
    ax1.plot(days["mean"], c="k", lw=0.8, label="medel (fast 0)" if MEAN_MODE == "noll" else "EWMA-medel")
    if not days["aktiv"].all():  # omscreening: perioder där paret inte klarade senaste screeningen
        ax1.fill_between(days.index, 0, 1, where=~days["aktiv"].astype(bool), transform=ax1.get_xaxis_transform(),
                         color="tab:red", alpha=0.07, step="post", label="ej godkänd (inga nya affärer)")
    entry_lbl = f"± dynamisk gräns ({ENTRY_FREQ:.0%} av dagarna utanför)" if ENTRY_FREQ else f"± {Z_ENTRY} std"
    ax1.fill_between(days.index, days["mean"] - days["z_entry"] * days["std"], days["mean"] + days["z_entry"] * days["std"],
                     color="grey", alpha=0.25, label=f"{entry_lbl} (inträde utanför)")
    for tr in v["trades"]:
        long = tr["riktning"] == "long spread"
        ax1.scatter(tr["in_datum"], tr["spread_in"], marker="^" if long else "v", s=60, zorder=3, c="green" if long else "red")
        stopped = tr["skäl"].startswith(("stop", "failsafe"))
        ax1.scatter(tr["ut_datum"], tr["spread_ut"], marker="x" if stopped else "o", s=45, zorder=3, c="red" if stopped else "k")
        if FAILSAFE_STD is not None:
            ax1.hlines(tr["failsafe"], tr["in_datum"], tr["ut_datum"], colors="red", linestyles="--", lw=0.9)
        if FREEZE_AT_ENTRY:  # låst exit-nivå: medel vid inträdet ± Z_EXIT std vid inträdet
            exit_level = tr["mean_in"] + (Z_EXIT if not long else -Z_EXIT) * tr["std_in"]
            ax1.hlines(exit_level, tr["in_datum"], tr["ut_datum"], colors="green", linestyles=":", lw=1.2)
    if FAILSAFE_STD is not None:
        ax1.plot([], [], "r--", lw=0.9, label="failsafe-nivå")
    if FREEZE_AT_ENTRY:
        ax1.plot([], [], "g:", lw=1.2, label="exit-nivå (låst vid inträde)")
    ax1.legend(fontsize=7, loc="upper left", ncol=3)
    medel = "medel=0" if MEAN_MODE == "noll" else f"span medel={SPAN_MEAN_MULT * v['half_life']:.1f} d"
    ax1.set_title(f"{a} / {b}:  korr={v['corr']:.2f}, β={v['beta']:.2f}, half-life={v['half_life']:.1f} d "
                  f"→ {medel}, std: {STD_MODE}   (▲ long in, ▼ short in, ● ut vid medel, ✕ stop/failsafe)", fontsize=10)

    # 2) Z-score med trösklar; grönt = long spread (köp A, blanka B), rött = short spread
    ax2.plot(days["z"], lw=0.8, c="k", label="z = spread / std" if MEAN_MODE == "noll" else "z (rörligt medel/std)")
    if FREEZE_AT_ENTRY:
        ax2.plot(days["z_trade"], lw=1.0, c="C1", label="z mot låst medel/std (styr exit)")
    ax2.plot(days["z_entry"], c="grey", ls="--", lw=0.8), ax2.plot(-days["z_entry"], c="grey", ls="--", lw=0.8)
    for level, style in [(Z_EXIT, ":")] + ([(Z_STOP, "-")] if Z_STOP is not None else []):
        ax2.axhline(level, c="grey", ls=style, lw=0.8), ax2.axhline(-level, c="grey", ls=style, lw=0.8)
    lim = max(Z_STOP or 0, np.nanmax(days["z_entry"]) + (FAILSAFE_STD or 4)) + 1
    ax2.fill_between(days.index, -lim, lim, where=days["pos"] > 0, color="green", alpha=0.15, step="post", label="long spread")
    ax2.fill_between(days.index, -lim, lim, where=days["pos"] < 0, color="red", alpha=0.15, step="post", label="short spread")
    ax2.set_ylim(-lim, lim), ax2.legend(fontsize=7, loc="upper left")
    in_txt = f"dynamisk {days['z_entry'].min():.2f}–{days['z_entry'].max():.2f}" if ENTRY_FREQ else f"±{Z_ENTRY}"
    ax2.set_title(f"z-score: in vid {in_txt} (streckad), ut vid ±{Z_EXIT} (prickad)"
                  + (f", stop-loss vid ±{Z_STOP} (heldragen)" if Z_STOP is not None else "")
                  + (f", failsafe {FAILSAFE_STD} std bortom inträdet" if FAILSAFE_STD is not None else ", ingen failsafe"),
                  fontsize=10)

    # 3) Ackumulerat resultat utan kostnader, med Lynx tidsförskjutning (position dag t ger avkastning från dag t+2)
    units = days[["units_a", "units_b"]].shift(evaluation.POSITION_LAG).values
    pnl = (units * prices[[a, b]].diff().loc[days.index].values).sum(axis=1)
    ax3.plot(days.index, 100 * np.nancumsum(pnl) / CAPITAL_PER_PAIR, c="C2")
    ax3.axhline(0, c="k", lw=0.6)
    ax3.set_title(f"Ackumulerat resultat i % av kapitalet, utan kostnader", fontsize=10)
    fig.tight_layout()


# ================================== KÖRNING ==================================
if __name__ == "__main__":
    # 0. Dataanalys (hela datan – bara för att titta, används inte av strategin)
    pairs = top_pairs(prices, TOP_PAIRS, verbose=PRINT_PAIRS)
    for show, df, name in [(PLOT_PRICES, prices, "Stängningspriser"), (PLOT_MACRO, macro, "Makrodata")]:
        if show:
            df.plot(subplots=True, layout=(-1, 5), figsize=(18, 2 * -(-df.shape[1] // 5)), sharex=True,
                    legend=False, title=list(df.columns), lw=0.8, xlabel="")
            plt.suptitle(name), plt.tight_layout(rect=(0, 0, 1, 0.97))
    if PLOT_CORR:
        plot_corr(prices, annotate=True)
    if PLOT_ROLL_CORR:
        plot_rolling_corr(prices, pairs)
    if PLOT_PAIR_PRICES:
        plot_pair_prices(prices, pairs)
    if PLOT_PAIR_LOG:
        plot_pair_prices(prices, pairs, log=True)

    # 1–4. Strategi
    if RUN_STRATEGY:
        results, positions, trades = run_strategy(prices)
        screening = screening_table(results)
        if PRINT_SCREENING:
            print_screening(screening)
        if SAVE_SCREENING:
            out = Path(__file__).parent / "pairs_screening.csv"
            screening.to_csv(out, index_label="par", float_format="%.4g")
            print(f"Screeningtabellen sparad i {out.name}\n")
        if PLOT_SCREENING:
            plot_screening(screening)
        if PRINT_DETAILS:
            print_details(results, positions, trades)
        print_summary(results, positions, trades)
        if SAVE_TRADES and not trades.empty:
            out = Path(__file__).parent / "pairs_trades.csv"
            trades.to_csv(out, index_label="affär", float_format="%.6g")
            print(f"Alla affärer sparade i {out.name}\n")
        if PLOT_RESIDUALS:
            plot_residuals(results)
        if PLOT_TRADES:
            for v in results:
                if v["traded"]:
                    plot_trades(prices, v)
    plt.show()
