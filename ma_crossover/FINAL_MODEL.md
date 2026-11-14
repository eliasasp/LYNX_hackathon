# Final trend model

Filen att dela är **`final_model.py`**. Den innehåller hela modellen och importerar inga lokala strategi-/analysmoduler. Endast `numpy` och `polars` krävs för positionsberäkning. Import av filen läser inga data, kör inga backtest och skriver inga filer.

## Frysta val

- MA10/250 med `tanh((SMA10 − SMA250)/(RMS-vol100 × √250))`.
- Invers 100-dagars RMS av absoluta prisförändringar per tillgång.
- Gemensam 100-dagars standardavvikelse av basstrategins observerade P&L med t−2-positioner. Minst 20 observationer; initial riskskala √18 och högst 3/√18 i skalningsfaktor.
- Alla 14 aktier/obligationer/valutor plus nanocapacitors, ultrapure_silicon, hydrocarbons och aerogel. Exakt lista finns som `INSTRUMENTS`; inga nya tillgångar inkluderas automatiskt.
- Handel över hela tillgängliga historiken efter vanlig uppvärmning. Ingen startspärr 2010 och ingen slutspärr 2016.
- Inget kapital-/procentavkastningsantagande: positioner är antal enheter enligt repoets P&L-konvention.

## Kör från repo-roten

```bash
.venv/bin/python ma_crossover/final_model.py
```

Standard: läser `close_prices_insample.csv` och sparar `ma_crossover/final_positions.csv`.

Verifiera och spara Quickstart-panelerna:

```bash
.venv/bin/python ma_crossover/final_model.py --check-causality --evaluate --plot
```

Detta använder repoets `evaluation.py` och sparar även `final_positions.checks.json`, `final_positions.metrics.json` och `final_positions.png`.

Kör på hela den nya historiken när full data kommer:

```bash
.venv/bin/python ma_crossover/final_model.py --prices /path/to/full_prices.csv --output /path/to/trend_positions.csv --check-causality
```

Läs in hela historiken, inte bara de nya dagarna, så MA-/riskuppvärmning och positioner över gränsen bevaras. CSV kan vara wide stängningspriser eller repoets OHLC-format med `|Close`-kolumner. Modellen förutsätter samma 18 instrument och kräver ändliga priser och unika stigande datum; felaktiga data ger ett tydligt fel i stället för tyst fyllning.

Om filen kopieras ut ur repon fungerar export med explicita `--prices` och `--output`. För valfri utvärdering anges även `--evaluation-module /path/to/evaluation.py`; den kräver evaluatorns vanliga beroenden.

## Använd i gemensam modell

```python
from ma_crossover.final_model import make_positions

trend_positions = make_positions(prices)
```

`prices` är en Polars DataFrame med `AsOfDate` och wide stängningspriser. Extra tillgångar ignoreras. Resultatet har `AsOfDate` och exakt 18 positionskolumner, utan tomma positioner. Datum kan vara Polars Date, Datetime eller ISO-datumsträngar. Filnamnet/modulen kan också importeras som `final_model` om den kopieras till en annan mapp.

**Positionerna dateras på signaldagen.** Skifta inte exporten; gemensam P&L-utvärdering ska tillämpa tvådagarslaggen en gång. Kostnader dras av vid utvärdering, inte ur positionsfilen. Modellen använder inga makroserier och inga andra strategiers positioner.

## Verifierat

- Positionsfilen matchar exakt `quickstart_18_candidate_positions.csv` från den körda notebooken.
- In-sample 1995–2016 enligt oförändrad evaluator: net Sharpe **1,041**, adjusted Sharpe **0,886**.
- Både förkortad historik och ändrad framtid, vid tre brytpunkter: sex godkända läckagekontroller med maxskillnad 0.
- Identiskt resultat från wide close och OHLC. Kontroll av uppvärmning, ändliga positioner, ogiltiga datum/priser och utökad historik efter 2016.

Resultatet är retrospektivt: universum och modellval använde samma in-sample. Kausala beräkningar eliminerar inte denna urvalsbias. Ingen kod har committats eller pushats automatiskt. Övriga strategifiler och MR-/pairs-modeller lämnas oförändrade.
