# Lynx Hackathon 2026: merged + pairs

Allt som behövs för att skapa positionsfilen finns i den här mappen. Kör från mappen:

```
pip install -r requirements.txt
python main.py                                       # insample-datan → output/submission.csv
```

## Out of sample (data utökad till 2026)

Lägg de nya filerna i mappen (eller ange en sökväg) och kör:

```
python main.py --prices <priser.csv> --macro <makro.csv>
```

`--prices` kan vara stängningspriser (`AsOfDate` + en kolumn per instrument, som
`close_prices_insample.csv`) eller OHLC-format (`<instrument>|Close`, som `prices_insample.csv`).
`--macro` har samma format som `macro_data_insample.csv`. Datan måste innehålla samma instrument som
insample-datan och ha priser för alla dagar (inga tomma värden).

## Utdata

`output/submission.csv`: kolumnen `AsOfDate` + en kolumn per instrument, en rad per handelsdag från
2010-01-01 till sista dagen i prisfilen. Positionen daterad T använder bara data t.o.m. T;
`evaluation.py` lägger på t+2-laggen. Övriga flaggor:

```
python main.py --plot          # visar dashboard, positions- och riskplottar
python main.py --save-plots    # sparar plottarna som PNG i output/
python main.py --no-eval       # hoppar över nyckeltalen
python main.py --help
```

## Test för framåtblickande data

```
python leakage_test.py                                     # insample + syntetisk förlängning till 2026
python leakage_test.py --prices <priser.csv> --macro <makro.csv>   # på den riktiga datan
```

Modellen körs på hela datan och på samma data avklippt vid 2013-12-31 och vid sista insample-dagen.
Positionerna t.o.m. respektive datum ska vara exakt lika, för varje strategi och kombinerat.

## Strategin

| Fil | Roll |
|---|---|
| `main.py` | kör strategierna, kombinerar dem (60 % merged / 40 % pairs i risk), volbroms och volmål på portföljen |
| `strategy/` | merged: trend (hysteres-momentum 100/250/500 d) + makro (BNP/KPI, ridge), lika risk per tillgångsslag |
| `ma_crossover/final_model.py` | merged: MA10/250-trend, 70 % av trendrisken |
| `pairs_trading/pairs_final.py` | pairs trading: kointegrerade par, omscreening var 50:e dag |
| `evaluation.py` | Lynx nyckeltal |
| `leakage_test.py` | test för framåtblickande data |

Ingen handel före 2010-01-01: fram till dess används datan bara för att skatta signaler och risk.
Mer detaljer finns överst i `main.py`.
