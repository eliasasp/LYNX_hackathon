# Trend + makro (justerad Sharpe 0.99 på insample-datan)

Positioner i `evaluation.py`-format: `AsOfDate` + en kolumn per instrument.

## Kör

```bash
python submission.py close_prices_insample.csv positions.csv --macro macro_data_insample.csv --eval
```

`--macro` är frivillig; utan den körs bara trenddelen. `--eval` kräver `evaluation.py` i samma mapp
och skriver ut nyckeltal för hela perioden, träning (1995–2011) och validering (2012–2016) samt
sparar två dashboards som PNG.

## Resultat på close_prices_insample.csv + macro_data_insample.csv

| Period | Justerad Sharpe | Sharpe | Kurtosis |
|---|---|---|---|
| Hela 1995–2016 | 0.99 | 1.26 | 5.63 |
| Träning 1995–2011 | 0.99 | 1.28 | 5.56 |
| Validering 2012–2016 | 1.01 | 1.19 | 5.93 |

Jämförelse: alltid lång med lika risk per instrument ger 0.55.

## Strategin i korthet

1. **Trend** (`trend.py`): tre tidsskalor (100, 250, 500 dagar), för varje en MA-crossover s/4 mot s
   med signal `tanh(2z)` och ett momentum-z med hysteres (in vid |z| > 0.5, ut vid teckenbyte).
2. **Makro** (`macro_joint.py`): BNP- och KPI-förändring över 20 och 120 dagar per region,
   z-normaliserade, i en gemensam ridge-regression per instrument (λ = 0.3 × medel(diag(X'X)))
   mot kommande 20 dagars avkastning. Walk-forward på ett rullande fönster om 2500 dagar.
   Signalen klipps vid ±2 standardavvikelser.
3. **Gruppvis sammanvägning** (`combine.py`): valutor, obligationer, aktier och råvaror. Inom varje
   grupp 67 % trend och 33 % makro, riskviktat, utom obligationer som får 100 % trend. Grupperna
   får sedan lika risk.
4. **Volatilitetsbroms** (`overlays.py`): när portföljens 20-dagarsvol överstiger 1.25 × 250-dagarsvolen
   skalas risken ner till den långa nivån. Aldrig uppskalning.

Alla steg använder bara data som var känd den aktuella dagen. Tidskonventionen följer `evaluation.py`:
en position daterad t tjänar avkastningen på raderna t+2 och framåt.

## Filer

| Fil | Roll |
|---|---|
| `submission.py` | sätter ihop allt, CLI, utvärdering. Alla inställningar är konstanter högst upp |
| `trend.py` | trendsignaler |
| `macro_joint.py` | gemensam ridge-regression för makro |
| `macro.py` | makroegenskaper, inläsning av makrodata, tillgångsslag |
| `combine.py` | riskviktad sammanvägning av strategier |
| `overlays.py` | volatilitetsbroms och andra överlägg |
| `lead_lag.py` | hjälpfunktioner (prisinläsning, volatilitet, skift) samt lead-lag, avstängd |
| `dynamic_allocation.py` | dynamiska vikter, avstängd |

Kräver `polars` och `numpy`.
