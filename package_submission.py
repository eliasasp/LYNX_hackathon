"""Exporterar exakt de filer som behövs för inlämningen till en fristående mapp och en zip-fil.

    .venv/bin/python package_submission.py      → lynx_submission/  och  output/lynx_submission.zip

Mappen kan köras som den är:  cd lynx_submission && python main.py
Kör om exporten efter ändringar i koden. Lägg till filer i FILES om main.py börjar använda fler moduler.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "lynx_submission"

FILES = [
    "main.py",                        # kör allt, kombinerar 60/40, skriver output/submission.csv och plottar
    "evaluation.py",                  # Lynx nyckeltal och dashboard
    "leakage_test.py",                # test för framåtblickande data
    "strategy/submission.py",         # merged: trend + makro, gruppvikter
    "strategy/combine.py",            # riskviktad kombination
    "strategy/dynamic_allocation.py",
    "strategy/lead_lag.py",           # gemensamma hjälpfunktioner (load_prices m.m.)
    "strategy/macro.py",
    "strategy/macro_joint.py",
    "strategy/overlays.py",           # volbroms
    "strategy/trend.py",
    "ma_crossover/final_model.py",    # merged: MA10/250-trend
    "pairs_trading/pairs_final.py",   # pairs trading
    "close_prices_insample.csv",      # standarddata (pairs_final läser dem när den importeras)
    "macro_data_insample.csv",
]

REQUIREMENTS = "matplotlib>=3.11\nnumpy>=2.0\npandas>=2.0\npolars>=1.44\npyarrow>=25.0\n"

README = """# Lynx Hackathon 2026: merged + pairs

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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=ROOT / NAME, help="mappen som skapas")
    parser.add_argument("--zip", type=Path, default=ROOT / "output" / f"{NAME}.zip", help="zip-filen som skapas")
    args = parser.parse_args()

    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"Saknas: {', '.join(missing)}")

    if args.dir.exists():
        if not (args.dir / "main.py").exists():
            raise SystemExit(f"{args.dir} finns redan och ser inte ut som en tidigare export; ta bort den själv")
        shutil.rmtree(args.dir)
    for f in FILES:
        (args.dir / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, args.dir / f)
    (args.dir / "requirements.txt").write_text(REQUIREMENTS)
    (args.dir / "README.md").write_text(README)
    print(f"Mapp: {args.dir} ({len(FILES) + 2} filer)")

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(args.zip.with_suffix("")), "zip", args.dir.parent, args.dir.name)
    print(f"Zip:  {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
