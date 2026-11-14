"""Test för framåtblickande data (data leakage), enligt Lynx instruktioner.

Två datamängder som är identiska t.o.m. dag T men olika efter T måste ge identiska positioner t.o.m. T.
Testet kör main.py:s modell (merged + pairs, kombinerad) på
    1. den fullständiga datan
    2. samma data avklippt vid varje CUTOFF
och kontrollerar att positionerna t.o.m. respektive cutoff är exakt lika, per strategi och kombinerat.

Utan --prices förlängs insample-datan (1995–2016) med syntetiska priser och makro t.o.m. 2026-12-31
(dagar ur historiken dras slumpmässigt, samma dag för alla instrument). Det testar samtidigt att modellen
klarar data utökad till 2026, och att positionerna t.o.m. 2016 inte påverkas av data efter 2016.

    python leakage_test.py                                         # insample + syntetisk förlängning till 2026
    python leakage_test.py --prices <priser.csv> --macro <makro.csv>   # på riktig (t.ex. out of sample-)data
"""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import main as model

DATE = model.DATE
SYNTHETIC_END = date(2026, 12, 31)


def extend_synthetic(prices: pl.DataFrame, macro: pl.DataFrame, end: date, seed: int = 0):
    """Förläng priser och makro med slumpvis dragna historiska dagar (log-förändringar) t.o.m. end."""
    rng = np.random.default_rng(seed)
    new_dates = pl.date_range(prices[DATE][-1], end, "1d", eager=True)[1:]
    new_dates = new_dates.filter(new_dates.dt.weekday() <= 5)

    def extend(df: pl.DataFrame) -> pl.DataFrame:
        cols = [c for c in df.columns if c != DATE]
        logs = np.diff(np.log(df.select(cols).to_numpy()), axis=0)
        draws = logs[rng.integers(0, len(logs), len(new_dates))]
        path = df.select(cols).to_numpy()[-1] * np.exp(np.cumsum(draws, axis=0))
        ext = pl.DataFrame(path, schema=cols).insert_column(0, new_dates.alias(DATE))
        return pl.concat([df.with_columns(pl.col(DATE).cast(pl.Date)), ext])

    return extend(prices), extend(macro)


def run(prices: pl.DataFrame, macro_path: str) -> dict[str, pl.DataFrame]:
    with contextlib.redirect_stdout(io.StringIO()):
        raw, combined = model.build_positions(prices, macro_path)
    return raw | {"KOMBINERAD": combined}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prices", help="prisfil (standard: insample + syntetisk förlängning till 2026)")
    parser.add_argument("--macro", help="makrofil (krävs med --prices)")
    parser.add_argument("--cutoffs", nargs="+", type=date.fromisoformat,
                        help="datum att klippa datan vid (standard: 2013-12-31 och sista insample-dagen)")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="leakage_"))
    if args.prices:
        if not args.macro:
            parser.error("--macro krävs tillsammans med --prices")
        prices = model.load_prices(args.prices)
        macro = pl.read_csv(args.macro, try_parse_dates=True, infer_schema_length=None)
        macro = macro.rename({macro.columns[0]: DATE}).with_columns(pl.col(DATE).cast(pl.Date))
        cutoffs = args.cutoffs or [date(2013, 12, 31), date(2016, 12, 30)]
    else:
        insample = model.load_prices(str(model.ROOT / "close_prices_insample.csv"))
        macro = pl.read_csv(model.ROOT / "macro_data_insample.csv", try_parse_dates=True, infer_schema_length=None)
        macro = macro.rename({macro.columns[0]: DATE}).with_columns(pl.col(DATE).cast(pl.Date))
        prices, macro = extend_synthetic(insample, macro, SYNTHETIC_END)
        cutoffs = args.cutoffs or [date(2013, 12, 31), insample[DATE][-1]]
        print(f"Syntetisk förlängning: insample t.o.m. {insample[DATE][-1]}, förlängd t.o.m. {prices[DATE][-1]}")

    macro.write_csv(tmp / "macro_full.csv")
    print(f"Fullständig data: {prices.height} dagar, {prices[DATE][0]} – {prices[DATE][-1]}")
    full = run(prices, str(tmp / "macro_full.csv"))
    traded = full["KOMBINERAD"].filter(pl.col(DATE) > max(cutoffs))
    inst = [c for c in traded.columns if c != DATE]
    print(f"  positioner efter {max(cutoffs)}: {traded.height} dagar, "
          f"{int((traded.select(inst).to_numpy() != 0).any(axis=1).sum())} med position\n")

    ok = True
    for cut in cutoffs:
        path = tmp / f"macro_{cut}.csv"
        macro.filter(pl.col(DATE) <= cut).write_csv(path)
        short = run(prices.filter(pl.col(DATE) <= cut), str(path))
        print(f"Avklippt vid {cut}: positioner t.o.m. {cut} jämförs med fullständig data")
        for name, pos in short.items():
            a = pos.filter(pl.col(DATE) <= cut)
            b = full[name].filter(pl.col(DATE) <= cut)
            cols = [c for c in a.columns if c != DATE]
            diff = np.abs(a.select(cols).to_numpy() - b.select(cols).to_numpy())
            same = a[DATE].equals(b[DATE]) and np.allclose(diff, 0, atol=1e-10)
            ok &= same
            first = "" if same else f", första avvikelse {a[DATE][int(np.argmax(diff.max(axis=1) > 1e-10))]}"
            print(f"  {name:12s} {'OK  ' if same else 'FEL '} max skillnad {diff.max():.2e}{first}")
    print("\nInget läckage: positionerna påverkas inte av senare data." if ok
          else "\nLÄCKAGE: positioner ändras när senare data läggs till.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
