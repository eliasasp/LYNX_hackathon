"""Frozen MA10/250 trend follower: 14 financial markets plus four materials.

Import ``make_positions(prices)`` or run this file to export desired positions.
Only numpy and polars are required. No data loading or evaluation on import.
Positions are dated on the signal day: the shared evaluator applies the t+2
P&L lag. No 2010 start gate or end-date cutoff is applied to this final model.

Universe and parameters were selected using supplied in-sample data. Historical
performance is retrospective; the model itself never uses future observations.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

DATE = "AsOfDate"
FAST = 10
SLOW = 250
VOL_WINDOW = 100
RISK_WINDOW = 100
INSTRUMENTS = (
    "ganymede_bonds",
    "ganymede_stocks",
    "ganymede_titan_fx",
    "io_bonds",
    "io_stocks",
    "io_titan_fx",
    "titan_bonds",
    "titan_stocks",
    "callisto_bonds",
    "callisto_stocks",
    "callisto_titan_fx",
    "europa_bonds",
    "europa_stocks",
    "europa_titan_fx",
    "nanocapacitors",
    "ultrapure_silicon",
    "hydrocarbons",
    "aerogel",
)


def _prices(prices: pl.DataFrame) -> pl.DataFrame:
    """Validate wide closing prices. Extra instruments are deliberately ignored."""
    missing = {DATE, *INSTRUMENTS} - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if prices.is_empty():
        raise ValueError("Price history is empty")
    prices = prices.select(DATE, *INSTRUMENTS)
    dtype = prices.schema[DATE]
    if dtype == pl.String:
        prices = prices.with_columns(pl.col(DATE).str.to_date(strict=True))
    elif dtype == pl.Datetime:
        prices = prices.with_columns(pl.col(DATE).cast(pl.Date))
    elif dtype != pl.Date:
        raise ValueError("AsOfDate must contain dates, datetimes or ISO date strings")
    dates = prices[DATE]
    if dates.null_count() or dates.n_unique() != prices.height or not dates.is_sorted():
        raise ValueError("Dates must be non-null, unique and increasing")
    if not all(prices.schema[c].is_numeric() for c in INSTRUMENTS):
        raise ValueError("Instrument columns must be numeric")
    if not np.isfinite(prices.select(INSTRUMENTS).to_numpy()).all():
        raise ValueError(
            "Prices must be finite; missing values require explicit handling"
        )
    return prices


def load_prices(path: str | Path) -> pl.DataFrame:
    """Read supplied wide close CSV or OHLC CSV with instrument|Close columns."""
    prices = pl.read_csv(path, try_parse_dates=True)
    close = [c for c in prices.columns if c.endswith("|Close")]
    if close:
        prices = prices.select(DATE, *close).rename(
            {c: c.removesuffix("|Close") for c in close}
        )
    return _prices(prices)


def make_positions(prices: pl.DataFrame) -> pl.DataFrame:
    """Return AsOfDate plus exactly 18 desired-position columns, without lagging.

    signal = tanh((SMA10-SMA250)/(RMS_vol100*sqrt(250))).
    Base positions = signal/RMS_vol100. Portfolio scaling is inverse trailing
    100-day std of base P&L using base positions from t-2. Before 20 observed
    P&L values use sqrt(18); cap scale at 3/sqrt(18). Undefined warmup positions
    are zero. All calculations use observations through the current day only.
    """
    prices = _prices(prices)
    cols = list(INSTRUMENTS)
    changes = prices.select(DATE, pl.col(cols).diff())
    vol = changes.select(DATE, pl.col(cols).pow(2).rolling_mean(VOL_WINDOW).sqrt())
    signal = prices.select(
        DATE,
        *[
            (
                (pl.col(c).rolling_mean(FAST) - pl.col(c).rolling_mean(SLOW))
                / (vol[c] * np.sqrt(SLOW))
            )
            .tanh()
            .alias(c)
            for c in cols
        ],
    )
    base = signal.select(
        DATE,
        *[
            pl.when(vol[c] > 1e-12).then(pl.col(c) / vol[c]).otherwise(None).alias(c)
            for c in cols
        ],
    )
    gross = base.select(
        DATE, *[(pl.col(c).shift(2) * changes[c]).alias(c) for c in cols]
    )
    pnl = gross.select(
        pl.when(pl.all_horizontal(pl.col(cols).is_null()))
        .then(None)
        .otherwise(pl.sum_horizontal(cols))
        .alias("pnl")
    )["pnl"]
    risk = pnl.rolling_std(RISK_WINDOW, min_samples=20)
    fallback = float(np.sqrt(len(cols)))
    scale = pl.DataFrame({"risk": risk}).select(
        (1 / pl.col("risk").fill_null(fallback).clip(lower_bound=fallback / 3)).alias(
            "scale"
        )
    )["scale"]
    return base.select(
        DATE, *[(pl.col(c) * scale).fill_nan(0.0).fill_null(0.0).alias(c) for c in cols]
    )


def check_causality(prices: pl.DataFrame) -> list[dict]:
    """Both required prefix/future-change checks with fixed model/universe."""
    prices = _prices(prices)
    full = make_positions(prices)
    rng = np.random.default_rng(73)
    cuts = sorted({min(150, prices.height - 1), prices.height // 2, prices.height - 20})
    results = []
    for cut in (c for c in cuts if 0 < c < prices.height):
        values = prices.select(INSTRUMENTS).to_numpy().copy()
        values[cut:] += rng.normal(0, 0.1, values[cut:].shape).cumsum(axis=0)
        altered = pl.DataFrame(values, schema=list(INSTRUMENTS)).insert_column(
            0, prices[DATE]
        )
        reference = full.head(cut).select(INSTRUMENTS).to_numpy()
        for label, history in (
            ("truncated_history", prices.head(cut)),
            ("changed_future", altered),
        ):
            actual = make_positions(history).head(cut).select(INSTRUMENTS).to_numpy()
            difference = float(np.abs(reference - actual).max())
            if difference > 1e-10:
                raise AssertionError(f"Causality failure at row {cut}: {label}")
            results.append(
                {"test": label, "prefix_rows": cut, "max_difference": difference}
            )
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", type=Path, default=root / "close_prices_insample.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("final_positions.csv")
    )
    parser.add_argument("--check-causality", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use the supplied evaluation.py and save metrics JSON",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save the Quickstart four-panel dashboard",
    )
    parser.add_argument(
        "--evaluation-module", type=Path, default=root / "evaluation.py"
    )
    args = parser.parse_args()
    prices = load_prices(args.prices)
    positions = make_positions(prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    positions.write_csv(args.output)
    print(
        f"MA10/250, vol100/risk100 | {len(INSTRUMENTS)} assets | {prices[DATE].min()}–{prices[DATE].max()}"
    )
    print(f"Positions saved: {args.output}")
    if args.check_causality:
        checks = check_causality(prices)
        args.output.with_suffix(".checks.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        print(f"Causality: {len(checks)} checks passed")
    if args.evaluate or args.plot:
        spec = importlib.util.spec_from_file_location(
            "lynx_evaluation", args.evaluation_module
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load evaluation.py; pass --evaluation-module")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        figures = evaluator.calc_key_figures(positions, prices)
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(figures, indent=2) + "\n"
        )
        print(
            f"Net Sharpe: {figures['sharpe']:.3f} | Adjusted Sharpe: {figures['adjusted_sharpe']:.3f}"
        )
        if args.plot:
            fig = evaluator.plot_key_figures(positions, prices)
            fig.suptitle("Final MA10/250 — 18 fixed assets, full supplied history")
            path = args.output.with_suffix(".png")
            fig.savefig(path, dpi=140, bbox_inches="tight")
            print(f"Dashboard saved: {path}")


if __name__ == "__main__":
    main()
