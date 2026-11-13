"""Strategy evaluation for hackathon participants.

Standalone module: depends on polars, numpy and matplotlib only, so it can be handed to
participants as a single file. Frames carry an ``AsOfDate`` column plus one column per instrument.
Positions at date ``t`` are traded on the close of ``t + 2``.
"""

from collections.abc import Callable, Sequence
from datetime import date
from typing import cast

import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.transforms import Bbox

DEFAULT_KEY_FIGURES = [
    "adjusted_sharpe",
    "av_holding",
    "kurtosis",
    "position_bias",
    "sharpe",
    "sharpe_1d_lag",
    "skewness",
]

ALL_KEY_FIGURES = [
    "adjusted_sharpe",
    "av_holding",
    "av_holding_per_instrument",
    "kurtosis",
    "position_bias",
    "position_bias_per_instrument",
    "returns",
    "sharpe",
    "sharpe_1d_lag",
    "sharpe_ex_slippage",
    "sharpe_per_instrument",
    "skewness",
    "slippages",
    "rolling_position_bias",
    "rolling_sharpe",
    "rolling_std",
]

DATE = "AsOfDate"
POSITION_LAG = 2
TRADING_DAYS_PER_YEAR = 260
VOLATILITY_WINDOW = 100
ANNUALIZATION = float(np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_key_figures(
    positions: pl.DataFrame,
    prices: pl.DataFrame,
    costs: float = -0.02,
    rolling_window: int = 260,
    key_figures: Sequence[str] | None = None,
) -> dict[str, object]:
    """Calculates key figures for the positions of a trading strategy over the full input period.

    Key figures:
        adjusted_sharpe: Yearly Sharpe ratio penalized for negative skewness and excess kurtosis.
        av_holding: Average number of days a position is held once taken. Fast strategies incur
            more slippage and are harder to trade at size.
        av_holding_per_instrument: ``av_holding`` per instrument.
        kurtosis: Daily kurtosis of total returns including slippage. Normal returns give 3.
        position_bias: Long/short tilt in [-1, 1]; -1 is short-only, 1 is long-only.
        position_bias_per_instrument: ``position_bias`` per instrument.
        returns: Backtested total return per day, summed over instruments.
        sharpe: Yearly Sharpe ratio including slippage, with no risk-free adjustment.
        sharpe_1d_lag: Sharpe for positions lagged one extra day. A large gap to ``sharpe``
            indicates short-term alpha or forward-looking bias.
        sharpe_ex_slippage: Sharpe without slippage. A large gap to ``sharpe`` indicates a strategy
            that is hard to execute at size.
        sharpe_per_instrument: Sharpe per instrument.
        skewness: Daily skewness of total returns including slippage.
        slippages: Estimated trading cost per day, summed over instruments.
        rolling_position_bias: ``position_bias`` on rolling windows.
        rolling_sharpe: Sharpe on rolling windows.
        rolling_std: Standard deviation on rolling windows.

    Args:
        positions: ``AsOfDate`` plus one column per instrument. A position dated ``t`` is traded on
            the close of ``t + 2``. The instrument columns define the universe and must all exist
            in ``prices``.
        prices: ``AsOfDate`` plus one price column per instrument.
        costs: Slippage per unit traded, in daily standard deviations.
        rolling_window: Window length, in days, for the rolling key figures.
        key_figures: Names of key figures to calculate; defaults to ``DEFAULT_KEY_FIGURES``. Valid
            names are listed in ``ALL_KEY_FIGURES``.

    Returns:
        The requested key figures keyed by name: scalars as ``float``, per-instrument figures as
        ``dict[str, float]``, time series as two-column frames ``AsOfDate`` plus the figure name.
    """
    if key_figures is None:
        key_figures = DEFAULT_KEY_FIGURES

    instruments = [column for column in positions.columns if column != DATE]
    positions = _dated(positions).with_columns(pl.col(instruments).shift(POSITION_LAG))

    returns = _dated(prices).select(DATE, pl.col(instruments).diff())
    volatility = returns.with_columns(
        pl.col(instruments).rolling_std(VOLATILITY_WINDOW, min_samples=10).fill_null(strategy="forward").shift(1)
    )

    start = max(returns[DATE][0], positions[DATE][0])
    end = min(returns[DATE][-1], positions[DATE][-1])
    window = pl.col(DATE).is_between(start, end)
    returns = returns.filter(window)
    # Back-filling the first few days of volatility is a small forward-looking approximation.
    volatility = volatility.filter(window).with_columns(pl.col(instruments).fill_null(strategy="backward"))
    positions = returns.select(DATE).join_asof(positions, on=DATE, strategy="backward")

    dates = returns.get_column(DATE)
    # Every frame below sits on the dates of ``returns``, so frame arithmetic is positional.
    returns, positions, volatility = (frame.select(instruments) for frame in (returns, positions, volatility))

    trades = positions.select(pl.all().diff().abs())

    norm_positions = positions * volatility
    norm_trades = trades * volatility

    model_returns = returns * positions
    slippages = norm_trades * costs

    tot_returns = _row_sum(model_returns + slippages)

    result: dict[str, object] = {}

    if "sharpe" in key_figures:
        result["sharpe"] = _calc_sharpe(tot_returns)

    if "sharpe_1d_lag" in key_figures:
        lagged = returns * positions.select(pl.all().shift(1)) + volatility * trades.select(pl.all().shift(1)) * costs
        result["sharpe_1d_lag"] = _calc_sharpe(_row_sum(lagged))

    if "sharpe_ex_slippage" in key_figures:
        result["sharpe_ex_slippage"] = _calc_sharpe(_row_sum(model_returns))

    if "sharpe_per_instrument" in key_figures:
        result["sharpe_per_instrument"] = _per_instrument(
            model_returns, lambda c: pl.col(c).mean() / pl.col(c).std() * ANNUALIZATION
        )

    if "av_holding" in key_figures:
        result["av_holding"] = 2 * _mean(_row_sum(_abs(norm_positions))) / _mean(_row_sum(_abs(norm_trades)))

    if "av_holding_per_instrument" in key_figures:
        result["av_holding_per_instrument"] = {
            c: 2 * _mean(norm_positions[c].abs()) / _mean(norm_trades[c].abs()) for c in instruments
        }

    if "position_bias" in key_figures:
        result["position_bias"] = _mean(_row_sum(norm_positions)) / _mean(_row_sum(_abs(norm_positions)))

    if "position_bias_per_instrument" in key_figures:
        result["position_bias_per_instrument"] = _per_instrument(
            norm_positions, lambda c: pl.col(c).mean() / pl.col(c).abs().mean()
        )

    if "returns" in key_figures:
        result["returns"] = _series(dates, "returns", tot_returns)

    if "slippages" in key_figures:
        result["slippages"] = _series(dates, "slippages", _row_sum(slippages))

    if "skewness" in key_figures:
        result["skewness"] = _skew(tot_returns)

    if "kurtosis" in key_figures:
        result["kurtosis"] = _excess_kurtosis(tot_returns) + 3

    if "adjusted_sharpe" in key_figures:
        result["adjusted_sharpe"] = _adjusted_sharpe(tot_returns)

    min_periods = rolling_window // 2

    if "rolling_std" in key_figures:
        rolling_std = tot_returns.rolling_std(rolling_window, min_samples=min_periods)
        result["rolling_std"] = _series(dates, "rolling_std", rolling_std)

    if "rolling_sharpe" in key_figures:
        rolling_mean = tot_returns.rolling_mean(rolling_window, min_samples=min_periods)
        rolling_std = tot_returns.rolling_std(rolling_window, min_samples=min_periods)
        result["rolling_sharpe"] = _series(dates, "rolling_sharpe", rolling_mean / rolling_std * ANNUALIZATION)

    if "rolling_position_bias" in key_figures:
        long_positions = norm_positions.select(pl.all().clip(lower_bound=0))
        daily_bias = 2 * _row_sum(long_positions) / _row_sum(_abs(norm_positions)) - 1
        rolling_bias = daily_bias.rolling_mean(rolling_window, min_samples=min_periods)
        result["rolling_position_bias"] = _series(dates, "rolling_position_bias", rolling_bias)

    return result


def plot_key_figures(
    positions: pl.DataFrame,
    prices: pl.DataFrame,
    costs: float = -0.02,
    rolling_window: int = 260,
) -> Figure:
    """Calculates and plots the key figures of a strategy as a four-panel dashboard.

    Args:
        positions: Positions per date and instrument; see ``calc_key_figures``.
        prices: Instrument prices per date, one column per instrument.
        costs: Slippage per unit traded, in daily standard deviations.
        rolling_window: Window length, in days, for the rolling key figures.

    Returns:
        The dashboard figure.
    """
    key_figures = calc_key_figures(
        positions, prices, costs=costs, rolling_window=rolling_window, key_figures=ALL_KEY_FIGURES
    )
    xlims = (cast(date, positions[DATE].min()), cast(date, positions[DATE].max()))

    fig = Figure(figsize=(16, 10))
    grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25)

    _plot_cumulative(fig.add_subplot(grid[0, 0]), key_figures, xlims)
    _plot_key_figure_table(fig.add_subplot(grid[0, 1]), key_figures)
    _plot_rolling(fig, grid[1, 0], key_figures, xlims, rolling_window)
    _plot_instrument_bars(fig, grid[1, 1], key_figures)
    return fig


def _dated(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(DATE).cast(pl.Date)).sort(DATE)


def _abs(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(pl.all().abs())


def _row_sum(frame: pl.DataFrame) -> pl.Series:
    """Sum over instruments per day; a day without any value is null, not zero."""
    total = pl.when(pl.all_horizontal(pl.all().is_null())).then(None).otherwise(pl.sum_horizontal(pl.all()))
    return frame.select(total.alias("total")).to_series()


def _mean(values: pl.Series) -> float:
    return float(values.mean())  # type: ignore[arg-type]


def _per_instrument(frame: pl.DataFrame, figure: Callable[[str], pl.Expr]) -> dict[str, float]:
    values = frame.select(figure(column).alias(column) for column in frame.columns).row(0)
    return dict(zip(frame.columns, (float(v) for v in values), strict=True))


def _series(dates: pl.Series, name: str, values: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({DATE: dates, name: values})


def _calc_sharpe(returns: pl.Series) -> float:
    return _mean(returns) / float(returns.std()) * ANNUALIZATION  # type: ignore[arg-type]


def _skew(returns: pl.Series) -> float:
    return float(returns.skew(bias=False))  # type: ignore[arg-type]


def _excess_kurtosis(returns: pl.Series) -> float:
    return float(returns.kurtosis(fisher=True, bias=False))  # type: ignore[arg-type]


def _adjusted_sharpe(returns: pl.Series) -> float:
    sharpe = _calc_sharpe(returns)
    return sharpe * (1 + sharpe / 6 * _skew(returns) - sharpe**2 / 24 * _excess_kurtosis(returns))


def _frame(key_figures: dict[str, object], name: str) -> pl.DataFrame:
    frame = key_figures[name]
    assert isinstance(frame, pl.DataFrame)
    return frame


def _plot_cumulative(ax: Axes, key_figures: dict[str, object], xlims: tuple[date, date]) -> None:
    for name, label in (("returns", "Return inc. slippage"), ("slippages", "Slippage")):
        frame = _frame(key_figures, name)
        ax.plot(frame[DATE].to_numpy(), frame[name].cum_sum().to_numpy(), label=label)
    ax.grid(zorder=-1.0)
    ax.set_title("Cumulative return and slippage")
    ax.set_xlim(xlims)  # type: ignore[arg-type]
    ax.set_xlabel("")
    ax.set_ylabel("USD")
    ax.legend()


def _plot_instrument_bars(fig: Figure | SubFigure, cell: gridspec.SubplotSpec, key_figures: dict[str, object]) -> None:
    grid = cell.subgridspec(3, 1, hspace=0.15)
    panels = (
        ("av_holding_per_instrument", "Av. holding"),
        ("sharpe_per_instrument", "Sharpe"),
        ("position_bias_per_instrument", "Position bias"),
    )
    first: Axes | None = None
    for row, (name, label) in enumerate(panels):
        ax = fig.add_subplot(grid[row, 0], sharex=first)
        values = key_figures[name]
        assert isinstance(values, dict)
        ax.bar(list(values), list(values.values()))
        ax.grid(axis="y", zorder=-1.0)
        ax.set_ylabel(label)
        if first is None:
            ax.set_title("Per instrument")
            first = ax
        ax.tick_params("x", labelbottom=row == len(panels) - 1, rotation=90)


def _plot_key_figure_table(ax: Axes, key_figures: dict[str, object]) -> None:
    rows = [
        ["Adjusted Sharpe (yearly)", f"{key_figures['adjusted_sharpe']:.2f}"],
        ["Sharpe (yearly)", f"{key_figures['sharpe']:.2f}"],
        ["Sharpe w. 1 day lagged pos. (yearly)", f"{key_figures['sharpe_1d_lag']:.2f}"],
        ["Sharpe ex. slippage (yearly)", f"{key_figures['sharpe_ex_slippage']:.2f}"],
        ["Skewness (daily)", f"{key_figures['skewness']:.2f}"],
        ["Kurtosis (daily)", f"{key_figures['kurtosis']:.2f}"],
        ["Average holding (in days)", f"{key_figures['av_holding']:.2f}"],
        ["Position bias", f"{key_figures['position_bias']:.2f}"],
    ]
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", bbox=Bbox.from_bounds(0.4, 0.3, 0.3, 0.7), edges="open")
    table.auto_set_font_size(False)
    table.set_fontsize(11)


def _plot_rolling(
    fig: Figure | SubFigure,
    cell: gridspec.SubplotSpec,
    key_figures: dict[str, object],
    xlims: tuple[date, date],
    rolling_window: int,
) -> None:
    grid = cell.subgridspec(3, 1, hspace=0.15)
    panels = (
        ("rolling_std", "Standard dev."),
        ("rolling_sharpe", "Sharpe"),
        ("rolling_position_bias", "Position bias"),
    )
    first: Axes | None = None
    for row, (name, label) in enumerate(panels):
        ax = fig.add_subplot(grid[row], sharex=first)
        frame = _frame(key_figures, name)
        ax.plot(frame[DATE].to_numpy(), frame[name].to_numpy())
        ax.set_ylabel(label)
        ax.grid(zorder=-1.0)
        ax.set_xlabel("")
        if first is None:
            ax.set_title(f"Rolling window ({rolling_window / TRADING_DAYS_PER_YEAR:.1g}y)")
            ax.set_xlim(xlims)  # type: ignore[arg-type]
            first = ax
        ax.tick_params("x", labelbottom=row == len(panels) - 1)
