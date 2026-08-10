from __future__ import annotations

from typing import Iterable
import pandas as pd


def load_returns_csv(file) -> pd.DataFrame:
    """
    Load a CSV containing dates plus asset return columns.

    The first column is treated as a date column when it can be parsed.
    Returns should be decimal returns, e.g. 0.01 = 1%.
    """
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    first = df.columns[0]
    parsed = pd.to_datetime(df[first], errors="coerce")

    if parsed.notna().mean() > 0.8:
        df[first] = parsed
        df = df.set_index(first)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all").dropna(axis=1, how="all").dropna()

    if df.shape[1] < 1:
        raise ValueError("No numeric return columns were found.")

    return df


def download_yahoo_returns(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    interval: str = "1mo",
    return_horizon_months: int | None = None,
) -> pd.DataFrame:
    """
    Download Yahoo Finance adjusted prices and compute returns.

    Parameters
    ----------
    tickers
        Ticker symbols.
    start, end
        Date range.
    interval
        Native Yahoo interval such as 1d, 1wk, or 1mo.
    return_horizon_months
        If 3, 6, or 12, monthly prices are downloaded and converted to
        non-overlapping 3-, 6-, or 12-month returns.
    """
    import yfinance as yf

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("Provide at least one ticker.")

    if return_horizon_months is not None:
        h = int(return_horizon_months)
        if h not in (3, 6, 12):
            raise ValueError("return_horizon_months must be 3, 6, or 12.")
        yahoo_interval = "1mo"
    else:
        h = None
        yahoo_interval = interval

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=yahoo_interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    # Extract close prices robustly for one or multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close prices in Yahoo Finance data.")
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        elif "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
        else:
            raise ValueError("Could not find Close prices in Yahoo Finance data.")

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.sort_index().dropna(how="all").ffill().dropna()

    if h is None:
        returns = prices.pct_change(fill_method=None)
    else:
        # Build non-overlapping h-month returns from monthly observations.
        # p[0], p[h], p[2h], ... gives one independent horizon block each time.
        sampled_prices = prices.iloc[::h].copy()
        returns = sampled_prices.pct_change(fill_method=None)

    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(how="all").dropna(axis=1, how="all").dropna()

    if len(returns) < 2:
        label = (
            f"{h}-month" if h is not None
            else str(interval)
        )
        raise ValueError(
            f"Not enough history to compute {label} returns. "
            "Use an earlier start date."
        )

    return returns


def download_yahoo_horizon_returns(
    tickers,
    start: str,
    horizon_months: int,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download monthly adjusted prices and compute OVERLAPPING h-month returns
    every month:

        R_t^(h) = P_t / P_{t-h} - 1

    This is used for model estimation / return-horizon studies.
    """
    import yfinance as yf

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("Provide at least one ticker.")

    h = int(horizon_months)
    if h not in (1, 3, 6, 12):
        raise ValueError("horizon_months must be one of 1, 3, 6, 12.")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError("Could not find close prices.")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        if col not in raw.columns:
            raise ValueError("Could not find close prices.")
        prices = raw[[col]].copy()
        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.sort_index().dropna(how="all").ffill().dropna()

    returns = prices.pct_change(periods=h, fill_method=None)
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna(how="all").dropna(axis=1, how="all").dropna()

    if len(returns) < 2:
        raise ValueError("Not enough history to compute overlapping horizon returns.")

    return returns


def download_yahoo_monthly_returns(
    tickers,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Convenience helper for monthly realized returns used in OOS evaluation."""
    return download_yahoo_horizon_returns(
        tickers=tickers,
        start=start,
        end=end,
        horizon_months=1,
    )
