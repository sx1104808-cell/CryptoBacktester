"""
Historical Backtesting Engine
==============================

Downloads historical OHLCV ("candlestick") data from a crypto exchange,
caches it in a local SQLite database, and backtests a moving-average
crossover strategy against it — with realistic execution (next-bar open,
fees, slippage), a full trade ledger, warmup period handling, and gap-checked data.

Note on Data Gaps
-----------------
The gap detection mechanism dynamically extrapolates the exchange's timestamp 
grid backward and forward from the known cache data. This strictly catches 
single missing boundary candles without falsely flagging user date requests 
that happen to fall between exchange intervals. 

Setup
-----
    pip install ccxt pandas matplotlib numpy

Usage
-----
    python backtest_engine.py --symbol BTC/USDT --timeframe 1h \
        --start 2023-01-01 --end 2024-01-01 --fast 20 --slow 50 --cash 10000
"""

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & Helpers
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "market_data.db"


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol    TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open      REAL NOT NULL,
            high      REAL NOT NULL,
            low       REAL NOT NULL,
            close     REAL NOT NULL,
            volume    REAL NOT NULL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol     TEXT NOT NULL,
            timeframe  TEXT NOT NULL,
            start_ms   INTEGER NOT NULL,
            end_ms     INTEGER NOT NULL,
            fetched_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_candles(conn: sqlite3.Connection, symbol: str, timeframe: str, rows) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO candles
            (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(symbol, timeframe, *row) for row in rows],
    )
    conn.commit()


def log_fetch(conn: sqlite3.Connection, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> None:
    conn.execute(
        "INSERT INTO fetch_log (symbol, timeframe, start_ms, end_ms, fetched_at) VALUES (?, ?, ?, ?, ?)",
        (symbol, timeframe, start_ms, end_ms, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _cached_timestamps(conn: sqlite3.Connection, symbol: str, timeframe: str, start_ms: int, end_ms: int):
    rows = conn.execute(
        "SELECT timestamp FROM candles WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp < ? ORDER BY timestamp ASC",
        (symbol, timeframe, start_ms, end_ms),
    ).fetchall()
    return [r[0] for r in rows]


def _missing_ranges(existing_ts, start_ms: int, end_ms: int, step_ms: int):
    if not existing_ts:
        return [(start_ms, end_ms)]

    gaps = []
    
    # Extrapolate grid backward to find the first expected candle >= start_ms
    aligned_start = existing_ts[0]
    if aligned_start > start_ms:
        steps = (aligned_start - start_ms) // step_ms
        aligned_start -= steps * step_ms
        
        if existing_ts[0] > aligned_start:
            gaps.append((aligned_start, existing_ts[0]))

    # Internal gaps
    for i in range(len(existing_ts) - 1):
        curr = existing_ts[i]
        nxt = existing_ts[i + 1]
        if nxt - curr > step_ms:
            gaps.append((curr + step_ms, nxt))

    # Extrapolate grid forward to find the last expected candle < end_ms
    aligned_end = existing_ts[-1]
    if end_ms > aligned_end:
        # Subtract 1 because end_ms is strictly an exclusive bound
        steps = ((end_ms - 1) - aligned_end) // step_ms
        aligned_end += steps * step_ms

        last_expected_next = existing_ts[-1] + step_ms
        if last_expected_next <= aligned_end:
            gaps.append((last_expected_next, min(aligned_end + step_ms, end_ms)))

    return gaps


def find_gaps(df: pd.DataFrame, start_ms: int, end_ms: int, step_ms: int):
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    existing_ts = ((df.index - epoch) // pd.Timedelta(milliseconds=1)).astype("int64").tolist()
    
    existing_ts = [ts for ts in existing_ts if start_ms <= ts < end_ms]
    return _missing_ranges(existing_ts, start_ms, end_ms, step_ms)


def load_candles(conn: sqlite3.Connection, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp ASC
        """,
        conn,
        params=(symbol, timeframe, start_ms, end_ms),
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


def validate_candle_data_integrity(df: pd.DataFrame) -> None:
    if df.empty:
        raise SystemExit("No candle data available to validate.")
    if df.index.duplicated().any():
        raise SystemExit("Duplicate timestamps found in loaded candles — data integrity issue.")
    if not df.index.is_monotonic_increasing:
        raise SystemExit("Candle data is not sorted by time.")

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    
    if df[ohlcv_cols].isnull().any().any():
        raise SystemExit("Candle data contains missing OHLCV values.")
    if not np.isfinite(df[ohlcv_cols].to_numpy()).all():
        raise SystemExit("Candle data contains non-finite (Infinity or NaN) values.")

    if (df["high"] < df["low"]).any():
        raise SystemExit("Candle data contains rows where high < low.")
    if (df["open"] < df["low"]).any() or (df["open"] > df["high"]).any():
        raise SystemExit("Open price falls outside the candle's high/low range.")
    if (df["close"] < df["low"]).any() or (df["close"] > df["high"]).any():
        raise SystemExit("Close price falls outside the candle's high/low range.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise SystemExit("Prices must be positive.")
    if (df["volume"] < 0).any():
        raise SystemExit("Volume cannot be negative.")


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def fetch_and_cache(exchange, step_ms: int, conn: sqlite3.Connection, symbol: str, timeframe: str, 
                    start_ms: int, end_ms: int, max_retries: int = 3) -> None:
    
    existing_ts = _cached_timestamps(conn, symbol, timeframe, start_ms, end_ms)
    gaps = _missing_ranges(existing_ts, start_ms, end_ms, step_ms)

    if not gaps:
        print(f"Cache already covers {symbol} {timeframe} for the requested range — skipping download.")
        return

    total_saved = 0
    for gap_start, gap_end in gaps:
        print(f"Fetching {symbol} {timeframe} from {exchange.id} "
              f"({_fmt_ts(gap_start)} -> {_fmt_ts(gap_end)})...")
        total_saved += _download_range(exchange, conn, symbol, timeframe, gap_start, gap_end, step_ms, max_retries)

    log_fetch(conn, symbol, timeframe, start_ms, end_ms)
    print(f"Done. {total_saved} new candles cached.")


def _download_range(exchange, conn, symbol, timeframe, range_start, range_end, step_ms, max_retries) -> int:
    cursor = range_start
    limit = 1000
    saved = 0

    while cursor < range_end:
        batch = None
        for attempt in range(1, max_retries + 1):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                break
            except Exception as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch {symbol} {timeframe} after {max_retries} attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt
                print(f"  fetch failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)

        if not batch:
            break

        raw_last_ts = batch[-1][0]
        filtered = [row for row in batch if range_start <= row[0] < range_end]
        if filtered:
            save_candles(conn, symbol, timeframe, filtered)
            saved += len(filtered)

        if raw_last_ts <= cursor:
            break
        cursor = raw_last_ts + step_ms

        if len(batch) < limit or cursor >= range_end:
            break

    return saved


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
def add_moving_averages(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl: float
    return_pct: float
    fees: float
    estimated_slippage: float
    forced_close: bool

    @property
    def winner(self) -> bool:
        return self.pnl > 0


def compute_buy_hold_equity(df: pd.DataFrame, starting_cash: float, fee_rate: float, slippage_rate: float) -> pd.Series:
    entry_price = df["open"].iloc[0] * (1 + slippage_rate)
    entry_fee = starting_cash * fee_rate
    holdings = (starting_cash - entry_fee) / entry_price

    equity = holdings * df["close"]

    exit_price = df["close"].iloc[-1] * (1 - slippage_rate)
    exit_proceeds = holdings * exit_price
    exit_fee = exit_proceeds * fee_rate
    equity.iloc[-1] = exit_proceeds - exit_fee
    return equity


def run_backtest(df: pd.DataFrame, starting_cash: float, fee_rate: float, slippage_rate: float, allocation: float, initial_signal: bool = False):
    df = df.copy()
    n = len(df)

    cash = starting_cash
    holdings = 0.0
    in_position = False
    entry_price = entry_time = entry_cash_used = entry_fee = None

    equity_curve = [starting_cash] * n
    trades: list[Trade] = []

    for i in range(n):
        ts = df.index[i]
        open_price = df["open"].iloc[i]

        want_in_position = initial_signal if i == 0 else bool(df["signal"].iloc[i - 1])

        if want_in_position and not in_position:
            buy_price = open_price * (1 + slippage_rate)
            spend = cash * allocation
            fee_cost = spend * fee_rate
            holdings = (spend - fee_cost) / buy_price
            cash -= spend
            in_position = True
            entry_price, entry_time, entry_cash_used, entry_fee = buy_price, ts, spend, fee_cost
            entry_slippage_cost = holdings * open_price * slippage_rate

        elif not want_in_position and in_position:
            sell_price = open_price * (1 - slippage_rate)
            proceeds = holdings * sell_price
            exit_fee = proceeds * fee_rate
            exit_slippage_cost = holdings * open_price * slippage_rate
            net_proceeds = proceeds - exit_fee
            cash += net_proceeds
            
            trade_pnl = net_proceeds - entry_cash_used
            trades.append(Trade(
                entry_time=entry_time, entry_price=entry_price,
                exit_time=ts, exit_price=sell_price,
                pnl=trade_pnl,
                return_pct=(trade_pnl / entry_cash_used) * 100,
                fees=entry_fee + exit_fee,
                estimated_slippage=entry_slippage_cost + exit_slippage_cost,
                forced_close=False,
            ))
            holdings = 0.0
            in_position = False
            entry_price = entry_time = entry_cash_used = entry_fee = None

        equity_curve[i] = cash + holdings * df["close"].iloc[i]

    if in_position:
        final_close = df["close"].iloc[-1]
        sell_price = final_close * (1 - slippage_rate)
        proceeds = holdings * sell_price
        exit_fee = proceeds * fee_rate
        exit_slippage_cost = holdings * final_close * slippage_rate
        net_proceeds = proceeds - exit_fee
        
        trade_pnl = net_proceeds - entry_cash_used
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            exit_time=df.index[-1], exit_price=sell_price,
            pnl=trade_pnl,
            return_pct=(trade_pnl / entry_cash_used) * 100,
            fees=entry_fee + exit_fee,
            estimated_slippage=entry_slippage_cost + exit_slippage_cost,
            forced_close=True,
        ))
        cash += net_proceeds
        equity_curve[-1] = cash

    df["equity"] = equity_curve
    df["buy_hold_equity"] = compute_buy_hold_equity(df, starting_cash, fee_rate, slippage_rate)
    return df, trades


def compute_metrics(df: pd.DataFrame, trades: list, starting_cash: float, avg_bar_seconds: float, has_gaps: bool = False, initial_signal: bool = False) -> dict:
    final_equity = float(df["equity"].iloc[-1])
    buy_hold_final = float(df["buy_hold_equity"].iloc[-1])
    total_return_pct = (final_equity / starting_cash - 1) * 100
    buy_hold_return_pct = (buy_hold_final / starting_cash - 1) * 100

    bar_returns = df["equity"].pct_change().fillna(0)
    seconds_per_year = 365.25 * 24 * 3600
    bars_per_year = seconds_per_year / avg_bar_seconds if avg_bar_seconds else float("nan")

    elapsed_seconds = (df.index[-1] - df.index[0]).total_seconds()
    years = elapsed_seconds / seconds_per_year if elapsed_seconds > 0 else float("nan")

    running_max = df["equity"].cummax()
    drawdown = (df["equity"] - running_max) / running_max
    max_drawdown_pct = float(drawdown.min() * 100)

    max_dd_streak = streak = 0
    for is_down in (drawdown < 0):
        streak = streak + 1 if is_down else 0
        max_dd_streak = max(max_dd_streak, streak)

    if has_gaps:
        cagr_pct = float("nan")
        vol_annual_pct = float("nan")
        sharpe = float("nan")
        sortino = float("nan")
        longest_underwater_days = float("nan")
    else:
        cagr_pct = ((final_equity / starting_cash) ** (1 / years) - 1) * 100 if years and years > 0 else float("nan")
        vol_annual_pct = bar_returns.std() * np.sqrt(bars_per_year) * 100 if bars_per_year == bars_per_year else float("nan")

        downside = bar_returns[bar_returns < 0]
        downside_std = downside.std() if len(downside) else 0.0
        sharpe = (bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year)) if bar_returns.std() else float("nan")
        sortino = (bar_returns.mean() / downside_std * np.sqrt(bars_per_year)) if downside_std else float("nan")
        longest_underwater_days = (max(0, max_dd_streak - 1) * avg_bar_seconds) / 86400 if avg_bar_seconds else float("nan")

    position_series = df["signal"].shift(1).fillna(int(initial_signal))
    time_in_market_pct = float(position_series.mean() * 100)

    num_trades = len(trades)
    wins = [t for t in trades if t.winner]
    losses = [t for t in trades if not t.winner]
    win_rate_pct = (len(wins) / num_trades * 100) if num_trades else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else float("nan")

    best_trade = max(trades, key=lambda t: t.return_pct) if trades else None
    worst_trade = min(trades, key=lambda t: t.return_pct) if trades else None

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "buy_hold_final_equity": buy_hold_final,
        "buy_hold_return_pct": buy_hold_return_pct,
        "outperformance_pct": total_return_pct - buy_hold_return_pct,
        "cagr_pct": cagr_pct,
        "annualized_volatility_pct": vol_annual_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_drawdown_pct,
        "longest_underwater_bars": max_dd_streak,
        "longest_underwater_days": longest_underwater_days,
        "time_in_market_pct": time_in_market_pct,
        "num_trades": num_trades,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "best_trade_pct": best_trade.return_pct if best_trade else None,
        "worst_trade_pct": worst_trade.return_pct if worst_trade else None,
        "total_fees": sum(t.fees for t in trades),
        "total_slippage": sum(t.estimated_slippage for t in trades),
        "equity_curve": df[["equity", "buy_hold_equity"]],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x, suffix="", decimals=2):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a" if np.isnan(x) else ("∞" if x > 0 else "-∞")
    return f"{x:,.{decimals}f}{suffix}"


def _clean_json_floats(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_floats(i) for i in obj]
    return obj


def print_report(symbol: str, timeframe: str, fast: int, slow: int, r: dict, has_eval_gaps: bool, has_warmup_gaps: bool) -> None:
    gap_note = " (Omitted due to data gaps)" if has_eval_gaps else ""

    print("\n" + "=" * 60)
    print(f"  {symbol}  ({timeframe})  —  MA({fast}) / MA({slow}) crossover")
    print("=" * 60)
    print(f"  Final equity:               ${_fmt(r['final_equity'])}")
    print(f"  Strategy return:            {_fmt(r['total_return_pct'], '%')}")
    print(f"  Buy & hold return (net):    {_fmt(r['buy_hold_return_pct'], '%')}")
    print(f"  Outperformance:             {_fmt(r['outperformance_pct'], '%')}")
    print(f"  CAGR:                       {_fmt(r['cagr_pct'], '%')}{gap_note}")
    print(f"  Annualized volatility:      {_fmt(r['annualized_volatility_pct'], '%')}{gap_note}")
    print(f"  Sharpe ratio:               {_fmt(r['sharpe_ratio'])}{gap_note}")
    print(f"  Sortino ratio:              {_fmt(r['sortino_ratio'])}{gap_note}")
    print(f"  Max drawdown:               {_fmt(r['max_drawdown_pct'], '%')}")
    print(f"  Longest time underwater:    {r['longest_underwater_bars']} bars (~{_fmt(r['longest_underwater_days'], ' days')}){gap_note}")
    print(f"  Time in market:             {_fmt(r['time_in_market_pct'], '%')}")
    print("-" * 60)
    print(f"  Number of trades:           {r['num_trades']}")
    print(f"  Win rate:                   {_fmt(r['win_rate_pct'], '%')}")
    print(f"  Profit factor:              {_fmt(r['profit_factor'])}")
    print(f"  Best trade:                 {_fmt(r['best_trade_pct'], '%')}")
    print(f"  Worst trade:                {_fmt(r['worst_trade_pct'], '%')}")
    print(f"  Total fees paid:            ${_fmt(r['total_fees'])}")
    print(f"  Est. total slippage:        ${_fmt(r['total_slippage'])}")
    print("-" * 60)
    
    if has_eval_gaps:
        print("  Time-dependent metrics were omitted because gaps were")
        print("  allowed and present in the evaluation data.")
    else:
        print("  Sharpe/Sortino assume a 0% risk-free rate and evenly spaced")
        print("  bars (checked by the gap validation above).")
        if has_warmup_gaps:
            print("  * Note: Warmup data contained gaps. Initial MAs may be slightly skewed.")
            
    print("=" * 60 + "\n")


def save_equity_chart(results: dict, symbol: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = results["equity_curve"]
    plt.figure(figsize=(10, 5))
    plt.plot(curve.index, curve["equity"], label="Strategy")
    plt.plot(curve.index, curve["buy_hold_equity"], label="Buy & hold (net of costs)", linestyle="--")
    plt.title(f"{symbol} — strategy vs buy & hold")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_path)
    plt.close()
    print(f"Chart saved -> {out_path}")


def save_results(results: dict, trades: list, args, db_path: Path, run_stamp: str, 
                 initial_signal: bool, has_warmup_gaps: bool, has_eval_gaps: bool) -> None:
    out_dir = db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.symbol.replace('/', '-')}_{args.timeframe}_ma{args.fast}-{args.slow}_{run_stamp}"

    config = {
        "symbol": args.symbol, "timeframe": args.timeframe, "start": args.start, "end": args.end,
        "fast": args.fast, "slow": args.slow, "cash": args.cash, "fee": args.fee,
        "slippage": args.slippage, "allocation": args.allocation, "exchange": args.exchange,
        "allow_gaps": args.allow_gaps,
        "strict_warmup": args.strict_warmup,
        "max_retries": args.max_retries,
        "warmup_bars": args.slow,
        "initial_signal": initial_signal,
        "has_warmup_gaps": has_warmup_gaps,
        "has_eval_gaps": has_eval_gaps,
    }
    
    metrics = {k: v for k, v in results.items() if k != "equity_curve"}

    if args.output == "json":
        path = out_dir / f"{base}_results.json"
        payload = _clean_json_floats({
            "config": config, 
            "metrics": metrics, 
            "trades": [asdict(t) for t in trades]
        })
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Results saved -> {path}")
    elif args.output == "csv":
        path = out_dir / f"{base}_trades.csv"
        pd.DataFrame([asdict(t) for t in trades]).to_csv(path, index=False)
        print(f"Trade ledger saved -> {path}")


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Backtest a moving-average crossover strategy.")
    p.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT")
    p.add_argument("--timeframe", default="1h", help="Candle size, e.g. 1m, 15m, 1h, 4h, 1d")
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    p.add_argument("--fast", type=int, default=20, help="Fast moving-average window")
    p.add_argument("--slow", type=int, default=50, help="Slow moving-average window")
    p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash")
    p.add_argument("--fee", type=float, default=0.001, help="Fee per fill, as a fraction (0.001 = 0.1%%)")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage per fill, as a fraction")
    p.add_argument("--allocation", type=float, default=1.0, help="Fraction of cash committed per trade, (0-1]")
    p.add_argument("--exchange", default="binance", help="ccxt exchange id")
    p.add_argument("--db", type=Path, default=None, help="Path to the SQLite database (default: data/market_data.db)")
    p.add_argument("--max-retries", type=int, default=3, help="Retries for a failed exchange request")
    p.add_argument("--allow-gaps", action="store_true", help="Proceed even if the candle data has missing bars")
    p.add_argument("--strict-warmup", action="store_true", help="Fail if the warmup period contains data gaps")
    p.add_argument("--plot", action="store_true", help="Save an equity curve chart next to the database")
    p.add_argument("--output", choices=["json", "csv"], help="Also save results (json) or the trade ledger (csv)")
    return p.parse_args()


def validate_args(args, start_ms: int, end_ms: int) -> None:
    numeric_args = [args.cash, args.fee, args.slippage, args.allocation]
    if not np.isfinite(numeric_args).all():
        raise SystemExit("Cash, fee, slippage, and allocation must be finite numbers.")
        
    if args.fast <= 0:
        raise SystemExit("--fast must be greater than zero")
    if args.slow <= 0:
        raise SystemExit("--slow must be greater than zero")
    if args.fast >= args.slow:
        raise SystemExit("--fast must be smaller than --slow")
    if args.cash <= 0:
        raise SystemExit("--cash must be greater than zero")
    if not (0 <= args.fee < 1):
        raise SystemExit("--fee must be between 0 (inclusive) and 1 (exclusive)")
    if not (0 <= args.slippage < 1):
        raise SystemExit("--slippage must be between 0 (inclusive) and 1 (exclusive)")
    if not (0 < args.allocation <= 1):
        raise SystemExit("--allocation must be between 0 (exclusive) and 1 (inclusive)")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be greater than zero")
    if end_ms <= start_ms:
        raise SystemExit("--end must be after --start")


def main():
    args = parse_args()

    try:
        start_ms = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError as exc:
        raise SystemExit(f"Could not parse --start/--end as YYYY-MM-DD: {exc}")

    validate_args(args, start_ms, end_ms)
    db_path = args.db if args.db else DEFAULT_DB_PATH

    if args.exchange not in ccxt.exchanges:
        raise SystemExit(f"Unsupported exchange: {args.exchange!r}. See ccxt.exchanges for supported ids.")

    exchange_class = getattr(ccxt, args.exchange)
    exchange = exchange_class({"enableRateLimit": True})

    try:
        exchange.load_markets()
        if args.symbol not in exchange.markets:
            raise SystemExit(f"Symbol {args.symbol!r} is not supported by {args.exchange}.")
            
        if exchange.timeframes and args.timeframe not in exchange.timeframes:
            raise SystemExit(f"Timeframe {args.timeframe!r} is not supported by {args.exchange}.")
            
        try:
            step_ms = exchange.parse_timeframe(args.timeframe) * 1000
        except Exception as exc:
            raise SystemExit(f"Unsupported timeframe {args.timeframe!r} for exchange {args.exchange!r}: {exc}")

        warmup_bars = args.slow
        warmup_ms = warmup_bars * step_ms
        fetch_start_ms = start_ms - warmup_ms

        conn = get_connection(db_path)
        try:
            fetch_and_cache(exchange, step_ms, conn, args.symbol, args.timeframe, fetch_start_ms, end_ms,
                             max_retries=args.max_retries)
            df = load_candles(conn, args.symbol, args.timeframe, fetch_start_ms, end_ms)
        finally:
            conn.close()
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    # Basic OHLCV sanity validation (NaNs, Sorting, Open/High/Low/Close structure)
    validate_candle_data_integrity(df)

    # Gap processing explicitly separated by window
    warmup_gaps = find_gaps(df, fetch_start_ms, start_ms, step_ms)
    eval_gaps = find_gaps(df, start_ms, end_ms, step_ms)
    
    has_warmup_gaps = len(warmup_gaps) > 0
    has_eval_gaps = len(eval_gaps) > 0

    if has_warmup_gaps:
        preview = ", ".join(f"{_fmt_ts(g[0])} -> {_fmt_ts(g[1])}" for g in warmup_gaps[:3])
        warn_msg = f"Initial warmup data has {len(warmup_gaps)} gap(s): {preview}"
        if args.strict_warmup:
            raise SystemExit(f"ERROR: {warn_msg}\n  Failing because --strict-warmup is enabled.")
        else:
            print(f"WARNING: {warn_msg}\n  Moving averages may take slightly longer to stabilize accurately.")
    
    if has_eval_gaps:
        preview = ", ".join(f"{_fmt_ts(g[0])} -> {_fmt_ts(g[1])}" for g in eval_gaps[:5])
        more = f" (+{len(eval_gaps) - 5} more)" if len(eval_gaps) > 5 else ""
        message = (f"{len(eval_gaps)} gap(s) exist in the evaluation window for {args.symbol} {args.timeframe}: "
                   f"{preview}{more}")
        if args.allow_gaps:
            print(f"WARNING: {message}\n  Proceeding anyway (--allow-gaps set) — time-dependent metrics will be omitted.")
        else:
            raise SystemExit(
                message + "\nThis usually means the exchange doesn't have data that far back, or a "
                "request failed. Pass --allow-gaps to proceed anyway."
            )

    df = add_moving_averages(df, args.fast, args.slow)
    df = generate_signals(df)

    # Capture the boundary signal right before trimming the warmup candles
    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    warmup_mask = df.index < start_dt
    
    if warmup_mask.any():
        warmup_signal = bool(df.loc[warmup_mask, "signal"].iloc[-1])
    else:
        warmup_signal = False

    # Trim warmup candles so evaluation rigorously matches the requested timeframe
    df = df[~warmup_mask].copy()

    if len(df) < 2:
        raise SystemExit("Not enough evaluation candles remain to run the backtest.")

    avg_bar_seconds = df.index.to_series().diff().median().total_seconds()
    df_out, trades = run_backtest(df, starting_cash=args.cash, fee_rate=args.fee,
                                   slippage_rate=args.slippage, allocation=args.allocation, 
                                   initial_signal=warmup_signal)
    
    results = compute_metrics(df_out, trades, args.cash, avg_bar_seconds, has_gaps=has_eval_gaps, initial_signal=warmup_signal)

    print_report(args.symbol, args.timeframe, args.fast, args.slow, results, has_eval_gaps=has_eval_gaps, has_warmup_gaps=has_warmup_gaps)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.plot:
        chart_name = f"{args.symbol.replace('/', '-')}_{args.timeframe}_ma{args.fast}-{args.slow}_{run_stamp}_equity.png"
        save_equity_chart(results, args.symbol, db_path.parent / chart_name)

    if args.output:
        save_results(results, trades, args, db_path, run_stamp, initial_signal=warmup_signal, has_warmup_gaps=has_warmup_gaps, has_eval_gaps=has_eval_gaps)


if __name__ == "__main__":
    main()