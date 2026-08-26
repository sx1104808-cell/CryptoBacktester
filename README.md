# Historical Backtesting Engine

Downloads historical OHLCV ("candlestick") data from a crypto exchange, caches it in a local SQLite database, and backtests a moving-average crossover strategy against it. It features realistic execution (next-bar open, fees, slippage), a full trade ledger, warmup period handling, and gap-checked data.

## Note on Data Gaps

The gap detection mechanism dynamically extrapolates the exchange's timestamp grid backward and forward from the known cache data. This strictly catches single missing boundary candles without falsely flagging user date requests that happen to fall between exchange intervals.

## Setup

Install the required dependencies:
```bash
pip install ccxt pandas matplotlib numpy

```

## Usage

Run the script via the command line:

```bash
python backtest_engine.py --symbol BTC/USDT --timeframe 1h \
    --start 2023-01-01 --end 2024-01-01 --fast 20 --slow 50 --cash 10000

```

### Command-Line Arguments

* **`--symbol`**: Trading pair, e.g. BTC/USDT (default: BTC/USDT).
* **`--timeframe`**: Candle size, e.g. 1m, 15m, 1h, 4h, 1d (default: 1h).
* **`--start`**: Start date, YYYY-MM-DD (Required).
* **`--end`**: End date, YYYY-MM-DD (Required).
* **`--fast`**: Fast moving-average window (default: 20).
* **`--slow`**: Slow moving-average window (default: 50).
* **`--cash`**: Starting cash (default: 10000.0).
* **`--fee`**: Fee per fill, as a fraction (0.001 = 0.1%) (default: 0.001).
* **`--slippage`**: Slippage per fill, as a fraction (default: 0.0005).
* **`--allocation`**: Fraction of cash committed per trade, (0-1] (default: 1.0).
* **`--exchange`**: ccxt exchange id (default: binance).
* **`--db`**: Path to the SQLite database (default: data/market_data.db).
* **`--max-retries`**: Retries for a failed exchange request (default: 3).
* **`--allow-gaps`**: Proceed even if the candle data has missing bars.
* **`--strict-warmup`**: Fail if the warmup period contains data gaps.
* **`--plot`**: Save an equity curve chart next to the database.
* **`--output`**: Also save results (json) or the trade ledger (csv).
