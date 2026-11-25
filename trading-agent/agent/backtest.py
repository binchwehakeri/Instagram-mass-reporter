"""Backtest the rules strategy plus risk limits on historical candles (no Claude calls).

    python -m agent.backtest --symbol BTC/USDT --timeframe 1h --bars 3000

Claude only filters signals, so if the rules alone lose money here, the agent
will most likely lose money live too. Check this before going live.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from .broker import PAPER_FEE, PAPER_SLIPPAGE, make_exchange
from .config import Settings
from .indicators import ohlcv_to_df
from .risk import Position, RiskManager, RiskState
from .strategy import MIN_BARS, generate_signal


def fetch_history(s: Settings, symbol: str, bars: int) -> pd.DataFrame:
    ex = make_exchange(s, with_keys=False)
    ms = ex.parse_timeframe(s.timeframe) * 1000
    since = ex.milliseconds() - bars * ms
    rows: list = []
    while len(rows) < bars:
        batch = ex.fetch_ohlcv(symbol, s.timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + ms
        if len(batch) < 2:
            break
    return ohlcv_to_df(rows).loc[lambda d: ~d.index.duplicated()]


def run(s: Settings, symbol: str, df: pd.DataFrame) -> dict:
    s.symbols = [symbol]
    rm = RiskManager(s, RiskState(), use_kill_switch=False)

    cash, qty, pos = s.paper_starting_cash, 0.0, None
    curve, trades = [], []
    for i in range(MIN_BARS + 2, len(df)):
        window = df.iloc[: i + 1]
        bar = window.iloc[-1]
        price = float(bar.close)
        equity = cash + qty * price
        rm.start_of_cycle(equity, now=bar.name.to_pydatetime())

        if pos:
            # intrabar stop / target check using the candle's range
            exit_px = None
            if bar.low <= pos.stop:
                exit_px = pos.stop
            elif bar.high >= pos.take_profit:
                exit_px = pos.take_profit
            if exit_px is None and generate_signal(symbol, window, True).action == "sell":
                exit_px = price
            if exit_px is not None:
                px = exit_px * (1 - PAPER_SLIPPAGE)
                cash += qty * px * (1 - PAPER_FEE)
                trades.append((px - pos.entry) / pos.entry - 2 * PAPER_FEE)
                qty, pos = 0.0, None
                rm.record_trade()
        else:
            sig = generate_signal(symbol, window, False)
            if sig.action == "buy" and rm.can_open()[0]:
                q, _ = rm.size_buy(symbol=symbol, price=price, atr=sig.atr, equity=equity, cash=cash,
                                   exposure={symbol: 0.0}, claude_fraction=1.0)
                if q > 0:
                    px = price * (1 + PAPER_SLIPPAGE)
                    cash -= q * px * (1 + PAPER_FEE)
                    qty = q
                    stop, tp = rm.stops_for(px, sig.atr)
                    pos = Position(q, px, stop, tp, str(bar.name))
                    rm.record_trade()
        curve.append(cash + qty * price)

    eq = pd.Series(curve)
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
    wins = [t for t in trades if t > 0]
    return SimpleNamespace(
        bars=len(df),
        start=str(df.index[0]),
        end=str(df.index[-1]),
        total_return=eq.iloc[-1] / s.paper_starting_cash - 1 if len(eq) else 0,
        buy_and_hold=float(df.close.iloc[-1] / df.close.iloc[MIN_BARS + 2] - 1),
        max_drawdown=float(dd),
        trades=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0,
        avg_trade=sum(trades) / len(trades) if trades else 0,
    ).__dict__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--bars", type=int, default=3000)
    a = ap.parse_args()
    s = Settings()
    if a.timeframe:
        s.timeframe = a.timeframe
    s.data_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_history(s, a.symbol, a.bars)
    r = run(s, a.symbol, df)
    print(f"\nBacktest {a.symbol} {s.timeframe} on {s.exchange}: {r['start']} -> {r['end']} ({r['bars']} bars)")
    print(f"  strategy return : {r['total_return']:+.2%}")
    print(f"  buy & hold      : {r['buy_and_hold']:+.2%}")
    print(f"  max drawdown    : {r['max_drawdown']:.2%}")
    print(f"  trades          : {r['trades']}  win rate {r['win_rate']:.0%}  avg {r['avg_trade']:+.2%}")
    print("  (note: the strategy only risks a slice of equity per trade, so compare drawdown, not just return)")


if __name__ == "__main__":
    main()
