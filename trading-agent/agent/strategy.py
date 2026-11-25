"""Rules-based signal generator.

Long-only spot strategy:
  BUY  when the 12 EMA crosses above the 26 EMA, price is above the 200 EMA
       (uptrend) and RSI is not overbought.
  SELL when the 12 EMA crosses below the 26 EMA, or RSI is very overbought.

Signals are only proposals: Claude reviews every BUY/SELL, and the risk
manager has the final say on whether and how much to trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import add_indicators

MIN_BARS = 210  # need enough history for the 200 EMA


@dataclass
class Signal:
    symbol: str
    action: str  # "buy" | "sell" | "hold"
    reason: str
    price: float
    atr: float
    metrics: dict = field(default_factory=dict)


def generate_signal(symbol: str, df: pd.DataFrame, in_position: bool) -> Signal:
    if len(df) < MIN_BARS:
        last = float(df["close"].iloc[-1]) if len(df) else 0.0
        return Signal(symbol, "hold", f"only {len(df)} bars, need {MIN_BARS}", last, 0.0)

    d = add_indicators(df)
    # use the last *closed* candle; the final row is still forming
    cur, prev = d.iloc[-2], d.iloc[-3]
    price = float(d["close"].iloc[-1])

    metrics = {
        "close": round(price, 6),
        "ema_fast": round(float(cur.ema_fast), 6),
        "ema_slow": round(float(cur.ema_slow), 6),
        "ema_trend": round(float(cur.ema_trend), 6),
        "rsi": round(float(cur.rsi), 2),
        "atr": round(float(cur.atr), 6),
        "atr_pct": round(float(cur.atr) / price * 100, 3),
        "volume_vs_avg": round(float(cur.volume / cur.vol_avg), 2) if cur.vol_avg else None,
        "change_24_bars_pct": round((price / float(d["close"].iloc[-25]) - 1) * 100, 2),
    }

    crossed_up = prev.ema_fast <= prev.ema_slow and cur.ema_fast > cur.ema_slow
    crossed_down = prev.ema_fast >= prev.ema_slow and cur.ema_fast < cur.ema_slow
    uptrend = cur.close > cur.ema_trend

    if not in_position and crossed_up and uptrend and cur.rsi < 70:
        return Signal(symbol, "buy", "EMA12 crossed above EMA26 in an uptrend, RSI < 70", price, float(cur.atr), metrics)
    if in_position and crossed_down:
        return Signal(symbol, "sell", "EMA12 crossed below EMA26", price, float(cur.atr), metrics)
    if in_position and cur.rsi > 80:
        return Signal(symbol, "sell", "RSI above 80 (overbought)", price, float(cur.atr), metrics)
    return Signal(symbol, "hold", "no setup", price, float(cur.atr), metrics)


def recent_candles_table(df: pd.DataFrame, n: int = 30) -> str:
    """Compact CSV of recent candles for Claude to read."""
    tail = df.tail(n)
    lines = ["time,open,high,low,close,volume"]
    for ts, r in tail.iterrows():
        lines.append(f"{ts:%Y-%m-%d %H:%M},{r.open:.6g},{r.high:.6g},{r.low:.6g},{r.close:.6g},{r.volume:.4g}")
    return "\n".join(lines)
