# Claude Crypto Trading Agent

An automated, long-only crypto spot trading agent. A rules-based strategy finds setups, Claude reviews each one, and hard-coded risk limits decide whether to trade and how much.

```
every LOOP_SECONDS, for each symbol:
  1. stop-loss / take-profit hit?  -> sell immediately (no AI, never blocked)
  2. rules strategy                -> BUY / SELL / HOLD
  3. Claude reviews the signal     -> approve or veto, size 0-100%, confidence
  4. risk manager                  -> caps the size, or blocks the trade
  5. broker                        -> paper fill or real market order
  6. journal                       -> every decision and fill logged to SQLite
```

> **Warning:** This is software, not financial advice. Most automated strategies lose money, and crypto is very volatile. Run it in paper mode for weeks first, and only trade live with money you can afford to lose.

## Setup

```bash
cd trading-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
```

## 1. Backtest the rules (free, no keys)

```bash
python -m agent.backtest --symbol BTC/USDT --timeframe 1h --bars 3000
```

Claude can only *filter* signals. If the rules lose badly here, change the strategy (`agent/strategy.py`) before doing anything else.

## 2. Paper trade (default)

```bash
python -m agent.main --once   # one cycle
python -m agent.main          # run continuously
```

Paper mode uses real exchange prices but simulated money (`PAPER_STARTING_CASH`, 0.1% fees and slippage). You only need `ANTHROPIC_API_KEY` for this.

## 3. Live trading

1. Create an exchange API key with **trading enabled and withdrawals disabled**. If the exchange supports it, restrict the key to your IP address.
2. Optional: try it with `EXCHANGE_SANDBOX=true` first (Binance and Bybit have testnets).
3. In `.env` set:
   ```
   MODE=live
   EXCHANGE_API_KEY=...
   EXCHANGE_API_SECRET=...
   LIVE_TRADING_CONFIRM=I_ACCEPT_THE_RISK
   ```
4. Start with small limits, for example `MAX_POSITION_PCT=0.02`.

The agent refuses to start in live mode without that exact confirmation value.

## Safety controls

| Control | Default | Effect |
|---|---|---|
| `MAX_POSITION_PCT` | 10% | Most equity in any one coin |
| `MAX_TOTAL_EXPOSURE_PCT` | 30% | Most equity in crypto overall |
| `RISK_PER_TRADE_PCT` | 1% | Size so that a stop-out loses about 1% |
| `MAX_DAILY_LOSS_PCT` | 3% | Blocks new buys for the rest of the UTC day |
| `MAX_TRADES_PER_DAY` | 6 | Blocks new buys after this many trades |
| `STOP_ATR_MULT` / `TAKE_PROFIT_ATR_MULT` | 2 / 4 | Stop and target distance, as multiples of ATR |
| `MIN_CLAUDE_CONFIDENCE` | 0.6 | Skips trades Claude isn't confident about |
| **Kill switch** | — | `touch data/STOP` blocks all new buys; exits still run |

Design rules:
- Claude can only shrink or veto a trade, never make it bigger than the limits allow.
- If the Claude call fails, buys are skipped, but exits still go through.
- Spot only: no leverage, no shorting, no withdrawals.
- Stops are checked once per loop (`LOOP_SECONDS`), not placed on the exchange. A fast crash can fill below the stop price. Lower `LOOP_SECONDS` if that matters to you.

## Reviewing what it did

```bash
sqlite3 data/journal.sqlite "select ts,symbol,side,qty,price,reason from fills order by ts desc limit 20"
sqlite3 data/journal.sqlite "select ts,symbol,action,detail,data from events where kind='review' order by ts desc limit 10"
```

## Tests

```bash
python -m pytest -q
```

The tests use a fake broker and a fake reviewer, so they need no network and no API key.

## Files

| File | Role |
|---|---|
| `agent/strategy.py` | Rules: EMA 12/26 crossover, 200 EMA trend filter, RSI filter |
| `agent/reviewer.py` | Claude review (structured output, with a fallback model if Claude declines) |
| `agent/risk.py` | Hard limits, position sizing, stops, daily state |
| `agent/broker.py` | ccxt paper broker and live broker |
| `agent/main.py` | The trading loop |
| `agent/backtest.py` | Historical test of rules plus risk limits |
| `agent/journal.py` | SQLite audit log |

## Cost

Claude is called only when the rules produce a BUY or SELL signal, not on every loop, which is typically a few times a day. The default model is `claude-opus-5-5` at `medium` effort. Set `CLAUDE_MODEL=claude-sonnet-5-5` in `.env` to make it cheaper.
