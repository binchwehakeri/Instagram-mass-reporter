import math

import numpy as np
import pandas as pd
import pytest

from agent.broker import Broker, Fill
from agent.config import Settings
from agent.indicators import atr, ema, rsi
from agent.main import TradingAgent
from agent.reviewer import Decision
from agent.risk import Position, RiskManager, RiskState
from agent.strategy import generate_signal


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "paper")
    s = Settings()
    s.data_dir = tmp_path
    s.symbols = ["BTC/USDT"]
    return s


def make_df(closes):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes, "volume": 100.0},
        index=idx,
    )


# ---------------- indicators ----------------
def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(up).iloc[-1] == 100
    down = pd.Series(np.arange(100, 1, -1, dtype=float))
    assert rsi(down).iloc[-1] < 1
    noisy = pd.Series(100 + np.sin(np.arange(300)) * 5)
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_ema_and_atr():
    s = pd.Series([10.0] * 50)
    assert ema(s, 12).iloc[-1] == pytest.approx(10)
    df = make_df([100.0] * 50)
    assert atr(df).iloc[-1] == pytest.approx(2.0, rel=0.05)  # high-low = 2% of 100


# ---------------- strategy ----------------
def test_buy_signal_on_crossover_in_uptrend():
    # long uptrend, a dip, then a recovery -> bullish EMA cross above the 200 EMA
    closes = list(np.linspace(100, 200, 230)) + list(np.linspace(200, 185, 15)) + list(np.linspace(185, 205, 12))
    for n in range(240, len(closes) + 1):
        sig = generate_signal("BTC/USDT", make_df(closes[:n]), in_position=False)
        if sig.action == "buy":
            break
    assert sig.action == "buy"
    assert sig.atr > 0


def test_not_enough_history_holds():
    assert generate_signal("BTC/USDT", make_df([100] * 50), False).action == "hold"


# ---------------- risk ----------------
def test_size_respects_every_cap(settings):
    rm = RiskManager(settings, RiskState())
    qty, why = rm.size_buy(symbol="BTC/USDT", price=100, atr=1, equity=10_000, cash=10_000,
                           exposure={"BTC/USDT": 0}, claude_fraction=1.0)
    # risk cap: 1% of 10k / (2*1) * 100 = 5000; symbol cap 10% = 1000 -> symbol cap binds
    assert qty * 100 == pytest.approx(1000)
    assert "symbol cap" in why

    qty, _ = rm.size_buy(symbol="BTC/USDT", price=100, atr=1, equity=10_000, cash=10_000,
                         exposure={"BTC/USDT": 0}, claude_fraction=0.5)
    assert qty * 100 == pytest.approx(500)

    # Claude cannot enlarge a trade
    qty, _ = rm.size_buy(symbol="BTC/USDT", price=100, atr=1, equity=10_000, cash=10_000,
                         exposure={"BTC/USDT": 0}, claude_fraction=5.0)
    assert qty * 100 == pytest.approx(1000)


def test_total_cap_and_minimum(settings):
    rm = RiskManager(settings, RiskState())
    qty, _ = rm.size_buy(symbol="BTC/USDT", price=100, atr=1, equity=10_000, cash=10_000,
                         exposure={"BTC/USDT": 0, "ETH/USDT": 2_950}, claude_fraction=1.0)
    assert qty * 100 == pytest.approx(50)
    qty, why = rm.size_buy(symbol="BTC/USDT", price=100, atr=1, equity=10_000, cash=10_000,
                           exposure={"BTC/USDT": 0, "ETH/USDT": 2_995}, claude_fraction=1.0)
    assert qty == 0 and "minimum" in why


def test_daily_loss_halts_and_kill_switch(settings):
    rm = RiskManager(settings, RiskState())
    rm.start_of_cycle(10_000)
    assert rm.can_open()[0]
    rm.start_of_cycle(9_600)  # -4% > 3% limit
    ok, why = rm.can_open()
    assert not ok and "daily loss" in why

    rm2 = RiskManager(settings, RiskState())
    rm2.start_of_cycle(10_000)
    settings.kill_switch.touch()
    assert not rm2.can_open()[0]


def test_max_trades_per_day(settings):
    rm = RiskManager(settings, RiskState())
    rm.start_of_cycle(10_000)
    for _ in range(settings.max_trades_per_day):
        rm.record_trade()
    assert not rm.can_open()[0]


def test_exit_hit():
    pos = Position(1, 100, 95, 110, "")
    assert "stop" in RiskManager.exit_hit(pos, 94)
    assert "take-profit" in RiskManager.exit_hit(pos, 111)
    assert RiskManager.exit_hit(pos, 100) is None


def test_live_mode_requires_confirmation(settings):
    settings.mode = "live"
    with pytest.raises(SystemExit):
        settings.validate()


# ---------------- end-to-end loop with fakes ----------------
class FakeBroker(Broker):
    def __init__(self, s, closes):
        self.s, self.closes, self.cash, self.held = s, closes, 10_000.0, {}

    def ohlcv(self, symbol, limit=300):
        df = make_df(self.closes)
        return [[int(ts.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume] for ts, r in df.iterrows()]

    def price(self, symbol):
        return float(self.closes[-1])

    def balances(self):
        return self.cash, dict(self.held)

    def buy(self, symbol, qty):
        p = self.price(symbol)
        self.cash -= qty * p
        self.held["BTC"] = self.held.get("BTC", 0) + qty
        return Fill(symbol, "buy", qty, p, 0.0)

    def sell(self, symbol, qty):
        p = self.price(symbol)
        self.cash += qty * p
        self.held["BTC"] -= qty
        return Fill(symbol, "sell", qty, p, 0.0)


class FakeReviewer:
    def __init__(self, decision):
        self.decision, self.calls = decision, 0

    def review(self, signal, candles, context):
        self.calls += 1
        return self.decision


def _buy_closes():
    closes = list(np.linspace(100, 200, 230)) + list(np.linspace(200, 185, 15)) + list(np.linspace(185, 205, 12))
    for n in range(240, len(closes) + 1):
        if generate_signal("BTC/USDT", make_df(closes[:n]), False).action == "buy":
            return closes[:n]
    raise AssertionError("no buy setup")


def test_agent_buys_when_claude_approves_and_stops_out(settings):
    closes = _buy_closes()
    broker = FakeBroker(settings, closes)
    reviewer = FakeReviewer(Decision(approve=True, size_fraction=1.0, confidence=0.9, reasoning="ok"))
    agent = TradingAgent(settings, broker=broker, reviewer=reviewer)
    agent.cycle()
    assert "BTC/USDT" in agent.state.positions
    assert broker.held["BTC"] * closes[-1] <= 10_000 * settings.max_position_pct + 1e-6

    # price crashes below the stop -> closed without asking Claude
    calls = reviewer.calls
    broker.closes = closes + [closes[-1] * 0.5]
    agent.cycle()
    assert "BTC/USDT" not in agent.state.positions
    assert reviewer.calls == calls


def test_agent_respects_veto_and_low_confidence(settings):
    closes = _buy_closes()
    for d in (Decision(approve=False, size_fraction=1, confidence=0.9, reasoning="choppy"),
              Decision(approve=True, size_fraction=1, confidence=0.3, reasoning="unsure")):
        broker = FakeBroker(settings, closes)
        agent = TradingAgent(settings, broker=broker, reviewer=FakeReviewer(d))
        agent.state.positions.clear()
        agent.cycle()
        assert not agent.state.positions
        assert broker.held == {}


def test_review_failure_skips_buy(settings):
    class Boom:
        def review(self, *a):
            raise RuntimeError("api down")

    broker = FakeBroker(settings, _buy_closes())
    agent = TradingAgent(settings, broker=broker, reviewer=Boom())
    agent.cycle()
    assert broker.held == {}
