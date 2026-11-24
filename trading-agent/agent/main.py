"""Trading loop: rules propose -> Claude reviews -> risk manager sizes -> broker executes.

Run:  python -m agent.main            (loop forever)
      python -m agent.main --once     (one cycle, then exit)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from .broker import Broker, make_broker
from .config import Settings
from .indicators import ohlcv_to_df
from .journal import Journal
from .reviewer import ClaudeReviewer
from .risk import Position, RiskManager, RiskState
from .strategy import generate_signal, recent_candles_table

log = logging.getLogger("agent")


class TradingAgent:
    def __init__(self, s: Settings, broker: Broker | None = None, reviewer: ClaudeReviewer | None = None):
        s.validate()
        self.s = s
        self.broker = broker or make_broker(s)
        self.reviewer = reviewer or ClaudeReviewer(s)
        self.state_path = s.data_dir / f"state_{s.mode}.json"
        self.state = RiskState.load(self.state_path)
        self.risk = RiskManager(s, self.state)
        self.journal = Journal(s.data_dir / "journal.sqlite")

    # ------------------------------------------------------------------
    def cycle(self) -> None:
        prices = {sym: self.broker.price(sym) for sym in self.s.symbols}
        equity, cash, exposure = self.broker.equity(prices)
        self.risk.start_of_cycle(equity)
        self._reconcile()
        self.journal.equity(self.s.mode, equity, cash)
        day_pnl = equity / self.state.day_start_equity - 1 if self.state.day_start_equity else 0
        log.info("equity %.2f %s | cash %.2f | day %+.2f%% | trades today %d%s",
                 equity, self.s.quote_currency, cash, day_pnl * 100, self.state.trades_today,
                 f" | HALTED: {self.state.halted_reason}" if self.state.halted_reason else "")

        for sym in self.s.symbols:
            try:
                self._handle_symbol(sym, prices[sym], equity, exposure)
                # refresh balances after any trade
                equity, cash, exposure = self.broker.equity({k: self.broker.price(k) for k in self.s.symbols})
            except Exception as e:  # noqa: BLE001 - one bad symbol must not stop the loop
                log.exception("%s: error: %s", sym, e)
                self.journal.event(self.s.mode, sym, "error", "", str(e))
        self.state.save(self.state_path)

    def _reconcile(self) -> None:
        """Drop tracked positions that no longer exist on the account (e.g. sold by hand)."""
        _, held = self.broker.balances()
        for sym in list(self.state.positions):
            base = sym.split("/")[0]
            pos = self.state.positions[sym]
            if held.get(base, 0.0) <= pos.qty * 0.01:
                log.warning("%s: position no longer on account, forgetting it", sym)
                del self.state.positions[sym]
            else:
                pos.qty = min(pos.qty, held[base])

    def _handle_symbol(self, sym: str, price: float, equity: float, exposure: dict[str, float]) -> None:
        pos = self.state.positions.get(sym)

        # 1. mechanical exits first - never reviewed, never blocked
        if pos:
            why = self.risk.exit_hit(pos, price)
            if why:
                self._close(sym, pos, why)
                return

        # 2. rules-based signal
        df = ohlcv_to_df(self.broker.ohlcv(sym))
        sig = generate_signal(sym, df, in_position=pos is not None)
        log.info("%s: %s (%s) price=%.6g rsi=%s", sym, sig.action, sig.reason, price, sig.metrics.get("rsi"))
        if sig.action == "hold":
            return

        if sig.action == "buy":
            ok, why = self.risk.can_open()
            if not ok:
                log.info("%s: buy blocked - %s", sym, why)
                self.journal.event(self.s.mode, sym, "blocked", "buy", why)
                return

        # 3. Claude review
        context = {
            "mode": self.s.mode,
            "equity": round(equity, 2),
            "exposure_by_symbol": {k: round(v, 2) for k, v in exposure.items()},
            "open_position": vars(pos) if pos else None,
            "trades_today": self.state.trades_today,
            "limits": {
                "max_position_pct": self.s.max_position_pct,
                "max_total_exposure_pct": self.s.max_total_exposure_pct,
                "risk_per_trade_pct": self.s.risk_per_trade_pct,
                "stop_atr_mult": self.s.stop_atr_mult,
                "take_profit_atr_mult": self.s.take_profit_atr_mult,
            },
        }
        try:
            d = self.reviewer.review(sig, recent_candles_table(df), context)
        except Exception as e:  # noqa: BLE001
            # fail toward less risk: skip buys, but still allow exits
            log.error("%s: Claude review failed (%s)", sym, e)
            self.journal.event(self.s.mode, sym, "review_error", sig.action, str(e))
            if sig.action == "sell" and pos:
                self._close(sym, pos, f"{sig.reason} (review unavailable)")
            return

        log.info("%s: Claude %s %s conf=%.2f size=%.2f - %s", sym, "APPROVES" if d.approve else "VETOES",
                 sig.action, d.confidence, d.size_fraction, d.reasoning)
        self.journal.event(self.s.mode, sym, "review", sig.action,
                           "approve" if d.approve else "veto", d.model_dump() | {"metrics": sig.metrics})
        if not d.approve or d.confidence < self.s.min_claude_confidence:
            return

        # 4. execute
        if sig.action == "sell" and pos:
            self._close(sym, pos, sig.reason)
        elif sig.action == "buy":
            _, cash, _ = self.broker.equity({k: self.broker.price(k) for k in self.s.symbols})
            qty, why = self.risk.size_buy(symbol=sym, price=price, atr=sig.atr, equity=equity, cash=cash,
                                          exposure=exposure, claude_fraction=d.size_fraction)
            log.info("%s: sizing -> %s", sym, why)
            if qty <= 0:
                return
            fill = self.broker.buy(sym, qty)
            stop, tp = self.risk.stops_for(fill.price, sig.atr)
            self.state.positions[sym] = Position(fill.qty, fill.price, stop, tp,
                                                 datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self.risk.record_trade()
            self.journal.fill(self.s.mode, fill, sig.reason)
            log.warning("%s: BOUGHT %.8g @ %.6g (stop %.6g, target %.6g)", sym, fill.qty, fill.price, stop, tp)

    def _close(self, sym: str, pos: Position, reason: str) -> None:
        fill = self.broker.sell(sym, pos.qty)
        pnl = (fill.price - pos.entry) * fill.qty - fill.fee
        del self.state.positions[sym]
        self.risk.record_trade()
        self.journal.fill(self.s.mode, fill, reason)
        log.warning("%s: SOLD %.8g @ %.6g - %s - P&L %+.2f", sym, fill.qty, fill.price, reason, pnl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude-reviewed crypto trading agent")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = Settings()
    agent = TradingAgent(s)
    if s.mode == "live":
        log.warning("*** LIVE TRADING on %s with real money: %s ***", s.exchange, ", ".join(s.symbols))
    else:
        log.info("Paper trading on %s prices: %s", s.exchange, ", ".join(s.symbols))

    while True:
        try:
            agent.cycle()
        except Exception as e:  # noqa: BLE001
            log.exception("cycle failed: %s", e)
        if args.once:
            break
        time.sleep(s.loop_seconds)


if __name__ == "__main__":
    main()
