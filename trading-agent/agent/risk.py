"""Hard risk limits. Nothing here can be overridden by Claude.

Claude can only *shrink* a trade (size_fraction in 0..1) or veto it.
Exits that reduce risk (stop-loss, take-profit, sell signals) are always
allowed, even when new trading is halted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings


@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    take_profit: float
    opened_at: str


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    trades_today: int = 0
    halted_reason: str = ""
    positions: dict[str, Position] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RiskState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        raw["positions"] = {k: Position(**v) for k, v in raw.get("positions", {}).items()}
        return cls(**raw)

    def save(self, path: Path) -> None:
        data = asdict(self)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)


class RiskManager:
    def __init__(self, settings: Settings, state: RiskState, use_kill_switch: bool = True):
        self.s = settings
        self.state = state
        self.use_kill_switch = use_kill_switch

    # ---- daily bookkeeping -------------------------------------------------
    def start_of_cycle(self, equity: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if self.state.day != today:
            self.state.day = today
            self.state.day_start_equity = equity
            self.state.trades_today = 0
            self.state.halted_reason = ""
        if self.state.day_start_equity > 0:
            pnl = equity / self.state.day_start_equity - 1
            if pnl <= -self.s.max_daily_loss_pct and not self.state.halted_reason:
                self.state.halted_reason = (
                    f"daily loss {pnl:.2%} hit limit -{self.s.max_daily_loss_pct:.2%}"
                )

    def record_trade(self) -> None:
        self.state.trades_today += 1

    # ---- gates -------------------------------------------------------------
    def can_open(self) -> tuple[bool, str]:
        if self.use_kill_switch and self.s.kill_switch.exists():
            return False, f"kill switch file present ({self.s.kill_switch})"
        if self.state.halted_reason:
            return False, f"halted: {self.state.halted_reason}"
        if self.state.trades_today >= self.s.max_trades_per_day:
            return False, f"max trades per day reached ({self.s.max_trades_per_day})"
        return True, "ok"

    # ---- sizing ------------------------------------------------------------
    def size_buy(
        self,
        *,
        symbol: str,
        price: float,
        atr: float,
        equity: float,
        cash: float,
        exposure: dict[str, float],
        claude_fraction: float,
    ) -> tuple[float, str]:
        """Return (base-currency quantity, explanation). Quantity 0 means don't trade."""
        if price <= 0 or atr <= 0 or equity <= 0:
            return 0.0, "invalid price/atr/equity"
        claude_fraction = max(0.0, min(1.0, claude_fraction))

        stop_dist = atr * self.s.stop_atr_mult
        by_risk = equity * self.s.risk_per_trade_pct / stop_dist * price  # in quote currency
        by_symbol = equity * self.s.max_position_pct - exposure.get(symbol, 0.0)
        by_total = equity * self.s.max_total_exposure_pct - sum(exposure.values())
        by_cash = cash * 0.98  # leave room for fees/slippage

        caps = {"risk": by_risk, "symbol cap": by_symbol, "total cap": by_total, "cash": by_cash}
        binding = min(caps, key=caps.get)
        notional = max(0.0, caps[binding]) * claude_fraction

        if notional < self.s.min_order_usd:
            return 0.0, f"order ${notional:.2f} below minimum (limited by {binding})"
        return notional / price, f"${notional:.2f} (limited by {binding}, Claude x{claude_fraction:.2f})"

    def stops_for(self, price: float, atr: float) -> tuple[float, float]:
        return price - atr * self.s.stop_atr_mult, price + atr * self.s.take_profit_atr_mult

    @staticmethod
    def exit_hit(pos: Position, price: float) -> str | None:
        if price <= pos.stop:
            return f"stop-loss hit ({price:.6g} <= {pos.stop:.6g})"
        if price >= pos.take_profit:
            return f"take-profit hit ({price:.6g} >= {pos.take_profit:.6g})"
        return None
