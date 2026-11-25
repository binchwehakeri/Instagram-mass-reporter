"""Exchange access through ccxt: a paper broker (simulated fills) and a live broker."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import ccxt

from .config import Settings

log = logging.getLogger(__name__)

PAPER_FEE = 0.001       # 0.1% per fill, typical taker fee
PAPER_SLIPPAGE = 0.0005  # 0.05%


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float  # in quote currency
    order_id: str = ""


def make_exchange(s: Settings, with_keys: bool) -> ccxt.Exchange:
    cls = getattr(ccxt, s.exchange, None)
    if cls is None:
        raise SystemExit(f"Unknown ccxt exchange id: {s.exchange}")
    params: dict = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if with_keys:
        params.update(apiKey=s.api_key, secret=s.api_secret)
        if s.api_password:
            params["password"] = s.api_password
    ex = cls(params)
    if s.sandbox:
        ex.set_sandbox_mode(True)
    return ex


class Broker:
    """Shared market-data methods."""

    def __init__(self, s: Settings, exchange: ccxt.Exchange):
        self.s = s
        self.ex = exchange
        self.ex.load_markets()
        for sym in s.symbols:
            if sym not in self.ex.markets:
                raise SystemExit(f"{sym} is not a market on {s.exchange}")

    def ohlcv(self, symbol: str, limit: int = 300) -> list[list[float]]:
        return self.ex.fetch_ohlcv(symbol, timeframe=self.s.timeframe, limit=limit)

    def price(self, symbol: str) -> float:
        t = self.ex.fetch_ticker(symbol)
        return float(t.get("last") or t["close"])

    def balances(self) -> tuple[float, dict[str, float]]:
        raise NotImplementedError

    def equity(self, prices: dict[str, float]) -> tuple[float, float, dict[str, float]]:
        """Return (equity, cash, exposure per symbol) in quote currency."""
        cash, held = self.balances()
        exposure = {sym: held.get(sym.split("/")[0], 0.0) * prices[sym] for sym in self.s.symbols}
        return cash + sum(exposure.values()), cash, exposure

    def buy(self, symbol: str, qty: float) -> Fill:
        raise NotImplementedError

    def sell(self, symbol: str, qty: float) -> Fill:
        raise NotImplementedError


class PaperBroker(Broker):
    """Real prices from the exchange, simulated fills and balances."""

    def __init__(self, s: Settings):
        super().__init__(s, make_exchange(s, with_keys=False))
        self.path: Path = s.data_dir / "paper_account.json"
        if self.path.exists():
            self.acct = json.loads(self.path.read_text())
        else:
            self.acct = {"cash": s.paper_starting_cash, "held": {}}
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.acct, indent=2))

    def balances(self) -> tuple[float, dict[str, float]]:
        return self.acct["cash"], dict(self.acct["held"])

    def buy(self, symbol: str, qty: float) -> Fill:
        px = self.price(symbol) * (1 + PAPER_SLIPPAGE)
        cost = qty * px
        fee = cost * PAPER_FEE
        if cost + fee > self.acct["cash"]:
            raise RuntimeError("paper account: insufficient cash")
        base = symbol.split("/")[0]
        self.acct["cash"] -= cost + fee
        self.acct["held"][base] = self.acct["held"].get(base, 0.0) + qty
        self._save()
        return Fill(symbol, "buy", qty, px, fee, "paper")

    def sell(self, symbol: str, qty: float) -> Fill:
        base = symbol.split("/")[0]
        qty = min(qty, self.acct["held"].get(base, 0.0))
        px = self.price(symbol) * (1 - PAPER_SLIPPAGE)
        proceeds = qty * px
        fee = proceeds * PAPER_FEE
        self.acct["held"][base] = self.acct["held"].get(base, 0.0) - qty
        self.acct["cash"] += proceeds - fee
        self._save()
        return Fill(symbol, "sell", qty, px, fee, "paper")


class LiveBroker(Broker):
    """Real market orders on a spot account. No leverage, no withdrawals."""

    def __init__(self, s: Settings):
        super().__init__(s, make_exchange(s, with_keys=True))

    def balances(self) -> tuple[float, dict[str, float]]:
        bal = self.ex.fetch_balance()
        free = bal.get("free", {}) or {}
        cash = float(free.get(self.s.quote_currency) or 0.0)
        held = {}
        for sym in self.s.symbols:
            base = sym.split("/")[0]
            held[base] = float(free.get(base) or 0.0)
        return cash, held

    def _order(self, symbol: str, side: str, qty: float) -> Fill:
        amount = float(self.ex.amount_to_precision(symbol, qty))
        min_amt = (self.ex.markets[symbol].get("limits", {}).get("amount", {}) or {}).get("min") or 0
        if amount <= 0 or amount < min_amt:
            raise RuntimeError(f"amount {amount} below exchange minimum {min_amt}")
        log.warning("LIVE ORDER: %s %s %s", side.upper(), amount, symbol)
        order = self.ex.create_order(symbol, "market", side, amount)
        # some exchanges return a sparse order; fetch it for fill details
        if not order.get("average") and order.get("id"):
            try:
                order = self.ex.fetch_order(order["id"], symbol)
            except Exception:  # noqa: BLE001 - details are best-effort
                pass
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or order.get("price") or self.price(symbol))
        fee = 0.0
        if order.get("fee") and order["fee"].get("cost") is not None:
            fee = float(order["fee"]["cost"])
        return Fill(symbol, side, filled, avg, fee, str(order.get("id", "")))

    def buy(self, symbol: str, qty: float) -> Fill:
        return self._order(symbol, "buy", qty)

    def sell(self, symbol: str, qty: float) -> Fill:
        _, held = self.balances()
        return self._order(symbol, "sell", min(qty, held.get(symbol.split("/")[0], 0.0)))


def make_broker(s: Settings) -> Broker:
    return LiveBroker(s) if s.mode == "live" else PaperBroker(s)
