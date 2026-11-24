"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

LIVE_CONFIRM_PHRASE = "I_ACCEPT_THE_RISK"


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    # mode
    mode: str = field(default_factory=lambda: os.getenv("MODE", "paper").lower())
    live_confirm: str = field(default_factory=lambda: os.getenv("LIVE_TRADING_CONFIRM", ""))

    # exchange
    exchange: str = field(default_factory=lambda: os.getenv("EXCHANGE", "binance"))
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    api_password: str = field(default_factory=lambda: os.getenv("EXCHANGE_PASSWORD", ""))
    sandbox: bool = field(default_factory=lambda: _b("EXCHANGE_SANDBOX", False))

    # universe
    symbols: list[str] = field(
        default_factory=lambda: [
            s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()
        ]
    )
    timeframe: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "1h"))
    loop_seconds: int = field(default_factory=lambda: _i("LOOP_SECONDS", 300))
    paper_starting_cash: float = field(default_factory=lambda: _f("PAPER_STARTING_CASH", 10_000))

    # hard risk limits
    max_position_pct: float = field(default_factory=lambda: _f("MAX_POSITION_PCT", 0.10))
    max_total_exposure_pct: float = field(default_factory=lambda: _f("MAX_TOTAL_EXPOSURE_PCT", 0.30))
    risk_per_trade_pct: float = field(default_factory=lambda: _f("RISK_PER_TRADE_PCT", 0.01))
    max_daily_loss_pct: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS_PCT", 0.03))
    max_trades_per_day: int = field(default_factory=lambda: _i("MAX_TRADES_PER_DAY", 6))
    stop_atr_mult: float = field(default_factory=lambda: _f("STOP_ATR_MULT", 2.0))
    take_profit_atr_mult: float = field(default_factory=lambda: _f("TAKE_PROFIT_ATR_MULT", 4.0))
    min_order_usd: float = field(default_factory=lambda: _f("MIN_ORDER_USD", 10))

    # Claude
    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: os.getenv("CLAUDE_EFFORT", "medium"))
    min_claude_confidence: float = field(default_factory=lambda: _f("MIN_CLAUDE_CONFIDENCE", 0.6))

    # files
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

    @property
    def kill_switch(self) -> Path:
        """Create this file (e.g. `touch data/STOP`) to halt all new trading."""
        return self.data_dir / "STOP"

    @property
    def quote_currency(self) -> str:
        return self.symbols[0].split("/")[1]

    def validate(self) -> None:
        if self.mode not in {"paper", "live"}:
            raise ValueError(f"MODE must be 'paper' or 'live', got {self.mode!r}")
        quotes = {s.split("/")[1] for s in self.symbols}
        if len(quotes) != 1:
            raise ValueError(f"All SYMBOLS must share one quote currency, got {quotes}")
        for name in ("max_position_pct", "max_total_exposure_pct", "risk_per_trade_pct", "max_daily_loss_pct"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {v}")
        if self.mode == "live":
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise SystemExit(
                    "Live mode refused: set LIVE_TRADING_CONFIRM="
                    f"{LIVE_CONFIRM_PHRASE} to trade real money."
                )
            if not (self.api_key and self.api_secret):
                raise SystemExit("Live mode needs EXCHANGE_API_KEY and EXCHANGE_API_SECRET.")
        self.data_dir.mkdir(parents=True, exist_ok=True)
