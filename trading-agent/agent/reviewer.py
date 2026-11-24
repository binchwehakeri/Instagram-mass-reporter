"""Claude reviews each rules-based signal and approves, vetoes or shrinks it."""

from __future__ import annotations

import json
import logging

import anthropic
from pydantic import BaseModel, Field

from .config import Settings
from .strategy import Signal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the risk-aware reviewer inside an automated, long-only crypto spot \
trading system. A rules-based strategy (EMA 12/26 crossover with a 200 EMA \
trend filter and RSI filter) proposes trades; you decide whether each one \
should go ahead.

What you control:
- approve: true to let the trade proceed, false to veto it.
- size_fraction: 0.0-1.0, a multiplier on the maximum size the risk manager \
allows. You can shrink a trade but never enlarge it. Use 1.0 only for a clean, \
high-quality setup.
- confidence: 0.0-1.0, how confident you are that your decision is right. \
Trades below the operator's confidence threshold are skipped.

What you do not control: position caps, stop-losses, take-profits, the daily \
loss limit and the kill switch are enforced in code after your review.

How to judge a BUY: look for confirmation in the recent candles (follow-through \
rather than a single spike), a reasonable distance from the 200 EMA, \
volatility (ATR%) that is not extreme, and volume that supports the move. \
Crossovers in choppy, sideways markets are frequent false signals; veto or \
shrink those. Preserving capital matters more than catching every move.

How to judge a SELL: these exit an existing position. Approve unless the data \
clearly shows the crossover is noise inside a strong ongoing uptrend.

Base your judgment only on the data provided. Keep reasoning to 2-4 sentences."""


class Decision(BaseModel):
    approve: bool
    size_fraction: float = Field(description="0.0 to 1.0 multiplier on the max allowed size")
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str


class ClaudeReviewer:
    def __init__(self, s: Settings):
        self.s = s
        self.client = anthropic.Anthropic()

    def review(self, signal: Signal, candles_csv: str, context: dict) -> Decision:
        prompt = (
            f"Proposed action: {signal.action.upper()} {signal.symbol}\n"
            f"Strategy reason: {signal.reason}\n\n"
            f"Indicators (last closed candle, timeframe {self.s.timeframe}):\n"
            f"{json.dumps(signal.metrics, indent=2)}\n\n"
            f"Portfolio and limits:\n{json.dumps(context, indent=2)}\n\n"
            f"Recent candles:\n{candles_csv}"
        )
        response = self.client.beta.messages.parse(
            model=self.s.claude_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=Decision,
            output_config={"effort": self.s.claude_effort},
            # If the primary model declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined to review: {response.stop_details}")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise RuntimeError(f"no usable decision (stop_reason={response.stop_reason})")
        d = response.parsed_output
        d.size_fraction = max(0.0, min(1.0, d.size_fraction))
        d.confidence = max(0.0, min(1.0, d.confidence))
        return d
