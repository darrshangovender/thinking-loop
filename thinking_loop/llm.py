"""Provider-portable LLM client. Anthropic + OpenAI, single chat() interface.

Strategy code stays clean: `self.llm.chat(messages, system=..., max_tokens=..., temperature=...)`.
The library knows nothing about provider SDKs — they're lazy-imported on first use.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any


# Per-1M-token USD pricing. Update quarterly.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5":   (3.00, 15.00),
    "claude-sonnet-4-6":   (3.00, 15.00),
    "claude-sonnet-4-7":   (3.00, 15.00),
    "claude-haiku-4-5":    (0.80, 4.00),
    "claude-opus-4-7":     (15.00, 75.00),
    "claude-opus-4-8":     (15.00, 75.00),
    "claude-fable-5":      (8.00, 40.00),
    "gpt-4o":              (2.50, 10.00),
    "gpt-4o-mini":         (0.15, 0.60),
}


def cost_for(model: str, ti: int, to: int) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    return (ti / 1_000_000) * p[0] + (to / 1_000_000) * p[1]


@dataclass
class Response:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_ms: int
    cost_usd: float | None


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.provider = "anthropic" if model.startswith("claude-") else "openai"

    async def chat(self, messages: list[dict], *, system: str = "", max_tokens: int = 1024, temperature: float = 0.4) -> Response:
        # We use asyncio.to_thread to keep strategies parallelizable even though
        # the underlying SDKs are sync. Cheaper than maintaining two code paths.
        return await asyncio.to_thread(self._chat_sync, messages, system, max_tokens, temperature)

    def _chat_sync(self, messages: list[dict], system: str, max_tokens: int, temperature: float) -> Response:
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}
            if system:
                kwargs["system"] = system
            resp = client.messages.create(**kwargs)
            ti, to = resp.usage.input_tokens, resp.usage.output_tokens
            content = resp.content[0].text
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            full = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(model=self.model, messages=full, max_tokens=max_tokens, temperature=temperature)
            ti, to = resp.usage.prompt_tokens, resp.usage.completion_tokens
            content = resp.choices[0].message.content or ""
        return Response(
            content=content, model=self.model,
            tokens_in=ti, tokens_out=to,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=cost_for(self.model, ti, to),
        )