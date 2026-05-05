"""
pipelines/llm_client.py
------------------------
Vendor-agnostic LLM client for OpenAI, Anthropic, Groq, and Google Gemini.

Every call returns a LLMResponse that includes:
  - answer text
  - prompt_tokens, completion_tokens, total_tokens   ← headline hackathon metric
  - cost_usd  (calculated from published pricing)
  - latency_ms

Set provider via env var:
    LLM_PROVIDER=openai | anthropic | groq | gemini
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import tiktoken

# ── Pricing table (USD per 1K tokens, as of mid-2025) ────────────────────────
# Update these if your provider changes pricing.
PRICING: dict[str, dict[str, float]] = {
    # model_id: {input: $/1K, output: $/1K}
    "gpt-4o-mini":                  {"input": 0.00015,  "output": 0.00060},
    "gpt-4o":                       {"input": 0.00250,  "output": 0.01000},
    "gpt-4.1-mini":                 {"input": 0.00040,  "output": 0.00160},
    "claude-haiku-4-5":             {"input": 0.00080,  "output": 0.00400},
    "claude-sonnet-4-20250514":     {"input": 0.00300,  "output": 0.01500},
    "llama-3.3-70b-versatile":      {"input": 0.00059,  "output": 0.00079},
    "llama-3.1-8b-instant":         {"input": 0.00005,  "output": 0.00008},
    "gemini-2.0-flash":             {"input": 0.00010,  "output": 0.00040},
    "gemini-1.5-flash":             {"input": 0.00007,  "output": 0.00030},
}

def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.001, "output": 0.002})
    return round((prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1000, 6)

def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate tokens via tiktoken (works well enough cross-provider)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    answer:            str
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    cost_usd:          float
    latency_ms:        int
    model:             str
    provider:          str
    extra:             dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer":            self.answer,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "cost_usd":          self.cost_usd,
            "latency_ms":        self.latency_ms,
            "model":             self.model,
            "provider":          self.provider,
        }


# ── Base interface ────────────────────────────────────────────────────────────

class LLMClient(ABC):
    def __init__(self, model: str, provider: str):
        self.model    = model
        self.provider = provider

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        m = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        super().__init__(m, "openai")
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def chat(self, messages, *, system=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model, messages=full,
            temperature=temperature, max_tokens=max_tokens,
        )
        ms = round((time.time() - t0) * 1000)

        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        return LLMResponse(
            answer=resp.choices[0].message.content.strip(),
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            cost_usd=_cost(self.model, pt, ct),
            latency_ms=ms, model=self.model, provider=self.provider,
        )


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic
        m = model or os.getenv("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        super().__init__(m, "anthropic")
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def chat(self, messages, *, system=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        t0 = time.time()
        resp = self._client.messages.create(**kwargs)
        ms = round((time.time() - t0) * 1000)

        pt = resp.usage.input_tokens
        ct = resp.usage.output_tokens
        return LLMResponse(
            answer=resp.content[0].text.strip(),
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            cost_usd=_cost(self.model, pt, ct),
            latency_ms=ms, model=self.model, provider=self.provider,
        )


# ── Groq ──────────────────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from groq import Groq
        m = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        super().__init__(m, "groq")
        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def chat(self, messages, *, system=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model, messages=full,
            temperature=temperature, max_tokens=max_tokens,
        )
        ms = round((time.time() - t0) * 1000)

        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        return LLMResponse(
            answer=resp.choices[0].message.content.strip(),
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            cost_usd=_cost(self.model, pt, ct),
            latency_ms=ms, model=self.model, provider=self.provider,
        )


# ── Google Gemini ─────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import google.generativeai as genai
        m = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        super().__init__(m, "gemini")
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self._genai = genai

    def chat(self, messages, *, system=None, temperature=0.2, max_tokens=1024) -> LLMResponse:
        model = self._genai.GenerativeModel(
            self.model,
            system_instruction=system or "",
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        # Flatten messages to a single prompt for simplicity
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

        t0 = time.time()
        resp = model.generate_content(prompt)
        ms = round((time.time() - t0) * 1000)

        text = resp.text.strip()
        # Gemini doesn't always return usage; estimate via tiktoken
        pt = _count_tokens(prompt)
        ct = _count_tokens(text)
        return LLMResponse(
            answer=text,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
            cost_usd=_cost(self.model, pt, ct),
            latency_ms=ms, model=self.model, provider=self.provider,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[LLMClient]] = {
    "openai":    OpenAIClient,
    "anthropic": AnthropicClient,
    "claude":    AnthropicClient,
    "groq":      GroqClient,
    "gemini":    GeminiClient,
    "google":    GeminiClient,
}

def get_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """
    Return an LLMClient for the given provider.

    Provider resolved from: argument → LLM_PROVIDER env var → "openai"
    """
    resolved = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    cls = _PROVIDERS.get(resolved)
    if cls is None:
        raise ValueError(f"Unknown provider '{resolved}'. Choose from: {list(_PROVIDERS)}")
    return cls(model=model, api_key=api_key)


# Singleton used by pipelines — reads LLM_PROVIDER at import time
default_client: LLMClient = get_client()