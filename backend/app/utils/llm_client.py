"""
LLM 客户端封装 — LiteLLM-backed.

Uses LiteLLM as a unified abstraction over 100+ providers (Gemini, Anthropic,
DeepSeek, Mistral, Groq, OpenAI, Ollama, …). The provider is selected by
prefix in the model string, e.g.:

    gemini/gemini-2.5-flash
    anthropic/claude-haiku-4-5
    deepseek/deepseek-chat
    mistral/mistral-small-latest
    openai/gpt-5.4

If LITELLM_MODEL is set in the environment, that wins. Otherwise we fall
back to the legacy OpenAI-compatible config (LLM_API_KEY / LLM_BASE_URL /
LLM_MODEL_NAME) routed through LiteLLM with an explicit api_base.

Provider API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
MISTRAL_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, …) are read by LiteLLM
directly from os.environ — no per-provider wiring required here.
"""

import json
import re
from typing import Any, Dict, List, Optional

import litellm

from ..config import Config


class LLMClient:
    """LLM client backed by LiteLLM.

    Public surface (chat, chat_json) is unchanged from the previous
    OpenAI-only implementation, so existing callers keep working.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Resolve model in priority order:
        #   1. explicit ctor arg
        #   2. LITELLM_MODEL env (via Config)
        #   3. legacy LLM_MODEL_NAME env (mapped to openai/<name>)
        if model:
            self.model = model
        elif Config.LITELLM_MODEL:
            self.model = Config.LITELLM_MODEL
        else:
            # Legacy mode — LLM_BASE_URL points at an OpenAI-compatible endpoint
            # (real OpenAI, DeepSeek, Mistral, Groq, OpenRouter, Ollama, …).
            # Wrap the model name with an explicit "openai/" prefix so LiteLLM
            # routes it through its OpenAI-compatible adapter.
            self.model = f"openai/{Config.LLM_MODEL_NAME}"

        # Per-call api_key / api_base overrides. When using a provider-prefixed
        # LITELLM_MODEL, leave these None and let LiteLLM read provider keys
        # from os.environ. When using the legacy openai/* path, pass the
        # explicit api_base/api_key from Config.
        self.api_key: Optional[str] = api_key or (
            None if Config.LITELLM_MODEL else (Config.LLM_API_KEY or None)
        )
        self.api_base: Optional[str] = base_url or (
            None if Config.LITELLM_MODEL else Config.LLM_BASE_URL
        )

    def _completion_kwargs(self, **overrides) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        kwargs.update(overrides)
        return kwargs

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a chat-completion request and return the text content."""
        kwargs = self._completion_kwargs(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        # Strip <think>…</think> blocks emitted by some reasoning models.
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat-completion request and parse the JSON reply."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Strip markdown code fences some models still emit even in JSON mode.
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {cleaned}") from exc


class ReportLLMClient(LLMClient):
    """LLM client preferring LITELLM_REPORT_MODEL for ReportAgent.

    The report agent benefits from a stronger model (long context, better
    synthesis), so we let operators configure it separately. Falls back to
    the same model the agent loop uses if LITELLM_REPORT_MODEL is unset.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if model is None:
            # Prefer the report-specific override; otherwise inherit the
            # default LLMClient resolution (LITELLM_MODEL → legacy openai/*).
            model = Config.LITELLM_REPORT_MODEL or None
        super().__init__(api_key=api_key, base_url=base_url, model=model)
