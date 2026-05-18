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


class LiteLLMOpenAIShim:
    """OpenAI-shaped facade over litellm.completion.

    Lets call sites that were written against the OpenAI SDK
    (`self.client.chat.completions.create(model=..., messages=..., ...)`)
    route through LiteLLM without code changes — LiteLLM returns
    OpenAI-compatible response objects, so `.choices[0].message.content`
    and `.finish_reason` continue to work.

    Used by oasis_profile_generator and simulation_config_generator, which
    keep intricate retry/JSON-repair logic around `.create(...)` calls
    that we don't want to disturb.
    """

    class _Completions:
        def __init__(self, api_key: Optional[str], api_base: Optional[str]):
            self._api_key = api_key
            self._api_base = api_base

        def create(self, **kwargs):
            if self._api_key and "api_key" not in kwargs:
                kwargs["api_key"] = self._api_key
            if self._api_base and "api_base" not in kwargs:
                kwargs["api_base"] = self._api_base
            return litellm.completion(**kwargs)

    class _Chat:
        def __init__(self, api_key: Optional[str], api_base: Optional[str]):
            self.completions = LiteLLMOpenAIShim._Completions(api_key, api_base)

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.chat = LiteLLMOpenAIShim._Chat(api_key, api_base)


def resolve_litellm_model(explicit: Optional[str]) -> Optional[str]:
    """Pick the model string to hand to LiteLLM.

    Priority: provider-prefixed explicit arg > LITELLM_MODEL env >
    legacy LLM_MODEL_NAME wrapped as `openai/<name>` > a bare explicit
    arg (assumed to be an OpenAI-style id). Returns None if nothing is
    configured — callers decide whether to raise.
    """
    if explicit and "/" in explicit:
        return explicit
    if Config.LITELLM_MODEL:
        return Config.LITELLM_MODEL
    if explicit:
        return f"openai/{explicit}"
    if Config.LLM_MODEL_NAME:
        return f"openai/{Config.LLM_MODEL_NAME}"
    return None


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
            # Heuristic: a well-formed JSON object/array ends with } or ]
            # after fence-stripping. If it doesn't, the model almost
            # certainly hit max_tokens and was truncated mid-string —
            # surface that explicitly instead of dumping a 5KB blob into
            # the error log.
            tail = cleaned[-1] if cleaned else ""
            if tail not in ("}", "]"):
                raise ValueError(
                    f"LLM JSON appears truncated (last char={tail!r}, "
                    f"length={len(cleaned)} chars). Likely hit max_tokens — "
                    f"increase the limit on this call. "
                    f"First 300 chars: {cleaned[:300]}..."
                ) from exc
            raise ValueError(
                f"LLM returned invalid JSON (length={len(cleaned)}). "
                f"First 500 chars: {cleaned[:500]}"
            ) from exc


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
