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
import logging
import re
from typing import Any, Dict, List, Optional

import litellm

from ..config import Config
from .openai_chat_compat import create_chat_completion, extract_chat_completion_text


logger = logging.getLogger(__name__)


class LLMResponseError(ValueError):
    """A safe, structured error for unusable model responses."""

    def __init__(self, message: str, *, finish_reason: Optional[str] = None):
        super().__init__(message)
        self.finish_reason = finish_reason


def _is_response_format_unsupported(error: Exception) -> bool:
    """Detect an explicit provider rejection of JSON response_format."""

    if getattr(error, "status_code", None) not in {400, 422}:
        return False

    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False

    details = body.get("error", body)
    if not isinstance(details, dict):
        return False

    param = str(details.get("param") or "").strip().lower()
    if param == "response_format" or param.startswith("response_format."):
        return True

    message = str(details.get("message") or "").lower()
    if "response_format" not in message:
        return False

    code = str(details.get("code") or "").lower()
    unsupported_codes = {
        "unsupported_parameter",
        "unsupported_value",
        "unknown_parameter",
        "invalid_parameter",
    }
    unsupported_phrases = (
        "not support",
        "unsupported",
        "unknown parameter",
        "unrecognized parameter",
    )
    return code in unsupported_codes or any(
        phrase in message for phrase in unsupported_phrases
    )


def _clean_chat_text(content: str) -> str:
    """Remove common reasoning wrappers and an outer Markdown JSON fence."""

    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    cleaned = cleaned.lstrip("\ufeff")
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


def _contains_additional_json_container(content: str) -> bool:
    """Return True when trailing text embeds another JSON object or array."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return True
    return False


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

        # Upstream calls the LLM through `create_chat_completion(self.client, ...)`,
        # which expects an OpenAI-SDK-shaped client. The shim gives it that shape
        # while routing every request through LiteLLM.
        self.client = LiteLLMOpenAIShim(api_key=self.api_key, api_base=self.api_base)
    def _create_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
    ) -> Any:
        """Send one raw Chat Completions request through the compatibility layer."""

        return create_chat_completion(
            self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        response = self._create_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        content = extract_chat_completion_text(response)
        return _clean_chat_text(content)
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        max_attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            max_attempts: 内容生成尝试次数（不含一次明确的JSON模式能力降级）
            
        Returns:
            解析后的JSON对象
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        response_format: Optional[Dict[str, str]] = {"type": "json_object"}
        request_max_tokens = max_tokens
        last_error: Optional[LLMResponseError] = None

        for attempt in range(1, max_attempts + 1):
            # JSON-mode capability negotiation is separate from content
            # regeneration. An explicit response_format rejection may add one
            # request, but it must not consume a content attempt.
            while True:
                try:
                    response = self._create_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=request_max_tokens,
                        response_format=response_format,
                    )
                except Exception as error:
                    if (
                        response_format is not None
                        and _is_response_format_unsupported(error)
                    ):
                        logger.warning(
                            "LLM provider explicitly rejected response_format; "
                            "retrying once with prompt-only JSON guidance"
                        )
                        response_format = None
                        continue
                    raise
                break

            try:
                return self._parse_json_response(response)
            except LLMResponseError as error:
                last_error = error
                if attempt >= max_attempts:
                    raise

                # A caller-supplied cap is the common cause of a partial JSON
                # object. Omit it for the one bounded retry so the provider can
                # use its model-specific output limit.
                had_token_cap = request_max_tokens is not None
                request_max_tokens = None
                logger.warning(
                    "LLM returned unusable JSON (finish_reason=%s); "
                    "retrying content generation%s",
                    error.finish_reason or "unknown",
                    " without an output token cap" if had_token_cap else "",
                )

        if last_error is not None:  # pragma: no cover - defensive loop guard
            raise last_error
        raise LLMResponseError("LLM did not produce a JSON response")

    @staticmethod
    def _parse_json_response(response: Any) -> Dict[str, Any]:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("LLM returned no choices")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMResponseError(
                "LLM JSON output was truncated at the token limit",
                finish_reason=finish_reason,
            )
        if finish_reason not in {None, "stop"}:
            raise LLMResponseError(
                f"LLM JSON generation stopped unexpectedly ({finish_reason})",
                finish_reason=finish_reason,
            )

        content = _clean_chat_text(extract_chat_completion_text(response))
        if not content:
            raise LLMResponseError(
                "LLM returned empty JSON content",
                finish_reason=finish_reason,
            )

        try:
            value = json.loads(content)
        except json.JSONDecodeError as strict_error:
            # Some compatible providers append a short explanation after an
            # otherwise complete JSON object. Accept only an object decoded
            # from the beginning; never repair or invent truncated JSON.
            try:
                value, end = json.JSONDecoder().raw_decode(content)
            except json.JSONDecodeError:
                raise LLMResponseError(
                    "LLM returned invalid JSON "
                    f"(line {strict_error.lineno}, column {strict_error.colno})",
                    finish_reason=finish_reason,
                ) from strict_error

            trailing = content[end:].strip()
            if trailing:
                if _contains_additional_json_container(trailing):
                    raise LLMResponseError(
                        "LLM returned multiple JSON values",
                        finish_reason=finish_reason,
                    )
                logger.warning("Ignoring text after a complete LLM JSON object")

        if not isinstance(value, dict):
            raise LLMResponseError(
                "LLM JSON response must be a top-level JSON object",
                finish_reason=finish_reason,
            )

        return value


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
            # default LLMClient resolution (LITELLM_MODEL -> legacy openai/*).
            model = Config.LITELLM_REPORT_MODEL or None
        super().__init__(api_key=api_key, base_url=base_url, model=model)
