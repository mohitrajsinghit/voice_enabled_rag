"""Dual-provider LLM client supporting Anthropic Claude and LM Studio."""
from __future__ import annotations

import json
import logging
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.app.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Error from LLM generation."""
    pass


class LLMClient:
    """Unified LLM client supporting Anthropic Claude and LM Studio (OpenAI-compatible).

    Selects provider based on the LLM_PROVIDER environment variable.
    Both providers expose the same interface: generate(system_prompt, user_prompt) -> str.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            provider: LLM provider (anthropic or lmstudio). Defaults to config.
            api_key: API key (for Anthropic). Defaults to config.
            base_url: Base URL (for LM Studio). Defaults to config.
            model: Model name. Defaults to config.
        """
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        if self.provider == LLMProvider.ANTHROPIC:
            self.api_key = self.api_key or settings.anthropic_api_key
            self.model = self.model or settings.anthropic_model
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
        elif self.provider in (LLMProvider.GEMINI, LLMProvider.GOOGLE):
            self.api_key = self.api_key or settings.google_api_key
            self.model = self.model or settings.gemini_model
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is required when using Gemini/Google provider")
        else:
            self.base_url = self.base_url or settings.lmstudio_base_url
            self.model = self.model or settings.lmstudio_model

        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key, timeout=15.0, max_retries=1)
        return self._anthropic_client

    def _get_openai_client(self):
        """Lazy-load OpenAI-compatible client for LM Studio."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=self.base_url,
                api_key="lm-studio",  # LM Studio doesn't need a real key
                timeout=10.0,
                max_retries=1,
            )
        return self._openai_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> tuple[str, float]:
        """Generate text using the configured LLM provider.

        Args:
            system_prompt: System/instruction prompt.
            user_prompt: User query/prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Tuple of (generated text, latency in ms).

        Raises:
            LLMError: If generation fails after retries.
        """
        t0 = time.perf_counter()

        try:
            if self.provider == LLMProvider.ANTHROPIC:
                text = self._generate_anthropic(system_prompt, user_prompt, max_tokens, temperature)
            elif self.provider in (LLMProvider.GEMINI, LLMProvider.GOOGLE):
                text = self._generate_gemini(system_prompt, user_prompt, max_tokens, temperature)
            else:
                text = self._generate_lmstudio(system_prompt, user_prompt, max_tokens, temperature)

            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"LLM generation ({self.provider.value}) in {latency_ms:.0f}ms, "
                f"output={len(text)} chars",
                extra={"stage": "generation", "latency_ms": latency_ms},
            )
            return text, latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"LLM generation failed ({self.provider.value}): {e}")
            raise LLMError(f"LLM generation failed: {e}") from e

    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Google Gemini REST API."""
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": f"{system_prompt}\n\nUser Query: {user_prompt}" if system_prompt else user_prompt}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        response = httpx.post(url, json=payload, timeout=25.0)
        if response.status_code != 200:
            raise LLMError(f"Gemini API error ({response.status_code}): {response.text}")

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini API response structure: {data}") from e

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Anthropic Claude API."""
        client = self._get_anthropic_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return message.content[0].text

    def _generate_lmstudio(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using LM Studio OpenAI-compatible API."""
        client = self._get_openai_client()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def parse_json_response(self, text: str) -> dict:
        """Parse a JSON response from the LLM, handling markdown code blocks.

        Args:
            text: LLM output text potentially containing JSON.

        Returns:
            Parsed dict.
        """
        # Strip markdown code blocks if present
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from the text
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}")
            return {"error": "Failed to parse response", "raw": text}
