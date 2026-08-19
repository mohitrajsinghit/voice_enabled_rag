"""Unified LLM client supporting Groq, Google Gemini, Anthropic Claude, and LM Studio."""
from __future__ import annotations

import json
import logging
import os
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.app.config import LLMProvider, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Error from LLM generation."""
    pass


class LLMClient:
    """Unified LLM client supporting Groq, Anthropic Claude, Gemini, and LM Studio (OpenAI-compatible).

    Selects provider based on the LLM_PROVIDER environment variable.
    All providers expose the same interface: generate(system_prompt, user_prompt) -> tuple[str, float].
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
            provider: LLM provider (groq, gemini, anthropic, or lmstudio). Defaults to config.
            api_key: API key. Defaults to config.
            base_url: Base URL (for LM Studio). Defaults to config.
            model: Model name. Defaults to config.
        """
        settings = get_settings()
        prov = provider or settings.llm_provider
        if isinstance(prov, str):
            prov = LLMProvider(prov.lower())
        self.provider = prov
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        if self.provider == LLMProvider.GROQ:
            self.api_key = self.api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
            self.model = self.model or settings.groq_model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            if not self.api_key:
                logger.warning("GROQ_API_KEY is not configured yet. Will check environment at query time.")
        elif self.provider == LLMProvider.ANTHROPIC:
            self.api_key = self.api_key or settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self.model = self.model or settings.anthropic_model
            if not self.api_key:
                logger.warning("ANTHROPIC_API_KEY is not configured yet. Will check environment at query time.")
        elif self.provider in (LLMProvider.GEMINI, LLMProvider.GOOGLE):
            self.api_key = self.api_key or settings.google_api_key or os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
            self.model = self.model or settings.gemini_model
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY is not configured yet. Will check environment at query time.")
        else:
            self.base_url = self.base_url or os.getenv("LMSTUDIO_BASE_URL", "") or settings.lmstudio_base_url
            self.model = self.model or os.getenv("LMSTUDIO_MODEL", "") or settings.lmstudio_model
            logger.info(f"Initialized local/remote OpenAI-compatible LLM client: base_url={self.base_url}, model={self.model}")

        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is required for Claude generation. Please set ANTHROPIC_API_KEY in .env.")
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=15.0, max_retries=1)
        return self._anthropic_client

    def _get_openai_client(self):
        """Lazy-load OpenAI-compatible client for LM Studio."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=self.base_url,
                api_key="lm-studio",  # LM Studio doesn't need a real key
                timeout=120.0,
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
            if self.provider == LLMProvider.GROQ:
                text = self._generate_groq(system_prompt, user_prompt, max_tokens, temperature)
            elif self.provider == LLMProvider.ANTHROPIC:
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

    def _generate_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Groq REST API (OpenAI-compatible)."""
        import httpx
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY", "") or self.api_key or get_settings().groq_api_key
        if not api_key or api_key in ("your_groq_api_key_here", ""):
            raise LLMError("GROQ_API_KEY is not configured in .env. Please set your Groq API key in .env.")
        model = os.getenv("GROQ_MODEL", "") or self.model or "openai/gpt-oss-120b"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = None
        for attempt in range(3):
            response = httpx.post(url, json=payload, headers=headers, timeout=25.0)
            if response.status_code == 429 and attempt < 2:
                time.sleep(1.2)
                continue
            break

        if response.status_code != 200:
            raise LLMError(f"Groq API error ({response.status_code}): {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Groq API response structure: {data}") from e

    def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate using Google Gemini REST API."""
        import httpx
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or self.api_key or get_settings().google_api_key
        if not api_key or api_key in ("your_google_gemini_api_key_here", "your_gemini_api_key_here"):
            raise LLMError("GOOGLE_API_KEY is not configured in .env. Please set your Google Gemini API key in .env.")
        model = os.getenv("GEMINI_MODEL", "") or self.model or "gemini-3.6-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
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
