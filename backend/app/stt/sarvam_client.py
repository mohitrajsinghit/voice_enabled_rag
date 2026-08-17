"""Sarvam AI STT API client with retry support."""
from __future__ import annotations

import io
import logging
import os
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.app.schemas import TranscriptResult

logger = logging.getLogger(__name__)


class SarvamSTTError(Exception):
    """Error from Sarvam STT API."""
    pass


class SarvamClient:
    """Async wrapper around Sarvam AI Speech-to-Text REST API.

    Uses the Sarvam STT endpoint with the saarika model for Indic language
    speech recognition. Includes retry logic with exponential backoff.
    """

    BASE_URL = "https://api.sarvam.ai"
    STT_ENDPOINT = "/speech-to-text"

    def __init__(self, api_key: str = ""):
        """Initialize the Sarvam client.

        Args:
            api_key: Sarvam AI API subscription key (optional at init).
        """
        self.api_key = api_key
        if not self.api_key:
            logger.info("SARVAM_API_KEY not configured at startup. Audio transcription will check environment on demand.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        reraise=True,
    )
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "unknown",
        model: str = "saarika:v2.5",
    ) -> TranscriptResult:
        """Transcribe audio bytes using Sarvam STT API.

        Args:
            audio_bytes: Raw audio data (WebM/WAV/MP3/OGG/FLAC).
            language_code: Language code for transcription (default: "unknown" for auto-detect, or "hi-IN", "en-IN", etc.).
            model: Sarvam STT model name (default: "saarika:v2.5").

        Returns:
            TranscriptResult with transcribed text and metadata.

        Raises:
            SarvamSTTError: If transcription fails after retries.
        """
        t0 = time.perf_counter()

        # Detect audio container format from magic bytes
        filename = "recording.webm"
        content_type = "audio/webm"
        if audio_bytes.startswith(b"RIFF"):
            filename = "recording.wav"
            content_type = "audio/wav"
        elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            filename = "recording.mp3"
            content_type = "audio/mp3"
        elif audio_bytes.startswith(b"OggS"):
            filename = "recording.ogg"
            content_type = "audio/ogg"
        elif audio_bytes.startswith(b"fLaC"):
            filename = "recording.flac"
            content_type = "audio/flac"

        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("SARVAM_API_KEY", "") or self.api_key
        if not api_key or api_key in ("dummy", "your_sarvam_ai_api_key_here", "your_sarvam_api_key_here"):
            raise SarvamSTTError("SARVAM_API_KEY is not configured in .env. Voice audio transcription requires a valid Sarvam AI API key. (You can also type questions using the text input below.)")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Prepare multipart form data
                files = {
                    "file": (filename, io.BytesIO(audio_bytes), content_type),
                }
                data = {
                    "language_code": language_code if language_code else "unknown",
                    "model": model,
                    "with_timestamps": "false",
                }

                response = await client.post(
                    f"{self.BASE_URL}{self.STT_ENDPOINT}",
                    headers={
                        "api-subscription-key": api_key,
                    },
                    files=files,
                    data=data,
                )
                response.raise_for_status()

            latency_ms = (time.perf_counter() - t0) * 1000
            result = response.json()

            transcript_text = result.get("transcript", "")
            language = result.get("language_code", language_code)
            confidence = result.get("confidence", 1.0)

            logger.info(
                f"STT transcription complete in {latency_ms:.0f}ms: '{transcript_text[:80]}...'",
                extra={"stage": "stt", "latency_ms": latency_ms},
            )

            return TranscriptResult(
                text=transcript_text,
                language=language,
                confidence=confidence if isinstance(confidence, float) else 1.0,
                latency_ms=round(latency_ms, 2),
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"Sarvam STT API error ({e.response.status_code}): {e.response.text}")
            raise SarvamSTTError(
                f"STT API returned {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("Sarvam STT API timeout")
            raise SarvamSTTError("STT API request timed out") from e
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"STT transcription failed: {e}")
            raise SarvamSTTError(f"STT transcription failed: {e}") from e

    async def transcribe_or_passthrough(
        self,
        audio_bytes: bytes | None = None,
        text_input: str | None = None,
        language_code: str = "unknown",
    ) -> TranscriptResult:
        """Transcribe audio or pass through text input.

        Supports text-mode for testing without audio/mic hardware.

        Args:
            audio_bytes: Raw audio data (optional).
            text_input: Direct text input bypassing STT (optional).
            language_code: Language code for transcription.

        Returns:
            TranscriptResult.
        """
        if text_input:
            return TranscriptResult(
                text=text_input,
                language="en",
                confidence=1.0,
                latency_ms=0.0,
            )

        if audio_bytes:
            return await self.transcribe(audio_bytes, language_code=language_code)

        raise SarvamSTTError("Either audio_bytes or text_input must be provided")
