"""MiniMax music generation provider.

This provider routes music generation jobs to the MiniMax ``music_generation``
HTTP API instead of the local HeartMuLa/HeartCodec pipelines. It supports both
the global (``api.minimax.io``) and China (``api.minimaxi.com``) regional
endpoints, the generation and cover model families, the regional request
fields, the synchronous status/audio response, and ``url``/``hex`` audio
parsing.

The provider is intentionally dependency-light: it only relies on ``requests``
(already a project dependency) so it can be imported and unit-tested without
the heavy torch/heartlib stack.
"""

import logging
import os
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Regional endpoints for the music_generation operation.
MINIMAX_MUSIC_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}

# Model families. Generation models create new music from a prompt/lyrics; cover
# models re-sing an existing reference track. Names come from the provider's
# published model list and are not hardcoded anywhere else in the codebase.
MINIMAX_GENERATION_MODELS = ["music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"]
MINIMAX_COVER_MODELS = ["music-cover", "music-cover-free"]
MINIMAX_DEFAULT_MODEL = "music-3.0"

# Output and audio format support.
MINIMAX_OUTPUT_FORMATS = ["url", "hex"]
MINIMAX_STREAM_OUTPUT_FORMATS = ["hex"]
MINIMAX_AUDIO_FORMATS = ["mp3", "wav", "pcm"]

# Regional request fields that only apply to specific regions.
MINIMAX_REGIONAL_FIELDS = {
    "global_en": [],
    "cn_zh": ["aigc_watermark"],
}

# Response field contract (from the provider reference).
STATUS_FIELD = "data.status"
STATUS_IN_PROGRESS = 1
STATUS_COMPLETED = 2
AUDIO_FIELD = "data.audio"
SUCCESS_CODE_FIELD = "base_resp.status_code"
SUCCESS_CODE = 0

# The music_generation call is synchronous: it holds the connection until the
# track is ready and returns the audio inline. There is no separate task id or
# query endpoint to poll.
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_URL_TTL_HOURS = 24


class MiniMaxMusicError(RuntimeError):
    """Raised when the MiniMax music API returns an unusable response."""


class MiniMaxMusicProvider:
    """MiniMax music_generation HTTP client.

    Configuration is read from the environment so the provider can be used
    standalone, but callers may also pass explicit overrides.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        output_format: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("MINIMAX_API_KEY", "")
        self.region = region if region is not None else os.environ.get("MINIMAX_REGION", "global_en")
        self.output_format = output_format if output_format is not None else os.environ.get(
            "MINIMAX_OUTPUT_FORMAT", "url"
        )
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS

        if not self.api_key:
            raise MiniMaxMusicError("MINIMAX_API_KEY is not configured")
        if self.region not in MINIMAX_MUSIC_ENDPOINTS:
            raise MiniMaxMusicError(
                f"Unknown MiniMax region '{self.region}'. "
                f"Expected one of: {', '.join(sorted(MINIMAX_MUSIC_ENDPOINTS))}"
            )
        if self.output_format not in MINIMAX_OUTPUT_FORMATS:
            raise MiniMaxMusicError(
                f"Unknown MiniMax output_format '{self.output_format}'. "
                f"Expected one of: {', '.join(MINIMAX_OUTPUT_FORMATS)}"
            )

    @property
    def endpoint(self) -> str:
        return MINIMAX_MUSIC_ENDPOINTS[self.region]

    @staticmethod
    def is_cover_model(model: str) -> bool:
        return model in MINIMAX_COVER_MODELS

    @staticmethod
    def is_generation_model(model: str) -> bool:
        return model in MINIMAX_GENERATION_MODELS

    @staticmethod
    def resolve_model(model: Optional[str], cover: bool = False) -> str:
        """Pick a valid model id, defaulting to the published default."""
        if model:
            if model not in (MINIMAX_GENERATION_MODELS + MINIMAX_COVER_MODELS):
                raise MiniMaxMusicError(f"Unknown MiniMax music model: {model}")
            return model
        if cover:
            return MINIMAX_COVER_MODELS[0]
        return MINIMAX_DEFAULT_MODEL

    def _headers(self) -> dict:
        # Authorization scheme is Bearer per the provider reference.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        model: str,
        prompt: Optional[str],
        lyrics: Optional[str],
        is_instrumental: bool = False,
        cover_feature_id: Optional[str] = None,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
    ) -> dict:
        """Build the request payload, applying regional fields per region."""
        payload = {
            "model": model,
            "output_format": self.output_format,
        }

        if prompt:
            payload["prompt"] = prompt
        if lyrics:
            payload["lyrics"] = lyrics
        if is_instrumental:
            payload["is_instrumental"] = True
        if cover_feature_id:
            payload["cover_feature_id"] = cover_feature_id

        # Cover models require exactly one reference-audio input.
        if self.is_cover_model(model):
            if audio_url:
                payload["audio_url"] = audio_url
            elif audio_base64:
                payload["audio_base64"] = audio_base64
            else:
                raise MiniMaxMusicError(
                    "Cover models require one of 'audio_url' or 'audio_base64'."
                )

        # Regional fields: only the China (cn_zh) region accepts aigc_watermark.
        for field_name in MINIMAX_REGIONAL_FIELDS.get(self.region, []):
            if field_name == "aigc_watermark":
                payload["aigc_watermark"] = os.environ.get("MINIMAX_AIGC_WATERMARK", "0")
        return payload

    def _post(self, payload: dict) -> dict:
        logger.info(
            "[MiniMax] POST %s (model=%s, output_format=%s)",
            self.endpoint,
            payload.get("model"),
            self.output_format,
        )
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise MiniMaxMusicError(f"MiniMax request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise MiniMaxMusicError(
                f"MiniMax returned non-JSON response (HTTP {resp.status_code}): {exc}"
            ) from exc

        return data

    @staticmethod
    def _get_nested(data: dict, dotted: str):
        """Fetch a dotted path like 'data.status' or 'base_resp.status_code'."""
        current = data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _parse_response(self, data: dict) -> Tuple[str, str]:
        """Validate the synchronous response and return (audio, audio_format).

        ``audio`` is either a download URL (output_format=url) or raw hex bytes
        decoded from the response (output_format=hex). ``audio_format`` is the
        file extension to use when saving the result.
        """
        status_code = self._get_nested(data, SUCCESS_CODE_FIELD)
        if status_code != SUCCESS_CODE:
            raise MiniMaxMusicError(
                f"MiniMax request failed (base_resp.status_code={status_code}): {data}"
            )

        status = self._get_nested(data, STATUS_FIELD)
        if status == STATUS_IN_PROGRESS:
            raise MiniMaxMusicError(
                "MiniMax returned in_progress status with no query endpoint"
            )
        if status != STATUS_COMPLETED:
            raise MiniMaxMusicError(f"Unexpected MiniMax status: {status}")

        audio = self._get_nested(data, AUDIO_FIELD)
        if not audio:
            raise MiniMaxMusicError(
                f"MiniMax response missing audio field '{AUDIO_FIELD}': {data}"
            )

        if self.output_format == "hex":
            # data.audio is a hex-encoded byte string of PCM audio.
            try:
                decoded = bytes.fromhex(audio)
            except (ValueError, TypeError) as exc:
                raise MiniMaxMusicError(f"Failed to decode hex audio: {exc}") from exc
            return decoded, "pcm"

        # output_format == "url": data.audio is a temporary download URL.
        return audio, "mp3"

    def generate(
        self,
        prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
        model: Optional[str] = None,
        is_instrumental: bool = False,
        cover_feature_id: Optional[str] = None,
        audio_url: Optional[str] = None,
        audio_base64: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Run a music generation/cover job and return (audio_bytes, ext).

        For generation models, ``prompt``/``lyrics`` drive the output. For cover
        models, one of ``audio_url``/``audio_base64`` is required as the source
        track to re-sing.
        """
        cover = self.is_cover_model(model) if model else False
        resolved_model = self.resolve_model(model, cover=cover)
        payload = self._build_payload(
            model=resolved_model,
            prompt=prompt,
            lyrics=lyrics,
            is_instrumental=is_instrumental,
            cover_feature_id=cover_feature_id,
            audio_url=audio_url,
            audio_base64=audio_base64,
        )
        data = self._post(payload)
        return self._parse_response(data)

    def download_audio(self, url: str) -> bytes:
        """Download a URL-mode audio result returned by the API."""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise MiniMaxMusicError(f"Failed to download MiniMax audio: {exc}") from exc
        return resp.content
