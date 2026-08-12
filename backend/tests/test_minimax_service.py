"""Tests for the MiniMax music generation provider.

These tests stub the network layer so they run without a MiniMax API key and
without the heavy torch/heartlib dependencies.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests as _requests

# Make ``backend`` importable when running from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.services.minimax_service import (  # noqa: E402
    MINIMAX_MUSIC_ENDPOINTS,
    MiniMaxMusicError,
    MiniMaxMusicProvider,
)


def _make_provider(**overrides):
    base = {"api_key": "test-key", "region": "global_en", "output_format": "url"}
    base.update(overrides)
    return MiniMaxMusicProvider(**base)


def test_endpoints_cover_both_regions():
    assert MINIMAX_MUSIC_ENDPOINTS["global_en"] == "https://api.minimax.io/v1/music_generation"
    assert MINIMAX_MUSIC_ENDPOINTS["cn_zh"] == "https://api.minimaxi.com/v1/music_generation"


def test_requires_api_key():
    with pytest.raises(MiniMaxMusicError):
        MiniMaxMusicProvider(api_key="", region="global_en")


def test_rejects_unknown_region():
    with pytest.raises(MiniMaxMusicError):
        MiniMaxMusicProvider(api_key="k", region="mars")


def test_rejects_unknown_output_format():
    with pytest.raises(MiniMaxMusicError):
        MiniMaxMusicProvider(api_key="k", region="global_en", output_format="flac")


def test_resolves_default_generation_model():
    assert _make_provider().resolve_model(None, cover=False) == "music-3.0"


def test_resolves_default_cover_model():
    assert _make_provider().resolve_model(None, cover=True) == "music-cover"


def test_rejects_unknown_model():
    with pytest.raises(MiniMaxMusicError):
        _make_provider().resolve_model("not-a-model")


def test_payload_generation_model_fields():
    p = _make_provider()
    payload = p._build_payload(
        model="music-3.0", prompt="upbeat electronic",
        lyrics="[Verse] la la la", is_instrumental=False,
    )
    assert payload["model"] == "music-3.0"
    assert payload["prompt"] == "upbeat electronic"
    assert payload["lyrics"] == "[Verse] la la la"
    assert payload["output_format"] == "url"
    assert "aigc_watermark" not in payload
    assert payload["stream"] is False


def test_payload_optional_generation_settings():
    p = _make_provider(output_format="hex")
    payload = p._build_payload(
        model="music-3.0", prompt="x", lyrics=None, stream=True,
        audio_setting={"format": "wav", "sample_rate": 44100},
        lyrics_optimizer=True,
    )
    assert payload["stream"] is True
    assert payload["audio_setting"] == {"format": "wav", "sample_rate": 44100}
    assert payload["lyrics_optimizer"] is True


def test_stream_requires_hex_output():
    with pytest.raises(MiniMaxMusicError, match="require output_format='hex'"):
        _make_provider()._build_payload(
            model="music-3.0", prompt="x", lyrics=None, stream=True,
        )


def test_payload_instrumental_flag():
    p = _make_provider()
    payload = p._build_payload(model="music-3.0", prompt=None, lyrics=None, is_instrumental=True)
    assert payload["is_instrumental"] is True
    assert "lyrics" not in payload
    assert "prompt" not in payload


def test_payload_cn_region_adds_aigc_watermark():
    p = _make_provider(region="cn_zh")
    payload = p._build_payload(model="music-3.0", prompt="x", lyrics=None)
    assert payload["aigc_watermark"] == os.environ.get("MINIMAX_AIGC_WATERMARK", "0")


def test_payload_global_region_omits_aigc_watermark():
    p = _make_provider(region="global_en")
    payload = p._build_payload(model="music-3.0", prompt="x", lyrics=None)
    assert "aigc_watermark" not in payload


def test_cover_model_requires_audio_input():
    p = _make_provider()
    with pytest.raises(MiniMaxMusicError):
        p._build_payload(model="music-cover", prompt="x", lyrics=None)


def test_cover_model_accepts_audio_url():
    p = _make_provider()
    payload = p._build_payload(
        model="music-cover", prompt=None, lyrics=None,
        audio_url="https://example.com/track.mp3",
        audio_duration_seconds=60,
        audio_size_bytes=1024,
    )
    assert payload["audio_url"] == "https://example.com/track.mp3"


def test_cover_model_accepts_audio_base64():
    p = _make_provider()
    payload = p._build_payload(
        model="music-cover", prompt=None, lyrics=None, audio_base64="AAAA",
        audio_duration_seconds=60,
    )
    assert payload["audio_base64"] == "AAAA"


def test_cover_model_accepts_preprocessed_feature():
    payload = _make_provider()._build_payload(
        model="music-cover", prompt=None, lyrics="replacement lyrics",
        cover_feature_id="feature-id",
    )
    assert payload["cover_feature_id"] == "feature-id"


def test_cover_model_rejects_both_audio_inputs():
    with pytest.raises(MiniMaxMusicError, match="exactly one"):
        _make_provider()._build_payload(
            model="music-cover", prompt=None, lyrics=None,
            audio_url="https://example.com/track.mp3", audio_base64="AAAA",
            audio_duration_seconds=60, audio_size_bytes=1024,
        )


def test_cover_model_rejects_direct_and_preprocessed_inputs():
    with pytest.raises(MiniMaxMusicError, match="exactly one"):
        _make_provider()._build_payload(
            model="music-cover", prompt=None, lyrics="replacement lyrics",
            audio_url="https://example.com/track.mp3", cover_feature_id="feature-id",
            audio_duration_seconds=60, audio_size_bytes=1024,
        )


@pytest.mark.parametrize("duration", [5, 361])
def test_cover_model_validates_duration(duration):
    with pytest.raises(MiniMaxMusicError, match="duration"):
        _make_provider()._build_payload(
            model="music-cover", prompt=None, lyrics=None,
            audio_base64="AAAA", audio_duration_seconds=duration,
        )


def test_cover_model_validates_size():
    with pytest.raises(MiniMaxMusicError, match="50 MB"):
        _make_provider()._build_payload(
            model="music-cover", prompt=None, lyrics=None,
            audio_url="https://example.com/track.mp3", audio_duration_seconds=60,
            audio_size_bytes=50 * 1024 * 1024 + 1,
        )


def test_parse_url_response():
    p = _make_provider(output_format="url")
    data = {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "https://cdn.example.com/song.mp3"}}
    audio, ext = p._parse_response(data)
    assert audio == "https://cdn.example.com/song.mp3"
    assert ext == "mp3"


def test_parse_hex_response():
    p = _make_provider(output_format="hex")
    data = {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "48656c6c6f"}}
    audio, ext = p._parse_response(data)
    assert audio == b"Hello"
    assert ext == "pcm"


def test_parse_response_rejects_nonzero_status_code():
    p = _make_provider()
    data = {"base_resp": {"status_code": 1004}, "data": {"status": 2, "audio": "x"}}
    with pytest.raises(MiniMaxMusicError):
        p._parse_response(data)


def test_parse_response_rejects_in_progress():
    p = _make_provider()
    data = {"base_resp": {"status_code": 0}, "data": {"status": 1}}
    with pytest.raises(MiniMaxMusicError):
        p._parse_response(data)


def test_parse_response_rejects_missing_audio():
    p = _make_provider()
    data = {"base_resp": {"status_code": 0}, "data": {"status": 2}}
    with pytest.raises(MiniMaxMusicError):
        p._parse_response(data)


def test_parse_response_rejects_invalid_hex():
    p = _make_provider(output_format="hex")
    data = {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "ZZ"}}
    with pytest.raises(MiniMaxMusicError):
        p._parse_response(data)


def _fake_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def test_generate_global_en_posts_to_global_endpoint():
    p = _make_provider(region="global_en")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _fake_response(json_data={
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": "https://cdn.example.com/song.mp3"},
        })

    with patch("backend.app.services.minimax_service.requests.post", side_effect=fake_post):
        audio, ext = p.generate(prompt="calm piano", lyrics=None, model="music-3.0")

    assert captured["url"] == "https://api.minimax.io/v1/music_generation"
    assert captured["json"]["model"] == "music-3.0"
    assert captured["json"]["prompt"] == "calm piano"
    assert audio == "https://cdn.example.com/song.mp3"
    assert ext == "mp3"


def test_generate_cn_zh_posts_to_cn_endpoint_and_adds_watermark():
    p = _make_provider(region="cn_zh")
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _fake_response(json_data={
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": "https://cdn.example.com/song.mp3"},
        })

    with patch("backend.app.services.minimax_service.requests.post", side_effect=fake_post):
        p.generate(prompt="x", lyrics=None, model="music-3.0")

    assert captured["url"] == "https://api.minimaxi.com/v1/music_generation"
    assert captured["json"]["aigc_watermark"] == os.environ.get("MINIMAX_AIGC_WATERMARK", "0")
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_generate_cover_model_sends_audio_url():
    p = _make_provider()
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _fake_response(json_data={
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": "https://cdn.example.com/cover.mp3"},
        })

    with patch("backend.app.services.minimax_service.requests.post", side_effect=fake_post):
        p.generate(prompt=None, lyrics=None, model="music-cover",
                   audio_url="https://example.com/src.mp3",
                   audio_duration_seconds=60, audio_size_bytes=1024)

    assert captured["json"]["model"] == "music-cover"
    assert captured["json"]["audio_url"] == "https://example.com/src.mp3"


def test_generate_propagates_api_error():
    p = _make_provider()
    with patch("backend.app.services.minimax_service.requests.post",
               return_value=_fake_response(json_data={"base_resp": {"status_code": 1004}, "data": {}})):
        with pytest.raises(MiniMaxMusicError):
            p.generate(prompt="x", lyrics=None, model="music-3.0")


def test_generate_handles_request_exception():
    p = _make_provider()
    with patch("backend.app.services.minimax_service.requests.post",
               side_effect=_requests.ConnectionError("boom")):
        with pytest.raises(MiniMaxMusicError):
            p.generate(prompt="x", lyrics=None, model="music-3.0")


def test_get_nested_handles_missing_path():
    assert MiniMaxMusicProvider._get_nested({"a": {"b": 1}}, "a.b") == 1
    assert MiniMaxMusicProvider._get_nested({"a": {}}, "a.b") is None
    assert MiniMaxMusicProvider._get_nested({}, "a.b") is None
    assert MiniMaxMusicProvider._get_nested({"base_resp": {"status_code": 0}}, "base_resp.status_code") == 0


def test_download_audio_returns_bytes():
    p = _make_provider()
    fake = MagicMock()
    fake.content = b"\x00\x01\x02"
    fake.raise_for_status.return_value = None
    with patch("backend.app.services.minimax_service.requests.get", return_value=fake):
        assert p.download_audio("https://cdn.example.com/song.mp3") == b"\x00\x01\x02"
