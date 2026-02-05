import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app
import base64
import numpy as np
import io
from pydub import AudioSegment

client = TestClient(app)

VALID_API_KEY = "demo_key_12345"
INVALID_API_KEY = "invalid_key"

def create_test_audio(duration_sec=1.0, sample_rate=16000):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    frequency = 440
    audio = np.sin(2 * np.pi * frequency * t) * 0.3

    audio_int16 = (audio * 32767).astype(np.int16)

    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    return buffer.getvalue()

class TestHealthEndpoints:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True

class TestAuthentication:

    def test_missing_api_key(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            json={"audio_base64": audio_base64}
        )
        assert response.status_code == 401

    def test_invalid_api_key(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": INVALID_API_KEY},
            json={"audio_base64": audio_base64}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_valid_api_key(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )
        assert response.status_code == 200

class TestDetectionEndpoint:

    def test_successful_detection(self):
        audio_bytes = create_test_audio(duration_sec=2.0)
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )

        assert response.status_code == 200
        data = response.json()

        assert "classification" in data
        assert "confidence" in data
        assert "language_detected" in data
        assert "processing_time_ms" in data

        assert data["classification"] in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["processing_time_ms"] > 0

    def test_detection_with_language(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={
                "audio_base64": audio_base64,
                "language": "tamil"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["language_detected"] == "tamil"

    def test_invalid_base64(self):
        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": "not_valid_base64!!!"}
        )

        assert response.status_code == 400

    def test_unsupported_language(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={
                "audio_base64": audio_base64,
                "language": "french"
            }
        )

        assert response.status_code == 400
        assert "Unsupported language" in response.json()["detail"]

    def test_audio_too_short(self):
        audio_bytes = create_test_audio(duration_sec=0.3)
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )

        assert response.status_code == 400
        assert "duration" in response.json()["detail"].lower()

    def test_all_supported_languages(self):
        languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        for lang in languages:
            response = client.post(
                "/detect",
                headers={"X-API-Key": VALID_API_KEY},
                json={
                    "audio_base64": audio_base64,
                    "language": lang
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["language_detected"] == lang

class TestSupportedLanguages:

    def test_get_supported_languages(self):
        response = client.get(
            "/supported-languages",
            headers={"X-API-Key": VALID_API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert "languages" in data

        languages = data["languages"]
        assert len(languages) == 5

        for lang in languages:
            assert "code" in lang
            assert "name" in lang

        codes = [lang["code"] for lang in languages]
        assert "tamil" in codes
        assert "english" in codes
        assert "hindi" in codes
        assert "malayalam" in codes
        assert "telugu" in codes

class TestResponseFormat:

    def test_response_schema(self):
        audio_bytes = create_test_audio()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )

        data = response.json()

        required_fields = ["classification", "confidence", "language_detected", "processing_time_ms"]
        for field in required_fields:
            assert field in data

        assert isinstance(data["classification"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["language_detected"], str)
        assert isinstance(data["processing_time_ms"], float)

class TestEdgeCases:

    def test_empty_audio(self):
        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": ""}
        )
        assert response.status_code == 400

    def test_very_long_audio(self):
        audio_bytes = create_test_audio(duration_sec=70.0)
        audio_base64 = base64.b64encode(audio_bytes).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )

        assert response.status_code == 400

    def test_silent_audio(self):
        audio_int16 = np.zeros(16000, dtype=np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1
        )

        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()

        response = client.post(
            "/detect",
            headers={"X-API-Key": VALID_API_KEY},
            json={"audio_base64": audio_base64}
        )

        assert response.status_code == 400
        assert "silent" in response.json()["detail"].lower()