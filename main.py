from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from typing import Literal
import logging
from utils.audio_processor import AudioProcessor
from utils.feature_extractor import FeatureExtractor
from models.classifier import VoiceClassifier
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Authenticity Detection API",
    description="Detects AI-generated vs human voice in multilingual audio",
    version="1.0.0"
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

VALID_API_KEYS = set(os.getenv("API_KEYS", "demo_key_12345,test_key_67890").split(","))

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

audio_processor = AudioProcessor()
feature_extractor = FeatureExtractor()
classifier = VoiceClassifier()

class DetectionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio")
    language: str | None = Field(None, description="Optional: tamil, english, hindi, malayalam, telugu, unknown")

class DetectionResponse(BaseModel):
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    language_detected: str | None = None
    processing_time_ms: float

@app.get("/")
async def root():
    return {
        "service": "Voice Authenticity Detection API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": classifier.is_loaded()}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice_authenticity(
    request: DetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    import time
    start_time = time.time()

    try:
        logger.info("Decoding audio from base64")
        audio_data = audio_processor.decode_base64(request.audio_base64)
        audio_array, sample_rate = audio_processor.load_audio(audio_data)

        audio_processor.validate_audio(audio_array, sample_rate)

        language = request.language
        if not language:
            language = audio_processor.detect_language(audio_array, sample_rate)
            logger.info(f"Detected language: {language}")

        supported_languages = ["tamil", "english", "hindi", "malayalam", "telugu", "unknown"]
        if language.lower() not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {language}. Supported: {supported_languages}"
            )

        logger.info("Extracting audio features")
        features = feature_extractor.extract_all_features(
            audio_array,
            sample_rate,
            language=language.lower()
        )

        logger.info("Running classification")
        classification, confidence = classifier.predict(features)

        processing_time = (time.time() - start_time) * 1000

        return DetectionResponse(
            classification=classification,
            confidence=round(confidence, 4),
            language_detected=language.lower(),
            processing_time_ms=round(processing_time, 2)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/supported-languages")
async def get_supported_languages(api_key: str = Depends(verify_api_key)):
    return {
        "languages": [
            {"code": "tamil", "name": "Tamil"},
            {"code": "english", "name": "English"},
            {"code": "hindi", "name": "Hindi"},
            {"code": "malayalam", "name": "Malayalam"},
            {"code": "telugu", "name": "Telugu"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)