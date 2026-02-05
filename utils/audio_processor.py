import base64
import io
import numpy as np
from pydub import AudioSegment
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:

    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def decode_base64(self, base64_string: str) -> bytes:
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]

            audio_bytes = base64.b64decode(base64_string)
            return audio_bytes
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {str(e)}")

    def load_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="mp3"
            )

            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)

            samples = np.array(audio_segment.get_array_of_samples())

            samples = samples.astype(np.float32) / (2**15)

            if audio_segment.frame_rate != self.target_sr:
                samples = librosa.resample(
                    samples,
                    orig_sr=audio_segment.frame_rate,
                    target_sr=self.target_sr
                )

            logger.info(f"Audio loaded: {len(samples)} samples at {self.target_sr}Hz")
            return samples, self.target_sr

        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")

    def detect_language(self, audio: np.ndarray, sr: int) -> str:
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

            centroid_mean = np.mean(spectral_centroid)
            rolloff_mean = np.mean(spectral_rolloff)

            if centroid_mean > 2500 and rolloff_mean > 5000:
                return "hindi"
            elif centroid_mean > 2200 and rolloff_mean > 4500:
                return "tamil"
            elif centroid_mean > 2000 and rolloff_mean > 4200:
                return "telugu"
            elif centroid_mean > 1800 and rolloff_mean > 4000:
                return "malayalam"
            else:
                return "unknown"

        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to unknown")
            return "unknown"

    def validate_audio(self, audio: np.ndarray, sr: int) -> bool:
        duration = len(audio) / sr

        if duration < 0.5 or duration > 60:
            raise ValueError(f"Audio duration must be between 0.5-60s, got {duration:.2f}s")

        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            raise ValueError("Audio is too quiet or silent")

        return True