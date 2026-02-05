import numpy as np
import librosa
from scipy import signal
from scipy.stats import skew, kurtosis
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:

    def __init__(self):
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512

    def extract_all_features(self, audio: np.ndarray, sr: int, language: str = "english") -> dict:
        features = {}

        features.update(self._extract_spectral_features(audio, sr))

        features.update(self._extract_prosodic_features(audio, sr))

        features.update(self._extract_temporal_features(audio, sr))

        features.update(self._extract_statistical_features(audio))

        features['language'] = self._encode_language(language)

        logger.info(f"Extracted {len(features)} features")
        return features

    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=self.hop_length)
        features['spectral_contrast_mean'] = np.mean(contrast)
        features['spectral_contrast_std'] = np.std(contrast)

        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)

        return features

    def _extract_prosodic_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}

        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        f0_voiced = f0[f0 > 0]

        if len(f0_voiced) > 0:
            features['f0_mean'] = np.mean(f0_voiced)
            features['f0_std'] = np.std(f0_voiced)
            features['f0_max'] = np.max(f0_voiced)
            features['f0_min'] = np.min(f0_voiced)
            features['f0_range'] = features['f0_max'] - features['f0_min']

            if len(f0_voiced) > 1:
                jitter = np.mean(np.abs(np.diff(f0_voiced)) / f0_voiced[:-1])
                features['jitter'] = jitter
            else:
                features['jitter'] = 0
        else:
            features.update({
                'f0_mean': 0, 'f0_std': 0, 'f0_max': 0,
                'f0_min': 0, 'f0_range': 0, 'jitter': 0
            })

        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)

        if len(rms[0]) > 1:
            shimmer = np.mean(np.abs(np.diff(rms[0])) / rms[0][:-1])
            features['shimmer'] = shimmer
        else:
            features['shimmer'] = 0

        return features

    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> dict:
        features = {}

        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        features['tempo'] = float(tempo[0])

        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        features['speech_rate'] = len(onset_frames) / (len(audio) / sr)

        return features

    def _extract_statistical_features(self, audio: np.ndarray) -> dict:
        features = {}

        features['signal_mean'] = np.mean(audio)
        features['signal_std'] = np.std(audio)
        features['signal_skewness'] = skew(audio)
        features['signal_kurtosis'] = kurtosis(audio)

        features['dynamic_range'] = np.max(audio) - np.min(audio)

        return features

    def _encode_language(self, language: str) -> float:
        language_map = {
            'tamil': 0.0,
            'english': 0.25,
            'hindi': 0.5,
            'malayalam': 0.75,
            'telugu': 1.0
        }
        return language_map.get(language.lower(), 0.25)

    def features_to_array(self, features: dict) -> np.ndarray:
        sorted_keys = sorted(features.keys())
        return np.array([features[k] for k in sorted_keys], dtype=np.float32)