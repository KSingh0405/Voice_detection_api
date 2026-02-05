#!/usr/bin/env python3

import librosa
import numpy as np
import base64
import io
from pydub import AudioSegment

def analyze_audio_features(audio_file):
    
    print(f"Analyzing: {audio_file}")

    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples.astype(np.float32) / (2**15)

    if audio_segment.frame_rate != 16000:
        samples = librosa.resample(samples, orig_sr=audio_segment.frame_rate, target_sr=16000)
        sr = 16000
    else:
        sr = audio_segment.frame_rate

    print(f"Sample rate: {sr}Hz")
    print(f"Samples: {len(samples)}")

    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)

    centroid_mean = np.mean(spectral_centroid)
    rolloff_mean = np.mean(spectral_rolloff)

    print(".2f")
    print(".2f")
    print(f"Spectral Centroid Mean: {centroid_mean:.2f}")
    print(f"Spectral Rolloff Mean: {rolloff_mean:.2f}")
    
    print("\nThreshold checks:")
    print(f"centroid > 2500 AND rolloff > 5000 (hindi): {centroid_mean > 2500 and rolloff_mean > 5000}")
    print(f"centroid > 2200 AND rolloff > 4500 (tamil): {centroid_mean > 2200 and rolloff_mean > 4500}")
    print(f"centroid > 2000 AND rolloff > 4200 (telugu): {centroid_mean > 2000 and rolloff_mean > 4200}")
    print(f"centroid > 1800 AND rolloff > 4000 (malayalam): {centroid_mean > 1800 and rolloff_mean > 4000}")
    print(f"else (unknown): True")

if __name__ == "__main__":
    analyze_audio_features("dataset/ai_generated/edge_english_001.mp3")