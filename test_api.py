import requests
import base64
import numpy as np
from pydub import AudioSegment
import io

def create_test_audio(duration_sec=1.0, frequency=440):
    """Create a simple test audio (sine wave)"""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    
    # Export to MP3
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    return buffer.getvalue()

def test_api():
    print("🧪 Testing Voice Authenticity Detection API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"    Failed: {e}")
        return
    
    # Test 2: Root Endpoint
    print("\n2. Testing Root Endpoint...")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # Test 3: Supported Languages
    print("\n3. Testing Supported Languages...")
    try:
        response = requests.get(
            "http://localhost:8000/supported-languages",
            headers={"X-API-Key": "demo_key_12345"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Languages: {response.json()}")
    except Exception as e:
        print(f"    Failed: {e}")
    
    # Test 4: Voice Detection
    print("\n4. Testing Voice Detection...")
    try:
        # Create test audio
        audio_bytes = create_test_audio(duration_sec=1.0)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        response = requests.post(
            "http://localhost:8000/detect",
            json={"audio_base64": audio_base64},
            headers={"X-API-Key": "demo_key_12345"}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Language: {result['language_detected']}")
            print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
        else:
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"    Failed: {e}")
    
    # Test 5: Error Cases
    print("\n5. Testing Error Cases...")
    
    # No API key
    try:
        response = requests.post("http://localhost:8000/detect", json={"audio_base64": "test"})
        print(f"   No API key: {response.status_code} (expected: 401)")
    except Exception as e:
        print(f"   No API key test failed: {e}")
    
    # Invalid base64
    try:
        response = requests.post(
            "http://localhost:8000/detect",
            json={"audio_base64": "invalid_base64"},
            headers={"X-API-Key": "demo_key_12345"}
        )
        print(f"   Invalid base64: {response.status_code} (expected: 400)")
    except Exception as e:
        print(f"   Invalid base64 test failed: {e}")
    
    print("\n API Testing Complete!")

if __name__ == "__main__":
    test_api()
