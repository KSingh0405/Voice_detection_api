#!/usr/bin/env python3

import sys
import os
import base64
import requests
import json

def test_audio_file(audio_file_path):

    if not os.path.exists(audio_file_path):
        print(f"Error: File '{audio_file_path}' not found!")
        return

    file_size = os.path.getsize(audio_file_path)
    print("Testing Voice Authenticity Detection API")
    print(f"Audio file: {audio_file_path}")
    print(f"File size: {file_size / 1024:.1f} KB")
    print()

    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        payload = {
            "audio_base64": audio_base64
        }

        print("Sending request to API...")

        response = requests.post(
            "http://localhost:8000/detect",
            json=payload,
            headers={
                "X-API-Key": "demo_key_12345",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(json.dumps(result, indent=2))

            classification = result.get('classification', 'Unknown')
            confidence = result.get('confidence', 0)
            language = result.get('language_detected', 'Unknown')
            processing_time = result.get('processing_time_ms', 0)

            print("\nResults:")
            print(f"   Classification: {classification}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Language: {language}")
            print(f"   Processing Time: {processing_time:.2f}ms")

        else:
            print("API Error:")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("Connection Error: Make sure the API server is running!")
        print("   Start it with: python3 main.py")
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_audio_python.py <audio_file.mp3>")
        print("Example: python3 test_audio_python.py sample.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]
    test_audio_file(audio_file)

if __name__ == "__main__":
    main()
