#!/bin/bash

# Test script for Voice Authenticity Detection API
# Usage: ./test_audio_api.sh <audio_file.mp3>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <audio_file.mp3>"
    echo "Example: $0 sample.mp3"
    exit 1
fi

AUDIO_FILE="$1"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: File '$AUDIO_FILE' not found!"
    exit 1
fi

echo " Testing Voice Authenticity Detection API"
echo " Audio file: $AUDIO_FILE"
echo " File size: $(du -h "$AUDIO_FILE" | cut -f1)"
echo

# Encode to base64 and create JSON payload
AUDIO_B64=$(base64 -w 0 "$AUDIO_FILE")
JSON_PAYLOAD="{\"audio_base64\": \"$AUDIO_B64\"}"

echo " Sending request to API..."

# Make the API call
RESPONSE=$(curl -s -X POST http://localhost:8000/detect \
  -H "X-API-Key: demo_key_12345" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD")

# Check if curl succeeded
if [ $? -eq 0 ]; then
    echo "✅ API Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo " Request failed!"
    echo "Make sure the API server is running: python3 main.py"
fi
