# API Reference

## Base URL

```
Production: https://api.yourapi.com
Development: http://localhost:8000
```

## Authentication

All API requests require authentication via API key in the request header:

```
X-API-Key: your_api_key_here
```

### Error Response

```json
{
  "detail": "Invalid API key"
}
```

Status Code: `401 Unauthorized`

---

## Endpoints

### 1. Health Check

Check API status and model availability.

**Endpoint**: `GET /health`

**Authentication**: Not required

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is down

---

### 2. Root Information

Get API information and version.

**Endpoint**: `GET /`

**Authentication**: Not required

**Response**:
```json
{
  "service": "Voice Authenticity Detection API",
  "version": "1.0.0",
  "status": "operational"
}
```

---

### 3. Voice Authenticity Detection

Detect if audio is AI-generated or human voice.

**Endpoint**: `POST /detect`

**Authentication**: Required

**Request Headers**:
```
X-API-Key: your_api_key_here
Content-Type: application/json
```

**Request Body**:
```json
{
  "audio_base64": "string (required)",
  "language": "string (optional)"
}
```

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio_base64 | string | Yes | Base64-encoded MP3 audio file |
| language | string | No | Language code: tamil, english, hindi, malayalam, telugu. If not provided, will be auto-detected. |

**Response**:
```json
{
  "classification": "HUMAN",
  "confidence": 0.87,
  "language_detected": "tamil",
  "processing_time_ms": 234.56
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| classification | string | "AI_GENERATED" or "HUMAN" |
| confidence | float | Confidence score between 0.0 and 1.0 |
| language_detected | string | Detected or provided language |
| processing_time_ms | float | Processing time in milliseconds |

**Status Codes**:
- `200 OK`: Successful detection
- `400 Bad Request`: Invalid audio or unsupported language
- `401 Unauthorized`: Invalid API key
- `422 Unprocessable Entity`: Invalid request format
- `500 Internal Server Error`: Server error

**Error Responses**:

```json
{
  "detail": "Audio duration must be between 0.5-60s, got 0.3s"
}
```

```json
{
  "detail": "Unsupported language: french. Supported: ['tamil', 'english', 'hindi', 'malayalam', 'telugu']"
}
```

---

### 4. Supported Languages

Get list of supported languages.

**Endpoint**: `GET /supported-languages`

**Authentication**: Required

**Response**:
```json
{
  "languages": [
    {
      "code": "tamil",
      "name": "Tamil"
    },
    {
      "code": "english",
      "name": "English"
    },
    {
      "code": "hindi",
      "name": "Hindi"
    },
    {
      "code": "malayalam",
      "name": "Malayalam"
    },
    {
      "code": "telugu",
      "name": "Telugu"
    }
  ]
}
```

---

## Rate Limits

- **Default**: 100 requests per hour per API key
- **Burst**: 10 requests per minute

Rate limit headers in response:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## Audio Requirements

### Supported Formats
- MP3 (recommended)
- Base64 encoded

### Audio Constraints
- **Duration**: 0.5 - 60 seconds
- **Sample Rate**: Any (will be resampled to 16kHz)
- **Channels**: Mono or Stereo (will be converted to mono)
- **File Size**: Maximum 10MB encoded

### Quality Recommendations
- Sample rate: 16kHz or higher
- Bitrate: 128kbps or higher
- Clear speech without excessive background noise

---

## Code Examples

### Python

```python
import requests
import base64

def detect_voice(audio_path, api_key, language=None):
    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    # Prepare request
    url = "http://localhost:8000/detect"
    headers = {"X-API-Key": api_key}
    payload = {"audio_base64": audio_base64}
    
    if language:
        payload["language"] = language
    
    # Make request
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Usage
result = detect_voice("sample.mp3", "demo_key_12345", "tamil")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript (Node.js)

```javascript
const fs = require('fs');
const axios = require('axios');

async function detectVoice(audioPath, apiKey, language = null) {
  // Read and encode audio
  const audioBuffer = fs.readFileSync(audioPath);
  const audioBase64 = audioBuffer.toString('base64');
  
  // Prepare request
  const url = 'http://localhost:8000/detect';
  const headers = { 'X-API-Key': apiKey };
  const data = { audio_base64: audioBase64 };
  
  if (language) {
    data.language = language;
  }
  
  // Make request
  try {
    const response = await axios.post(url, data, { headers });
    return response.data;
  } catch (error) {
    throw new Error(`Error: ${error.response.status} - ${error.response.data.detail}`);
  }
}

// Usage
detectVoice('sample.mp3', 'demo_key_12345', 'tamil')
  .then(result => {
    console.log(`Classification: ${result.classification}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  })
  .catch(console.error);
```

### cURL

```bash
#!/bin/bash

AUDIO_FILE="sample.mp3"
API_KEY="demo_key_12345"
LANGUAGE="tamil"

# Encode audio to base64
AUDIO_BASE64=$(base64 -w 0 "$AUDIO_FILE")

# Make request
curl -X POST "http://localhost:8000/detect" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "'"$AUDIO_BASE64"'",
    "language": "'"$LANGUAGE"'"
  }' | jq .
```

### PHP

```php
<?php

function detectVoice($audioPath, $apiKey, $language = null) {
    // Read and encode audio
    $audioContent = file_get_contents($audioPath);
    $audioBase64 = base64_encode($audioContent);
    
    // Prepare request
    $url = 'http://localhost:8000/detect';
    $headers = [
        'X-API-Key: ' . $apiKey,
        'Content-Type: application/json'
    ];
    
    $data = ['audio_base64' => $audioBase64];
    if ($language) {
        $data['language'] = $language;
    }
    
    // Make request
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode === 200) {
        return json_decode($response, true);
    } else {
        throw new Exception("Error: $httpCode - $response");
    }
}

// Usage
$result = detectVoice('sample.mp3', 'demo_key_12345', 'tamil');
echo "Classification: " . $result['classification'] . "\n";
echo "Confidence: " . ($result['confidence'] * 100) . "%\n";
?>
```

### Go

```go
package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

type DetectRequest struct {
    AudioBase64 string `json:"audio_base64"`
    Language    string `json:"language,omitempty"`
}

type DetectResponse struct {
    Classification   string  `json:"classification"`
    Confidence       float64 `json:"confidence"`
    LanguageDetected string  `json:"language_detected"`
    ProcessingTimeMs float64 `json:"processing_time_ms"`
}

func detectVoice(audioPath, apiKey, language string) (*DetectResponse, error) {
    // Read and encode audio
    audioBytes, err := ioutil.ReadFile(audioPath)
    if err != nil {
        return nil, err
    }
    audioBase64 := base64.StdEncoding.EncodeToString(audioBytes)
    
    // Prepare request
    reqBody := DetectRequest{
        AudioBase64: audioBase64,
        Language:    language,
    }
    jsonData, _ := json.Marshal(reqBody)
    
    // Make request
    req, _ := http.NewRequest("POST", "http://localhost:8000/detect", bytes.NewBuffer(jsonData))
    req.Header.Set("X-API-Key", apiKey)
    req.Header.Set("Content-Type", "application/json")
    
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    // Parse response
    var result DetectResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }
    
    return &result, nil
}

func main() {
    result, err := detectVoice("sample.mp3", "demo_key_12345", "tamil")
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Classification: %s\n", result.Classification)
    fmt.Printf("Confidence: %.2f%%\n", result.Confidence*100)
}
```

---

## Webhooks (Coming Soon)

Subscribe to detection results via webhooks.

**Configuration**:
```json
{
  "webhook_url": "https://your-server.com/webhook",
  "events": ["detection.completed"],
  "secret": "your_webhook_secret"
}
```

**Webhook Payload**:
```json
{
  "event": "detection.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "classification": "AI_GENERATED",
    "confidence": 0.92,
    "language_detected": "tamil"
  }
}
```

---

## Best Practices

1. **Reuse connections**: Keep HTTP connections alive for better performance
2. **Handle rate limits**: Implement exponential backoff for rate limit errors
3. **Cache results**: Cache detection results for identical audio
4. **Batch processing**: Use async/parallel requests for multiple files
5. **Error handling**: Always check status codes and handle errors gracefully
6. **Audio optimization**: Compress audio to reduce upload time
7. **Language specification**: Provide language hint for faster processing

---

## Support

For API support:
- Email: support@yourapi.com
- Documentation: https://docs.yourapi.com
- Status Page: https://status.yourapi.com