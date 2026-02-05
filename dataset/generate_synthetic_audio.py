import os
import argparse
from pathlib import Path
import asyncio
from typing import List
import logging

from gtts import gTTS
import pyttsx3
try:
    import edge_tts
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False
    print("edge-tts not available. Install with: pip install edge-tts")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXTS = {
    'tamil': [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "இன்று வானிலை மிகவும் அழகாக இருக்கிறது",
        "நான் தமிழ் மொழியை மிகவும் விரும்புகிறேன்",
        "உங்கள் பெயர் என்ன?",
        "எனக்கு புத்தகங்கள் படிக்க மிகவும் பிடிக்கும்",
        "இசை என் வாழ்க்கையின் ஒரு முக்கிய பகுதி",
        "நான் ஒவ்வொரு நாளும் உடற்பயிற்சி செய்கிறேன்",
        "உணவு சமைப்பது என் பொழுதுபோக்கு",
        "தொழில்நுட்பம் உலகை மாற்றுகிறது",
        "கல்வி மிகவும் முக்கியமானது",
        "குடும்பம் எனக்கு எல்லாம்",
        "நான் பயணம் செய்ய விரும்புகிறேன்",
        "விளையாட்டு ஆரோக்கியத்திற்கு நல்லது",
        "இயற்கை அழகானது",
        "நட்பு விலைமதிப்பற்றது",
        "நேரம் விலைமதிப்பற்றது",
        "கடின உழைப்பு வெற்றிக்கு வழிவகுக்கிறது",
        "கனவுகள் நனவாகும்",
        "நாளை சிறந்த நாளாக இருக்கும்",
        "வாழ்க்கை அழகானது"
    ],
    'english': [
        "Hello, how are you doing today?",
        "The weather is beautiful outside",
        "I love learning new things every day",
        "What is your name?",
        "Reading books is my favorite hobby",
        "Music is an important part of my life",
        "I exercise every morning",
        "Cooking is my passion",
        "Technology is changing the world",
        "Education is very important",
        "Family means everything to me",
        "I love to travel and explore",
        "Sports are good for health",
        "Nature is beautiful",
        "Friendship is precious",
        "Time is valuable",
        "Hard work leads to success",
        "Dreams can come true",
        "Tomorrow will be better",
        "Life is wonderful"
    ],
    'hindi': [
        "नमस्ते, आप कैसे हैं?",
        "आज का मौसम बहुत सुहावना है",
        "मुझे हिंदी भाषा बहुत पसंद है",
        "आपका नाम क्या है?",
        "मुझे किताबें पढ़ना बहुत पसंद है",
        "संगीत मेरे जीवन का एक महत्वपूर्ण हिस्सा है",
        "मैं हर दिन व्यायाम करता हूं",
        "खाना बनाना मेरा शौक है",
        "प्रौद्योगिकी दुनिया को बदल रही है",
        "शिक्षा बहुत महत्वपूर्ण है",
        "परिवार मेरे लिए सब कुछ है",
        "मुझे यात्रा करना पसंद है",
        "खेल स्वास्थ्य के लिए अच्छे हैं",
        "प्रकृति सुंदर है",
        "दोस्ती अनमोल है",
        "समय बहुमूल्य है",
        "कड़ी मेहनत सफलता की ओर ले जाती है",
        "सपने सच हो सकते हैं",
        "कल बेहतर होगा",
        "जीवन सुंदर है"
    ],
    'malayalam': [
        "ഹലോ, നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?",
        "ഇന്ന് കാലാവസ്ഥ വളരെ മനോഹരമാണ്",
        "എനിക്ക് മലയാളം ഭാഷ വളരെ ഇഷ്ടമാണ്",
        "നിങ്ങളുടെ പേര് എന്താണ്?",
        "എനിക്ക് പുസ്തകങ്ങൾ വായിക്കാൻ വളരെ ഇഷ്ടമാണ്",
        "സംഗീതം എന്റെ ജീവിതത്തിന്റെ പ്രധാന ഭാഗമാണ്",
        "ഞാൻ എല്ലാ ദിവസവും വ്യായാമം ചെയ്യുന്നു",
        "പാചകം എന്റെ ഹോബിയാണ്",
        "സാങ്കേതികവിദ്യ ലോകത്തെ മാറ്റുന്നു",
        "വിദ്യാഭ്യാസം വളരെ പ്രധാനമാണ്",
        "കുടുംബം എനിക്ക് എല്ലാം ആണ്",
        "എനിക്ക് യാത്ര ചെയ്യാൻ ഇഷ്ടമാണ്",
        "കായികം ആരോഗ്യത്തിന് നല്ലതാണ്",
        "പ്രകൃതി സുന്ദരമാണ്",
        "സൗഹൃദം വിലമതിക്കാനാവാത്തതാണ്",
        "സമയം വിലപ്പെട്ടതാണ്",
        "കഠിനാധ്വാനം വിജയത്തിലേക്ക് നയിക്കുന്നു",
        "സ്വപ്നങ്ങൾ യാഥാർത്ഥ്യമാകും",
        "നാളെ മികച്ചതായിരിക്കും",
        "ജീവിതം മനോഹരമാണ്"
    ],
    'telugu': [
        "హలో, మీరు ఎలా ఉన్నారు?",
        "ఈరోజు వాతావరణం చాలా అందంగా ఉంది",
        "నాకు తెలుగు భాష చాలా ఇష్టం",
        "మీ పేరు ఏమిటి?",
        "నాకు పుస్తకాలు చదవడం చాలా ఇష్టం",
        "సంగీతం నా జీవితంలో ముఖ్యమైన భాగం",
        "నేను ప్రతిరోజు వ్యాయామం చేస్తాను",
        "వంట చేయడం నా అభిరుచి",
        "సాంకేతికత ప్రపంచాన్ని మారుస్తోంది",
        "విద్య చాలా ముఖ్యం",
        "కుటుంబం నాకు అంతా",
        "నాకు ప్రయాణం చేయడం ఇష్టం",
        "క్రీడలు ఆరోగ్యానికి మంచివి",
        "ప్రకృతి అందంగా ఉంది",
        "స్నేహం అమూల్యమైనది",
        "సమయం విలువైనది",
        "కష్టపడి పనిచేయడం విజయానికి దారితీస్తుంది",
        "కలలు నిజం అవుతాయి",
        "రేపు మంచిది అవుతుంది",
        "జీవితం అందంగా ఉంది"
    ]
}

# Voice configurations for different TTS engines
EDGE_VOICES = {
    'tamil': 'ta-IN-PallaviNeural',
    'english': 'en-US-JennyNeural',
    'hindi': 'hi-IN-SwaraNeural',
    'malayalam': 'ml-IN-SobhanaNeural',
    'telugu': 'te-IN-ShrutiNeural'
}

GTTS_LANGS = {
    'tamil': 'ta',
    'english': 'en',
    'hindi': 'hi',
    'malayalam': 'ml',
    'telugu': 'te'
}

class SyntheticAudioGenerator:
    """Generate synthetic AI audio samples"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_with_gtts(self, language: str, num_samples: int = 20):
        logger.info(f"Generating {num_samples} samples for {language} using gTTS...")
        
        texts = SAMPLE_TEXTS[language]
        lang_code = GTTS_LANGS[language]
        
        for i in range(num_samples):
            text = texts[i % len(texts)]
            filename = self.output_dir / f"gtts_{language}_{i+1:03d}.mp3"
            
            try:
                tts = gTTS(text=text, lang=lang_code, slow=False)
                tts.save(str(filename))
                logger.info(f"  Created: {filename.name}")
            except Exception as e:
                logger.error(f"  Failed to create {filename.name}: {e}")
    
    def generate_with_pyttsx3(self, language: str, num_samples: int = 20):
        logger.info(f"Generating {num_samples} samples for {language} using pyttsx3...")
        
        texts = SAMPLE_TEXTS[language]
        engine = pyttsx3.init()
        
        # Adjust voice properties
        voices = engine.getProperty('voices')
        engine.setProperty('rate', 150)  # Speed
        engine.setProperty('volume', 0.9)  # Volume
        
        for i in range(num_samples):
            text = texts[i % len(texts)]
            filename = self.output_dir / f"pyttsx3_{language}_{i+1:03d}.mp3"
            
            try:
                engine.save_to_file(text, str(filename))
                engine.runAndWait()
                logger.info(f"  Created: {filename.name}")
            except Exception as e:
                logger.error(f"  Failed to create {filename.name}: {e}")
    
    async def generate_with_edge_tts(self, language: str, num_samples: int = 20):
        if not EDGE_AVAILABLE:
            logger.warning(f"Skipping Edge TTS for {language} - not installed")
            return
        
        logger.info(f"Generating {num_samples} samples for {language} using Edge TTS...")
        
        texts = SAMPLE_TEXTS[language]
        voice = EDGE_VOICES[language]
        
        for i in range(num_samples):
            text = texts[i % len(texts)]
            filename = self.output_dir / f"edge_{language}_{i+1:03d}.mp3"
            
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(filename))
                logger.info(f"  Created: {filename.name}")
            except Exception as e:
                logger.error(f"  Failed to create {filename.name}: {e}")
    
    def generate_all(self, samples_per_language: int = 100):
        languages = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
        
        for language in languages:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Language: {language.upper()}")
            logger.info(f"{'='*60}\n")
            
            gtts_samples = int(samples_per_language * 0.4)
            pyttsx3_samples = int(samples_per_language * 0.3)
            edge_samples = samples_per_language - gtts_samples - pyttsx3_samples
            
            self.generate_with_gtts(language, gtts_samples)
            
            self.generate_with_pyttsx3(language, pyttsx3_samples)
            
            if EDGE_AVAILABLE:
                asyncio.run(self.generate_with_edge_tts(language, edge_samples))
            else:
                self.generate_with_gtts(language, edge_samples)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {samples_per_language * 5}")
        logger.info(f"Output directory: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic AI audio for training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./dataset/ai_generated',
        help='Output directory for audio files'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples per language (default: 100)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=['tamil', 'english', 'hindi', 'malayalam', 'telugu', 'all'],
        default=['all'],
        help='Languages to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    generator = SyntheticAudioGenerator(args.output_dir)
    
    if 'all' in args.languages:
        generator.generate_all(args.samples)
    else:
        for language in args.languages:
            generator.generate_with_gtts(language, int(args.samples * 0.4))
            generator.generate_with_pyttsx3(language, int(args.samples * 0.3))
            if EDGE_AVAILABLE:
                asyncio.run(generator.generate_with_edge_tts(language, int(args.samples * 0.3)))
    
    total_files = len(list(Path(args.output_dir).glob('*.mp3')))
    print(f"\nSummary:")
    print(f"   Total files generated: {total_files}")
    print(f"   Location: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"   1. Collect human voice samples")
    print(f"   2. Run: python train_model.py --data_dir ./dataset")

if __name__ == "__main__":
    main()