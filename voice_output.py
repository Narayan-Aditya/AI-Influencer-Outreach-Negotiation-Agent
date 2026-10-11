"""
Voice output engine supporting Clean Native Hindi/Hinglish (Neural), Indian English, and KittenML TTS.
Features:
- Dynamic Language Detection (Hindi vs English)
- Paced speech rate (+18% speed)
- Human conversational filler support ('Hmm...', 'Haan...')
- Clean number/pricing pronunciation (e.g., '10k' -> 'दस हज़ार रुपये' in Hindi / 'ten thousand rupees' in English)
"""

import os
import re
import asyncio
import tempfile
import threading
import subprocess
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)

import edge_tts


# Number mappings for clean, natural speech pronunciation
HINDI_NUMBERS = {
    "1": "एक", "2": "दो", "3": "तीन", "4": "चार", "5": "पाँच",
    "6": "छह", "7": "सात", "8": "आठ", "9": "नौ", "10": "दस",
    "12": "बारह", "15": "पंद्रह", "16": "सोलह", "18": "अठारह",
    "20": "बीस", "22": "बाईस", "25": "पच्चीस", "28": "अट्ठाईस",
    "30": "तीस", "35": "पैंतीस", "38": "अड़तीस", "40": "चालीस",
    "45": "पैंतालीस", "50": "पचास", "60": "साठ", "70": "सत्तर",
    "75": "पिचहत्तर", "80": "अस्सी", "90": "नब्बे", "100": "सौ"
}

ENG_NUMBERS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "12": "twelve", "15": "fifteen", "16": "sixteen", "18": "eighteen",
    "20": "twenty", "22": "twenty-two", "25": "twenty-five", "28": "twenty-eight",
    "30": "thirty", "35": "thirty-five", "38": "thirty-eight", "40": "forty",
    "45": "forty-five", "50": "fifty", "60": "sixty", "70": "seventy",
    "75": "seventy-five", "80": "eighty", "90": "ninety", "100": "one hundred"
}

HINDI_HINTS = {
    'haan', 'bhai', 'kya', 'hai', 'mera', 'meri', 'hum', 'chahiye', 'kar', 
    'do', 'karo', 'toh', 'aur', 'pe', 'kitna', 'batao', 'liye', 'hoga', 
    'sakte', 'nahi', 'accha', 'acha', 'theek', 'kaise', 'kab', 'kuch', 'bhi',
    'bhejo', 'dekho', 'lekin', 'paise', 'rupaye', 'ka', 'ke', 'ki', 'samjhein',
    'taaki', 'rahega', 'dena', 'diya', 'baat', 'apna', 'apni'
}


def detect_language(text: str) -> str:
    """Detects whether text is Hindi/Hinglish or English."""
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    matches = words.intersection(HINDI_HINTS)
    return 'hindi' if len(matches) >= 1 else 'english'


class VoiceSpeaker:
    def __init__(self, mode: str = "auto", rate: str = "+18%", enabled: bool = True):
        """
        Modes:
        - 'auto': Automatically mirrors language (Hindi -> Madhur, English -> Prabhat)
        - 'hindi_male': hi-IN-MadhurNeural
        - 'hindi_female': hi-IN-SwaraNeural
        - 'english_male': en-IN-PrabhatNeural
        - 'kitten': KittenTTS Nano
        """
        self.mode = mode
        self.rate = rate  # Increased speed by default (+18%)
        self.enabled = enabled
        self._current_process = None
        self._kitten_tts = None

    def _get_kitten(self):
        if self._kitten_tts is None:
            try:
                import espeakng_loader
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
                from kittentts import KittenTTS
                self._kitten_tts = KittenTTS()
            except Exception as e:
                print(f"⚠️ KittenTTS load error: {e}")
        return self._kitten_tts

    def clean_text_for_speech(self, text: str, lang: str = "hindi") -> str:
        """Expands numbers, pricing ('10k' -> 'दस हज़ार रुपये' or 'ten thousand rupees'), and cleans formatting."""
        t = re.sub(r'[*_~`#\[\]]', '', text)

        # Human fillers smoothing: make sure 'Hmm...', 'Haan...' have a natural comma pause
        t = re.sub(r'\b(hmm|haan|acha|dekho|well|got it)\.\.\.', r'\1, ', t, flags=re.IGNORECASE)

        if lang == "hindi":
            def replace_k_hi(m):
                num = m.group(1)
                word = HINDI_NUMBERS.get(num, num)
                return f" {word} हज़ार रुपये "

            # ₹10k, 10k, INR 10,000 -> दस हज़ार रुपये
            t = re.sub(r'(?:INR|₹)?\s*(\d+)k\b', replace_k_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+),000', replace_k_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 रुपये ', t, flags=re.IGNORECASE)
            # Dates: 25th -> 25 तारीख
            t = re.sub(r'(\d+)(?:st|nd|rd|th)\b', r'\1 तारीख', t, flags=re.IGNORECASE)
        else:
            def replace_k_en(m):
                num = m.group(1)
                word = ENG_NUMBERS.get(num, num)
                return f" {word} thousand rupees "

            # ₹10k, 10k -> ten thousand rupees
            t = re.sub(r'(?:INR|₹)?\s*(\d+)k\b', replace_k_en, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+),000', replace_k_en, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 rupees ', t, flags=re.IGNORECASE)

        # Clean special chars / emojis
        t = re.sub(r'[^\w\s.,?!:;\-]', ' ', t)
        return " ".join(t.split()).strip()

    def speak(self, text: str, blocking: bool = False):
        if not self.enabled:
            return

        lang = detect_language(text)
        cleaned = self.clean_text_for_speech(text, lang=lang)
        if not cleaned:
            return

        def _worker():
            try:
                # Stop any currently playing audio
                if self._current_process and self._current_process.poll() is None:
                    self._current_process.terminate()

                if self.mode == "kitten":
                    k = self._get_kitten()
                    if k:
                        import soundfile as sf
                        audio = k.generate(cleaned, voice="expr-voice-2-m", speed=1.15)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio, 24000)
                        self._current_process = subprocess.Popen(["afplay", temp_path])
                        self._current_process.wait()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    # Select voice based on mode & language
                    if self.mode == "auto":
                        voice_name = "hi-IN-MadhurNeural" if lang == "hindi" else "en-IN-PrabhatNeural"
                    elif self.mode == "hindi_female":
                        voice_name = "hi-IN-SwaraNeural"
                    elif self.mode == "english_male":
                        voice_name = "en-IN-PrabhatNeural"
                    else:
                        voice_name = "hi-IN-MadhurNeural"

                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_path = f.name

                    communicate = edge_tts.Communicate(cleaned, voice=voice_name, rate=self.rate)
                    asyncio.run(communicate.save(temp_path))

                    self._current_process = subprocess.Popen(["afplay", temp_path])
                    self._current_process.wait()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception:
                pass

        if blocking:
            _worker()
        else:
            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

    def set_mode(self, mode: str):
        self.mode = mode

    def set_rate(self, rate: str):
        self.rate = rate
