"""
Voice output engine supporting High-Fidelity Human Conversational Prosody, Native Hindi/Hinglish Phonetics, Indian English, and KittenML TTS.
Features:
- Dynamic Language Detection (Hindi vs English)
- Native Devanagari Phonetic mapping for natural, human acoustic cadence (eliminates robotic letter-reading)
- SSML Micro-Breathing pauses & pitch contour for authentic human feeling
- Multiple natural voice personas (Salman: warm/casual male, Swara: soft/expressive female, Madhur: crisp male, Prabhat: Indian English)
- Ultra-Low Latency Streaming Audio Pipeline (synthesizes sentence-by-sentence while Gemini streams)
- Clean number/pricing pronunciation (e.g., '₹10k' -> 'दस हज़ार रुपये' in Hindi / 'ten thousand rupees' in English)
"""

import os
import re
import queue
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
    "11": "ग्यारह", "12": "बारह", "13": "तेरह", "14": "चौदह", "15": "पंद्रह",
    "16": "सोलह", "17": "सत्रह", "18": "अठारह", "19": "उन्नीस", "20": "बीस",
    "22": "बाईस", "25": "पच्चीस", "28": "अट्ठाईस", "30": "तीस", "35": "पैंतीस",
    "38": "अड़तीस", "40": "चालीस", "45": "पैंतालीस", "50": "पचास", "60": "साठ",
    "70": "सत्तर", "75": "पिचहत्तर", "80": "अस्सी", "90": "नब्बे", "100": "सौ"
}

ENG_NUMBERS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
    "22": "twenty-two", "25": "twenty-five", "28": "twenty-eight", "30": "thirty",
    "35": "thirty-five", "38": "thirty-eight", "40": "forty", "45": "forty-five",
    "50": "fifty", "60": "sixty", "70": "seventy", "75": "seventy-five", "80": "eighty",
    "90": "ninety", "100": "one hundred"
}

HINDI_HINTS = {
    'haan', 'bhai', 'kya', 'hai', 'mera', 'meri', 'hum', 'chahiye', 'kar', 
    'do', 'karo', 'toh', 'aur', 'pe', 'kitna', 'batao', 'liye', 'hoga', 
    'sakte', 'nahi', 'accha', 'acha', 'theek', 'kaise', 'kab', 'kuch', 'bhi',
    'bhejo', 'dekho', 'lekin', 'paise', 'rupaye', 'ka', 'ke', 'ki', 'samjhein',
    'taaki', 'rahega', 'dena', 'diya', 'baat', 'apna', 'apni', 'arey', 'suno'
}

# Phonetic mapping to convert Hinglish vocabulary to native Devanagari
# This completely eliminates robotic Latin-reading stutter and unlocks native human speech intonation
PHONETIC_MAP = {
    'arey': 'अरे', 'haan': 'हाँ', 'han': 'हाँ', 'bhai': 'भाई', 'bro': 'ब्रो',
    'bilkul': 'बिल्कुल', 'toh': 'तो', 'to': 'तो', 'aur': 'और', 'bhi': 'भी',
    'kya': 'क्या', 'hai': 'है', 'hain': 'हैं', 'ho': 'हो', 'hoga': 'होगा',
    'karein': 'करें', 'karo': 'करो', 'kar': 'कर', 'do': 'दो', 'dena': 'देना',
    'deal': 'डील', 'lock': 'लॉक', 'confirm': 'कन्फर्म', 'budget': 'बजट',
    'reel': 'रील', 'reels': 'रील्स', 'story': 'स्टोरी', 'stories': 'स्टोरीज़',
    'short': 'शॉर्ट', 'shorts': 'शॉर्ट्स', 'post': 'पोस्ट', 'posts': 'पोस्ट्स',
    'video': 'वीडियो', 'videos': 'वीडियोज़', 'draft': 'ड्राफ्ट', 'script': 'स्क्रिप्ट',
    'advance': 'एडवांस', 'payment': 'पेमेंट', 'product': 'प्रोडक्ट', 'dispatch': 'डिस्पैच',
    'ka': 'का', 'ke': 'के', 'ki': 'की', 'ko': 'को', 'se': 'से', 'me': 'में',
    'mein': 'में', 'pe': 'पे', 'par': 'पर', 'hum': 'हम', 'hamara': 'हमारा',
    'hamare': 'हमारे', 'hamari': 'हमारी', 'mera': 'मेरा', 'meri': 'मेरी', 'mere': 'मेरे',
    'aapka': 'आपका', 'aapke': 'आपके', 'aapki': 'आपकी', 'aap': 'आप',
    'right': 'राइट', 'total': 'टोटल', 'banta': 'बनता', 'theek': 'ठीक',
    'acha': 'अच्छा', 'accha': 'अच्छा', 'dekho': 'देखो', 'suno': 'सुनो',
    'badhiya': 'बढ़िया', 'badiya': 'बढ़िया', 'awesome': 'ऑसम', 'perfect': 'परफेक्ट',
    'done': 'डन', 'rate': 'रेट', 'quote': 'कोट', 'package': 'पैकेज',
    'terms': 'टर्म्स', 'approval': 'अप्रूवल', 'collab': 'कोलैब', 'collaboration': 'कोलैबोरेशन',
    'brand': 'ब्रांड', 'campaign': 'कैंपेन', 'launch': 'लॉन्च', 'link': 'लिंक',
    'code': 'कोड', 'discount': 'डिस्काउंट', 'timeline': 'टाइमलाइन', 'address': 'एड्रेस',
    'share': 'शेयर', 'send': 'सेंड', 'bhejo': 'भेजो', 'bhej': 'भेज', 'mila': 'मिला',
    'sirf': 'सिर्फ', 'lekin': 'लेकिन', 'agar': 'अगर', 'paise': 'पैसे', 'rupaye': 'रुपये',
    'got': 'गॉट', 'it': 'इट', 'well': 'वेल', 'sure': 'श्योर', 'so': 'सो',
    'na': 'ना', 'nahi': 'नहीं', 'chalega': 'चलेगा', 'kare': 'करे', 'rahega': 'रहेगा'
}


def detect_language(text: str) -> str:
    """Detects whether text is Hindi/Hinglish or English."""
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    matches = words.intersection(HINDI_HINTS)
    return 'hindi' if len(matches) >= 1 else 'english'


def build_conversational_ssml(text: str, voice_name: str, lang: str = "hindi", rate: str = "+12%") -> str:
    """
    Constructs an emotionally resonant, humanized SSML document with:
    - Native Devanagari phonetics (eradicating robotic Latin reading)
    - Natural breath pauses (<break time="..."/>) after fillers and clauses
    - Warm vocal prosody & pitch contour
    """
    t = re.sub(r'[*_~`#\[\]]', '', text)

    if lang == "hindi":
        # 1. Expand currency and numbers to Hindi words
        def rep_k(m):
            num = m.group(1)
            w = HINDI_NUMBERS.get(num, num)
            return f" {w} हज़ार रुपये "
        t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK]\b', rep_k, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000\b', rep_k, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)000\b', rep_k, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 रुपये ', t)
        t = t.replace('₹', 'रुपये')

        # Deliverables count: 1 Reel -> एक रील, 2 Stories -> दो स्टोरीज़
        def rep_deliv(m):
            cnt = m.group(1)
            item = m.group(2).lower()
            w = HINDI_NUMBERS.get(cnt, cnt)
            if 'reel' in item:
                item_hi = 'रील'
            elif 'stori' in item:
                item_hi = 'स्टोरीज़'
            elif 'story' in item:
                item_hi = 'स्टोरी'
            elif 'short' in item:
                item_hi = 'शॉर्ट्स'
            elif 'post' in item:
                item_hi = 'पोस्ट'
            elif 'video' in item:
                item_hi = 'वीडियो'
            else:
                item_hi = item
            return f"{w} {item_hi}"
        t = re.sub(r'\b(\d+)\s*(Reels?|Stories|Story|Posts?|Shorts?|Videos?)\b', rep_deliv, t, flags=re.IGNORECASE)

        # 2. Transliterate common Hinglish words into native Devanagari phonetics
        words = t.split()
        phonetic_words = []
        for w in words:
            clean = re.sub(r'[^\w]', '', w.lower())
            p_b = re.match(r'^[^\w]+', w)
            p_a = re.search(r'[^\w]+$', w)
            pb_str = p_b.group(0) if p_b else ''
            pa_str = p_a.group(0) if p_a else ''
            rep = PHONETIC_MAP.get(clean, clean if re.search(r'[\u0900-\u097F]', w) else w)
            phonetic_words.append(f"{pb_str}{rep}{pa_str}")
        t = " ".join(phonetic_words)

        # 3. Clean special characters, preserving Devanagari range
        t = re.sub(r'[^\w\s.,?!:;\-\u0900-\u097F]', ' ', t)
        t = " ".join(t.split()).strip()

        # 4. Insert natural micro-breathing breaks after conversational markers
        t = re.sub(r'\b(hmm|हम्म)\b[.,!]*', r'हम्म... <break time="220ms"/>', t, flags=re.IGNORECASE)
        t = re.sub(r'\b(हाँ\s*भाई|haan|haan\s*bhai|arey|अरे|suno|सुनो|dekho|देखो)[,.]*', r'\1, <break time="180ms"/>', t, flags=re.IGNORECASE)
        t = re.sub(r'([,;])\s*', r'\1 <break time="160ms"/> ', t)
        t = re.sub(r'([?])\s*', r'? <break time="200ms"/> ', t)

        # Natural warm pitch
        pitch = "-1Hz" if "Salman" in voice_name else "+0Hz"
        xml_lang = "hi-IN"
    else:
        # English expansion
        def rep_k_en(m):
            num = m.group(1)
            w = ENG_NUMBERS.get(num, num)
            return f" {w} thousand rupees "
        t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK]\b', rep_k_en, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000\b', rep_k_en, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)000\b', rep_k_en, t)
        t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 rupees ', t)
        t = t.replace('₹', 'rupees')

        def rep_deliv_en(m):
            cnt = m.group(1)
            item = m.group(2)
            w = ENG_NUMBERS.get(cnt, cnt)
            return f"{w} {item}"
        t = re.sub(r'\b(\d+)\s*(Reels?|Stories|Story|Posts?|Shorts?|Videos?)\b', rep_deliv_en, t, flags=re.IGNORECASE)

        t = re.sub(r'[^\w\s.,?!:;\-]', ' ', t)
        t = " ".join(t.split()).strip()

        t = re.sub(r'\b(hmm|got it|well|sure)[.,!]*', r'\1, <break time="200ms"/>', t, flags=re.IGNORECASE)
        t = re.sub(r'([,;])\s*', r'\1 <break time="160ms"/> ', t)
        t = re.sub(r'([?])\s*', r'? <break time="200ms"/> ', t)

        pitch = "-0.5Hz"
        xml_lang = "en-IN"

    return f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{xml_lang}">
    <voice name="{voice_name}">
        <prosody rate="{rate}" pitch="{pitch}">
            {t}
        </prosody>
    </voice>
</speak>"""


class StreamingVoicePipeline:
    """
    Pipelined audio streaming engine.
    Synthesizes sentence-by-sentence in a background thread using humanized phonetic SSML
    and plays via afplay while Gemini is still streaming tokens or while user starts typing!
    """
    def __init__(self, speaker: "VoiceSpeaker"):
        self.speaker = speaker
        self.buffer = ""
        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_active = True
        self.temp_files = []
        self._lock = threading.Lock()

        # Start synthesis worker and playback worker
        self.synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.synth_thread.start()
        self.play_thread.start()

    def _synth_worker(self):
        while self.is_active:
            try:
                item = self.sentence_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.audio_queue.put(None)
                break

            text = item.strip()
            if not text or not self.speaker.enabled:
                self.sentence_queue.task_done()
                continue

            lang = detect_language(text)

            try:
                if self.speaker.mode == "kitten":
                    k = self.speaker._get_kitten()
                    if k:
                        import soundfile as sf
                        cleaned = self.speaker.clean_text_for_speech(text, lang=lang)
                        audio = k.generate(cleaned, voice="expr-voice-2-m", speed=1.2)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio, 24000)
                        with self._lock:
                            self.temp_files.append(temp_path)
                        self.audio_queue.put(temp_path)
                else:
                    voice_name = self.speaker._resolve_voice_name(lang)
                    ssml_content = build_conversational_ssml(
                        text=text,
                        voice_name=voice_name,
                        lang=lang,
                        rate=self.speaker.rate
                    )

                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_path = f.name

                    try:
                        communicate = edge_tts.Communicate(ssml_content, voice=voice_name)
                        asyncio.run(communicate.save(temp_path))
                    except Exception:
                        # Fallback to plain cleaned text if SSML has an unexpected edge case
                        cleaned = self.speaker.clean_text_for_speech(text, lang=lang)
                        communicate = edge_tts.Communicate(cleaned, voice=voice_name, rate=self.speaker.rate)
                        asyncio.run(communicate.save(temp_path))

                    with self._lock:
                        self.temp_files.append(temp_path)
                    self.audio_queue.put(temp_path)
            except Exception:
                pass
            finally:
                self.sentence_queue.task_done()

    def _play_worker(self):
        while self.is_active:
            try:
                item = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            temp_path = item
            if not self.is_active:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except Exception: pass
                self.audio_queue.task_done()
                continue

            try:
                proc = subprocess.Popen(["afplay", temp_path])
                self.speaker._current_process = proc
                proc.wait()
            except Exception:
                pass
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except Exception: pass
                self.audio_queue.task_done()

    def push_chunk(self, chunk: str):
        """Buffers text tokens and enqueues completed clauses/sentences."""
        if not self.speaker.enabled:
            return

        self.buffer += chunk

        # Look for sentence boundaries: [.?!।\n]
        while True:
            match = re.search(r'([.!?।\n]+(?:\s+|$))', self.buffer)
            if not match:
                break
            end_pos = match.end()
            candidate = self.buffer[:end_pos].strip()
            words = candidate.split()

            # If phrase is too brief (e.g. 'Hmm...'), lookahead to merge with following sentence
            if len(words) < 3:
                sub = re.search(r'([.!?।\n]+(?:\s+|$))', self.buffer[end_pos:])
                if sub:
                    end_pos = end_pos + sub.end()
                    candidate = self.buffer[:end_pos].strip()
                else:
                    if len(candidate) < 20:
                        break

            if candidate:
                self.sentence_queue.put(candidate)
            self.buffer = self.buffer[end_pos:]

    def finish(self, wait: bool = False):
        """Flushes remaining text buffer and signals completion."""
        if self.buffer.strip():
            self.sentence_queue.put(self.buffer.strip())
            self.buffer = ""
        self.sentence_queue.put(None)

        if wait:
            self.synth_thread.join()
            self.play_thread.join()

    def stop(self):
        """Immediately halts current playback and cleans up."""
        self.is_active = False
        if self.speaker._current_process and self.speaker._current_process.poll() is None:
            try:
                self.speaker._current_process.terminate()
            except Exception:
                pass
        with self._lock:
            for path in self.temp_files:
                if os.path.exists(path):
                    try: os.remove(path)
                    except Exception: pass
            self.temp_files.clear()


class VoiceSpeaker:
    def __init__(self, mode: str = "auto", rate: str = "+12%", enabled: bool = True):
        """
        Modes:
        - 'auto': Mirrors language using warm, casual human voices (Hindi -> Salman, English -> Prabhat)
        - 'salman': ur-IN-SalmanNeural (Deep, warm casual male tone - highly recommended for realism)
        - 'swara': hi-IN-SwaraNeural (Soft, expressive, natural female tone)
        - 'madhur': hi-IN-MadhurNeural (Crisp, clean Hindi male tone)
        - 'english': en-IN-PrabhatNeural (Indian English male tone)
        - 'kitten': KittenTTS Nano
        """
        self.mode = mode
        self.rate = rate  # +12% to +15% gives the most relaxed, natural human speech
        self.enabled = enabled
        self._current_process = None
        self._kitten_tts = None
        self._active_pipeline = None

    def _resolve_voice_name(self, lang: str) -> str:
        if self.mode == "salman":
            return "ur-IN-SalmanNeural"
        elif self.mode == "swara":
            return "hi-IN-SwaraNeural"
        elif self.mode == "madhur":
            return "hi-IN-MadhurNeural"
        elif self.mode == "english":
            return "en-IN-PrabhatNeural"
        elif self.mode == "auto":
            return "ur-IN-SalmanNeural" if lang == "hindi" else "en-IN-PrabhatNeural"
        else:
            return "ur-IN-SalmanNeural" if lang == "hindi" else "en-IN-PrabhatNeural"

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

    def create_streaming_pipeline(self) -> StreamingVoicePipeline:
        """Stops any existing playback and creates a fresh streaming pipeline."""
        self.stop()
        self._active_pipeline = StreamingVoicePipeline(self)
        return self._active_pipeline

    def stop(self):
        """Immediately stops all audio playback."""
        if self._active_pipeline:
            self._active_pipeline.stop()
            self._active_pipeline = None
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def clean_text_for_speech(self, text: str, lang: str = "hindi") -> str:
        """Fallback clean text without SSML."""
        t = re.sub(r'[*_~`#\[\]]', '', text)
        if lang == "hindi":
            def replace_k_hi(m):
                num = m.group(1)
                w = HINDI_NUMBERS.get(num, num)
                return f" {w} हज़ार रुपये "
            t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK]\b', replace_k_hi, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000\b', replace_k_hi, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)000\b', replace_k_hi, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 रुपये ', t)
            t = t.replace('₹', 'रुपये')
        else:
            def replace_k_en(m):
                num = m.group(1)
                w = ENG_NUMBERS.get(num, num)
                return f" {w} thousand rupees "
            t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK]\b', replace_k_en, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000\b', replace_k_en, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)000\b', replace_k_en, t)
            t = re.sub(r'(?:INR|₹)\s*(\d+)', r' \1 rupees ', t)
            t = t.replace('₹', 'rupees')

        t = re.sub(r'[^\w\s.,?!:;\-\u0900-\u097F]', ' ', t)
        return " ".join(t.split()).strip()

    def speak(self, text: str, blocking: bool = False):
        """One-shot speaker using humanized phonetic SSML."""
        if not self.enabled:
            return

        lang = detect_language(text)

        def _worker():
            try:
                self.stop()
                if self.mode == "kitten":
                    k = self._get_kitten()
                    if k:
                        import soundfile as sf
                        cleaned = self.clean_text_for_speech(text, lang=lang)
                        audio = k.generate(cleaned, voice="expr-voice-2-m", speed=1.2)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio, 24000)
                        self._current_process = subprocess.Popen(["afplay", temp_path])
                        self._current_process.wait()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    voice_name = self._resolve_voice_name(lang)
                    ssml_content = build_conversational_ssml(
                        text=text,
                        voice_name=voice_name,
                        lang=lang,
                        rate=self.rate
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_path = f.name
                    try:
                        communicate = edge_tts.Communicate(ssml_content, voice=voice_name)
                        asyncio.run(communicate.save(temp_path))
                    except Exception:
                        cleaned = self.clean_text_for_speech(text, lang=lang)
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
