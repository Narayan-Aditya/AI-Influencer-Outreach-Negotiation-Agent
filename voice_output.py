"""
Voice output engine supporting Clean Native Hindi/Hinglish (Neural), Indian English, and KittenML TTS.
Features:
- Dynamic Language Detection (Hindi vs English)
- PURE DIALOGUE ONLY: Zero XML tags, zero voice description, zero assistant preambles
- Ultra-Low Latency Streaming Audio Pipeline (synthesizes sentence-by-sentence while Gemini streams)
- Clean number/pricing pronunciation ('₹10k' -> 'दस हज़ार रुपये' / 'ten thousand rupees')
- Clean deliverable counts ('1 Reel' -> 'एक Reel', '2 Stories' -> 'दो Stories')
- Natural conversational pacing (+16% speed default)
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


def detect_language(text: str) -> str:
    """Detects whether text is Hindi/Hinglish or English."""
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    matches = words.intersection(HINDI_HINTS)
    return 'hindi' if len(matches) >= 1 else 'english'


class StreamingVoicePipeline:
    """
    Pipelined audio streaming engine.
    Synthesizes sentence-by-sentence in a background thread and plays via afplay
    while Gemini is still streaming tokens or while user starts typing!
    Ensures ZERO XML tags and ZERO voice descriptions are ever vocalized.
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
            cleaned = self.speaker.clean_text_for_speech(text, lang=lang)
            if not cleaned:
                self.sentence_queue.task_done()
                continue

            try:
                if self.speaker.mode == "kitten":
                    k = self.speaker._get_kitten()
                    if k:
                        import soundfile as sf
                        audio = k.generate(cleaned, voice="expr-voice-2-m", speed=1.2)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio, 24000)
                        with self._lock:
                            self.temp_files.append(temp_path)
                        self.audio_queue.put(temp_path)
                else:
                    voice_name = self.speaker._resolve_voice_name(lang)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_path = f.name

                    # Pass PURE CLEAN TEXT to edge_tts (zero XML tags)
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

    def clean_candidate(self, text: str) -> str:
        """Strips assistant preambles, XML artifacts, and numbered lists so only true spoken dialogue is voiced."""
        # Strip any XML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        # Strip assistant preambles like "Sure, here is:", "Aarav ke liye ye options hain:", "Here are some options:"
        text = re.sub(r'^(?:sure,?\s*(?:here (?:is|are)[^:\n]*:)?|here (?:is|\'s)[^:\n]*:|[\w\s]+ ke liye (?:aap )?inmein se[^:\n]*:|inmein se koi bhi[^:\n]*:)\s*', '', text, flags=re.IGNORECASE)
        # Strip leading numbers like '1. ', '2. ', 'Option 1:'
        text = re.sub(r'^(?:\d+[\.\)]|option\s*\d+:?)\s*[\"\']?', '', text, flags=re.IGNORECASE)
        text = text.strip().strip('"\'*')
        # If phrase is assistant meta-advice rather than dialogue, drop it
        if re.search(r'\b(inmein se|choose one|sabse accha lage|copy karke bhej)\b', text, flags=re.IGNORECASE):
            return ""
        return text

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

            cleaned = self.clean_candidate(candidate)
            if cleaned:
                self.sentence_queue.put(cleaned)
            self.buffer = self.buffer[end_pos:]

    def finish(self, wait: bool = False):
        """Flushes remaining text buffer and signals completion."""
        if self.buffer.strip():
            cleaned = self.clean_candidate(self.buffer.strip())
            if cleaned:
                self.sentence_queue.put(cleaned)
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
        Refined Voices:
        - 'auto': Mirrors language (Swara for Hindi, Neerja for English)
        - 'neerja': en-IN-NeerjaNeural (Top Indian Female Voice - Highly Recommended)
        - 'swara': hi-IN-SwaraNeural (Soft, Natural Hindi Female Voice)
        - 'madhur': hi-IN-MadhurNeural (Clean Hindi Male Voice)
        - 'prabhat': en-IN-PrabhatNeural (Indian English Male Voice)
        """
        self.mode = mode
        self.rate = rate  # +12% gives crisp, human-paced speech without rushing
        self.enabled = enabled
        self._current_process = None
        self._active_pipeline = None

    def _resolve_voice_name(self, lang: str) -> str:
        if self.mode in ["neerja"]:
            return "en-IN-NeerjaNeural"
        elif self.mode in ["swara", "hindi_female"]:
            return "hi-IN-SwaraNeural"
        elif self.mode in ["madhur", "hindi_male"]:
            return "hi-IN-MadhurNeural"
        elif self.mode in ["prabhat", "english_male"]:
            return "en-IN-PrabhatNeural"
        elif self.mode == "auto":
            return "hi-IN-SwaraNeural" if lang == "hindi" else "en-IN-NeerjaNeural"
        else:
            return "hi-IN-SwaraNeural" if lang == "hindi" else "en-IN-NeerjaNeural"

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
        """
        Cleans text for audio synthesis:
        - Removes all XML/HTML tags
        - Removes markdown formatting
        - Expands numbers & pricing ('₹10k' -> 'दस हजार रुपये' in Hindi / 'ten thousand rupees' in English)
        - Preserves Devanagari Unicode
        """
        # Strip ANY xml/html tags completely
        t = re.sub(r'<[^>]+>', ' ', text)
        t = re.sub(r'[*_~`#\[\]]', '', t)

        # Human fillers smoothing
        t = re.sub(r'\b(hmm+|haan|acha|accha|dekho|arey|well|got it)\.\.\.', r'\1, ', t, flags=re.IGNORECASE)

        if lang == "hindi":
            # Devanagari terms mapping for native, buttery smooth Hindi TTS
            hindi_phonetic_map = {
                r'\breels?\b': 'रील',
                r'\bstor(?:y|ies)\b': 'स्टोरीज़',
                r'\bposts?\b': 'पोस्ट',
                r'\bvideos?\b': 'वीडियो',
                r'\bshorts?\b': 'शॉर्ट्स',
                r'\bcollab(?:oration)?\b': 'कोलैब',
                r'\bconfirm\b': 'कन्फर्म',
                r'\bbudget\b': 'बजट',
                r'\bdeal\b': 'डील',
                r'\btotal\b': 'टोटल',
                r'\bpackage\b': 'पैकेज',
                r'\bhaz[aā]+r\b': 'हजार',
                r'\bhaan\b': 'हाँ',
                r'\barey\b': 'अरे',
                r'\bhmm+\b': 'हम्म',
                r'\bachha\b|\bacha\b': 'अच्छा',
                r'\btheek\b': 'ठीक',
                r'\bperfect\b': 'परफेक्ट',
            }

            # Expand 10k, ₹10k, 10,000 -> दस हजार रुपये (using clean 'हजार' without nukta)
            # Avoid repeating 'रुपये' if user/agent already wrote 'rupaye' or 'rupees' right after
            def replace_k_hi(m):
                num = m.group(1)
                word = HINDI_NUMBERS.get(num, num)
                return f" {word} हजार रुपये "

            # Match 10k with optional trailing rupaye/rupees
            t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK](?:\s*(?:rupaye?|rupees|रुपये))?\b', replace_k_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000(?:\s*(?:rupaye?|rupees|रुपये))?\b', replace_k_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)000(?:\s*(?:rupaye?|rupees|रुपये))?\b', replace_k_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)(?:\s*(?:rupaye?|rupees|रुपये))?\b', r' \1 रुपये ', t, flags=re.IGNORECASE)
            t = t.replace('₹', 'रुपये')
            # Clean any double 'रुपये'
            t = re.sub(r'(?:रुपये\s*){2,}', 'रुपये ', t)

            # Deliverables count: 1 Reel -> एक रील, 2 Stories -> दो स्टोरीज़
            def replace_deliverables_hi(m):
                cnt = m.group(1)
                item = m.group(2).lower()
                w = HINDI_NUMBERS.get(cnt, cnt)
                if 'reel' in item:
                    item_hi = 'रील'
                elif 'stor' in item:
                    item_hi = 'स्टोरीज़'
                elif 'post' in item:
                    item_hi = 'पोस्ट'
                elif 'short' in item:
                    item_hi = 'शॉर्ट्स'
                elif 'video' in item:
                    item_hi = 'वीडियो'
                else:
                    item_hi = item
                return f"{w} {item_hi}"

            t = re.sub(r'\b(\d+)\s*(Reels?|Story|Stories|Posts?|Shorts?|Videos?)\b', replace_deliverables_hi, t, flags=re.IGNORECASE)
            t = re.sub(r'(\d+)(?:st|nd|rd|th)\b', r'\1 तारीख', t, flags=re.IGNORECASE)

            # Apply conversational phonetic map for Hindi voices
            for pattern, replacement in hindi_phonetic_map.items():
                t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        else:
            def replace_k_en(m):
                num = m.group(1)
                word = ENG_NUMBERS.get(num, num)
                return f" {word} thousand rupees "

            t = re.sub(r'(?:INR|₹)?\s*(\d+)\s*[kK](?:\s*(?:rupaye?|rupees))?\b', replace_k_en, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)[,.]000(?:\s*(?:rupaye?|rupees))?\b', replace_k_en, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)000(?:\s*(?:rupaye?|rupees))?\b', replace_k_en, t, flags=re.IGNORECASE)
            t = re.sub(r'(?:INR|₹)\s*(\d+)(?:\s*(?:rupaye?|rupees))?\b', r' \1 rupees ', t, flags=re.IGNORECASE)
            t = t.replace('₹', 'rupees')
            t = re.sub(r'(?:rupees\s*){2,}', 'rupees ', t, flags=re.IGNORECASE)

            def replace_deliverables_en(m):
                cnt = m.group(1)
                item = m.group(2)
                w = ENG_NUMBERS.get(cnt, cnt)
                return f"{w} {item}"

            t = re.sub(r'\b(\d+)\s*(Reels?|Story|Stories|Posts?|Shorts?|Videos?)\b', replace_deliverables_en, t, flags=re.IGNORECASE)

        # Preserve Devanagari Unicode
        t = re.sub(r'[^\w\s.,?!:;\-\u0900-\u097F]', ' ', t)
        return " ".join(t.split()).strip()

    def speak(self, text: str, blocking: bool = False):
        """Fallback one-shot speaker for static messages."""
        if not self.enabled:
            return

        lang = detect_language(text)
        cleaned = self.clean_text_for_speech(text, lang=lang)
        if not cleaned:
            return

        def _worker():
            try:
                self.stop()
                if self.mode == "kitten":
                    k = self._get_kitten()
                    if k:
                        import soundfile as sf
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
