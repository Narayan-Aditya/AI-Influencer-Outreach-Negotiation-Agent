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
import numpy as np
import scipy.signal as signal
import soundfile as sf


def create_ambiance(num_samples: int, sr: int = 24000, ambiance_type: str = "ac_fan", level_db: float = -32.0) -> np.ndarray:
    """
    Synthesizes ultra-realistic room tone presence to mask digital TTS silence:
    - 'ac_fan' / 'ac': Split AC room tone (smooth air flow + subtle compressor hum + presence)
    - 'fan': Ceiling/Desk fan (airflow + rhythmic blade pulsation + motor hum)
    - 'crowd': Distant coffee shop / office crowd murmur + ambient room air
    """
    white = np.random.randn(num_samples)
    t = np.arange(num_samples) / sr

    if ambiance_type == "fan":
        # Ceiling/Desk fan: airflow + rhythmic blade pulsation (~4.5 Hz) + 50Hz hum
        sos = signal.butter(3, [60, 450], btype="bandpass", fs=sr, output="sos")
        air = signal.sosfilt(sos, white)
        air = air / (np.max(np.abs(air)) + 1e-9)
        mod = 1.0 + 0.25 * np.sin(2 * np.pi * 4.5 * t)
        air = air * mod
        hum = 0.35 * np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 100 * t)
        amb = 0.7 * air + 0.3 * hum
    elif ambiance_type == "crowd":
        # Distant coffee shop / office crowd murmur: bandpass filtered babble + room resonance
        sos1 = signal.butter(3, [200, 1800], btype="bandpass", fs=sr, output="sos")
        murmur = signal.sosfilt(sos1, white)
        env = np.abs(signal.sosfilt(signal.butter(2, 1.5, fs=sr, output="sos"), np.random.randn(num_samples)))
        env = env / (np.max(env) + 1e-9)
        murmur = murmur * (0.5 + 0.5 * env)
        murmur = murmur / (np.max(np.abs(murmur)) + 1e-9)

        sos_room = signal.butter(2, [80, 500], btype="bandpass", fs=sr, output="sos")
        room = signal.sosfilt(sos_room, white)
        room = room / (np.max(np.abs(room)) + 1e-9)
        amb = 0.6 * murmur + 0.4 * room
    else:  # 'ac_fan' or 'ac'
        # Split AC room tone: smooth air rush + low compressor drone + faint presence
        sos_ac = signal.butter(3, [70, 550], btype="bandpass", fs=sr, output="sos")
        ac_rush = signal.sosfilt(sos_ac, white)
        ac_rush = ac_rush / (np.max(np.abs(ac_rush)) + 1e-9)
        drone = 0.4 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)
        sos_hiss = signal.butter(2, [1200, 3200], btype="bandpass", fs=sr, output="sos")
        hiss = 0.1 * signal.sosfilt(sos_hiss, white)
        hiss = hiss / (np.max(np.abs(hiss)) + 1e-9)
        amb = 0.75 * ac_rush + 0.18 * drone + 0.07 * hiss

    amb = amb / (np.max(np.abs(amb)) + 1e-9)
    amplitude = 10.0 ** (level_db / 20.0)
    amb = amb * amplitude

    # Smooth edges fade (40ms)
    fade = int(min(sr * 0.04, num_samples / 10))
    if fade > 0:
        c = np.linspace(0, 1, fade)
        amb[:fade] *= c
        amb[-fade:] *= c[::-1]
    return amb


def apply_phone_call_filter(data: np.ndarray, sr: int = 24000) -> np.ndarray:
    """
    Applies real phone call / WhatsApp voice note acoustics (AMR-WB HD Voice + MEMS Mic):
    - Cuts unnatural studio sub-bass below 180 Hz
    - Low-pass filters frequencies above 7000 Hz to simulate phone telecom bandwidth
    - Adds mid-frequency smartphone capsule presence around 2.4 kHz
    - Adds subtle warm harmonic saturation typical of smartphone microphones
    """
    if len(data.shape) > 1:
        data = data[:, 0]
    # 1. High-pass filter at 180 Hz
    sos_hp = signal.butter(2, 180, btype="highpass", fs=sr, output="sos")
    f = signal.sosfilt(sos_hp, data)
    # 2. Low-pass filter at 7000 Hz (AMR-WB telecom bandwidth)
    sos_lp = signal.butter(2, 7000, btype="lowpass", fs=sr, output="sos")
    f = signal.sosfilt(sos_lp, f)
    # 3. Smartphone microphone capsule presence (2.4 kHz boost)
    b, a = signal.iirpeak(2400, 1.5, fs=sr)
    boosted = signal.lfilter(b, a, f)
    f = 0.82 * f + 0.18 * boosted
    # 4. MEMS mic gentle harmonic warmth
    f = np.tanh(f * 1.05) / 1.05
    return f


def generate_micro_breath(sr: int = 24000, duration_sec: float = 0.16, level_db: float = -31.0) -> np.ndarray:
    """
    Synthesizes a realistic biological micro-inhale before speech:
    - Shaped oral cavity aspiration noise (800Hz - 4200Hz)
    - Natural asymmetric inhalation envelope (gradual air intake, quick taper into phonation)
    """
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, 1, num_samples)
    noise = np.random.randn(num_samples)
    sos = signal.butter(3, [800, 4200], btype="bandpass", fs=sr, output="sos")
    aspiration = signal.sosfilt(sos, noise)
    b, a = signal.iirpeak(2100, 2.0, fs=sr)
    peak_res = signal.lfilter(b, a, aspiration)
    aspiration = 0.75 * aspiration + 0.25 * peak_res
    aspiration = aspiration / (np.max(np.abs(aspiration)) + 1e-9)
    env = (t ** 1.6) * ((1.0 - t) ** 0.5)
    env = env / (np.max(env) + 1e-9)
    breath = aspiration * env
    amp = 10.0 ** (level_db / 20.0)
    return breath * amp


def mix_audio_with_ambiance(
    file_path: str,
    ambiance_type: str = "ac_fan",
    level_db: float = -32.0,
    phone_filter: bool = True,
    add_breath: bool = False,
    post_pause_sec: float = 0.0
) -> str:
    """Mixes subtle background acoustic room tone, optional micro-breath, post-pause, and phone call mic filter with speech audio."""
    try:
        data, sr = sf.read(file_path)
        if len(data.shape) > 1:
            data = data[:, 0]

        # Prepend subtle biological inhale before speech turn
        if add_breath:
            breath = generate_micro_breath(sr=sr, duration_sec=0.16, level_db=-31.0)
            gap = np.zeros(int(sr * 0.035))  # 35ms natural pre-vocal pause
            data = np.concatenate([breath, gap, data])

        # Append post-pause for deliberate consideration after pricing/numbers
        if post_pause_sec > 0:
            pause = np.zeros(int(sr * post_pause_sec))
            data = np.concatenate([data, pause])

        # Apply phone call / WhatsApp microphone acoustic filter if enabled
        if phone_filter:
            data = apply_phone_call_filter(data, sr=sr)

        # Mix background room tone
        if ambiance_type and ambiance_type != "off":
            amb = create_ambiance(len(data), sr=sr, ambiance_type=ambiance_type, level_db=level_db)
            data = data + amb

        peak = np.max(np.abs(data))
        if peak > 0.96:
            data = (data / peak) * 0.96

        out_path = file_path.rsplit(".", 1)[0] + "_proc.wav"
        sf.write(out_path, data, sr)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        return out_path
    except Exception:
        return file_path



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
        self.is_first_chunk = True
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

                    # Determine dynamic conversational pacing
                    chunk_rate = self.speaker.get_dynamic_rate(cleaned)
                    has_deal_terms = bool(re.search(r'\b(हजार|रुपये|thousand|rupees|budget|deal|reel|stories|post|₹|\d+k?)\b', cleaned, flags=re.IGNORECASE))
                    post_pause = 0.22 if has_deal_terms else 0.0

                    # Pass PURE CLEAN TEXT to edge_tts with dynamic rate and lowered pitch (-2Hz)
                    communicate = edge_tts.Communicate(
                        cleaned,
                        voice=voice_name,
                        rate=chunk_rate,
                        pitch=self.speaker.pitch
                    )
                    asyncio.run(communicate.save(temp_path))

                    # Prepend biological micro-breath on first chunk of turn
                    should_add_breath = self.is_first_chunk and self.speaker.breath_enabled
                    if self.is_first_chunk:
                        self.is_first_chunk = False

                    # Apply real phone mic / WhatsApp audio note filter & room ambiance
                    temp_path = mix_audio_with_ambiance(
                        temp_path,
                        ambiance_type=self.speaker.ambiance,
                        level_db=self.speaker.ambiance_level_db,
                        phone_filter=self.speaker.phone_filter,
                        add_breath=should_add_breath,
                        post_pause_sec=post_pause
                    )

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
    def __init__(
        self,
        mode: str = "auto",
        rate: str = "+12%",
        pitch: str = "-2Hz",
        ambiance: str = "ac_fan",
        ambiance_level_db: float = -32.0,
        phone_filter: bool = True,
        breath_enabled: bool = True,
        enabled: bool = True
    ):
        """
        Refined Voices:
        - 'auto': Mirrors language (Swara for Hindi, Neerja for English)
        - 'neerja': en-IN-NeerjaNeural (Top Indian Female Voice - Highly Recommended)
        - 'swara': hi-IN-SwaraNeural (Soft, Natural Hindi Female Voice)
        - 'madhur': hi-IN-MadhurNeural (Clean Hindi Male Voice)
        - 'prabhat': en-IN-PrabhatNeural (Indian English Male Voice)

        Acoustic Presence:
        - pitch: '-2Hz' (1-2 points lower for warm, non-digital chest resonance)
        - ambiance: 'ac_fan' (subtle AC/fan room tone masking digital silence), 'fan', 'crowd', or 'off'
        - ambiance_level_db: -32.0 dB (ultra-subtle, natural background presence)
        - phone_filter: True (AMR-WB HD Voice + MEMS phone capsule filter for realistic phone calls)
        - breath_enabled: True (Subtle biological oral/tracheal inhale cue before speech turn)
        """
        self.mode = mode
        self.rate = rate  # +12% gives crisp, human-paced speech without rushing
        self.pitch = pitch  # -2Hz gives warm, grounded human tone
        self.ambiance = ambiance
        self.ambiance_level_db = ambiance_level_db
        self.phone_filter = phone_filter
        self.breath_enabled = breath_enabled
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

        # Human fillers smoothing: keep '...' after thinking fillers for 250ms hesitation pause
        t = re.sub(r'\b(hmm+|haan|acha|accha|dekho|dekhiye|suno|arey|well|wait|got it)\s*[.]{2,}', r'\1... ', t, flags=re.IGNORECASE)

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
                r'\bdekhiye\b': 'देखिए',
                r'\bdekho\b': 'देखो',
                r'\bsuno\b': 'सुनो',
                r'\bactually\b': 'एक्चुअली',
                r'\bwait\b': 'वेट',
                r'\bwell\b': 'वेल',
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

        # Natural conversational cadence around pricing & deliverables
        if lang == "hindi":
            # Add breath comma after deliverable and after currency terms
            t = re.sub(r'(\b(?:एक|दो|तीन|चार|पाँच|\d+)\s*(?:रील|स्टोरीज़|पोस्ट|वीडियो|शॉर्ट्स))\s+(?:ka|ke|ki|का|के|की)\s+', r'\1, ', t, flags=re.IGNORECASE)
            t = re.sub(r'(हजार रुपये|रुपये)\s+(aur|मिला के|टोटल|confirm|theek|kaisa)', r'\1, \2', t, flags=re.IGNORECASE)
        else:
            t = re.sub(r'(\b(?:one|two|three|four|five|\d+)\s*(?:Reels?|Stor(?:y|ies)|Posts?|Videos?))\s+(?:for|that\'s|is)\s+', r'\1, ', t, flags=re.IGNORECASE)
            t = re.sub(r'(thousand rupees|rupees)\s+(and|total|makes|confirm|deal)', r'\1, \2', t, flags=re.IGNORECASE)

        # Preserve Devanagari Unicode and conversational apostrophes (that's, don't, let's)
        t = re.sub(r"[^\w\s.,?!:;\-'\u0900-\u097F]", ' ', t)
        # Clean double commas and spaces before punctuation
        t = re.sub(r',\s*,+', ', ', t)
        t = re.sub(r'\s+([.,?!:;])', r'\1', t)
        return " ".join(t.split()).strip()

    def get_dynamic_rate(self, text: str) -> str:
        """
        Calculates dynamic speech rate:
        - Numbers, pricing & deliverables: Deliberate +6% pace for crisp negotiation clarity.
        - Conversational greetings & general dialogue: Standard brisk conversational rate (+12%).
        """
        has_deal_terms = re.search(r'\b(हजार|रुपये|thousand|rupees|budget|deal|reel|stories|post|₹|\d+k?)\b', text, flags=re.IGNORECASE)
        if has_deal_terms:
            return "+6%"
        return self.rate

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

                    chunk_rate = self.get_dynamic_rate(cleaned)
                    has_deal_terms = bool(re.search(r'\b(हजार|रुपये|thousand|rupees|budget|deal|reel|stories|post|₹|\d+k?)\b', cleaned, flags=re.IGNORECASE))
                    post_pause = 0.22 if has_deal_terms else 0.0

                    communicate = edge_tts.Communicate(
                        cleaned,
                        voice=voice_name,
                        rate=chunk_rate,
                        pitch=self.pitch
                    )
                    asyncio.run(communicate.save(temp_path))

                    temp_path = mix_audio_with_ambiance(
                        temp_path,
                        ambiance_type=self.ambiance,
                        level_db=self.ambiance_level_db,
                        phone_filter=self.phone_filter,
                        add_breath=self.breath_enabled,
                        post_pause_sec=post_pause
                    )

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

    def set_pitch(self, pitch: str):
        self.pitch = pitch

    def set_ambiance(self, ambiance: str, level_db: float = -32.0):
        self.ambiance = ambiance
        self.ambiance_level_db = level_db

    def set_phone_filter(self, enabled: bool):
        self.phone_filter = enabled

    def set_breath(self, enabled: bool):
        self.breath_enabled = enabled
