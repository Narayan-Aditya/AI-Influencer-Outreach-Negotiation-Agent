"""
Voice output engine supporting Clean Native Hindi/Hinglish (Neural) and KittenML TTS.
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


class VoiceSpeaker:
    def __init__(self, mode: str = "hindi_male", speed: float = 1.0, enabled: bool = True):
        """
        Modes:
        - 'hindi_male': hi-IN-MadhurNeural (Natural clean Hindi/Hinglish male voice)
        - 'hindi_female': hi-IN-SwaraNeural (Natural clean Hindi/Hinglish female voice)
        - 'kitten': KittenTTS Nano (English American experimental)
        """
        self.mode = mode
        self.speed = speed
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

    def clean_text_for_speech(self, text: str) -> str:
        """Strips markdown asterisks, brackets, and prepares natural Hindi speech text."""
        t = re.sub(r'[*_~`#\[\]]', '', text)
        # Currency: INR 16,000 or ₹16,000 -> 16 हजार रुपये
        t = re.sub(r'(?:INR|₹)\s*(\d+),000', r'\1 हजार रुपये', t, flags=re.IGNORECASE)
        t = re.sub(r'(?:INR|₹)\s*(\d+)', r'\1 रुपये', t, flags=re.IGNORECASE)
        # Suffix k: 5k -> 5 हजार
        t = re.sub(r'(\d+)k\b', r'\1 हजार', t, flags=re.IGNORECASE)
        # Dates: 25th -> 25 तारीख
        t = re.sub(r'(\d+)(?:st|nd|rd|th)\b', r'\1 तारीख', t, flags=re.IGNORECASE)
        # Clean special chars / emojis
        t = re.sub(r'[^\w\s.,?!:;\-]', ' ', t)
        return " ".join(t.split()).strip()

    def speak(self, text: str, blocking: bool = False):
        if not self.enabled:
            return

        cleaned = self.clean_text_for_speech(text)
        if not cleaned:
            return

        def _worker():
            try:
                # Stop any currently playing audio
                if self._current_process and self._current_process.poll() is None:
                    self._current_process.terminate()

                if self.mode == "kitten":
                    # KittenTTS Nano fallback
                    k = self._get_kitten()
                    if k:
                        import soundfile as sf
                        audio = k.generate(cleaned, voice="expr-voice-2-m")
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio, 24000)
                        self._current_process = subprocess.Popen(["afplay", temp_path])
                        self._current_process.wait()
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                else:
                    # Native Clean Indian Hindi / Hinglish Voice (hi-IN-MadhurNeural)
                    voice_name = "hi-IN-MadhurNeural" if self.mode == "hindi_male" else "hi-IN-SwaraNeural"
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_path = f.name

                    communicate = edge_tts.Communicate(cleaned, voice=voice_name)
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
