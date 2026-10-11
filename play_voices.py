#!/usr/bin/env python3
"""
Refined Voice Player:
Plays the approved high-quality voice personas with full conversational phonetics:
  1. Neerja - Indian Female (Clear, Natural & Professional) - 90%+ Ready
  2. Swara - Hindi Female (Soft, Expressive & Conversational) - Boosted with Devanagari
  3. Madhur - Hindi Male (Clean, Business Casual Tone) - Boosted with Devanagari
  4. Prabhat - Indian English Male (Friendly, Relaxed Pacing)

Usage:
  ./venv/bin/python3 play_voices.py       # Plays all 4 voices
  ./venv/bin/python3 play_voices.py 1     # Plays Neerja
  ./venv/bin/python3 play_voices.py 2     # Plays Swara
  ./venv/bin/python3 play_voices.py 3     # Plays Madhur
  ./venv/bin/python3 play_voices.py 4     # Plays Prabhat
"""

import sys
import os
import time
import tempfile
import asyncio
import subprocess
import edge_tts
from voice_output import VoiceSpeaker

speaker_helper = VoiceSpeaker()

HINDI_RAW = "Arey haan Aarav! 1 Reel ka ₹10k aur 2 Stories ka ₹6k mila ke total ₹16k confirm hai na? Deal theek hai toh batana."
ENGLISH_RAW = "Hey Aarav! Aman here from Bolt Audio. So for 1 Reel that's ₹10k, and for 2 Stories ₹6k. Total comes to ₹16k, right? Deal confirmed if this works for you."

VOICE_LIST = [
    {
        "id": "1",
        "name": "Neerja (Indian Female - Clear & Conversational)",
        "voice": "en-IN-NeerjaNeural",
        "raw_text": ENGLISH_RAW,
        "lang": "english",
        "rate": "+12%"
    },
    {
        "id": "2",
        "name": "Swara (Hindi Female - Soft, Expressive Devanagari)",
        "voice": "hi-IN-SwaraNeural",
        "raw_text": HINDI_RAW,
        "lang": "hindi",
        "rate": "+12%"
    },
    {
        "id": "3",
        "name": "Madhur (Hindi Male - Natural Business Tone)",
        "voice": "hi-IN-MadhurNeural",
        "raw_text": HINDI_RAW,
        "lang": "hindi",
        "rate": "+12%"
    },
    {
        "id": "4",
        "name": "Prabhat (Indian English Male - Friendly & Relaxed)",
        "voice": "en-IN-PrabhatNeural",
        "raw_text": ENGLISH_RAW,
        "lang": "english",
        "rate": "+10%"
    }
]


async def generate_and_play(v):
    cleaned = speaker_helper.clean_text_for_speech(v['raw_text'], lang=v['lang'])
    print("\n" + "=" * 60)
    print(f"▶️ PLAYING [{v['id']}] {v['name']}")
    print(f"   Original: \"{v['raw_text']}\"")
    print(f"   Phonetic: \"{cleaned}\"")
    print(f"   Rate:     {v['rate']}")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        c = edge_tts.Communicate(cleaned, voice=v['voice'], rate=v['rate'])
        await c.save(temp_path)
        subprocess.run(["afplay", temp_path])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def main():
    target = sys.argv[1].strip() if len(sys.argv) > 1 else None

    if target:
        selected = [v for v in VOICE_LIST if v['id'] == target]
        if not selected:
            print(f"Voice ID '{target}' not found. Choose 1 to 4.")
            return
        await generate_and_play(selected[0])
    else:
        print("\n🎧 PLAYING REFINED 4 VOICES (WITH 90%+ PHONETIC UPGRADES):")
        for v in VOICE_LIST:
            await generate_and_play(v)
            time.sleep(1.0)
        print("\n✅ Finished playing refined voices!")


if __name__ == "__main__":
    asyncio.run(main())
