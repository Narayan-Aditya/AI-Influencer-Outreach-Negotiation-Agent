#!/usr/bin/env python3
"""
Refined Voice Player (with Room Ambiance & Warmer Lowered Pitch):
Plays the approved voice personas with acoustic background sound (AC/fan/crowd) and -2Hz pitch:
  1. Neerja - Indian Female (Clear, Natural & Professional)
  2. Swara - Hindi Female (Soft, Expressive Devanagari)
  3. Madhur - Hindi Male (Natural Business Tone)
  4. Prabhat - Indian English Male (Friendly, Relaxed Pacing)

Usage:
  ./venv/bin/python3 play_voices.py               # Plays all 4 voices sequentially
  ./venv/bin/python3 play_voices.py 1             # Plays Neerja (with AC/room ambiance)
  ./venv/bin/python3 play_voices.py 2             # Plays Swara (with AC/room ambiance)
  ./venv/bin/python3 play_voices.py 3             # Plays Madhur (with AC/room ambiance)
  ./venv/bin/python3 play_voices.py 4             # Plays Prabhat (with AC/room ambiance)

Change Ambiance on the fly:
  ./venv/bin/python3 play_voices.py 1 fan         # Play Neerja with ceiling fan sound
  ./venv/bin/python3 play_voices.py 1 crowd       # Play Neerja with subtle office/cafe murmur
  ./venv/bin/python3 play_voices.py 1 ac          # Play Neerja with AC room sound
  ./venv/bin/python3 play_voices.py 1 off         # Play Neerja without background sound
"""

import sys
import os
import time
import tempfile
import asyncio
import subprocess
import edge_tts
from voice_output import VoiceSpeaker, mix_audio_with_ambiance

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
        "rate": "+12%",
        "pitch": "-2Hz"
    },
    {
        "id": "2",
        "name": "Swara (Hindi Female - Soft, Expressive Devanagari)",
        "voice": "hi-IN-SwaraNeural",
        "raw_text": HINDI_RAW,
        "lang": "hindi",
        "rate": "+12%",
        "pitch": "-2Hz"
    },
    {
        "id": "3",
        "name": "Madhur (Hindi Male - Natural Business Tone)",
        "voice": "hi-IN-MadhurNeural",
        "raw_text": HINDI_RAW,
        "lang": "hindi",
        "rate": "+12%",
        "pitch": "-2Hz"
    },
    {
        "id": "4",
        "name": "Prabhat (Indian English Male - Friendly & Relaxed)",
        "voice": "en-IN-PrabhatNeural",
        "raw_text": ENGLISH_RAW,
        "lang": "english",
        "rate": "+10%",
        "pitch": "-2Hz"
    }
]


async def generate_and_play(v, ambiance_type="ac_fan"):
    cleaned = speaker_helper.clean_text_for_speech(v['raw_text'], lang=v['lang'])
    print("\n" + "=" * 65)
    print(f"▶️ PLAYING [{v['id']}] {v['name']}")
    print(f"   Speech:   \"{cleaned}\"")
    print(f"   Pitch:    {v['pitch']} (1 point lowered for warm chest resonance)")
    print(f"   Rate:     {v['rate']}")
    print(f"   Ambiance: {ambiance_type.upper()} (Natural room acoustic sound)")
    print("=" * 65)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        c = edge_tts.Communicate(cleaned, voice=v['voice'], rate=v['rate'], pitch=v['pitch'])
        await c.save(temp_path)

        play_path = temp_path
        if ambiance_type != "off":
            play_path = mix_audio_with_ambiance(temp_path, ambiance_type=ambiance_type, level_db=-31.0)

        subprocess.run(["afplay", play_path])
        if os.path.exists(play_path):
            os.remove(play_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def main():
    target = sys.argv[1].strip() if len(sys.argv) > 1 else None
    ambiance = "ac_fan"

    # Support ambiance flag as 2nd arg (e.g. ./play_voices.py 1 crowd)
    if len(sys.argv) > 2:
        arg2 = sys.argv[2].strip().lower()
        if arg2 in ["ac", "ac_fan"]:
            ambiance = "ac_fan"
        elif arg2 in ["fan", "ceiling_fan"]:
            ambiance = "fan"
        elif arg2 in ["crowd", "office", "cafe"]:
            ambiance = "crowd"
        elif arg2 in ["off", "none", "silent"]:
            ambiance = "off"

    if target:
        selected = [v for v in VOICE_LIST if v['id'] == target]
        if not selected:
            print(f"Voice ID '{target}' not found. Choose 1 to 4.")
            return
        await generate_and_play(selected[0], ambiance_type=ambiance)
    else:
        print("\n🎧 PLAYING REFINED 4 VOICES (WITH ROOM AMBIANCE & LOWERED PITCH):")
        for v in VOICE_LIST:
            await generate_and_play(v, ambiance_type=ambiance)
            time.sleep(1.0)
        print("\n✅ Finished playing refined voices!")


if __name__ == "__main__":
    asyncio.run(main())
