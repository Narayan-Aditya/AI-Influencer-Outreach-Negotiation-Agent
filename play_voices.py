#!/usr/bin/env python3
"""
Refined Voice Player (Feature 1: Real Phone Call / WhatsApp Voice Note Filter):
Plays the approved voice personas with acoustic phone mic filter, room ambiance, and -2Hz pitch:
  1. Neerja - Indian Female (Clear, Natural & Professional)
  2. Swara - Hindi Female (Soft, Expressive Devanagari)
  3. Madhur - Hindi Male (Natural Business Tone)
  4. Prabhat - Indian English Male (Friendly, Relaxed Pacing)

Usage:
  ./venv/bin/python3 play_voices.py               # Plays all 4 voices (Phone Call mode)
  ./venv/bin/python3 play_voices.py 1             # Plays Neerja (Phone Call mode)
  ./venv/bin/python3 play_voices.py 2             # Plays Swara (Phone Call mode)
  ./venv/bin/python3 play_voices.py 3             # Plays Madhur (Phone Call mode)
  ./venv/bin/python3 play_voices.py 4             # Plays Prabhat (Phone Call mode)

Side-by-Side Comparison (Studio vs Phone Call):
  ./venv/bin/python3 play_voices.py 1 compare     # Plays Neerja Studio first, then Phone Call
  ./venv/bin/python3 play_voices.py 2 compare     # Plays Swara Studio first, then Phone Call

Ambiance Modes:
  ./venv/bin/python3 play_voices.py 1 studio      # Raw Studio mode (no filter, no ambiance)
  ./venv/bin/python3 play_voices.py 1 fan         # Phone Call with Ceiling Fan sound
  ./venv/bin/python3 play_voices.py 1 crowd       # Phone Call with Office/Cafe murmur
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


async def generate_and_play(v, ambiance_type="ac_fan", phone_filter=True, tag=""):
    cleaned = speaker_helper.clean_text_for_speech(v['raw_text'], lang=v['lang'])
    mode_desc = "PHONE CALL / WHATSAPP FILTER ON" if phone_filter else "STUDIO MODE (Raw Digital)"
    if tag:
        mode_desc = f"{tag} - {mode_desc}"

    print("\n" + "=" * 65)
    print(f"▶️ PLAYING [{v['id']}] {v['name']}")
    print(f"   Mode:     {mode_desc}")
    print(f"   Speech:   \"{cleaned}\"")
    print(f"   Pitch:    {v['pitch']} (Warm chest resonance)")
    print(f"   Ambiance: {ambiance_type.upper()}")
    print("=" * 65)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        c = edge_tts.Communicate(cleaned, voice=v['voice'], rate=v['rate'], pitch=v['pitch'])
        await c.save(temp_path)

        play_path = mix_audio_with_ambiance(
            temp_path,
            ambiance_type=ambiance_type,
            level_db=-31.0,
            phone_filter=phone_filter
        )

        subprocess.run(["afplay", play_path])
        if os.path.exists(play_path):
            os.remove(play_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def main():
    target = sys.argv[1].strip() if len(sys.argv) > 1 else None
    mode = sys.argv[2].strip().lower() if len(sys.argv) > 2 else "phone"

    ambiance = "ac_fan"
    phone_filter = True

    if mode == "studio":
        phone_filter = False
        ambiance = "off"
    elif mode in ["fan", "ceiling_fan"]:
        ambiance = "fan"
        phone_filter = True
    elif mode in ["crowd", "office"]:
        ambiance = "crowd"
        phone_filter = True
    elif mode in ["ac", "phone"]:
        ambiance = "ac_fan"
        phone_filter = True

    if target:
        selected = [v for v in VOICE_LIST if v['id'] == target]
        if not selected:
            print(f"Voice ID '{target}' not found. Choose 1 to 4.")
            return

        item = selected[0]
        if mode == "compare":
            print(f"\n🔄 RUNNING SIDE-BY-SIDE COMPARISON FOR [{item['name']}]:")
            print("1️⃣ Playing Studio Mode first...")
            await generate_and_play(item, ambiance_type="off", phone_filter=False, tag="1/2 STUDIO")
            time.sleep(1.2)
            print("\n2️⃣ Now playing Phone Call / WhatsApp Filter Mode...")
            await generate_and_play(item, ambiance_type="ac_fan", phone_filter=True, tag="2/2 PHONE CALL")
        else:
            await generate_and_play(item, ambiance_type=ambiance, phone_filter=phone_filter)
    else:
        print("\n🎧 PLAYING REFINED 4 VOICES (WITH FEATURE 1: PHONE CALL FILTER):")
        for v in VOICE_LIST:
            await generate_and_play(v, ambiance_type=ambiance, phone_filter=phone_filter)
            time.sleep(1.0)
        print("\n✅ Finished playing refined voices!")


if __name__ == "__main__":
    asyncio.run(main())
