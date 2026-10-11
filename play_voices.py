#!/usr/bin/env python3
"""
Refined Voice Player (Feature 1: Phone Mic Filter + Feature 2: Micro-Breath):
Plays the approved voice personas with biological inhale, phone mic filter, room ambiance, and -2Hz pitch:
  1. Neerja - Indian Female (Clear, Natural & Professional)
  2. Swara - Hindi Female (Soft, Expressive Devanagari)
  3. Madhur - Hindi Male (Natural Business Tone)
  4. Prabhat - Indian English Male (Friendly, Relaxed Pacing)

Usage:
  ./venv/bin/python3 play_voices.py               # Plays all 4 voices (Full Human Mode)
  ./venv/bin/python3 play_voices.py 1             # Plays Neerja (Full Human Mode: Breath + Phone Filter + AC Ambiance)
  ./venv/bin/python3 play_voices.py 2             # Plays Swara (Full Human Mode)
  ./venv/bin/python3 play_voices.py 3             # Plays Madhur (Full Human Mode)
  ./venv/bin/python3 play_voices.py 4             # Plays Prabhat (Full Human Mode)

Side-by-Side Comparison (Studio vs Full Human Mode):
  ./venv/bin/python3 play_voices.py 1 compare     # Plays Neerja Studio first, then Full Human Mode
  ./venv/bin/python3 play_voices.py 2 compare     # Plays Swara Studio first, then Full Human Mode

Other Modes:
  ./venv/bin/python3 play_voices.py 1 studio      # Raw Studio mode (no filter, no breath, no ambiance)
  ./venv/bin/python3 play_voices.py 1 nobreath    # Phone Call mode without breath
  ./venv/bin/python3 play_voices.py 1 fan         # Full Human Mode with Ceiling Fan sound
  ./venv/bin/python3 play_voices.py 1 crowd       # Full Human Mode with Office/Cafe murmur
"""

import sys
import os
import re
import time
import tempfile
import asyncio
import subprocess
import edge_tts
from voice_output import VoiceSpeaker, mix_audio_with_ambiance

speaker_helper = VoiceSpeaker()

HINDI_RAW = "Hmmmm... dekhiye Aarav, 1 Reel ka ₹10k... aur 2 Stories ka ₹6k mila ke total ₹16k confirm hai na? Deal theek hai toh batana."
ENGLISH_RAW = "Hmmmm... well look Aarav, for 1 Reel that's ₹10k... and for 2 Stories ₹6k. Total comes to ₹16k, right? Deal confirmed if this works for you."

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


async def generate_and_play(v, ambiance_type="ac_fan", phone_filter=True, add_breath=True, tag=""):
    cleaned = speaker_helper.clean_text_for_speech(v['raw_text'], lang=v['lang'])
    features = []
    if phone_filter: features.append("Phone Filter")
    if add_breath: features.append("Micro-Breath Inhale")
    if ambiance_type != "off": features.append(f"{ambiance_type.upper()} Ambiance")
    mode_desc = " + ".join(features) if features else "STUDIO MODE (Raw Digital)"
    if tag:
        mode_desc = f"{tag} - {mode_desc}"

    dyn_rate = speaker_helper.get_dynamic_rate(cleaned) if phone_filter else v['rate']
    has_deal_terms = bool(re.search(r'\b(हजार|रुपये|thousand|rupees|budget|deal|reel|stories|post|₹|\d+k?)\b', cleaned, flags=re.IGNORECASE))
    post_pause = 0.22 if (has_deal_terms and phone_filter) else 0.0

    print("\n" + "=" * 65)
    print(f"▶️ PLAYING [{v['id']}] {v['name']}")
    print(f"   Pipeline: {mode_desc}")
    print(f"   Speech:   \"{cleaned}\"")
    print(f"   Pitch:    {v['pitch']} (Warm chest resonance)")
    print(f"   Pacing:   {dyn_rate} ({'Deliberate deal cadence with 220ms pause' if post_pause > 0 else 'Brisk conversation'})")
    print(f"   Breath:   {'ON (Biological 0.16s Inhale)' if add_breath else 'OFF'}")
    print("=" * 65)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        c = edge_tts.Communicate(cleaned, voice=v['voice'], rate=dyn_rate, pitch=v['pitch'])
        await c.save(temp_path)

        play_path = mix_audio_with_ambiance(
            temp_path,
            ambiance_type=ambiance_type,
            level_db=-31.0,
            phone_filter=phone_filter,
            add_breath=add_breath,
            post_pause_sec=post_pause
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
    add_breath = True

    if mode == "studio":
        phone_filter = False
        add_breath = False
        ambiance = "off"
    elif mode in ["nobreath", "no_breath"]:
        phone_filter = True
        add_breath = False
        ambiance = "ac_fan"
    elif mode in ["fan", "ceiling_fan"]:
        ambiance = "fan"
        phone_filter = True
        add_breath = True
    elif mode in ["crowd", "office"]:
        ambiance = "crowd"
        phone_filter = True
        add_breath = True
    elif mode in ["ac", "phone"]:
        ambiance = "ac_fan"
        phone_filter = True
        add_breath = True

    if target:
        selected = [v for v in VOICE_LIST if v['id'] == target]
        if not selected:
            print(f"Voice ID '{target}' not found. Choose 1 to 4.")
            return

        item = selected[0]
        if mode == "compare":
            print(f"\n🔄 RUNNING SIDE-BY-SIDE COMPARISON FOR [{item['name']}]:")
            print("1️⃣ Playing Studio Mode first (No Filter, No Breath)...")
            await generate_and_play(item, ambiance_type="off", phone_filter=False, add_breath=False, tag="1/2 STUDIO")
            time.sleep(1.2)
            print("\n2️⃣ Now playing Full Human Pipeline (Micro-Breath + Phone Filter + AC Ambiance)...")
            await generate_and_play(item, ambiance_type="ac_fan", phone_filter=True, add_breath=True, tag="2/2 FULL HUMAN")
        else:
            await generate_and_play(item, ambiance_type=ambiance, phone_filter=phone_filter, add_breath=add_breath)
    else:
        print("\n🎧 PLAYING REFINED 4 VOICES (WITH FEATURE 1: PHONE FILTER + FEATURE 2: MICRO-BREATH):")
        for v in VOICE_LIST:
            await generate_and_play(v, ambiance_type=ambiance, phone_filter=phone_filter, add_breath=add_breath)
            time.sleep(1.0)
        print("\n✅ Finished playing refined voices!")


if __name__ == "__main__":
    asyncio.run(main())
