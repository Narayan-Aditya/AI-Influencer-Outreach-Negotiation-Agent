#!/usr/bin/env python3
"""
Interactive CLI for testing and refining the Influencer Outreach & Negotiation Agent.
"""

import os
import sys

# Auto-switch to virtual environment if running with system python
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "bin", "python3")
if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)

import json
import select
import threading
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)
from dotenv import load_dotenv

load_dotenv()

from agent import InfluencerAgentEngine
from sample_briefs import SAMPLE_BRIEFS
from voice_output import VoiceSpeaker


def print_banner():
    print("=" * 65)
    print(" 🎯 AI INFLUENCER OUTREACH & NEGOTIATION AGENT (TEST BENCH)")
    print("=" * 65)


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not found in environment or .env file.")
        api_key = input("👉 Please enter your Gemini API Key: ").strip()
        if not api_key:
            print("❌ API Key is required to run the agent. Exiting.")
            sys.exit(1)
        save = input("Save this API key to .env file for future runs? (y/n): ").strip().lower()
        if save == 'y':
            with open(".env", "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            print("✅ Saved to .env!")
    return api_key


def select_brief():
    print("\nSelect a Brand Brief to test with:")
    for key, data in SAMPLE_BRIEFS.items():
        print(f"  [{key}] {data['title']}")
    print("  [4] Enter my own custom brand brief")

    choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"

    if choice in SAMPLE_BRIEFS:
        print(f"\n✅ Selected: {SAMPLE_BRIEFS[choice]['title']}")
        return SAMPLE_BRIEFS[choice]["content"]
    elif choice == "4":
        print("\nPaste your Brand Brief below (press Enter twice or type 'DONE' on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip() == "DONE":
                break
            if not line and lines and lines[-1] == "":
                break
            lines.append(line)
        content = "\n".join(lines).strip()
        if not content:
            print("Empty brief provided! Using default brief 1.")
            return SAMPLE_BRIEFS["1"]["content"]
        return content
    else:
        print("Invalid choice, defaulting to 1.")
        return SAMPLE_BRIEFS["1"]["content"]


def interruptible_input(prompt: str, speaker: VoiceSpeaker) -> str:
    """
    Listens for user input with instant full-duplex barge-in.
    The moment any keystroke or newline is detected on stdin, speaker.stop()
    is triggered within 50ms, immediately silencing the bot.
    """
    stop_event = threading.Event()

    def _monitor():
        while not stop_event.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    speaker.stop()
                    break
            except Exception:
                break

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    try:
        val = input(prompt).strip()
    finally:
        stop_event.set()
        speaker.stop()
    return val


def run_chat_session():
    print_banner()
    api_key = get_api_key()

    print("\n⏳ Initializing Gemini Engine...")
    try:
        engine = InfluencerAgentEngine(api_key=api_key)
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return

    # 1. Get Brief
    brief_text = select_brief()

    print("\n⏳ Parsing Brand Brief with Gemini...")
    try:
        parsed_brief = engine.parse_brief(brief_text)
    except Exception as e:
        print(f"❌ Failed to parse brief: {e}")
        return

    print("\n" + "-" * 50)
    print(f"📋 PARSED BRAND BRIEF: {parsed_brief.brand_name.upper()}")
    print("-" * 50)
    print(f"• Campaign: {parsed_brief.campaign_name}")
    print(f"• Target Audience: {parsed_brief.target_audience}")
    print("• Deliverables Required:")
    for d in parsed_brief.deliverables_list:
        print(f"   ▫️ {d}")
    print(f"• Timeline: {parsed_brief.timeline}")
    print(f"• Indicative Budget: {parsed_brief.budget_guideline}")
    print(f"• Key USPs: {parsed_brief.key_messaging}")
    print("-" * 50)

    # 2. Setup Creator Persona
    print("\nSetup Creator to test with:")
    creator_name = input("Creator Name / Handle (e.g. 'Aarav Sharma' / @tech_aarav): ").strip() or "Aarav Sharma"
    creator_niche = input("Creator Niche (e.g. Tech, Fitness, Lifestyle): ").strip() or "Tech & Lifestyle"

    print("\n" + "=" * 65)
    print(f" 💬 CHAT SESSION STARTED WITH BOT (Representing {parsed_brief.brand_name})")
    print(" 💡 Commands:")
    print("    - Type normally to respond as the Creator")
    print("    - Type '/summary' to see current deal extraction")
    print("    - Type '/brief' to view the brand requirements again")
    print("    - Type '/voice off' or '/voice on' to toggle audio playback")
    print("    - Type '/voice neerja' (Top Indian Female Voice - Highly Recommended)")
    print("    - Type '/voice swara' (Soft, natural Hindi female tone)")
    print("    - Type '/voice madhur' (Clean Hindi male tone)")
    print("    - Type '/voice prabhat' (Indian English male tone)")
    print("    - Type '/voice auto' (Auto-mirrors language: Swara for Hindi, Neerja for English)")
    print("    - Type '/voice fast' (+20%), '/voice relaxed' (+14%), or '/voice normal' (+0%)")
    print("    - Type '/ambiance ac', '/ambiance fan', '/ambiance crowd', or '/ambiance off'")
    print("    - Type '/pitch lower' (-2Hz, warm), '/pitch normal' (+0Hz)")
    print("    - Type 'exit' or 'quit' to end chat and finalize deal sheet")
    print("=" * 65 + "\n")

    # Initialize Smart Voice Speaker (Auto mirror: Swara for Hindi, Neerja for English, with warm -2Hz pitch and clean background)
    speaker = VoiceSpeaker(mode="auto", rate="+12%", pitch="-2Hz", ambiance="off", enabled=True)
    if speaker.enabled:
        print("🔊 Smart Voice Playback is ON (Auto Hindi/English mirror, -2Hz pitch, Clean background)\n")

    chat = engine.create_creator_chat(parsed_brief, creator_name=creator_name, creator_niche=creator_niche)
    history = []

    # Initial Agent Greeting (short & human-like, streamed)
    try:
        greeting_prompt = (
            f"You are on a live call with {creator_name} ({creator_niche}). Say your warm 1-2 sentence opening greeting directly to them now introducing yourself from {parsed_brief.brand_name}. "
            f"Do NOT provide options, preamble, or explanations. Output ONLY your spoken words to {creator_name}."
        )
        print("🤖 Bot (Agency Manager):\n", end="", flush=True)
        full_agent_msg = []
        for chunk in chat.send_message_stream(greeting_prompt):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_agent_msg.append(chunk)
        print("\n")
        agent_msg = "".join(full_agent_msg).strip()
        history.append({"role": "agent", "text": agent_msg})

        # Play greeting as seamless unified audio track (zero pause/play fragmentation)
        if speaker.enabled and agent_msg:
            speaker.speak(agent_msg, blocking=False)
    except Exception as e:
        print(f"❌ Error starting conversation: {e}")
        return

    # Conversation Loop
    while True:
        try:
            user_input = interruptible_input(f"👤 {creator_name}: ", speaker)
        except (KeyboardInterrupt, EOFError):
            speaker.stop()
            print("\nExiting chat...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            speaker.stop()
            break

        if user_input.lower() == "/brief":
            print("\n📋 ACTIVE DELIVERABLES:")
            for d in parsed_brief.deliverables_list:
                print(f"  • {d}")
            print(f"• Budget Guide: {parsed_brief.budget_guideline}\n")
            continue

        if user_input.lower() == "/voice off":
            speaker.enabled = False
            speaker.stop()
            print("🔇 Voice playback disabled.\n")
            continue

        if user_input.lower() == "/voice on":
            speaker.enabled = True
            print("🔊 Voice playback enabled.\n")
            continue

        if user_input.lower() in ["/voice neerja", "/voice female"]:
            speaker.set_mode("neerja")
            print("🎙️ Voice set to Neerja (Top Indian Female Voice - Highly Recommended).\n")
            continue

        if user_input.lower() in ["/voice swara", "/voice hindi"]:
            speaker.set_mode("swara")
            print("🎙️ Voice set to Swara (Soft, natural Hindi female tone).\n")
            continue

        if user_input.lower() in ["/voice madhur", "/voice male"]:
            speaker.set_mode("madhur")
            print("🎙️ Voice set to Madhur (Clean Hindi male tone).\n")
            continue

        if user_input.lower() in ["/voice prabhat", "/voice english"]:
            speaker.set_mode("prabhat")
            print("🎙️ Voice set to Prabhat (Indian English male tone).\n")
            continue

        if user_input.lower() == "/voice auto":
            speaker.set_mode("auto")
            print("🎙️ Voice set to Auto (Swara for Hindi, Neerja for English).\n")
            continue

        if user_input.lower() == "/voice fast":
            speaker.set_rate("+20%")
            print("⚡ Voice speed set to Fast (+20%).\n")
            continue

        if user_input.lower() == "/voice relaxed":
            speaker.set_rate("+12%")
            print("🍃 Voice speed set to Relaxed (+12%).\n")
            continue

        if user_input.lower() == "/voice normal":
            speaker.set_rate("+0%")
            print("⏱️ Voice speed set to Normal (+0%).\n")
            continue

        if user_input.lower().startswith("/ambiance"):
            parts = user_input.lower().split()
            amb_type = parts[1] if len(parts) > 1 else "ac"
            if amb_type in ["ac", "ac_fan"]:
                speaker.set_ambiance("ac_fan")
                print("🌬️ Acoustic ambiance set to AC Room Tone.\n")
            elif amb_type in ["fan", "ceiling_fan"]:
                speaker.set_ambiance("fan")
                print("🌀 Acoustic ambiance set to Ceiling Fan Tone.\n")
            elif amb_type in ["crowd", "office", "cafe"]:
                speaker.set_ambiance("crowd")
                print("☕ Acoustic ambiance set to Soft Office/Cafe Murmur.\n")
            elif amb_type in ["off", "none", "silent"]:
                speaker.set_ambiance("off")
                print("🔇 Background ambiance turned OFF.\n")
            else:
                speaker.set_ambiance("ac_fan")
                print("🌬️ Acoustic ambiance set to AC Room Tone.\n")
            continue

        if user_input.lower().startswith("/pitch"):
            parts = user_input.lower().split()
            pitch_type = parts[1] if len(parts) > 1 else "lower"
            if pitch_type in ["lower", "-2hz", "-2"]:
                speaker.set_pitch("-2Hz")
                print("🎙️ Pitch set to Lowered (-2Hz, warm human tone).\n")
            elif pitch_type in ["deep", "-4hz", "-4"]:
                speaker.set_pitch("-4Hz")
                print("🎙️ Pitch set to Deep (-4Hz).\n")
            elif pitch_type in ["normal", "+0hz", "0"]:
                speaker.set_pitch("+0Hz")
                print("🎙️ Pitch set to Normal (+0Hz).\n")
            else:
                speaker.set_pitch(pitch_type)
                print(f"🎙️ Pitch set to {pitch_type}.\n")
            continue

        if user_input.lower().startswith("/filter"):
            parts = user_input.lower().split()
            f_type = parts[1] if len(parts) > 1 else "phone"
            if f_type in ["phone", "whatsapp", "call", "on"]:
                speaker.set_phone_filter(True)
                print("📱 Phone Call / WhatsApp MEMS mic acoustic filter ENABLED.\n")
            else:
                speaker.set_phone_filter(False)
                print("🎧 Phone filter DISABLED (Pure studio mode).\n")
            continue

        if user_input.lower().startswith("/breath"):
            parts = user_input.lower().split()
            b_val = parts[1] if len(parts) > 1 else "on"
            if b_val in ["on", "true", "1"]:
                speaker.set_breath(True)
                print("🌬️ Biological micro-breath inhale ENABLED.\n")
            else:
                speaker.set_breath(False)
                print("🔇 Biological micro-breath inhale DISABLED.\n")
            continue

        if user_input.lower() == "/summary":
            print("\n⏳ Extracting live deal summary...")
            try:
                summary = engine.extract_deal_summary(parsed_brief, creator_name, history)
                print(json.dumps(summary.model_dump(), indent=2))
            except Exception as e:
                print(f"Error extracting summary: {e}")
            print()
            continue

        history.append({"role": "creator", "text": user_input})

        try:
            print("\n🤖 Bot (Agency Manager):\n", end="", flush=True)
            full_agent_msg = []
            for chunk in chat.send_message_stream(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_agent_msg.append(chunk)
            print("\n")
            agent_msg = "".join(full_agent_msg).strip()
            history.append({"role": "agent", "text": agent_msg})

            # Voice synthesis of complete unified turn (Zero pause/play interruptions, seamless continuous audio)
            if speaker.enabled and agent_msg:
                speaker.speak(agent_msg, blocking=False)
        except Exception as e:
            print(f"\n❌ Error getting bot response: {e}\n")

    # Final Deal Summary
    if len(history) > 1:
        print("\n" + "=" * 65)
        print(" 📊 FINAL DEAL EXTRACTION REPORT")
        print("=" * 65)
        try:
            deal_summary = engine.extract_deal_summary(parsed_brief, creator_name, history)
            print(f"• Creator: {deal_summary.creator_name}")
            print(f"• Status: {deal_summary.status}")
            print(f"• Initial Quote: {deal_summary.initial_quoted_price}")
            print(f"• Final Agreed Price: {deal_summary.final_agreed_price}")
            print(f"• Deliverables Agreed:")
            for d in deal_summary.deliverables_agreed:
                print(f"   ▫️ {d}")
            print(f"• Notes: {deal_summary.negotiation_notes}")
            print(f"• Next Steps: {deal_summary.next_steps}")
            print("=" * 65)

            # Save report JSON
            out_file = f"deal_summary_{creator_name.replace(' ', '_').lower()}.json"
            with open(out_file, "w") as f:
                json.dump(deal_summary.model_dump(), f, indent=2)
            print(f"\n💾 Saved structured deal summary to: {out_file}")

            # Append to cumulative CSV deals tracker
            import csv
            from datetime import datetime
            csv_file = "deals_tracker.csv"
            file_exists = os.path.exists(csv_file)
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Brand", "Creator", "Status", "Initial Quote", "Final Price", "Deliverables", "Notes", "Next Steps"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    parsed_brief.brand_name,
                    deal_summary.creator_name,
                    deal_summary.status,
                    deal_summary.initial_quoted_price,
                    deal_summary.final_agreed_price,
                    " | ".join(deal_summary.deliverables_agreed),
                    deal_summary.negotiation_notes,
                    deal_summary.next_steps
                ])
            print(f"📈 Appended deal to campaign tracker: {csv_file}")
        except Exception as e:
            print(f"Could not generate final deal summary: {e}")
    else:
        print("\nChat ended without messages.")


if __name__ == "__main__":
    run_chat_session()
