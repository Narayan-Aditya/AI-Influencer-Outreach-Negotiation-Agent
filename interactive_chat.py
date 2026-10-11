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
    print("    - Type '/voice salman' (Warm, casual human male tone - Recommended)")
    print("    - Type '/voice swara' (Soft, expressive natural female tone)")
    print("    - Type '/voice madhur' (Clean Hindi male tone)")
    print("    - Type '/voice english' (Indian English male tone)")
    print("    - Type '/voice auto' (Auto-mirrors language with SSML human prosody)")
    print("    - Type '/voice fast' (+20%), '/voice relaxed' (+12%), or '/voice normal' (+0%)")
    print("    - Type 'exit' or 'quit' to end chat and finalize deal sheet")
    print("=" * 65 + "\n")

    # Initialize Smart Humanized Voice Speaker (Salman tone, relaxed +12% cadence, SSML breath pauses)
    speaker = VoiceSpeaker(mode="auto", rate="+12%", enabled=True)
    if speaker.enabled:
        print("🔊 Humanized Voice Playback is ON (Native Phonetics, Micro-Breathing SSML, Casual Voice Note Tone)\n")

    chat = engine.create_creator_chat(parsed_brief, creator_name=creator_name, creator_niche=creator_niche)
    history = []

    # Initial Agent Greeting (short & human-like, streamed)
    try:
        greeting_prompt = (
            f"Send a natural, short 1-2 sentence opening message to {creator_name} ({creator_niche}) "
            f"introducing yourself from {parsed_brief.brand_name} and asking if they are open for a quick collab."
        )
        print("🤖 Bot (Agency Manager):\n", end="", flush=True)
        pipeline = speaker.create_streaming_pipeline()
        full_agent_msg = []
        for chunk in chat.send_message_stream(greeting_prompt):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            pipeline.push_chunk(chunk)
            full_agent_msg.append(chunk)
        print("\n")
        pipeline.finish(wait=False)
        agent_msg = "".join(full_agent_msg).strip()
        history.append({"role": "agent", "text": agent_msg})
    except Exception as e:
        print(f"❌ Error starting conversation: {e}")
        return

    # Conversation Loop
    while True:
        try:
            user_input = input(f"👤 {creator_name}: ").strip()
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

        if user_input.lower() == "/voice salman":
            speaker.set_mode("salman")
            print("🎙️ Voice set to Salman (Warm, casual human male tone - Recommended).\n")
            continue

        if user_input.lower() in ["/voice swara", "/voice female"]:
            speaker.set_mode("swara")
            print("🎙️ Voice set to Swara (Soft, expressive natural female tone).\n")
            continue

        if user_input.lower() in ["/voice madhur", "/voice hindi"]:
            speaker.set_mode("madhur")
            print("🎙️ Voice set to Madhur (Clean Hindi male tone).\n")
            continue

        if user_input.lower() in ["/voice english", "/voice prabhat"]:
            speaker.set_mode("english")
            print("🎙️ Voice set to Prabhat (Indian English male tone).\n")
            continue

        if user_input.lower() == "/voice auto":
            speaker.set_mode("auto")
            print("🎙️ Voice set to Auto (Mirrors Hindi/English with native phonetics).\n")
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

        if user_input.lower() == "/voice kitten":
            speaker.set_mode("kitten")
            print("🎙️ Voice set to KittenTTS Nano.\n")
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
            pipeline = speaker.create_streaming_pipeline()
            full_agent_msg = []
            for chunk in chat.send_message_stream(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                pipeline.push_chunk(chunk)
                full_agent_msg.append(chunk)
            print("\n")
            pipeline.finish(wait=False)
            agent_msg = "".join(full_agent_msg).strip()
            history.append({"role": "agent", "text": agent_msg})
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
