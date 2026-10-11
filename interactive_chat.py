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
    print("    - Type '/voice off' or '/voice on' to toggle TTS audio playback")
    print("    - Type '/voice auto' (Auto Hindi/English mirror), '/voice fast', or '/voice normal'")
    print("    - Type 'exit' or 'quit' to end chat and finalize deal sheet")
    print("=" * 65 + "\n")

    # Initialize Smart Auto-Language Voice Speaker (+18% human pacing speed)
    speaker = VoiceSpeaker(mode="auto", rate="+18%", enabled=True)
    if speaker.enabled:
        print("🔊 Smart Voice Playback is ON (Auto Hindi/English mirror, faster +18% human pacing)\n")

    chat = engine.create_creator_chat(parsed_brief, creator_name=creator_name, creator_niche=creator_niche)
    history = []

    # Initial Agent Greeting (short & human-like)
    try:
        initial_response = chat.send_message(
            f"Send a natural, short 1-2 sentence opening message to {creator_name} ({creator_niche}) introducing yourself from {parsed_brief.brand_name} and asking if they are open for a quick collab."
        )
        agent_msg = initial_response.text
        print(f"🤖 Bot (Agency Manager):\n{agent_msg}\n")
        speaker.speak(agent_msg)
        history.append({"role": "agent", "text": agent_msg})
    except Exception as e:
        print(f"❌ Error starting conversation: {e}")
        return

    # Conversation Loop
    while True:
        try:
            user_input = input(f"👤 {creator_name}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input.lower() == "/brief":
            print("\n📋 ACTIVE DELIVERABLES:")
            for d in parsed_brief.deliverables_list:
                print(f"  • {d}")
            print(f"• Budget Guide: {parsed_brief.budget_guideline}\n")
            continue

        if user_input.lower() == "/voice off":
            speaker.enabled = False
            print("🔇 Voice playback disabled.\n")
            continue

        if user_input.lower() == "/voice on":
            speaker.enabled = True
            print("🔊 Voice playback enabled.\n")
            continue

        if user_input.lower() == "/voice auto":
            speaker.set_mode("auto")
            print("🎙️ Voice set to Auto (Mirrors Hindi / English automatically).\n")
            continue

        if user_input.lower() == "/voice fast":
            speaker.set_rate("+25%")
            print("⚡ Voice speed set to Fast (+25%).\n")
            continue

        if user_input.lower() == "/voice normal":
            speaker.set_rate("+0%")
            print("⏱️ Voice speed set to Normal (+0%).\n")
            continue

        if user_input.lower() == "/voice hindi":
            speaker.set_mode("hindi_male")
            print("🎙️ Voice set to Clean Hindi Male (Madhur).\n")
            continue

        if user_input.lower() == "/voice female":
            speaker.set_mode("hindi_female")
            print("🎙️ Voice set to Clean Hindi Female (Swara).\n")
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
            response = chat.send_message(user_input)
            agent_msg = response.text
            print(f"\n🤖 Bot (Agency Manager):\n{agent_msg}\n")
            speaker.speak(agent_msg)
            history.append({"role": "agent", "text": agent_msg})
        except Exception as e:
            print(f"❌ Error getting bot response: {e}\n")

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
