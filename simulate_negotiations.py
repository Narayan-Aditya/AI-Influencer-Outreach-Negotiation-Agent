#!/usr/bin/env python3
"""
Automated Multi-Scenario Test Suite for Influencer Negotiation Agent.
Simulates realistic creator dialogues and tests negotiation intelligence.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from agent import InfluencerAgentEngine
from sample_briefs import SAMPLE_BRIEFS


def run_scenario(scenario_name, creator_name, creator_niche, brief_content, creator_turns):
    print("\n" + "=" * 70)
    print(f"🧪 RUNNING TEST SCENARIO: {scenario_name}")
    print(f"👤 Creator: {creator_name} ({creator_niche})")
    print("=" * 70)

    engine = InfluencerAgentEngine()
    brief = engine.parse_brief(brief_content)
    chat = engine.create_creator_chat(brief, creator_name=creator_name, creator_niche=creator_niche)

    history = []
    # Turn 0: Opening
    init_res = chat.send_message(
        f"Send a natural, short 1-2 sentence opening message to {creator_name} ({creator_niche}) introducing yourself from {brief.brand_name} and asking if they are open for a quick collab."
    )
    bot_msg = init_res.text.strip()
    print(f"\n🤖 Bot:\n   {bot_msg}")
    history.append({"role": "agent", "text": bot_msg})

    for i, user_msg in enumerate(creator_turns, 1):
        print(f"\n👤 {creator_name} (Turn {i}):\n   {user_msg}")
        history.append({"role": "creator", "text": user_msg})

        bot_res = chat.send_message(user_msg)
        bot_reply = bot_res.text.strip()
        print(f"\n🤖 Bot:\n   {bot_reply}")
        history.append({"role": "agent", "text": bot_reply})

    print("\n⏳ Extracting final deal summary...")
    deal = engine.extract_deal_summary(brief, creator_name, history)
    print("\n📊 EXTRACTED DEAL OUTCOME:")
    print(f"• Status: {deal.status}")
    print(f"• Initial Quote: {deal.initial_quoted_price}")
    print(f"• Final Agreed: {deal.final_agreed_price}")
    print(f"• Deliverables: {deal.deliverables_agreed}")
    print(f"• Notes: {deal.negotiation_notes}")
    print(f"• Next Steps: {deal.next_steps}")
    return deal


def main():
    print("🚀 STARTING AUTOMATED NEGOTIATION SUITE...")

    # Scenario 1: High Quote Negotiation
    run_scenario(
        scenario_name="High Price Negotiation (70k Quote -> 38k Agreed)",
        creator_name="Kabir Tech",
        creator_niche="Tech Reviews",
        brief_content=SAMPLE_BRIEFS["1"]["content"],
        creator_turns=[
            "Haan bhai bolo, kya campaign hai aur kya deliverables hain?",
            "Bhai mera full package (Reel + 2 Stories + Short) ka commercial quote ₹75,000 hoga.",
            "Hmm, budget tight hai toh best ₹38,000 me kar dunga pura package agar payment time pe ho.",
            "Haan right, 38k all-inclusive locked. Address aur brief bhej do."
        ]
    )

    # Scenario 2: Questions about Creative Freedom & Payment
    run_scenario(
        scenario_name="Creator Questions on Creative Freedom & Advance Payment",
        creator_name="Pooja Verma",
        creator_niche="Fitness & Lifestyle",
        brief_content=SAMPLE_BRIEFS["2"]["content"],
        creator_turns=[
            "Hi Aman! Sure, PlantStrong ke sath collab ke liye interested hu. What are the requirements?",
            "Reel, 1 Story aur YT integration ke liye mera rate ₹40,000 hoga. Lekin script freedom kitni hogi?",
            "Great! Payment terms kya rahenge aapke?",
            "Perfect! 40k done hai, I agree to the terms. Deal confirm samjhein."
        ]
    )


if __name__ == "__main__":
    main()
