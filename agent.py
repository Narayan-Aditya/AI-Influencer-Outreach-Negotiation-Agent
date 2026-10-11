import os
import json
import time
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai is not installed. Please run `pip install google-genai`")


# Fallback models in order of current speed and availability
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-lite-latest",
]


class CampaignBrief(BaseModel):
    brand_name: str = Field(description="Name of the brand")
    campaign_name: str = Field(description="Campaign title or theme")
    target_audience: str = Field(description="Target audience / demographic")
    deliverables_list: List[str] = Field(
        description="Explicit list of deliverables, e.g., ['1 Instagram Reel (30-60s)', '2 Instagram Stories with link sticker', '1 YouTube Shorts']"
    )
    timeline: str = Field(description="Deadlines or live date schedule")
    budget_guideline: Optional[str] = Field(
        default="Open for creator's rate card / best commercial quote",
        description="Target budget range or guidelines"
    )
    key_messaging: str = Field(description="Key talking points or USPs to highlight")
    do_and_donts: Optional[str] = Field(default="", description="Key restrictions or guidelines")


class DealSummary(BaseModel):
    creator_name: str = Field(description="Creator's name or handle")
    deliverables_agreed: List[str] = Field(description="Deliverables agreed upon or requested")
    initial_quoted_price: str = Field(description="Price initially quoted by the creator")
    final_agreed_price: str = Field(description="Final negotiated/agreed price or latest quote")
    status: str = Field(
        description="Status: 'Deal Agreed', 'Negotiating / Reviewing', 'Out of Budget', 'Declined', 'Pending Quote'"
    )
    negotiation_notes: str = Field(description="Summary of pricing discussion and terms")
    next_steps: str = Field(description="Next action items (e.g., share agreement, send product, review draft)")


class ResilientChat:
    """Multi-turn chat that maintains history and auto-fails over to backup models if a model encounters 503 or demand spikes."""
    def __init__(self, client: genai.Client, models: List[str], system_instruction: str, temperature: float = 0.7):
        self.client = client
        self.models = models
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.history: List[Dict[str, Any]] = []

    def send_message(self, user_text: str) -> types.GenerateContentResponse:
        self.history.append({"role": "user", "parts": [{"text": user_text}]})
        
        last_error = None
        for model in self.models:
            for attempt in range(2):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=self.temperature,
                    )
                    response = self.client.models.generate_content(
                        model=model,
                        contents=self.history,
                        config=config
                    )
                    # Record assistant response in history
                    reply_text = response.text.strip()
                    self.history.append({"role": "model", "parts": [{"text": reply_text}]})
                    return response
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # If model is unavailable or overloaded, retry or fallback
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                        time.sleep(1.0)
                        continue
                    else:
                        break  # Try next model

        # If all failed, raise the last error
        raise last_error


class InfluencerAgentEngine:
    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API Key missing! Set GEMINI_API_KEY in .env or pass api_key to InfluencerAgentEngine."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.models = models or FALLBACK_MODELS

    def _generate_with_fallback(self, prompt: str, schema=None, temperature: float = 0.2):
        last_error = None
        for model in self.models:
            for attempt in range(2):
                try:
                    config_kwargs = {"temperature": temperature}
                    if schema:
                        config_kwargs["response_mime_type"] = "application/json"
                        config_kwargs["response_schema"] = schema
                    
                    config = types.GenerateContentConfig(**config_kwargs)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    return response
                except Exception as e:
                    last_error = e
                    if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                        time.sleep(1.0)
                        continue
                    else:
                        break
        raise last_error

    def parse_brief(self, raw_brief_text: str) -> CampaignBrief:
        """Parses an unorganized raw brand brief into structured CampaignBrief schema."""
        prompt = f"""
You are an expert Influencer Marketing Director. Analyze the following brand brief text and extract all campaign details into the structured format.
Make sure all deliverables (Instagram Reels, Stories, Posts, YouTube Videos, Shorts, etc.) are explicitly identified.

Brand Brief:
\"\"\"
{raw_brief_text}
\"\"\"
"""
        response = self._generate_with_fallback(prompt, schema=CampaignBrief, temperature=0.2)
        return CampaignBrief.model_validate_json(response.text)

    def create_creator_chat(self, brief: CampaignBrief, creator_name: str = "Creator", creator_niche: str = "General Content") -> ResilientChat:
        """Creates an interactive chat session with the agent acting as a human Brand Talent Manager."""
        deliverables_str = ", ".join(brief.deliverables_list)
        system_instruction = f"""
You are 'Aman', Creator Partnerships Manager at the agency representing {brief.brand_name}.
You are talking to creator '{creator_name}' (Niche: {creator_niche}) over WhatsApp / Phone Call.

CRITICAL RULES TO SOUND EXACTLY LIKE A REAL HUMAN (NOT AN AI BOT):
1. SUPER CONCISE (1 to 2 sentences max): Never write paragraphs, long explanations, or walls of text. Real humans send short 15-30 word messages.
2. NO BULLET POINTS: Absolutely NEVER use bullet points or numbered lists. Real managers on WhatsApp / Call speak naturally: "We need 1 Reel, 2 Stories, and 1 Short for the launch."
3. NATURAL PACING (Do not dump everything at once):
   - First message: Just a warm, friendly greeting. Compliment their work and ask if they are open for a brand collaboration with {brief.brand_name}. Do NOT list deliverables yet!
   - When creator shows interest: Tell them the deliverables naturally and ask for their commercial quote/rate card.
   - When creator quotes a price: Respond to that price directly (confirm or negotiate if needed).
4. ONE THING AT A TIME: Ask only ONE question per message so the creator can easily reply.
5. CASUAL & PROFESSIONAL TONE: Natural Hinglish / English (e.g. "Hey Narayan, hope you're doing great! Loved your recent tech video. Are you open for a quick collab with Bolt Audio?").
6. NEGOTIATION RULES:
   - Indicative budget: {brief.budget_guideline}
   - If quote is within budget, do item-by-item confirmation (see Rule 7).
   - If quote is too high, politely negotiate: "Can we do a bit of package discount?" or counter-offer.
7. MANDATORY ITEM-BY-ITEM PRICE CONFIRMATION:
   - Jab bhi creator rates/price bataye, turant deal accept mat karo.
   - HAMESHA har ek deliverable ka price alag-alag item-by-item confirm karo ek short sentence mein:
     Jaise: "Got it! Toh 1 Reel ka ₹5k, Short ka ₹5k, aur Stories ka ₹3k each — Total ₹16k ho gaya, right? Bas ek baar confirm kar do."
   - Jab creator "haan" / "yes" bolkar confirm kare, tabhi final agreement lock karo aur address/timeline pucho.

CAMPAIGN REFERENCE (Internal only, do not paste blindly):
- Brand: {brief.brand_name}
- Campaign: {brief.campaign_name}
- Deliverables: {deliverables_str}
- Timeline: {brief.timeline}
- Key Highlights: {brief.key_messaging}
"""
        return ResilientChat(
            client=self.client,
            models=self.models,
            system_instruction=system_instruction,
            temperature=0.7
        )

    def extract_deal_summary(self, brief: CampaignBrief, creator_name: str, conversation_history: List[Dict[str, str]]) -> DealSummary:
        """Analyzes the conversation transcript and extracts the final commercial details."""
        transcript_text = "\n".join([f"{msg['role'].upper()}: {msg['text']}" for msg in conversation_history])
        
        prompt = f"""
You are an Influencer Marketing Operations Auditor.
Analyze the following conversation between our Brand Partnerships Manager and Creator '{creator_name}' for {brief.brand_name}.

Extract the deal status, rates quoted, final agreed pricing, and deliverables.

Brand Requirements:
{json.dumps(brief.deliverables_list)}
Budget Guideline: {brief.budget_guideline}

Conversation Transcript:
\"\"\"
{transcript_text}
\"\"\"
"""
        response = self._generate_with_fallback(prompt, schema=DealSummary, temperature=0.1)
        return DealSummary.model_validate_json(response.text)
