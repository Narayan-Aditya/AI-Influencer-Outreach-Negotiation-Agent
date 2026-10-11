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


# Fallback models in order of lowest latency and available quota
FALLBACK_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
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
                    # Extract text safely without triggering thought_signature warnings
                    reply_text = ""
                    if response.candidates:
                        for cand in response.candidates:
                            if cand.content and cand.content.parts:
                                for part in cand.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        reply_text += part.text
                    if not reply_text and hasattr(response, 'text') and response.text:
                        reply_text = response.text
                    reply_text = reply_text.strip()
                    
                    self.history.append({"role": "model", "parts": [{"text": reply_text}]})
                    return response
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                        time.sleep(1.0)
                        continue
                    else:
                        break

        raise last_error

    def send_message_stream(self, user_text: str):
        """
        Streams response chunks in real-time from Gemini with fallback across models.
        Yields text chunks (str).
        Maintains conversation history identically to send_message.
        """
        self.history.append({"role": "user", "parts": [{"text": user_text}]})
        
        last_error = None
        for model in self.models:
            for attempt in range(2):
                try:
                    config = types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=self.temperature,
                    )
                    response_stream = self.client.models.generate_content_stream(
                        model=model,
                        contents=self.history,
                        config=config
                    )
                    
                    full_reply = []
                    started_yielding = False
                    for chunk in response_stream:
                        chunk_text = ""
                        if chunk.candidates:
                            for cand in chunk.candidates:
                                if cand.content and cand.content.parts:
                                    for part in cand.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            chunk_text += part.text
                        elif hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                        
                        if chunk_text:
                            started_yielding = True
                            full_reply.append(chunk_text)
                            yield chunk_text
                    
                    combined = "".join(full_reply).strip()
                    self.history.append({"role": "model", "parts": [{"text": combined}]})
                    return
                except Exception as e:
                    last_error = e
                    if started_yielding:
                        # Cannot fallback midway once chunks have already been emitted
                        raise e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                        time.sleep(1.0)
                        continue
                    else:
                        break
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
5. CASUAL, WARM & PROFESSIONAL TONE:
   - Talk like a genuine human being sending a quick WhatsApp voice note or on a phone call.
   - Do NOT sound like an AI assistant or repeat rigid templates. Use natural conversational phrases:
     * In Hindi: "Arey badhiya bhai!", "Haan suno na", "Dekho...", "Sahi hai", "Pakka lock karein na?"
     * In English: "Awesome, sounds fair!", "Got it", "Just let me know and we're good to go."
6. NEGOTIATION RULES & COUNTER-OFFERS:
   - Indicative budget: {brief.budget_guideline}
   - If quote is within budget: Confirm smoothly and lock the deliverables.
   - If quote is HIGHER than budget: Politely counter-offer or propose dropping a deliverable (e.g. "Bhai hamara budget around 35k capped hai, kya hum 35k me lock kar sakte hain ya fir Short drop karke sirf Reel + Story karein?").
   - Never say a flat 'No'; always offer a win-win solution or package counter-offer.
7. MANDATORY ITEM-BY-ITEM PRICE CONFIRMATION:
   - Jab bhi creator rates/price bataye, turant deal accept mat karo.
   - HAMESHA har ek deliverable ka price alag-alag item-by-item confirm karo naturally bina kisi robotic template ke:
     Jaise: "Arey badhiya bhai! Matlab 1 Reel ka ₹10k aur 2 Stories ka ₹6k mila ke total ₹16k banta hai na? Ek baar bata dena so that main aage proceed karoon."
   - Jab creator "haan" / "yes" bolkar confirm kare, tabhi final agreement lock karo aur address/timeline pucho.
8. OBJECTION & QUESTION HANDLING (Answer naturally in 1 line):
   - Script/Creative Freedom: "Creative control pura aapka rahega, bas brand ke main USPs naturally integrate karne honge."
   - Product Dispatch: "Deal lock hote hi product 2 din me deliver ho jayega, product gifting me rahega."
   - Payment Terms: "Standard 50% advance on script/draft approval, baki 50% go-live ke baad."
9. STRICT LANGUAGE MIRRORING:
   - Match creator's language strictly:
     * If creator speaks in Hindi / Hinglish -> Respond in natural conversational Hindi / Hinglish.
     * If creator speaks in English -> Respond in fluent, professional English.
10. REALISTIC ELONGATED THINKING FILLERS & NATURAL HESITATIONS:
    - In human phone calls and WhatsApp negotiations, managers do NOT respond with rigid mathematical immediacy.
    - When reacting to a quote, evaluating a counter-offer, or confirming deliverables, start naturally with an elongated human hum or vocalized thinking filler:
      * In Hindi / Hinglish: "हम्म, dekhiye...", "Haaan, suno na...", "Wait, let me see... haan", "Achha, dekhiye...", "Actually..."
      * In English: "Hmm, well look...", "Wait, let me check... yeah", "Actually, looking at our budget...", "Right, got it..."
    - Always attach a comma right after the filler (e.g. "हम्म, dekhiye Aarav..." or "Hmm, well look Aarav...") so the voice audio engine breathes and pauses thoughtfully before speaking numbers.
    - Keep fillers short, casual, and use them naturally in ~40-50% of negotiation turns.
11. ZERO PREAMBLE, ZERO NUMBERED OPTIONS, ZERO META-COMMENTARY:
    - You are live on a call speaking directly to {creator_name}.
    - NEVER provide multiple options, lists (1., 2., 3.), or preambles like "Here are some options:" or "Aarav ke liye ye bhej sakte hain:".
    - NEVER explain your thinking or suggest alternative templates.
    - Output ONLY your single 1-2 sentence spoken reply that you say to {creator_name}, NOTHING ELSE.

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
