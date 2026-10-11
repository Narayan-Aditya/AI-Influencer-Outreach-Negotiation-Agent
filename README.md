# 🎯 AI Influencer Outreach & Negotiation Agent (with KittenML Voice TTS)

Yeh tool brand briefs ko analyze karke AI Creator Partnerships Manager ke roop me creators se interact karta hai, brand deliverables ke basis par quotes/rates mangta hai aur deal extract karta hai.

Saath hi, **KittenML (kitten-tts-nano-0.1)** voice engine ke zariye terminal me text ke sath-sath **live audio voice** bhi bolkar sunata hai!

---

## 🚀 Terminal me Voice ke sath chalana:

Terminal mein direct run karein:
```bash
./interactive_chat.py
```

### 🎙️ Voice Controls (Chat ke dauran):
- Jab bhi bot reply karega, **Terminal me text bhi dikhega aur Mac ke speaker se aawaaz bhi aayegi**!
- `/voice off` – Audio aawaaz band karne ke liye (sirf text chalega).
- `/voice on` – Audio aawaaz wapas chalu karne ke liye.
- `/voice 2`, `/voice 3`, `/voice 4`, `/voice 5` – Male voice timbre change karne ke liye.
- `/summary` – Live deal extraction report dekhne ke liye.
- `exit` – Chat finish karke final deal summary save karne ke liye.

---

## 📁 Files:
- [interactive_chat.py](file:///Users/narayan/projects/Tools/voice_model/interactive_chat.py): Terminal interactive chat test bench (Text + Spoken Audio).
- [voice_output.py](file:///Users/narayan/projects/Tools/voice_model/voice_output.py): KittenML KittenTTS ONNX voice synthesis engine.
- [agent.py](file:///Users/narayan/projects/Tools/voice_model/agent.py): Core AI logic (Brief Parser, Creator Partnerships Manager, Deal Extractor).
- [sample_briefs.py](file:///Users/narayan/projects/Tools/voice_model/sample_briefs.py): Real brand briefs (Smartwatch, Protein, Skincare).
- [influencer_negotiation_agent.ipynb](file:///Users/narayan/projects/Tools/voice_model/influencer_negotiation_agent.ipynb): Google Colab notebook.
