# AI Voice Agent for Real Estate


DEMO - https://www.linkedin.com/posts/mahammed-rayyan-233a581b1_started-as-a-curiosity-now-its-a-real-time-activity-7402044845307142144-lBt0?utm_source=share&utm_medium=member_desktop&rcm=ACoAADF0lEcBzHTjyLYe2m3sUH3sBE5A1NGQl4s

This project is a sophisticated, real-time AI voice agent designed to act as an appointment setter for a real estate firm. It can handle both inbound and outbound calls, qualify leads, provide information on properties, and book site visits directly on a Google Calendar.

## Recent Improvements 

- Hardened LLM response parsing in `main.py` (`_to_text`) to gracefully handle varied Vertex AI / LangChain payload shapes (strings, dicts, objects, lists, candidate arrays) and avoid `AttributeError` crashes.
- Suppressed noisy third‑party deprecation and syntax warnings for cleaner logs (see warning filters in `main.py` and `llm_test.py`).
- Added defensive fallbacks so the agent always emits a speakable response even if the LLM returns an unexpected structure.

## Core Features

- **Real-Time Conversation**: Utilizes a low-latency stack (Deepgram, ElevenLabs, Twilio) for natural, real-time voice interaction.
- **Inbound & Outbound Calls**: Can receive calls from interested clients and proactively call leads from a list.
- **Intelligent Appointment Booking**: The agent checks a Google Calendar for availability in real-time and books appointments using LangChain tools.
- **Context-Aware**: The agent is aware of the specific real estate properties the firm offers and can discuss them with leads.
- **24/7 Availability**: The agent can book appointments around the clock, finding the next available slots.
- **Modular & Scalable**: Built with FastAPI, the system is designed to be scalable and allows for easy integration with other lead sources like CRMs.

