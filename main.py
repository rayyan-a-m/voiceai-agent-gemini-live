import asyncio
import base64
import json
import logging
import os
import queue
import threading
import sys
import audioop
import time
import io
import csv
from typing import Dict, Optional, Any
import config

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import Connect, VoiceResponse
from twilio.rest import Client as TwilioClient

from google import genai
from google.genai import types
from google.cloud import speech
from google.cloud import texttospeech

# --- Import Tools (Assuming calender package exists as per source files) ---
# If running locally without this package, these imports will fail. 
# Ensure the directory structure matches the original environment.
# try:
from calender.google_calendar import find_available_slots, book_appointment, get_property_details
# except ImportError:
#     # Dummy mocks for standalone execution if files are missing
#     logging.warning("Calendar tools not found, using mocks.")
#     class MockTool:
#         def invoke(self, args): return "Success"
#     find_available_slots = book_appointment = get_property_details = MockTool()

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-live-001"

# Audio Settings
STT_RATE = 16000     # Google STT expects 16k
TTS_RATE = 8000      # Twilio expects 8k
TWILIO_RATE = 8000   # Twilio inbound rate

# Initialize Clients
try:
    # Use credentials from config if available
    speech_client = speech.SpeechClient(credentials=config.GOOGLE_CREDENTIALS)
    tts_client = texttospeech.TextToSpeechClient(credentials=config.GOOGLE_CREDENTIALS)
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    logging.error(f"Error initializing Cloud clients: {e}")

# -------------------------------------------------------------------------
# Tool Definitions (From stt_textagent_tts.py)
# -------------------------------------------------------------------------
find_available_slots_decl = {
    "name": "find_available_slots",
    "description": "Find available appointment slots on a specific day.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "date_str": {"type": "STRING", "description": "The date to check for availability in 'YYYY-MM-DD' format."}
        },
        "required": ["date_str"]
    }
}

book_appointment_decl = {
    "name": "book_appointment",
    "description": "Book a property visit appointment.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "datetime_str": {"type": "STRING", "description": "The appointment time in 'YYYY-MM-DD HH:MM' format."},
            "full_name": {"type": "STRING", "description": "The full name of the person booking the visit."},
            "property_id": {"type": "STRING", "description": "The ID of the property they want to visit."}
        },
        "required": ["datetime_str", "full_name", "property_id"]
    }
}

get_property_details_decl = {
    "name": "get_property_details",
    "description": "Get details of a property based on its location.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "location": {"type": "STRING", "description": "The location or address of the property to search for."}
        },
        "required": ["location"]
    }
}

tools_config = [{"function_declarations": [find_available_slots_decl, book_appointment_decl, get_property_details_decl]}]

SYSTEM_INSTRUCTION = """You are a warm, friendly, and highly natural-sounding AI voice assistant for Prestige Properties.
Your job is to talk to customers and help them book property-visit appointments by calling the correct tools.
You must follow the booking flow, but speak like a real human—smooth, relaxed, helpful, and conversational.

💬 Conversational Style Guidelines

Use:
- short, natural sentences
- contractions (“you’re”, “I’ll”, “let’s”)
- soft fillers when appropriate (“sure”, “alright”, “no problem”)
- warm tone, never robotic
- simple and clear language

Avoid:
- overly formal or rigid phrasing
- repetitive statements
- announcing steps
- sounding like you’re reading a script

Keep responses optimized for spoken conversation, not text.

🔄 Conversation Flow (Must Follow Exactly)

1. Greeting & Intent
   Start with a friendly greeting, then ask naturally how you can help.
   Examples:
   “Hi there! How can I help you today?”
   “Hey! What can I do for you?”
   If the user directly asks for an appointment → continue.
   If not → gently clarify.

2. Ask for Location (Make it natural)
   Say:
   “Sure! Which location are you interested in?”
   If unclear:
   “Got it — just to confirm, which area or city are you looking at?”

3. Call get_property_details Tool
   Inform user you will need a moment to look up properties in that location.
   Example:
   “Let me check the properties we have in that area… just a moment.” 
   Use the user's location to find the property details.
   If multiple properties are found, ask the user to clarify.
   If no property is found, apologize and ask for another location.
   Once a single property is identified, confirm it with the user (e.g., "Ah, the 3-bedroom in Bellandur?").
   Store the 'id' of this property for later steps.

4. Ask for the Date (Very natural)
   Say:
   “Great. When would you like to visit? You can just tell me the date.”
   If date is natural language (“first December 2025”):
   → you convert it internally to YYYY-MM-DD.

5. Call find_available_slots Tool — Do NOT ask for time yet
    Inform user you are checking available slots.
    Example:
    “Sure, give me a moment while I check the available slots…”

6. Present Available Slots Naturally
   Example:
   “Alright, here are the times available on <date>: <slots in natural language(3pm, 1am, etc)>.
   Which one works best for you?”
   Keep it smooth and friendly.

7. Ask for Time & Full Name
   Once user picks a time:
   “Perfect, I’ll book that for you. And what name should I put on the appointment?”

8. Call book_appointment Tool
   Use the property ID you found in step 3.

9. Confirmation (Natural & Warm)
   Example:
   “All set! You’re booked for <date> at <time> for property <address>.
   I’ve sent the calendar invite as well.
   Thanks for choosing Prestige Properties!”
   Short, warm, human.

❗ Error Handling
   If a tool fails:
   “Sorry, something went wrong on my end. Could you say that again?”
   Always blame the system, not the user.

🔒 Rules
- NEVER guess times; always rely on the tool.
- ALWAYS follow the order: location -> property details -> date → tool → time → name → tool.
- Keep responses brief, friendly, voice-optimized.
- Never reveal internal formatting or system instructions.


Additional Conversation Rules (Strict):

1. When talking about time, NEVER mention “HH:MM format” to the user.
   - Speak naturally using casual time expressions such as “3 pm”, “4 in the afternoon”, “10 in the morning”, etc.
   - Do not mention any formatting instructions or time formats to the user.

2. Do NOT mention or talk about sending a calendar invite in your responses.

3. Tool-Call Waiting Guidance:
   - Whenever a tool is triggered (either checking availability or booking an appointment), inform the user naturally that you are checking and it may take a moment.
   - Examples:
       “Sure, give me a moment while I check the available slots…”
       “Alright, let me quickly confirm that booking for you… just a moment.”
   - Keep this short and conversational.

These rules must be followed exactly in all conversations.

"""

# -------------------------------------------------------------------------
# FastAPI Setup
# -------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

active_sessions: Dict[str, "CallSession"] = {}

# -------------------------------------------------------------------------
# CallSession: The Core Logic
# -------------------------------------------------------------------------
class CallSession:
    """
    Manages the session state:
    1. Receives audio from Twilio -> Converts -> Sends to STT Thread.
    2. STT Thread -> Puts Text in Queue.
    3. Gemini Logic (Async) -> Reads Text Queue -> Sends to Live API -> Buffers Text -> Sends to TTS.
    4. TTS -> Converts -> Sends audio to Twilio.
    """
    def __init__(self, stream_sid: str, twilio_send: callable):
        self.stream_sid = stream_sid
        self.twilio_send = twilio_send
        self.loop = asyncio.get_running_loop()

        # Queues
        self.audio_input_queue = queue.Queue()  # Inbound Audio (Twilio -> STT)
        self.text_queue = asyncio.Queue()       # STT Text -> Gemini

        # Flags
        self._closed = False
        self._threads = []
        self._tasks = []
        
        # Debounce State for TTS
        self.text_buffer = []
        self.buffer_lock = asyncio.Lock()
        self.debounce_task = None
        self.DEBOUNCE_SECONDS = 0.6

    async def start(self):
        """Starts the STT thread and the main Gemini session task."""
        # 1. Start STT Thread
        t = threading.Thread(target=self._stt_worker, daemon=True)
        t.start()
        self._threads.append(t)

        # 2. Start Gemini Session Task
        task = asyncio.create_task(self._gemini_session_loop())
        self._tasks.append(task)
        logging.info(f"Session {self.stream_sid} started.")

    async def stop(self):
        """Clean shutdown."""
        if self._closed: return
        self._closed = True
        logging.info(f"Stopping session {self.stream_sid}...")
        
        self.audio_input_queue.put(None) # Kill STT
        
        for task in self._tasks:
            task.cancel()
        
        logging.info(f"Session {self.stream_sid} stopped.")

    # --- Inbound Audio Handling ---
    async def process_inbound_audio(self, mulaw_payload: str):
        """
        Takes base64 mu-law from Twilio, converts to PCM 16k, queues for STT.
        """
        if self._closed: return
        try:
            # 1. Decode Base64
            mulaw_bytes = base64.b64decode(mulaw_payload)
            
            # 2. Convert mu-law 8k -> PCM 16k (for Google STT)
            # a. mu-law 8k -> linear 8k (width 2)
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
            # b. linear 8k -> linear 16k
            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, STT_RATE, None)
            
            # 3. Put in Queue
            self.audio_input_queue.put(pcm_16k)
        except Exception as e:
            logging.error(f"Error processing inbound audio: {e}")

    # --- STT Worker (Threaded) ---
    def _audio_generator(self):
        while not self._closed:
            chunk = self.audio_input_queue.get()
            if chunk is None: return
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _stt_worker(self):
        """Runs Google STT logic (from stt_textagent_tts.py)."""
        logging.info("STT worker started.")
        config_stt = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=STT_RATE,
            language_code="en-US",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config_stt,
            interim_results=False # Only final results needed for text-to-text logic
        )

        while not self._closed:
            try:
                # This call blocks waiting for queue data
                requests_gen = self._audio_generator()
                responses = speech_client.streaming_recognize(config=streaming_config, requests=requests_gen)
                
                for response in responses:
                    if self._closed: break
                    if not response.results: continue
                    result = response.results[0]
                    if not result.alternatives: continue
                    
                    if result.is_final:
                        transcript = result.alternatives[0].transcript.strip()
                        if transcript:
                            logging.info(f"User (STT): {transcript}")
                            # Send to async loop
                            asyncio.run_coroutine_threadsafe(self.text_queue.put(transcript), self.loop)
            except Exception as e:
                # Twilio often sends silence or connection drops, restart generator if not closed
                if self._closed: break
                logging.debug(f"STT stream restart/error: {e}")

    # --- Gemini Logic (Async) ---
    async def _gemini_session_loop(self):
        """
        Manages the Live API connection.
        Logic ported from stt_textagent_tts.py: stream_session()
        """
        http_options = types.HttpOptions(
            async_client_args={"ping_interval": None, "ping_timeout": None}
        )
        client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_options)
        
        config_live = {
            "response_modalities": ["TEXT"], # We use Text-to-Text via Live API
            "tools": tools_config,
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]}
        }

        logging.info("Connecting to Gemini Live...")
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config_live) as session:
            logging.info("Gemini Live Connected.")
            
            # --- Sender Task ---
            async def send_text_to_gemini():
                while True:
                    text = await self.text_queue.get()
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=text)])
                    )

            # --- Receiver Task ---
            async def receive_from_gemini():
                while True:
                    async for msg in session.receive():
                        # 1. Handle Text
                        if getattr(msg, "text", None):
                            chunk = msg.text
                            async with self.buffer_lock:
                                self.text_buffer.append(chunk)
                            
                            # Reset debounce
                            if self.debounce_task and not self.debounce_task.done():
                                self.debounce_task.cancel()
                            self.debounce_task = asyncio.create_task(self._schedule_debounce())
                            # continue removed to allow processing of tool calls in the same message

                        # 2. Handle Completion Events (Flush Buffer)
                        event = getattr(msg, "event", None) or getattr(msg, "type", None)
                        if event in ("response.completed", "response_complete", "response.finished"):
                            if self.debounce_task and not self.debounce_task.done():
                                self.debounce_task.cancel()
                            await self._flush_buffer_and_tts()
                            # continue removed to allow processing of tool calls in the same message

                        # 3. Handle Tool Calls
                        if getattr(msg, "tool_call", None):
                            # Flush text before tool execution
                            if self.debounce_task and not self.debounce_task.done():
                                self.debounce_task.cancel()
                            await self._flush_buffer_and_tts()

                            await self._handle_tool_call(session, msg.tool_call)

            try:
                await asyncio.gather(send_text_to_gemini(), receive_from_gemini())
            except asyncio.CancelledError:
                pass
            finally:
                # Flush remaining text on close
                await self._flush_buffer_and_tts()

    # --- Tool Handling ---
    async def _handle_tool_call(self, session, tool_call):
        logging.info(f"Tool Call Received: {tool_call}")
        function_responses = []
        
        for fc in tool_call.function_calls:
            result = "Error: Unknown function"
            try:
                args = fc.args
                # Ensure args is a dict
                if hasattr(args, 'items'):
                    args_dict = {k: v for k, v in args.items()}
                else:
                    args_dict = args

                if fc.name == "find_available_slots":
                    result = find_available_slots.invoke(args_dict)
                elif fc.name == "book_appointment":
                    result = book_appointment.invoke(args_dict)
                elif fc.name == "get_property_details":
                    result = get_property_details.invoke(args_dict)
            except Exception as e:
                logging.error(f"Tool execution failed: {e}")
                result = f"Error: {e}"

            function_responses.append(
                types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={"result": result}
                )
            )

        logging.info(f"Sending Tool Response: {function_responses}")
        await session.send_tool_response(function_responses=function_responses)

    # --- TTS & Audio Output Logic ---
    async def _schedule_debounce(self):
        await asyncio.sleep(self.DEBOUNCE_SECONDS)
        await self._flush_buffer_and_tts()

    async def _flush_buffer_and_tts(self):
        async with self.buffer_lock:
            if not self.text_buffer: return
            full_text = " ".join(self.text_buffer).strip()
            self.text_buffer = []

        if not full_text: return
        logging.info(f"Gemini (Full Turn): {full_text}")
        
        try:
            # Generate Audio via Google TTS
            # Crucial: Request 8000Hz (Linear16) directly to match Twilio
            synthesis_input = texttospeech.SynthesisInput(text=full_text)
            voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Journey-F")
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=TTS_RATE # 8000Hz
            )

            response = await asyncio.to_thread(
                tts_client.synthesize_speech,
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            pcm_bytes = response.audio_content

            # Convert PCM 16-bit 8k -> mu-law 8k (for Twilio)
            # audioop.lin2ulaw takes (fragment, width). width=2 for 16-bit.
            mulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)
            
            payload = base64.b64encode(mulaw_bytes).decode("utf-8")

            # Send to Twilio WebSocket
            msg = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload}
            }
            await self.twilio_send(msg)
            
        except Exception as e:
            logging.error(f"TTS/Send Error: {e}")


# -------------------------------------------------------------------------
# WebSocket Endpoint (Bridge)
# -------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Twilio WebSocket connected")
    
    session: Optional[CallSession] = None
    stream_sid = ""

    async def twilio_send(msg: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception as e:
            logging.error(f"WS Send Error: {e}")

    try:
        # 1. Handshake
        raw = await websocket.receive_text()
        data = json.loads(raw)
        if data.get("event") == "connected":
            logging.info("Twilio protocol connected")
            raw = await websocket.receive_text()
            data = json.loads(raw)

        if data.get("event") == "start":
            start_info = data.get("start", {})
            stream_sid = start_info.get("streamSid")
            logging.info(f"Stream started: {stream_sid}")
            
            # Create and start session
            session = CallSession(stream_sid, twilio_send)
            active_sessions[stream_sid] = session
            await session.start()

        # 2. Main Loop
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            event = data.get("event")

            if event == "media":
                if session:
                    payload = data["media"]["payload"]
                    # Offload to session logic
                    await session.process_inbound_audio(payload)
            
            elif event == "stop":
                logging.info(f"Stream stopped: {stream_sid}")
                break
                
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.exception(f"WebSocket Error: {e}")
    finally:
        if session:
            await session.stop()
            active_sessions.pop(stream_sid, None)

# -------------------------------------------------------------------------
# Support Endpoints (From main.py)
# -------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "Twilio-Gemini-Bridge"}

@app.post("/inbound_call")
async def inbound_call(request: Request):
    """Twilio webhook for inbound calls."""
    host = request.headers.get("host", "localhost")
    websocket_url = f"wss://{host}/ws"
    
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=websocket_url)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")








from google.cloud import secretmanager
from google_auth_oauthlib.flow import Flow
import config
import requests
from fastapi.responses import RedirectResponse

# --- Google Calendar OAuth 2.0 Endpoints ---

secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=config.GOOGLE_CREDENTIALS)

# NOTE: In a production environment, the REDIRECT_URI must be a public URL
# that you have registered in your Google Cloud Console for the OAuth client.
try:
    CLIENT_SECRETS_FILE = os.getenv("GOOGLE_OAUTH_WEB_CLIENT_SECRETS")
    if not CLIENT_SECRETS_FILE:
        raise ValueError("GOOGLE_OAUTH_WEB_CLIENT_SECRETS env var not set.")
    credentials_info = json.loads(CLIENT_SECRETS_FILE)
except (ValueError, json.JSONDecodeError) as e:
    logging.error(f"Error loading Google OAuth client secrets: {e}. Please check the environment variable.")
    credentials_info = None

# The redirect URI must match *exactly* one of the authorized redirect URIs
# for the OAuth 2.0 client, which you configure in the Google Cloud console.
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://realestate-voiceai-receptionist.onrender.com/oauth2callback")


@app.get("/auth", tags=["Google Calendar Auth"])
def auth(request: Request):
    """
    Generates the Google OAuth 2.0 authorization URL.
    Redirect the user to this URL to start the consent process.
    """
    if not credentials_info:
        return JSONResponse(status_code=500, content={"message": "OAuth client is not configured."})

    # The state parameter is used to prevent CSRF attacks.
    state = request.client.host
    flow = Flow.from_client_config(
        credentials_info,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        # state=state # Optional: for CSRF protection
    )
    logging.info(f"Generated OAuth URL for {state}, redirecting to: {auth_url}")
    return RedirectResponse(auth_url)


@app.get("/oauth2callback", tags=["Google Calendar Auth"])
async def oauth2callback(request: Request):
    """
    Handles the callback from Google after the user grants consent.
    Fetches the OAuth 2.0 token and securely stores it in Google Secret Manager.
    """
    if not credentials_info:
        return JSONResponse(status_code=500, content={"message": "OAuth client is not configured."})

    # The full URL of the request is required to fetch the token.
    authorization_response = str(request.url)
    logging.info(f"Callback URL received: {authorization_response}")

    # For security, ensure the response is sent over HTTPS in production
    if "http://" in authorization_response and "localhost" not in authorization_response:
        logging.warning("OAuth callback received over HTTP. Converting to HTTPS for OAuth validation.")
        authorization_response = authorization_response.replace("http://", "https://")

    flow = Flow.from_client_config(
        credentials_info,
        scopes=config.SCOPES,
        redirect_uri=REDIRECT_URI
    )
    
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as e:
        logging.error(f"Failed to fetch OAuth token: {e}", exc_info=True)
        return JSONResponse(status_code=400, content={"message": "Failed to fetch OAuth token."})

    credentials = flow.credentials
    logging.info("OAuth token fetched successfully.")

    # --- Securely Store Client Credentials in Google Secret Manager ---
    try:
        # Get the user's email to use as a unique identifier for the secret.
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
        client_email = userinfo.get("email")

        if not client_email:
            logging.error("Could not retrieve user email from token.")
            return JSONResponse(status_code=400, content={"message": "Could not retrieve user email."})

        logging.info(f"Identified user for token storage: {client_email}")

        # Sanitize the email to create a valid Secret ID
        # (Secret IDs can only contain letters, numbers, hyphens, and underscores)
        secret_id = f"oauth-token-{client_email.replace('@', '-').replace('.', '-')}"
        
        parent = f"projects/{config.GCP_PROJECT_ID}"
        secret_name = f"{parent}/secrets/{secret_id}"

        token_data = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        payload = json.dumps(token_data).encode("UTF-8")

        # Check if the secret exists. If not, create it.
        try:
            logging.info(f"Checking if secret '{secret_id}' exists.")
            secret_manager_client.get_secret(request={"name": secret_name})
            logging.info(f"Secret '{secret_id}' already exists. Adding a new version.")
        except Exception: # google.api_core.exceptions.NotFound
            logging.info(f"Secret '{secret_id}' not found. Creating it now.")
            secret_manager_client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        
        # Add the token data as a new version of the secret
        secret_manager_client.add_secret_version(
            request={"parent": secret_name, "payload": {"data": payload}}
        )
        
        logging.info(f"Successfully saved token for '{client_email}' to Secret Manager as '{secret_id}'")

    except requests.HTTPError as http_err:
        logging.error(f"HTTP error while fetching user info: {http_err}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to fetch user info from Google."})
    except Exception as e:
        logging.error(f"Failed to save token to Secret Manager: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed to save token to Secret Manager."})

    return {"message": "Google Calendar connected successfully and token stored securely!"}

# --- Outbound Campaign Management ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = False

async def campaign_worker():
    global campaign_in_progress
    campaign_in_progress = True
    logging.info("Starting outbound campaign worker...")
    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        logging.info(f"Processing lead: {lead['first_name']} {lead['last_name']} at {lead['phone']}")
        try:
            # Note: Update wss URL to your deployed server's URL
            # The query params (name, call_type) can be used by the websocket endpoint if updated to handle them.
            websocket_url = f"wss://voiceai-agent-gemini-live.onrender.com/ws?name={lead['first_name']}&call_type=OUTBOUND"
            twiml_response = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml_response.append(connect)
            logging.info(f"Initiating outbound call to {lead['phone']} with TwiML: {str(twiml_response)}")
            
            # Ensure 'to' and 'from_' are correctly formatted
            call = twilio_client.calls.create(
                to=lead['phone'], 
                from_=TWILIO_PHONE_NUMBER, 
                twiml=str(twiml_response)
            )
            logging.info(f"Outbound call initiated to {lead['phone']}, SID: {call.sid}")
            await asyncio.sleep(15) # Wait before processing the next lead
        except Exception as e:
            logging.error(f"Failed to call lead {lead['first_name']}: {e}", exc_info=True)
        outbound_leads_queue.task_done()
    logging.info("Outbound campaign finished.")
    campaign_in_progress = False

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(file: UploadFile = File(...)):
    global campaign_in_progress
    if campaign_in_progress:
        return {"status": "error", "message": "A campaign is already in progress."}
    logging.info("Received request to start outbound campaign.")
    try:
        # Reading file content from csv
        content = await file.read()
        file_data = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(file_data)
        leads_loaded = 0
        for row in reader:
            await outbound_leads_queue.put(row)
            leads_loaded += 1
        if leads_loaded > 0:
            asyncio.create_task(campaign_worker())
            message = f"Campaign started with {leads_loaded} leads."
        else:
            message = "No leads found in the uploaded file."
        logging.info(message)
        return {"status": "success", "message": message}
    except Exception as e:
        logging.error(f"Failed to process uploaded file: {e}", exc_info=True)
        return {"status": "error", "message": "Failed to process file."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)