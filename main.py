# app_twilio_gemini.py
import asyncio
import base64
import json
import logging
import os
import uuid
import audioop
from typing import Any, Dict, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse
from google.genai import types

# Load .env if present
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----- Configuration (from env) -----
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025")
GEMINI_WS_HOST = "generativelanguage.googleapis.com"
GEMINI_WS_PATH = "ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
# We'll target 16000 Hz PCM for Gemini input/output for easy resampling
GEMINI_AUDIO_RATE = 16000

OUTBOUND_CALL_INTERVAL_SECONDS = int(os.getenv("OUTBOUND_CALL_INTERVAL_SECONDS", "2"))

# --- Twilio client
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Active sessions: streamSid -> CallSession
active_sessions: Dict[str, "CallSession"] = {}

# Utility: Gemini WS URI
def gemini_ws_uri(api_key: str) -> str:
    return f"wss://{GEMINI_WS_HOST}/{GEMINI_WS_PATH}?key={api_key}"

# -------------------------
# CallSession: manages Gemini websocket and audio queues
# -------------------------
class CallSession:
    """
    Manages a bidirectional stream between Twilio WebSocket and Gemini Live WebSocket
    - inbound queue: audio from Twilio to send to Gemini (linear PCM @ GEMINI_AUDIO_RATE)
    - receives Gemini audio and forwards transformed audio back to Twilio as mulaw@8k
    """
    def __init__(self, call_sid: str, stream_sid: str, twilio_send: callable):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.twilio_send = twilio_send  # coroutine fn to send messages to Twilio WS
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._gemini_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._tasks: list[asyncio.Task] = []
        self._closed = asyncio.Event()
        self._media_chunk_counter = 0  # for Twilio outbound chunk
        logging.info("CallSession created for stream %s", stream_sid)

    async def start(self):
        """Open Gemini websocket and start sender/receiver tasks"""
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured in environment")

        uri = gemini_ws_uri(GEMINI_API_KEY)
        logging.info("Connecting to Gemini Live: %s", uri)
        # connect to Gemini (bidirectional)
        self._gemini_ws = await websockets.connect(uri, max_size=None)
        # Send initial setup message to Gemini
        setup_msg = {
            "setup": {
                "model": f"models/{GEMINI_MODEL}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "realtime_input_config": {
                        "automatic_activity_detection": {
                            "disabled": False, # default
                            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
                            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
                            "prefix_padding_ms": 20,
                            "silence_duration_ms": 100,
                        }
                    },
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Orus"
                            }
                        },
                        # Request Gemini to produce PCM at GEMINI_AUDIO_RATE
                        "audio_format": {"mime_type": f"audio/pcm;rate={GEMINI_AUDIO_RATE}"}
                    }
                }
            }
        }
        await self._gemini_ws.send(json.dumps(setup_msg))
        logging.info("Sent Gemini setup message")

        # start tasks
        self._tasks = [
            asyncio.create_task(self._sender_loop(), name=f"session-sender-{self.stream_sid}"),
            asyncio.create_task(self._receiver_loop(), name=f"session-recv-{self.stream_sid}")
        ]

    async def stop(self):
        logging.info("Stopping CallSession %s", self.stream_sid)
        # mark closed so loops can exit
        self._closed.set()
        for t in self._tasks:
            t.cancel()
        # close gemini ws
        try:
            if self._gemini_ws and not self._gemini_ws.closed:
                await self._gemini_ws.close()
        except Exception:
            pass
        logging.info("CallSession %s stopped", self.stream_sid)

    async def send_audio_from_twilio(self, pcm16_bytes: bytes):
        """Called by outer websocket handler when Twilio sends inbound audio.
        We put PCM16 @ GEMINI_AUDIO_RATE bytes into the queue; sender loop will base64-encode into Gemini message.
        """
        await self.input_queue.put(pcm16_bytes)

    async def _sender_loop(self):
        """Consume input_queue and send to Gemini as realtime_input media_chunks (base64 of PCM)"""
        try:
            while not self._closed.is_set():
                pcm_chunk: bytes = await self.input_queue.get()
                if pcm_chunk is None:
                    break
                # base64 encode PCM and send as realtime_input message
                b64_data = base64.b64encode(pcm_chunk).decode("utf-8")
                msg = {
                    "realtime_input": {
                        "media_chunks": [{
                            "mime_type": f"audio/pcm;rate={GEMINI_AUDIO_RATE}",
                            "data": b64_data
                        }]
                    }
                }
                try:
                    await self._gemini_ws.send(json.dumps(msg))
                except Exception as e:
                    logging.error("Failed to send audio to Gemini: %s", e)
                    break
        except asyncio.CancelledError:
            logging.debug("Sender loop cancelled")
        except Exception as e:
            logging.exception("Sender loop exception: %s", e)

    async def _receiver_loop(self):
        """
        Receive messages from Gemini. When audio is received (inlineData.data),
        decode base64 -> PCM bytes -> convert to mulaw@8k -> send to Twilio as 'media' message.
        """
        try:
            async for raw_msg in self._gemini_ws:
                if self._closed.is_set():
                    break
                try:
                    response = json.loads(raw_msg)
                except Exception:
                    continue

                server_content = response.get("serverContent")
                if not server_content:
                    continue

                model_turn = server_content.get("modelTurn")
                if not model_turn:
                    continue

                # iterate parts to find inlineData audio
                for part in model_turn.get("parts", []):
                    inline = part.get("inlineData")
                    if not inline:
                        continue
                    # inline contains {"mime_type":"audio/pcm;rate=16000","data":"<b64>"}
                    mime = inline.get("mime_type", "")
                    b64 = inline.get("data")
                    if not b64:
                        continue
                    try:
                        pcm_bytes = base64.b64decode(b64)
                    except Exception as e:
                        logging.exception("Failed to decode Gemini audio b64: %s", e)
                        continue

                    # If Gemini audio sample rate != 8000, we need to resample and convert
                    # We requested GEMINI_AUDIO_RATE. Expect pcm16 linear16 samples.
                    # Convert to mu-law 8k for Twilio:
                    try:
                        # Ensure pcm bytes are 16-bit little-endian signed integers
                        sample_width = 2  # bytes per sample
                        # If GEMINI_AUDIO_RATE != 8000, resample to 8000
                        if GEMINI_AUDIO_RATE != 8000:
                            # audioop.ratecv requires mono PCM, width in bytes
                            # It returns (newfragment, state)
                            new_frames, _ = audioop.ratecv(pcm_bytes, sample_width, 1, GEMINI_AUDIO_RATE, 8000, None)
                            pcm_for_mulaw = new_frames
                        else:
                            pcm_for_mulaw = pcm_bytes

                        # Convert linear16 PCM to mu-law (returns bytes of mu-law)
                        mulaw_bytes = audioop.lin2ulaw(pcm_for_mulaw, 2)
                    except Exception as e:
                        logging.exception("Error converting Gemini PCM to mu-law: %s", e)
                        continue

                    # Send to Twilio websocket as a media message
                    self._media_chunk_counter += 1
                    media_msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            # Twilio expects pure mu-law bytes base64 encoded
                            "payload": base64.b64encode(mulaw_bytes).decode("ascii")
                        }
                    }
                    # Optionally include a mark so Twilio will notify when playback completes
                    try:
                        await self.twilio_send(media_msg)
                        # send a mark to be notified when audio finishes playing
                        mark = {
                            "event": "mark",
                            "streamSid": self.stream_sid,
                            "sequenceNumber": str(self._media_chunk_counter),
                            "mark": {"name": f"msg-{self._media_chunk_counter}-{uuid.uuid4().hex[:6]}"}
                        }
                        await self.twilio_send(mark)
                    except Exception as e:
                        logging.exception("Failed to send media to Twilio: %s", e)
                        # if twilio send fails, continue (Twilio may have disconnected)
        except asyncio.CancelledError:
            logging.debug("Receiver loop cancelled")
        except Exception as e:
            logging.exception("Receiver loop error: %s", e)

# -------------------------
# FastAPI endpoints
# -------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, name: str = "Valued Customer", call_type: str = "INBOUND"):
    """
    Handles Twilio Media Streams websocket (bidirectional)
    - Reads Twilio messages (start, media, stop)
    - For media inbound, convert from mu-law@8k -> pcm16@GEMINI_AUDIO_RATE and pass to CallSession
    - For outbound, CallSession will send 'media' messages back to Twilio via twilio_ws_send
    """
    await websocket.accept()
    logging.info("Accepted websocket from Twilio. call_type=%s name=%s", call_type, name)
    call_sid = "unknown"
    stream_sid = "unknown"
    session: Optional[CallSession] = None

    async def twilio_ws_send(msg: Dict[str, Any]):
        # send JSON text to Twilio websocket
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            logging.exception("Failed to send to Twilio websocket")

    try:
        # First messages: connected, start
        raw_connected = await websocket.receive_text()
        connected = json.loads(raw_connected)
        if connected.get("event") != "connected":
            logging.error("Expected connected event, got %s", connected.get("event"))
            await websocket.close()
            return
        logging.info("Twilio connected: %s", connected.get("protocol"))

        raw_start = await websocket.receive_text()
        start = json.loads(raw_start)
        if start.get("event") != "start":
            logging.error("Expected start event, got %s", start.get("event"))
            await websocket.close()
            return

        start_info = start.get("start", {})
        call_sid = start_info.get("callSid", "unknown")
        stream_sid = start_info.get("streamSid", "unknown")
        logging.info("Stream started: call=%s stream=%s", call_sid, stream_sid)

        # create session and start Gemini connection
        session = CallSession(call_sid=call_sid, stream_sid=stream_sid, twilio_send=twilio_ws_send)
        await session.start()
        active_sessions[stream_sid] = session

        # Now process incoming messages
        while True:
            msg_text = await websocket.receive_text()
            data = json.loads(msg_text)
            event = data.get("event")
            if event == "media":
                # Twilio sends a base64 payload which is mu-law @ 8000Hz
                streamSid = data.get("streamSid", stream_sid)
                media = data.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue
                try:
                    mulaw_bytes = base64.b64decode(payload_b64)
                    # convert mu-law@8k -> linear16@16k for Gemini
                    # 1) ulaw2lin -> PCM16@8k
                    pcm8 = audioop.ulaw2lin(mulaw_bytes, 2)
                    # 2) resample 8k -> GEMINI_AUDIO_RATE (e.g., 16000)
                    if GEMINI_AUDIO_RATE != 8000:
                        pcm16, _ = audioop.ratecv(pcm8, 2, 1, 8000, GEMINI_AUDIO_RATE, None)
                    else:
                        pcm16 = pcm8
                    # forward PCM16 bytes to CallSession (which will base64 and send to Gemini)
                    await session.send_audio_from_twilio(pcm16)
                except Exception:
                    logging.exception("Failed to convert inbound Twilio payload")
            elif event == "stop":
                logging.info("Twilio stop event, stream=%s", stream_sid)
                break
            elif event == "mark":
                # Twilio returns marks for our sent media messages playback completion
                logging.debug("Received mark from Twilio: %s", data.get("mark"))
            elif event == "dtmf":
                logging.info("Received DTMF: %s", data)
            else:
                logging.debug("Unhandled Twilio event: %s", event)

    except WebSocketDisconnect:
        logging.info("Twilio websocket disconnected for call %s", call_sid)
    except Exception as e:
        logging.exception("Error in Twilio websocket handler: %s", e)
    finally:
        # cleanup
        if session:
            await session.stop()
        if stream_sid in active_sessions:
            active_sessions.pop(stream_sid, None)
        try:
            await websocket.close()
        except Exception:
            pass
        logging.info("Cleaned up session for %s", stream_sid)


# --- Outbound campaign support (CSV upload) ---
outbound_leads_queue = asyncio.Queue()
campaign_in_progress = asyncio.Event()

async def campaign_worker(server_host: str):
    logging.info("Starting outbound campaign worker...")
    campaign_in_progress.set()
    while not outbound_leads_queue.empty():
        lead = await outbound_leads_queue.get()
        first_name = lead.get("first_name", "").strip()
        phone_number = lead.get("phone", "").strip()
        if not phone_number:
            outbound_leads_queue.task_done()
            continue
        try:
            websocket_url = f"wss://{server_host}/ws?name={first_name}&call_type=OUTBOUND"
            twiml = VoiceResponse()
            connect = Connect()
            connect.stream(url=websocket_url)
            twiml.append(connect)
            call = twilio_client.calls.create(
                to=phone_number,
                from_=TWILIO_PHONE_NUMBER,
                twiml=str(twiml)
            )
            logging.info("Started outbound call to %s SID=%s", phone_number, call.sid)
            await asyncio.sleep(OUTBOUND_CALL_INTERVAL_SECONDS)
        except Exception:
            logging.exception("Failed to place outbound call")
        finally:
            outbound_leads_queue.task_done()
    campaign_in_progress.clear()
    logging.info("Outbound campaign finished.")

@app.post("/start_outbound_campaign")
async def start_outbound_campaign(request: Request, file: UploadFile = File(...)):
    if campaign_in_progress.is_set():
        return JSONResponse(status_code=409, content={"status": "error", "message": "Campaign already in progress"})
    content = await file.read()
    text = content.decode("utf-8-sig")
    import csv, io
    reader = csv.DictReader(io.StringIO(text))
    leads = list(reader)
    if not leads:
        return JSONResponse(status_code=400, content={"status":"error","message":"No leads"})
    for lead in leads:
        await outbound_leads_queue.put(lead)
    server_host = request.headers.get("host", "localhost")
    asyncio.create_task(campaign_worker(server_host))
    return {"status":"success", "message": f"Started campaign with {len(leads)} leads."}


@app.post("/inbound_call")
async def inbound_call(request: Request):
    """
    Twilio webhook that returns TwiML to connect the call to our WebSocket endpoint.
    Twilio will then open a WebSocket and stream audio events there.
    """
    host = request.headers.get("host", "localhost")
    websocket_url = f"wss://{host}/ws?call_type=INBOUND"
    response = VoiceResponse()
    connect = Connect()
    # use Connect.stream for bidirectional: Twilio will send inbound audio to /ws and we can send audio back from /ws.
    connect.stream(url=websocket_url)
    response.append(connect)
    logging.info("Inbound TwiML generated for host %s", host)
    return Response(content=str(response), media_type="application/xml")



# If run directly:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_twilio_gemini:app", host="0.0.0.0", port=8000, reload=False)
