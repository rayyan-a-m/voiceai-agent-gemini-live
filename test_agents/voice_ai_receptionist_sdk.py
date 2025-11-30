import asyncio
import base64
import os
import sys

import pyaudio
from google import genai
from google.genai import types

# Add parent directory to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import tools
from calender.google_calendar import find_available_slots, book_appointment
from config import TIMEZONE



import json

def to_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"

    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # Attempt to pull __dict__ for complex objects
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)

    return str(obj)  # fallback




# Tool definitions
find_available_slots_decl = {
    "name": "find_available_slots",
    "description": "Find available appointment slots on a specific day.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "date_str": {
                "type": "STRING",
                "description": "The date to check for availability in 'YYYY-MM-DD' format."
            }
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
            "datetime_str": {
                "type": "STRING",
                "description": "The appointment time in 'YYYY-MM-DD HH:MM' format."
            },
            "full_name": {
                "type": "STRING",
                "description": "The full name of the person booking the visit."
            },
            "email": {
                "type": "STRING",
                "description": "The email address of the person."
            },
            "property_id": {
                "type": "STRING",
                "description": "The ID of the property they want to visit."
            }
        },
        "required": ["datetime_str", "full_name", "email", "property_id"]
    }
}

tools_config = [{"function_declarations": [find_available_slots_decl, book_appointment_decl]}]

# 1. Configuration
# Ensure you have your API_KEY set in your environment variables
# export API_KEY="your_api_key_here"
API_KEY = "AIzaSyA8adOsgTI2Id6tF8FL0Gm6UtCqwKicSTA" #os.environ.get("API_KEY")
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

# Audio Settings
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512


async def stream_session(client: genai.Client, mic_stream, spk_stream):
    config = {
        "response_modalities": ["AUDIO"],
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 300,
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Charon"
                }
            }
        },
        "context_window_compression": {
            "sliding_window": {
                "target_tokens": 20000
            }
        },

        # Transcribe what the caller says AND what the AI says
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "enable_affective_dialog": True,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are an AI voice booking agent for Prestige Properties. "
                        "Your job is to speak with customers over phone calls and "
                        "book property site-visit appointments. "
                        "Be polite, professional, and efficient. "
                        "Follow these rules:\n"
                        "1. Greet the caller warmly and ask how you can help.\n"
                        "2. If they want to book a site visit, collect:\n"
                        "   - Full name\n"
                        "   - Phone number\n"
                        f"   - Preferred date & time in {TIMEZONE}\n"
                        "   - Property they want to visit\n"
                        "3. Confirm availability using the backend tools (if provided).\n"
                        "4. Speak naturally, clearly, and keep answers short.\n"
                        "5. If the customer asks about pricing or details, give a brief summary.\n"
                        "6. Avoid long monologues; always end with a question to keep the call moving.\n"
                        "7. If the user asks any irrelevant/unrelated topic, gently redirect to the booking process.\n"
                        "8. Close the call by summarizing the appointment and thanking them.\n"
                    )
                }
            ]
        },
        "proactivity": {
            "proactive_audio": True
        },
        "realtime_input_config": {
            "automatic_activity_detection": {
                "silence_duration_ms": 1200
            }
        },
        "tools": tools_config,
    }


    print("Connecting to Gemini Live via SDK...")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("Connected. Speak now!")
        

        async def send_audio():
            try:
                while True:
                    data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    # print(f"Sending {len(data)} bytes of audio data")
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={INPUT_RATE}"
                        )
                    )
                    await asyncio.sleep(0)
            except Exception as exc:
                print(f"Error sending: {exc}")

        async def receive_audio():
            try:
                while True:
                    async for msg in session.receive():
                        try:
                            import json
                            # Handle tool calls
                            tool_call = getattr(msg, "tool_call", None)
                            safe_msg = to_serializable(msg)
                            print(json.dumps(safe_msg, indent=2))
                            if tool_call:
                                print(f"\nTool call received: {tool_call}")
                                function_responses = []
                                for fc in tool_call.function_calls:
                                    result = "Error: Unknown function"
                                    try:
                                        if fc.name == "find_available_slots":
                                            print(f"Executing find_available_slots with args: {fc.args}")
                                            result = find_available_slots.invoke(fc.args)
                                            print(f"find_available_slots result: {result}")
                                        elif fc.name == "book_appointment":
                                            print(f"Executing book_appointment with args: {fc.args}")
                                            result = book_appointment.invoke(fc.args)
                                            print(f"book_appointment result: {result}")
                                    except Exception as e:
                                        result = f"Error executing {fc.name}: {str(e)}"
                                        print(result)
                                    
                                    function_responses.append(types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": str(result)} 
                                    ))
                                
                                print(f"Sending tool responses...")
                                await session.send_tool_response(function_responses=function_responses)

                            server_content = getattr(msg, "server_content", None)
                            # print("Received server content:", server_content)
                            if not server_content:
                                continue

                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn:
                                parts = getattr(model_turn, "parts", None)
                                if parts:
                                    for part in parts:
                                        if part.text:
                                            print(f"Agent: {part.text}")
                                        inline = getattr(part, "inline_data", None)
                                        if inline and getattr(inline, "data", None):
                                            audio_bytes = inline.data
                                            await asyncio.to_thread(spk_stream.write, audio_bytes)

                            input_transcription = getattr(server_content, "input_transcription", None)
                            if input_transcription:
                                print(f"User: {input_transcription.text}")
                            
                            output_transcription = getattr(server_content, "output_transcription", None)
                            if output_transcription:
                                print(f"Agent (Transcription): {output_transcription.text}")
                        except Exception as exc:
                            print(f"Error processing message: {exc}")
            except Exception as exc:
                print(f"Error receiving: {exc}")

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(receive_audio())
        done, pending = await asyncio.wait(
            [send_task, recv_task], return_when=asyncio.FIRST_EXCEPTION
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception():
                raise task.exception()


async def main():
    if not API_KEY:
        print("Error: API_KEY environment variable not set (use API_KEY or GEMINI_API_KEY).")
        return

    # Configure HTTP options to disable keepalive pings for WebSocket connection
    # This prevents "keepalive ping timeout" errors during long streaming sessions
    http_options = types.HttpOptions(
        api_version='v1alpha',
        async_client_args={
            "ping_interval": None,
            "ping_timeout": None,
        }
    )

    client = genai.Client(api_key=API_KEY, http_options=http_options)

    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    spk_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=OUTPUT_RATE,
        output=True
    )

    try:
        while True:
            try:
                await stream_session(client, mic_stream, spk_stream)
            except Exception as exc:
                print(f"Session error: {exc}. Reconnecting in 2 seconds...")
                await asyncio.sleep(2)
            else:
                break
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        spk_stream.stop_stream()
        spk_stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected.")