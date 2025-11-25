import asyncio
import base64
import json
import os
import pyaudio
import websockets

# 1. Configuration
# Ensure you have your API_KEY set in your environment variables
# export API_KEY="your_api_key_here"
API_KEY = "AIzaSyCa2DFbbLDw4JDNzwCs9ap_SKwh7MEuVd4" #os.environ.get("API_KEY")
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

# The BidiStreaming endpoint for the Gemini API
URI = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={API_KEY}"

# Audio Settings
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512

async def main():
    if not API_KEY:
        print("Error: API_KEY environment variable not set.")
        return

    # Initialize Audio
    p = pyaudio.PyAudio()
    
    # Microphone Input Stream (16kHz)
    mic_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=INPUT_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # Speaker Output Stream (24kHz)
    spk_stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=OUTPUT_RATE,
        output=True
    )

    print(f"Connecting to WebSocket: {URI}")

    async with websockets.connect(URI) as ws:
        
        # 2. Send Setup Message
        # This configures the session with the model and audio capabilities
        setup_msg = {
            "setup": {
                "model": f"models/{MODEL}",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Orus" 
                            }
                        }
                    }
                }
            }
        }
        await ws.send(json.dumps(setup_msg))
        print("Connected. Speak now!")

        # 3. Task: Send Audio from Mic to WebSocket
        async def send_audio():
            while True:
                try:
                    # Read raw PCM data from microphone
                    data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    
                    # Base64 encode the audio
                    b64_data = base64.b64encode(data).decode("utf-8")
                    
                    # Construct RealtimeInput message
                    msg = {
                        "realtime_input": {
                            "media_chunks": [{
                                "mime_type": "audio/pcm;rate=16000",
                                "data": b64_data
                            }]
                        }
                    }
                    await ws.send(json.dumps(msg))
                except Exception as e:
                    print(f"Error sending: {e}")
                    break

        # 4. Task: Receive Audio from WebSocket to Speaker
        async def receive_audio():
            async for raw_msg in ws:
                try:
                    response = json.loads(raw_msg)
                    
                    # Check for server content (audio response)
                    server_content = response.get("serverContent")
                    if server_content:
                        model_turn = server_content.get("modelTurn")
                        if model_turn:
                            for part in model_turn.get("parts", []):
                                inline_data = part.get("inlineData")
                                if inline_data:
                                    # Decode base64 audio and play it
                                    audio_bytes = base64.b64decode(inline_data["data"])
                                    await asyncio.to_thread(spk_stream.write, audio_bytes)
                        
                        # Handle interruption or turn completion if needed
                        if server_content.get("interrupted"):
                            print("Model interrupted")
                            
                except Exception as e:
                    print(f"Error receiving: {e}")
                    break

        # Run both tasks concurrently
        await asyncio.gather(send_audio(), receive_audio())

    # Cleanup
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