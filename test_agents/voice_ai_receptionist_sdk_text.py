import asyncio
import os
import sys
import json

from google import genai
from google.genai import types

# Add parent directory to path to import tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import tools
from calender.google_calendar import find_available_slots, book_appointment

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
MODEL = "gemini-2.0-flash-001"

async def stream_session(client: genai.Client):
    config = {
        "response_modalities": ["TEXT"],
        "tools": tools_config,
    }

    print("Connecting to Gemini Live via SDK (Text Mode)...")

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("Connected. Type your message and press Enter. Type 'q' to quit.")

        async def send_text():
            try:
                while True:
                    text = await asyncio.to_thread(input, "You: ")
                    if text.lower() in ["q", "quit", "exit"]:
                        break

                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=text)]
                        )
                    )
            except EOFError:
                pass

        async def receive_text():
            try:
                while True:
                    async for msg in session.receive():
                        text = msg.text
                        if text:
                            print(text, end="", flush=True)
                        
                        server_content = getattr(msg, "server_content", None)
                        if server_content and server_content.turn_complete:
                            print("\n") # Add newline after turn is complete

                        # Handle tool calls
                        tool_call = getattr(msg, "tool_call", None)
                        if tool_call:
                            print(f"\nTool call received: {tool_call}")
                            function_responses = []
                            for fc in tool_call.function_calls:
                                result = "Error: Unknown function"
                                try:
                                    if fc.name == "find_available_slots":
                                        print(f"Executing find_available_slots with args: {fc.args}")
                                        # LangChain tool invoke expects a dict or single arg
                                        result = find_available_slots.invoke(fc.args)
                                    elif fc.name == "book_appointment":
                                        print(f"Executing book_appointment with args: {fc.args}")
                                        result = book_appointment.invoke(fc.args)
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

            except Exception as exc:
                print(f"Error receiving: {exc}")

        send_task = asyncio.create_task(send_text())
        recv_task = asyncio.create_task(receive_text())
        
        done, pending = await asyncio.wait(
            [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
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
        async_client_args={
            "ping_interval": None,
            "ping_timeout": None,
        }
    )

    client = genai.Client(api_key=API_KEY, http_options=http_options)

    try:
        while True:
            try:
                await stream_session(client)
                break # Exit after session ends (user typed q)
            except Exception as exc:
                print(f"Session error: {exc}. Reconnecting in 2 seconds...")
                await asyncio.sleep(2)
    finally:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected.")