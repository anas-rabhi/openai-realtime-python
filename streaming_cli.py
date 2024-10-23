import asyncio
import os

from pynput import keyboard
from client_api import RealtimeClient, TurnDetectionMode, AudioHandler, InputHandler

# Add your own tools here!
# NOTE: FunctionTool parses the docstring to get description, the tool name is the function name
def get_phone_number(name: str) -> str:
    """Get my phone number."""
    if name == "Jerry":
        return "1234567890"
    elif name == "Logan":
        return "0987654321"
    else:
        return "Unknown"


rag_tool= {'name': 'get_toulouse_information', 
 'description': 'get_toulouse_information(query: str) -> str\nGet the information about the city of Toulouse.', 
 'parameters': {'properties': {'query': {'title': 'Query', 'type': 'string'}}, 
                'required': ['query'], 'type': 'object'}, 
                'type': 'function'}
# rag_tool = {
#         "type": "function",
#         'name': 'get_toulouse_information',
#         "function": {
#             "name": "RAG_tool",
#             "description": "Get information about toulouse",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {
#                         "type": "string",
#                         "description": f"""
#                                 Text query to search in the RAG database.
#                                 """,
#                     }
#                 },
#                 "required": ["query"],
#             },
#         }
#     }



# tools = [rag_tool]
tools = [rag_tool]

async def main():
    audio_handler = AudioHandler()
    input_handler = InputHandler()
    input_handler.loop = asyncio.get_running_loop()
    print('1 Hi')
    client = RealtimeClient(
        instructions="Tu es un guide touristique qui répond à des questions en utilisant l'outil RAG et les informations que cet outil te donne. Si l'information n'est pas dans le RAG, tu dis que tu ne sais pas.",
        api_key=os.environ.get("OPENAI_API_KEY"),
        on_text_delta=lambda text: print(f"\nAssistant: {text}", end="", flush=True),
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately(),
        turn_detection_mode=TurnDetectionMode.SERVER_VAD,
        tools=tools,
    )
    print('2 Hi')

    # Start keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=input_handler.on_press)
    listener.start()
    print('3 Hi')
    
    try:
        await client.connect()
        message_handler = asyncio.create_task(client.handle_messages())
        print('4 Hi')
        
        print("Connected to OpenAI Realtime API!")
        print("Audio streaming will start automatically.")
        print("Press 'q' to quit")
        print("")
        
        print('5 Hi')
        # Start continuous audio streaming
        streaming_task = asyncio.create_task(audio_handler.start_streaming(client))
        
        # Simple input loop for quit command
        while True:
            command, _ = await input_handler.command_queue.get()
            
            if command == 'q':
                break
        print('6 Hi')
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_handler.stop_streaming()
        audio_handler.cleanup()
        await client.close()
        print('7 Hi')
if __name__ == "__main__":
    print("Starting Realtime API CLI with Server VAD...")
    asyncio.run(main())
