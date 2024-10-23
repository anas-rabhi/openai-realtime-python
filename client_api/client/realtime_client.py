import asyncio
import websockets
import json
import base64
import io

from typing import Optional, Callable, List, Dict, Any
from enum import Enum
from pydub import AudioSegment



class TurnDetectionMode(Enum):
    SERVER_VAD = "server_vad"
    MANUAL = "manual"

class RealtimeClient:
    """
    A client for interacting with the OpenAI Realtime API.

    This class provides methods to connect to the Realtime API, send text and audio data,
    handle responses, and manage the WebSocket connection.

    Attributes:
        api_key (str): 
            The API key for authentication.
        model (str): 
            The model to use for text and audio processing.
        voice (str): 
            The voice to use for audio output.
        instructions (str): 
            The instructions for the chatbot.
        turn_detection_mode (TurnDetectionMode): 
            The mode for turn detection.
        tools (List[BaseTool]): 
            The tools to use for function calling.
        on_text_delta (Callable[[str], None]): 
            Callback for text delta events. 
            Takes in a string and returns nothing.
        on_audio_delta (Callable[[bytes], None]): 
            Callback for audio delta events. 
            Takes in bytes and returns nothing.
        on_interrupt (Callable[[], None]): 
            Callback for user interrupt events, should be used to stop audio playback.
        extra_event_handlers (Dict[str, Callable[[Dict[str, Any]], None]]): 
            Additional event handlers. 
            Is a mapping of event names to functions that process the event payload.
    """
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        instructions: str = "You are a helpful, witty, and friendly AI. Act like a human, but remember that you aren't a human and that you can't do human things in the real world. Your voice and personality should be warm and engaging, with a lively and playful tone.",
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.SERVER_VAD,
        tools: Optional[List] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        extra_event_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.ws = None
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_interrupt = on_interrupt
        self.instructions = instructions
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.extra_event_handlers = extra_event_handlers or {}
        self.turn_detection_mode = turn_detection_mode
        
        # Properly format tools with required fields
        self.tools = tools
        print(self.tools)


        # Track current response state
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        
        # Adjust audio buffer settings for smoother streaming
        self._audio_buffer = b''
        self._min_audio_chunk_size = 3200  # Reduced for more frequent chunks
        self._last_audio_time = 0
        self._audio_chunk_delay = 0.05  # 50ms delay between chunks
        self._playback_rate = 0.95  # Slightly slower than realtime
        
        self._current_content_index = 0
        self._current_audio_sample_count = 0

    async def connect(self) -> None:
        """Establish WebSocket connection with the Realtime API."""
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        self.ws = await websockets.connect(url, extra_headers=headers)
        
        # Set up default session configuration
        base_config = {
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "tools": self.tools,
            "tool_choice": "auto",
            "temperature": 0.7,
        }
        
        if self.turn_detection_mode == TurnDetectionMode.SERVER_VAD:
            base_config["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 200
            }
        
        await self.update_session(base_config)

    async def update_session(self, config: Dict[str, Any]) -> None:
        """Update session configuration."""
        event = {
            "type": "session.update",
            "session": config
        }
        await self.ws.send(json.dumps(event))

    async def send_text(self, text: str) -> None:
        """Send text message to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }
        await self.ws.send(json.dumps(event))
        await self.create_response()

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the API."""
        # Convert audio to required format (24kHz, mono, PCM16 little-endian)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        pcm_data = base64.b64encode(audio.raw_data).decode()

        # Append audio to buffer
        append_event = {
            "type": "input_audio_buffer.append",
            "audio": pcm_data
        }
        await self.ws.send(json.dumps(append_event))
        
        # Commit the buffer
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        await self.ws.send(json.dumps(commit_event))
        
        # In manual mode, we need to explicitly request a response
        if self.turn_detection_mode == TurnDetectionMode.MANUAL:
            await self.create_response()

    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Stream raw audio data to the API."""
        audio_b64 = base64.b64encode(audio_chunk).decode()
        
        append_event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        await self.ws.send(json.dumps(append_event))

    async def create_response(self, functions: Optional[List[Dict[str, Any]]] = None) -> None:
        """Request a response from the API."""
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        }
        
        if functions:
            event["response"]["tools"] = functions
            
        await self.ws.send(json.dumps(event))

    async def send_function_result(self, call_id: str, result: Any) -> None:
        """Send function call result back to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }
        await self.ws.send(json.dumps(event))
        await self.create_response()  # Explicitly request new response after function call

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        event = {
            "type": "response.cancel"
        }
        await self.ws.send(json.dumps(event))
    
    async def truncate_response(self):
        """
        Truncate the conversation item to match what was actually played.
        This is important for handling interruptions correctly.
        """
        if self._current_item_id:
            # Calculate milliseconds of audio played
            # PCM16 at 24kHz = 48000 bytes per second (2 bytes per sample * 24000 samples)
            audio_ms = (self._current_audio_sample_count * 1000) // 24000

            event = {
                "type": "conversation.item.truncate",
                "item_id": self._current_item_id,
                "content_index": self._current_content_index,
                "audio_end_ms": audio_ms
            }
            await self.ws.send(json.dumps(event))

    async def call_tool(self, call_id: str,tool_name: str, tool_arguments: Dict[str, Any]) -> None:
        from .utils import rag_pipeline
        tool_result = rag_pipeline(tool_arguments['query'])
    
        await self.send_function_result(call_id, str(tool_result))

    async def handle_interruption(self):
        """Handle user interruption of the current response."""
        if not self._is_responding:
            return
            
        print("\n[Handling interruption]")
        
        # 1. Cancel the current response
        if self._current_response_id:
            await self.cancel_response()
        
        # 2. Truncate the conversation item to what was actually played
        if self._current_item_id:
            await self.truncate_response()
            
        self._is_responding = False
        self._current_response_id = None
        self._current_item_id = None

    async def handle_messages(self) -> None:
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                if event_type == "error":
                    print(f"Error: {event.get('error', {}).get('message', 'Unknown error')}")
                    print(f"Code: {event.get('error', {}).get('code')}")
                    print(f"Event ID: {event.get('error', {}).get('event_id')}")
                    continue
                
                # Track response state
                elif event_type == "response.created":
                    self._current_response_id = event.get("response", {}).get("id")
                    self._is_responding = True
                    self._audio_buffer = b''  # Reset audio buffer at start of response
                
                elif event_type == "response.output_item.added":
                    self._current_item_id = event.get("item", {}).get("id")
                
                elif event_type == "response.done":
                    # Play any remaining audio in the buffer
                    if self._audio_buffer and self.on_audio_delta:
                        final_chunk = self._audio_buffer
                        self._audio_buffer = b''
                        self.on_audio_delta(final_chunk)
                    
                    self._is_responding = False
                    self._current_response_id = None
                    self._current_item_id = None
                
                # Handle interruptions
                elif event_type == "input_audio_buffer.speech_started":
                    print("\n[Speech detected]")
                    if self._is_responding:
                        await self.handle_interruption()

                    if self.on_interrupt:
                        self.on_interrupt()

                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("\n[Speech ended]")
                
                # Handle normal response events
                elif event_type == "response.text.delta":
                    if self.on_text_delta:
                        self.on_text_delta(event["delta"])
                        
                elif event_type == "response.audio.delta":
                    if self.on_audio_delta:
                        audio_bytes = base64.b64decode(event["delta"])
                        self._audio_buffer += audio_bytes
                        
                        current_time = asyncio.get_event_loop().time()
                        time_since_last = current_time - self._last_audio_time
                        
                        # Calculate ideal chunk duration (in seconds)
                        # PCM16 at 24kHz = 48000 bytes per second
                        chunk_duration = len(self._audio_buffer) / 48000
                        target_delay = chunk_duration * self._playback_rate
                        
                        if (len(self._audio_buffer) >= self._min_audio_chunk_size and 
                            time_since_last >= target_delay):
                            chunk_to_play = self._audio_buffer
                            self._audio_buffer = b''
                            self._last_audio_time = current_time
                            
                            # Update audio sample count for truncation
                            self._current_audio_sample_count += len(chunk_to_play) // 2
                            
                            # Add small delay proportional to chunk size
                            await asyncio.sleep(target_delay * 0.1)  # 10% of chunk duration
                            self.on_audio_delta(chunk_to_play)

                elif event_type == "response.function_call_arguments.done":
                    await self.call_tool(event["call_id"], event['name'], json.loads(event['arguments']))

                elif event_type in self.extra_event_handlers:
                    self.extra_event_handlers[event_type](event)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {str(e)}")
        except Exception as e:
            print(f"Error in message handling: {str(e)}")
            raise

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()

