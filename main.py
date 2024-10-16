import websocket
import json
import pyaudio
import base64
import threading
import time
import os
import numpy as np
import queue
import logging
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Replace with your actual API key

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000

# VAD parameters
SILENCE_THRESHOLD = 250
SILENCE_DURATION = 0.3

# WebSocket connection
ws = None

# Audio interface
p = pyaudio.PyAudio()

# Flags to control recording and processing
is_speaking = False
is_processing = False

# Add a global variable for the audio output stream
audio_output_stream = None
audio_queue = queue.Queue()

# Add a new global flag for user speech detection
is_user_speaking = False

def on_message(ws, message):
    try:
        event = json.loads(message)
        logger.info(f"Received event: {event['type']}")
        
        if event['type'] == 'response.audio.delta':
            play_audio(base64.b64decode(event['delta']))
        elif event['type'] == 'response.text.delta':
            logger.info(f"Text response: {event.get('delta', '')}")
        elif event['type'] == 'error':
            logger.error(f"Error from API: {event.get('message', 'Unknown error')}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode message: {message}")
    except KeyError as e:
        logger.error(f"Missing key in event: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in on_message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    logger.info("WebSocket connection opened")
    # Set up the initial session
    
    ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "input_audio_transcription": True,
            "turn_detection": "server_vad",
            "modalities": ["text", "audio"],
        "instructions": "AT EACH MESSAGE SAY 'HELLO'.",
        "voice": "alloy",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1"
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200
        },
        "tools": [
        ]
        }
    }))

def connect_websocket():
    global ws
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        header={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1"
        },
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    return ws_thread

def is_silent(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(audio_array)))
    return rms < SILENCE_THRESHOLD

def record_and_stream_audio():
    global is_speaking, is_processing, is_user_speaking
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    logger.info("Listening...")

    silence_start = None
    audio_buffer = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            if is_silent(data):
                if is_speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        is_speaking = False
                        is_user_speaking = False
                        if not is_processing:
                            is_processing = True
                            threading.Thread(target=process_audio, args=(audio_buffer,)).start()
                        audio_buffer = []
                        silence_start = None
            else:
                silence_start = None
                if not is_speaking:
                    is_speaking = True
                    is_user_speaking = True
                    logger.info("Speech detected, listening...")
                
            if is_speaking:
                audio_buffer.append(data)
    except Exception as e:
        logger.error(f"Error in record_and_stream_audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()

def process_audio(audio_buffer):
    global is_processing
    logger.info("Processing audio...")
    if ws and ws.sock and ws.sock.connected:
        # Send the accumulated audio data
        send_audio(b''.join(audio_buffer))
        # Commit the audio buffer
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # Request a response
        ws.send(json.dumps({
            "event_id": "event_123",
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Please assist the user.",
            }
        }))
    is_processing = False

def send_audio(audio_data):
    if ws and ws.sock and ws.sock.connected:
        # Convert the audio data to base64
        base64_audio = base64.b64encode(audio_data).decode()
        ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }))

def play_audio(audio_data):
    audio_queue.put(audio_data)

def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

def audio_playback_thread():
    global audio_output_stream
    
    while True:
        try:
            if is_user_speaking:
                clear_audio_queue()
                if audio_output_stream:
                    audio_output_stream.stop_stream()
                time.sleep(0.05)
                continue

            audio_chunk = audio_queue.get(timeout=0.05)
            if audio_output_stream is None or not audio_output_stream.is_active():
                audio_output_stream = p.open(format=FORMAT,
                                             channels=CHANNELS,
                                             rate=RATE,
                                             output=True)
            audio_output_stream.write(audio_chunk)
        except queue.Empty:
            if audio_output_stream and audio_output_stream.is_active():
                audio_output_stream.stop_stream()
        except Exception as e:
            logger.error(f"Error in audio playback: {e}")

def main():
    ws_thread = connect_websocket()
    time.sleep(2)  # Wait for WebSocket connection to establish
    ws.send(json.dumps({
        "event_id": "event_123",
        "type": "session.update",
        "session": {
            "input_audio_transcription": True,
            "turn_detection": "server_vad",
            "modalities": ["text", "audio"],
        "instructions": "AT EACH MESSAGE SAY 'HELLO'.",
        "voice": "alloy",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {
            "model": "whisper-1"
        },
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200
        },
        "tools": [
        ]
        }
    }))
    record_thread = threading.Thread(target=record_and_stream_audio)
    record_thread.start()

    playback_thread = threading.Thread(target=audio_playback_thread)
    playback_thread.daemon = True
    playback_thread.start()

    try:
        while True:
            time.sleep(0.1)
            if is_user_speaking and audio_output_stream and audio_output_stream.is_active():
                logger.info("User started speaking, interrupting playback...")
                clear_audio_queue()
                audio_output_stream.stop_stream()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        if ws:
            ws.close()
        if audio_output_stream:
            audio_output_stream.stop_stream()
            audio_output_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
