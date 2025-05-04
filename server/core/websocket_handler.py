# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebSocket message handling for Gemini Multimodal Live Proxy Server
"""

import logging
import json
import asyncio
import base64
import traceback
import wave
import os
import datetime
from typing import Any, Optional, List
from google.genai import types

from core.tool_handler import execute_tool
from core.session import create_session, remove_session, SessionState
from core.gemini_client import create_gemini_session

logger = logging.getLogger(__name__)

# Define the output directory relative to this script's location
AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output_audio')
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True) # Ensure the directory exists

def save_audio_to_wav(session_id: str, audio_bytes_list: List[bytes]):
    """Saves the collected audio bytes to a WAV file."""
    if not audio_bytes_list:
        logger.info(f"No audio bytes to save for session {session_id}")
        return

    combined_audio_bytes = b"".join(audio_bytes_list)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = os.path.join(AUDIO_OUTPUT_DIR, f"turn_audio_{session_id}_{timestamp}.wav")

    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)       # Mono
            wf.setsampwidth(2)       # 16-bit PCM = 2 bytes
            wf.setframerate(24000)   # 24kHz sample rate
            wf.writeframes(combined_audio_bytes)
        logger.info(f"Saved turn audio for session {session_id} to {filename}")
    except Exception as e:
        logger.error(f"Error saving WAV file {filename}: {e}")

async def send_error_message(websocket: Any, error_data: dict) -> None:
    """Send formatted error message to client."""
    try:
        await websocket.send(json.dumps({
            "type": "error",
            "data": error_data
        }))
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

async def cleanup_session(session: Optional[SessionState], session_id: str) -> None:
    """Clean up session resources."""
    try:
        if session:
            # Cancel any running tasks
            if session.current_tool_execution:
                session.current_tool_execution.cancel()
                try:
                    await session.current_tool_execution
                except asyncio.CancelledError:
                    pass
            
            # Close Gemini session
            if session.genai_session:
                try:
                    await session.genai_session.close()
                except Exception as e:
                    logger.error(f"Error closing Gemini session: {e}")
            
            # Remove session from active sessions
            remove_session(session_id)
            logger.info(f"Session {session_id} cleaned up and ended")
    except Exception as cleanup_error:
        logger.error(f"Error during session cleanup: {cleanup_error}")

async def handle_messages(websocket: Any, session: SessionState) -> None:
    """Handles bidirectional message flow between client and Gemini."""
    client_task = None
    gemini_task = None
    
    try:
        async with asyncio.TaskGroup() as tg:
            # Task 1: Handle incoming messages from client
            client_task = tg.create_task(handle_client_messages(websocket, session))
            # Task 2: Handle responses from Gemini
            gemini_task = tg.create_task(handle_gemini_responses(websocket, session))
    except* Exception as eg:
        handled = False
        for exc in eg.exceptions:
            if "Quota exceeded" in str(exc):
                logger.info("Quota exceeded error occurred")
                try:
                    # Send error message for UI handling
                    await send_error_message(websocket, {
                        "message": "Quota exceeded.",
                        "action": "Please wait a moment and try again in a few minutes.",
                        "error_type": "quota_exceeded"
                    })
                    # Send text message to show in chat
                    await websocket.send(json.dumps({
                        "type": "text",
                        "data": "⚠️ Quota exceeded. Please wait a moment and try again in a few minutes."
                    }))
                    handled = True
                    break
                except Exception as send_err:
                    logger.error(f"Failed to send quota error message: {send_err}")
            elif "connection closed" in str(exc).lower():
                logger.info("WebSocket connection closed")
                handled = True
                break
        
        if not handled:
            # For other errors, log and re-raise
            logger.error(f"Error in message handling: {eg}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    finally:
        # Cancel tasks if they're still running
        if client_task and not client_task.done():
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass
        
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass

async def handle_client_messages(websocket: Any, session: SessionState) -> None:
    """Handle incoming messages from the client."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if "type" in data:
                    msg_type = data["type"]
                    if msg_type == "audio":
                        logger.debug("Client -> Gemini: Sending audio data...")
                    elif msg_type == "image":
                        logger.debug("Client -> Gemini: Sending image data...")
                    else:
                        # Replace audio data with placeholder in debug output
                        debug_data = data.copy()
                        if "data" in debug_data and debug_data["type"] == "audio":
                            debug_data["data"] = "<audio data>"
                        logger.debug(f"Client -> Gemini: {json.dumps(debug_data, indent=2)}")
                
                # Handle different types of input
                if "type" in data:
                    if data["type"] == "audio":
                        logger.debug("Sending audio to Gemini...")
                        await session.genai_session.send(input={
                            "data": data.get("data"),
                            "mime_type": "audio/pcm"
                        }, end_of_turn=True)
                        logger.debug("Audio sent to Gemini")
                    elif data["type"] == "image":
                        logger.info("Sending image to Gemini...")
                        await session.genai_session.send(input={
                            "data": data.get("data"),
                            "mime_type": "image/jpeg"
                        })
                        logger.info("Image sent to Gemini")
                    elif data["type"] == "text":
                        logger.info("Sending text to Gemini...")
                        await session.genai_session.send(input=data.get("data"), end_of_turn=True)
                        logger.info("Text sent to Gemini")
                    elif data["type"] == "end":
                        logger.info("Received end signal")
                    elif data["type"] == "setup":
                        # Setup message is handled initially in handle_client
                        # We can potentially handle mid-session setup changes here if needed
                        logger.info(f"Received setup message during active session (currently ignored): {data.get('data')}")
                    else:
                        logger.warning(f"Unsupported message type: {data.get('type')}")
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
    except Exception as e:
        if "connection closed" not in str(e).lower():  # Don't log normal connection closes
            logger.error(f"WebSocket connection error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise  # Re-raise to let the parent handle cleanup

async def handle_gemini_responses(websocket: Any, session: SessionState) -> None:
    """Handle responses from Gemini."""
    tool_queue = asyncio.Queue()  # Queue for tool responses
    
    # Start a background task to process tool calls
    tool_processor = asyncio.create_task(process_tool_queue(tool_queue, websocket, session))
    
    try:
        while True:
            async for response in session.genai_session.receive():
                try:
                    # Replace audio data with placeholder in debug output
                    debug_response = str(response)
                    if 'data=' in debug_response and 'mime_type=\'audio/pcm' in debug_response:
                        debug_response = debug_response.split('data=')[0] + 'data=<audio data>' + debug_response.split('mime_type=')[1]
                    logger.debug(f"Received response from Gemini: {debug_response}")
                    
                    # If there's a tool call, add it to the queue and continue
                    if response.tool_call:
                        await tool_queue.put(response.tool_call)
                        continue  # Continue processing other responses while tool executes
                    
                    # Process server content (including audio, text, and transcriptions)
                    # Pass session_id for potential saving
                    await process_server_content(websocket, session, response.server_content)
                    
                except Exception as e:
                    logger.error(f"Error handling Gemini response: {e}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        # Cancel and clean up tool processor
        if tool_processor and not tool_processor.done():
            tool_processor.cancel()
            try:
                await tool_processor
            except asyncio.CancelledError:
                pass
        
        # Clear any remaining items in the queue
        while not tool_queue.empty():
            try:
                tool_queue.get_nowait()
                tool_queue.task_done()
            except asyncio.QueueEmpty:
                break

async def process_tool_queue(queue: asyncio.Queue, websocket: Any, session: SessionState):
    """Process tool calls from the queue."""
    while True:
        tool_call = await queue.get()
        try:
            function_responses = []
            for function_call in tool_call.function_calls:
                # Store the tool execution in session state
                session.current_tool_execution = asyncio.current_task()
                
                # Send function call to client (for UI feedback)
                await websocket.send(json.dumps({
                    "type": "function_call",
                    "data": {
                        "name": function_call.name,
                        "args": function_call.args
                    }
                }))
                
                tool_result = await execute_tool(function_call.name, function_call.args)
                
                # Send function response to client
                await websocket.send(json.dumps({
                    "type": "function_response",
                    "data": tool_result
                }))
                
                function_responses.append(
                    types.FunctionResponse(
                        name=function_call.name,
                        id=function_call.id,
                        response=tool_result
                    )
                )
                
                session.current_tool_execution = None
            
            if function_responses:
                tool_response = types.LiveClientToolResponse(
                    function_responses=function_responses
                )
                await session.genai_session.send(input=tool_response)
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
        finally:
            queue.task_done()

async def process_server_content(websocket: Any, session: SessionState, server_content: Any):
    """Process server content including audio, text, and transcriptions."""
    # Check for interruption first
    if hasattr(server_content, 'interrupted') and server_content.interrupted:
        logger.info("Interruption detected from Gemini")
        await websocket.send(json.dumps({
            "type": "interrupted",
            "data": {
                "message": "Response interrupted by user input"
            }
        }))
        session.is_receiving_response = False
        return

    if server_content.model_turn:
        session.received_model_response = True
        session.is_receiving_response = True
        for part in server_content.model_turn.parts:
            if part.inline_data and hasattr(part.inline_data, 'data'): # Check if data attribute exists
                audio_bytes = part.inline_data.data
                # Append raw bytes to buffer
                session.audio_buffer.append(audio_bytes)
                # Send base64 encoded audio to client for playback
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_base64
                }))
            elif part.text:
                await websocket.send(json.dumps({
                    "type": "text",
                    "data": part.text
                }))

    # Handle Input Transcription
    if hasattr(server_content, 'input_transcription') and server_content.input_transcription and hasattr(server_content.input_transcription, 'text'):
        input_text = server_content.input_transcription.text
        if input_text and input_text.strip():
            logger.debug(f"Received input transcription: {input_text}")
            await websocket.send(json.dumps({
                "type": "input_transcription",
                "data": input_text
            }))

    # Handle Output Transcription
    if hasattr(server_content, 'output_transcription') and server_content.output_transcription and hasattr(server_content.output_transcription, 'text'):
        output_text = server_content.output_transcription.text
        if output_text and output_text.strip():
            logger.debug(f"Received output transcription: {output_text}")
            await websocket.send(json.dumps({
                "type": "output_transcription",
                "data": output_text
            }))

    if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
        # Save accumulated audio from the buffer to a WAV file
        save_audio_to_wav(session.session_id, session.audio_buffer)
        # Clear the buffer for the next turn
        session.audio_buffer.clear()

        await websocket.send(json.dumps({
            "type": "turn_complete"
        }))
        session.received_model_response = False
        session.is_receiving_response = False

async def handle_client(websocket: Any) -> None:
    """Handles a new client connection."""
    session_id = str(id(websocket))
    session = create_session(session_id)
    
    try:
        # --- Send Ready Signal FIRST --- 
        logger.info(f"Sending ready signal to client {session_id}")
        await websocket.send(json.dumps({"ready": True}))
        # ------------------------------

        # --- Wait for initial setup message --- 
        logger.info(f"Waiting for setup message from client {session_id}...")
        setup_message_str = await websocket.recv()
        setup_data = json.loads(setup_message_str)
        
        enable_input_transcription = False
        enable_output_transcription = False

        if setup_data.get("type") == "setup" and "data" in setup_data:
            client_data = setup_data["data"]
            if "modality" in client_data:
                session.response_modality = client_data["modality"]
                if session.response_modality not in ["AUDIO", "TEXT"]:
                    logger.warning(f"Invalid modality '{session.response_modality}' received. Defaulting to AUDIO.")
                    session.response_modality = "AUDIO"
                logger.info(f"Received setup for session {session_id}. Response modality set to: {session.response_modality}")
            else:
                logger.warning(f"Modality missing in setup data for {session_id}. Defaulting to AUDIO.")
                session.response_modality = "AUDIO"
            
            # Check for transcription flags
            if "input_audio_transcription" in client_data and client_data["input_audio_transcription"] is True:
                enable_input_transcription = True
                logger.info(f"Enabling input transcription for session {session_id}")
            if "output_audio_transcription" in client_data and client_data["output_audio_transcription"] is True:
                enable_output_transcription = True
                logger.info(f"Enabling output transcription for session {session_id}")
        else:
            logger.error(f"Invalid or missing setup message from client {session_id}. Closing connection.")
            await send_error_message(websocket, {
                "message": "Invalid setup message.",
                "action": "Please ensure the client sends a valid setup message first.",
                "error_type": "setup_error"
            })
            return # Exit early
        # --- End Wait for setup message ---

        # Create and initialize Gemini session *after* getting modality and transcription flags
        async with await create_gemini_session(
            response_modality=session.response_modality,
            enable_input_transcription=enable_input_transcription,
            enable_output_transcription=enable_output_transcription
        ) as gemini_session:
            session.genai_session = gemini_session
            
            logger.info(f"New session started: {session_id}")
            
            try:
                # Start message handling
                await handle_messages(websocket, session)
            except Exception as e:
                if "code = 1006" in str(e) or "connection closed abnormally" in str(e).lower():
                    logger.info(f"Browser disconnected or refreshed for session {session_id}")
                    await send_error_message(websocket, {
                        "message": "Connection closed unexpectedly",
                        "action": "Reconnecting...",
                        "error_type": "connection_closed"
                    })
                else:
                    raise
            
    except asyncio.TimeoutError:
        logger.info(f"Session {session_id} timed out - this is normal for long idle periods")
        await send_error_message(websocket, {
            "message": "Session timed out due to inactivity.",
            "action": "You can start a new conversation.",
            "error_type": "timeout"
        })
    except Exception as e:
        logger.error(f"Error in handle_client: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        if "connection closed" in str(e).lower() or "websocket" in str(e).lower():
            logger.info(f"WebSocket connection closed for session {session_id}")
            # No need to send error message as connection is already closed
        else:
            await send_error_message(websocket, {
                "message": "An unexpected error occurred.",
                "action": "Please try again.",
                "error_type": "general"
            })
    finally:
        # Always ensure cleanup happens
        await cleanup_session(session, session_id) 