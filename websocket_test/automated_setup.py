import asyncio
import json
import websockets
import ssl
import certifi
import google.auth
from google.auth.transport.requests import Request
import base64
import io
from PIL import Image
import logging
from enum import Enum, auto
import pyaudio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

GEMINI_LIVE_API_URL = "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
MODEL_NAME_ID = "gemini-2.0-flash-live-preview-04-09"
IMAGE_PATH = "websocket_test/test_image.png"
OUTPUT_MODALITY = "AUDIO"
SYSTEM_INSTRUCTIONS_PATH = "websocket_test/system-instruction_original.txt"

DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_FRAMERATE = 24000

PRINT_BLACKBOARD_TOOL_SCHEMA = {
    "name": "print_blackboard",
    "description": "Output the summary of the key concepts and knowledge of the current round of conversation to the user in text form and display them on the blackboard. Scientific formulas should be in standard latex with '$' surroundings and the explanations should include necessary explanations. Also, bold some important terms with '**' surroundings according to the explanation. You cannot and must not output 'Step x'on the blackboard, just the key knowledge of the current step.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "content": {
                "type": "STRING",
                "description": "The content need to be displayed on the blackboard, showed to students."
            },
            "explanation_is_finish": {
                "type": "BOOLEAN",
                "description": "Indicates whether the teacher's explanation is over. True means explanation is finish, False otherwise"
            }
        }
    }
}

class ScriptState(Enum):
    INITIAL = auto()
    CONNECTING = auto()
    WAITING_FOR_SETUP_COMPLETE = auto()
    PROCESSING_ACTIONS = auto()
    WAITING_FOR_MODEL_RESPONSE = auto()
    CLOSING = auto()
    FINISHED = auto()
    ERROR = auto()

async def get_access_token():
    try:
        creds, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if not creds.valid:
            auth_req = Request()
            creds.refresh(auth_req)
        logger.info(f"Successfully obtained access token for project: {project_id}")
        return creds.token, project_id
    except Exception as e:
        logger.error(f"Error getting access token or project ID: {e}", exc_info=True)
        raise

def load_system_instructions(file_path=SYSTEM_INSTRUCTIONS_PATH):
    try:
        with open(file_path, "r") as f:
            system_instructions_text = f.read()
        logger.info(f"Successfully loaded system instructions from {file_path}")
        return system_instructions_text
    except FileNotFoundError:
        logger.error(f"System instructions file {file_path} not found. Using a default system instruction.")
        return "You are a helpful assistant."
    except Exception as e:
        logger.error(f"Error reading system instructions file {file_path}: {e}. Using a default system instruction.")
        return "You are a helpful assistant."

def prepare_image_chunk_for_realtime_input(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_bytes = image_io.getvalue()
        
        encoded_image_string = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"Successfully prepared image {image_path} as JPEG for realtimeInput chunk.")
        return {"mimeType": "image/jpeg", "data": encoded_image_string}
    except FileNotFoundError:
        logger.error(f"Image file {image_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Error preparing image {image_path} for realtimeInput: {e}", exc_info=True)
        return None

async def process_websocket_action_sequence(action_sequence: list):
    logger.info(f"Starting query with {len(action_sequence)} actions. Connecting to {GEMINI_LIVE_API_URL}...")
    
    bearer_token, project_id = await get_access_token()
    if not bearer_token or not project_id:
        logger.error("Failed to get bearer token or project ID. Exiting.")
        return

    location = "us-central1"
    publisher = "google"
    dynamic_model_name = f"projects/{project_id}/locations/{location}/publishers/{publisher}/models/{MODEL_NAME_ID}"
    logger.info(f"Using model: {dynamic_model_name}")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    current_script_state = ScriptState.INITIAL
    action_index = 0
    
    current_turn_generation_has_completed = False
    current_turn_has_completed = False
    last_sent_action_description = "N/A"

    pya = None
    output_stream = None

    try:
        if OUTPUT_MODALITY == "AUDIO":
            logger.info("Audio output modality selected. Initializing PyAudio.")
            pya = pyaudio.PyAudio()
            output_stream = await asyncio.to_thread(
                pya.open,
                format=pyaudio.paInt16,
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_FRAMERATE,
                output=True
            )
            logger.info("PyAudio output stream opened.")

        current_script_state = ScriptState.CONNECTING
        async with websockets.connect(
            GEMINI_LIVE_API_URL,
            extra_headers=headers,
            ssl=ssl_context
        ) as websocket:
            logger.info("Connected to WebSocket.")
            current_script_state = ScriptState.WAITING_FOR_SETUP_COMPLETE

            system_instructions_text = load_system_instructions()
            
            setup_message = {
                "setup": {
                    "model": dynamic_model_name,
                    "generationConfig": {"responseModalities": [OUTPUT_MODALITY]},
                    "systemInstruction": {
                        "parts": [{
                            "text": system_instructions_text
                        }]
                    },
                    "tools": [
                        {"functionDeclarations": [PRINT_BLACKBOARD_TOOL_SCHEMA]}
                    ]
                }
            }
            logger.info(f"Sending setup message: {json.dumps(setup_message)}")
            await websocket.send(json.dumps(setup_message))

            while current_script_state not in [ScriptState.FINISHED, ScriptState.CLOSING, ScriptState.ERROR]:
                
                if current_script_state == ScriptState.PROCESSING_ACTIONS:
                    while action_index < len(action_sequence) and current_script_state == ScriptState.PROCESSING_ACTIONS:
                        current_action = action_sequence[action_index]
                        action_type = current_action.get("action")
                        last_sent_action_description = current_action.get("description", f"Action {action_index + 1} ({action_type})")
                        logger.info(f"Processing action {action_index + 1}/{len(action_sequence)}: {last_sent_action_description}")

                        if action_type == "send_image_realtime":
                            image_path_for_action = current_action.get("image_path", IMAGE_PATH)
                            image_chunk = prepare_image_chunk_for_realtime_input(image_path_for_action)
                            if not image_chunk:
                                logger.error(f"Could not prepare image chunk for action: {last_sent_action_description}. Aborting.")
                                current_script_state = ScriptState.ERROR
                                break
                            
                            realtime_image_payload = {"realtimeInput": {"mediaChunks": [image_chunk]}}
                            loggable_realtime_payload = {
                                "realtimeInput": {
                                    "mediaChunks_count": 1,
                                    "mediaChunk_mimeType": image_chunk.get("mimeType"),
                                    "mediaChunk_data_length": len(image_chunk.get("data", ""))
                                }
                            }
                            logger.info(f"Sending image (realtimeInput) for '{last_sent_action_description}': {json.dumps(loggable_realtime_payload)}")
                            await websocket.send(json.dumps(realtime_image_payload))
                            action_index += 1

                        elif action_type == "send_text_clientcontent":
                            prompt_text = current_action.get("prompt")
                            if prompt_text is None:
                                logger.error(f"No prompt provided for send_text_clientcontent action: {last_sent_action_description}. Aborting.")
                                current_script_state = ScriptState.ERROR
                                break
                            
                            client_text_payload = {
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": prompt_text}]}],
                                    "turnComplete": True 
                                }
                            }
                            logger.info(f"Sending text (clientContent) for '{last_sent_action_description}': {json.dumps(client_text_payload)}")
                            await websocket.send(json.dumps(client_text_payload))
                            action_index += 1

                            if current_action.get("expect_response", False):
                                logger.info(f"Sent text for '{last_sent_action_description}'. Now waiting for model response.")
                                current_script_state = ScriptState.WAITING_FOR_MODEL_RESPONSE
                                current_turn_generation_has_completed = False
                                current_turn_has_completed = False
                            else:
                                logger.info(f"Sent text for '{last_sent_action_description}'. Not expecting an immediate dedicated response.")
                        else:
                            logger.error(f"Unknown action type: {action_type} in action: {last_sent_action_description}. Aborting.")
                            current_script_state = ScriptState.ERROR
                            break
                
                if current_script_state == ScriptState.ERROR:
                    break

                if current_script_state not in [ScriptState.FINISHED, ScriptState.CLOSING, ScriptState.ERROR]:
                    try:
                        timeout_duration = 60.0
                        if current_script_state == ScriptState.PROCESSING_ACTIONS and action_index >= len(action_sequence):
                            timeout_duration = 10.0 
                        elif current_script_state == ScriptState.WAITING_FOR_SETUP_COMPLETE:
                            timeout_duration = 60.0
                        elif current_script_state == ScriptState.WAITING_FOR_MODEL_RESPONSE:
                            timeout_duration = 120.0

                        message_str = await asyncio.wait_for(websocket.recv(), timeout=timeout_duration)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for server message (State: {current_script_state.name}). Still connected: {websocket.closed}")
                        if websocket.closed:
                            logger.error("Connection closed during timeout.")
                            current_script_state = ScriptState.CLOSING
                            break
                        
                        if current_script_state == ScriptState.PROCESSING_ACTIONS and action_index >= len(action_sequence):
                            logger.info("Timeout after all actions sent and no final server messages. Finishing.")
                            current_script_state = ScriptState.FINISHED
                        elif current_script_state == ScriptState.WAITING_FOR_MODEL_RESPONSE:
                            logger.error(f"Timeout waiting for model response for action: '{last_sent_action_description}'. Aborting sequence.")
                            current_script_state = ScriptState.ERROR
                        elif current_script_state == ScriptState.WAITING_FOR_SETUP_COMPLETE:
                            logger.error("Timeout waiting for setupComplete from server. Aborting.")
                            current_script_state = ScriptState.ERROR
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Connection closed by server.")
                        current_script_state = ScriptState.CLOSING
                        break
                    
                    message = json.loads(message_str)
                    logger.debug(f"Raw server message: {json.dumps(message, indent=2)}")

                    if "setupComplete" in message and current_script_state == ScriptState.WAITING_FOR_SETUP_COMPLETE:
                        logger.info("Session setup complete. Moving to process actions.")
                        current_script_state = ScriptState.PROCESSING_ACTIONS
                    
                    elif "toolCall" in message:
                        tool_info = message["toolCall"]
                        if "functionCalls" in tool_info and len(tool_info["functionCalls"]) > 0:
                            function_call = tool_info["functionCalls"][0]
                            tool_name = function_call.get("name", "unknown")
                            tool_args = function_call.get("args", {})
                            
                            logger.info(f"Received tool call: {tool_name} with args: {json.dumps(tool_args)}")
                            if tool_name == "print_blackboard":
                                content = tool_args.get("content", "")
                                explanation_is_finish = tool_args.get("explanation_is_finish", False)
                                print(f"\n[BLACKBOARD] {content}\n")
                                logger.info(f"BLACKBOARD CONTENT (explanation_is_finish={explanation_is_finish}): {content}")
                    
                    elif "serverContent" in message:
                        server_content = message["serverContent"]
                        response_text_part = ""
                        if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                            for part in server_content["modelTurn"]["parts"]:
                                if "text" in part:
                                    response_text_part += part['text']
                        
                        if response_text_part:
                            print(f"Gemini (for '{last_sent_action_description}'): {response_text_part}")

                        if current_script_state == ScriptState.WAITING_FOR_MODEL_RESPONSE:
                            if server_content.get("generationComplete", False):
                                logger.debug(f"Received generationComplete: true for '{last_sent_action_description}'.")
                                current_turn_generation_has_completed = True
                            
                            if server_content.get("turnComplete", False):
                                logger.debug(f"Received turnComplete: true for '{last_sent_action_description}'.")
                                current_turn_has_completed = True

                            if current_turn_has_completed and current_turn_generation_has_completed:
                                logger.info(f"Model response for '{last_sent_action_description}' fully complete. Ready for next action.")
                                current_script_state = ScriptState.PROCESSING_ACTIONS
                                current_turn_generation_has_completed = False
                                current_turn_has_completed = False

                            if OUTPUT_MODALITY == "AUDIO":
                                if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                                    for part in server_content["modelTurn"]["parts"]:
                                        if part.get("inlineData") and part["inlineData"].get("data"):
                                            mime_type = part["inlineData"].get("mimeType", "N/A")
                                            if "audio" in mime_type.lower():
                                                logger.debug(f"Received audio chunk. MimeType: {mime_type}")
                                                try:
                                                    b64_data = part["inlineData"]["data"]
                                                    decoded_bytes = base64.b64decode(b64_data)
                                                    if output_stream:
                                                        await asyncio.to_thread(output_stream.write, decoded_bytes)
                                                    else:
                                                        logger.warning("Output stream not available for audio playback.")
                                                except Exception as e:
                                                    logger.error(f"Error decoding or playing audio: {e}", exc_info=True)
                        else:
                            if response_text_part:
                                 logger.info("Received unsolicited server content.")
                    
                    elif "error" in message or ("status" in message and message.get("status", {}).get("code") != 0):
                        logger.error(f"Received error from server: {json.dumps(message, indent=2)}")
                        current_script_state = ScriptState.ERROR
                        break
                    elif "goAway" in message:
                        logger.info(f"Server sent GoAway: {message['goAway']}. Closing connection.")
                        current_script_state = ScriptState.CLOSING
                        break
            
            if current_script_state not in [ScriptState.FINISHED, ScriptState.CLOSING, ScriptState.ERROR]:
                 logger.warning(f"Main loop exited unexpectedly in state: {current_script_state.name}")
            elif current_script_state == ScriptState.FINISHED:
                 logger.info("All actions processed successfully and responses received as expected.")

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket connection failed: Status {e.status_code}, Headers: {e.headers}", exc_info=True)
        current_script_state = ScriptState.ERROR
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e.code} {e.reason}", exc_info=True)
        current_script_state = ScriptState.ERROR
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        current_script_state = ScriptState.ERROR
    finally:
        logger.info(f"Script finished with state: {current_script_state.name}.")
        if output_stream:
            try:
                await asyncio.to_thread(output_stream.stop_stream)
                await asyncio.to_thread(output_stream.close)
                logger.info("PyAudio output stream closed.")
            except Exception as e:
                logger.error(f"Error closing PyAudio output stream: {e}", exc_info=True)
        if pya:
            try:
                await asyncio.to_thread(pya.terminate)
                logger.info("PyAudio instance terminated.")
            except Exception as e:
                logger.error(f"Error terminating PyAudio instance: {e}", exc_info=True)

if __name__ == "__main__":
    if not IMAGE_PATH:
        logger.error("CRITICAL: IMAGE_PATH is not set.")
        exit(1)
        
    try:
        with open(IMAGE_PATH, 'rb') as f:
            pass 
        logger.info(f"Default image for actions (if not overridden): {IMAGE_PATH}")
    except FileNotFoundError:
        logger.warning(f"Default image file '{IMAGE_PATH}' not found. Ensure paths in actions are correct if this is used.")

    action_sequence = [
        {
            "action": "send_image_realtime",
            "description": "Send initial test image via realtimeInput",
            "expect_response": False
        },
        {
            "action": "send_text_clientcontent",
            "prompt": "Here is the image.",
            "expect_response": True, 
            "description": "Image sent"
        },
        {
            "action": "send_text_clientcontent",
            "prompt": "Describe the image in detail.",
            "expect_response": True,
            "description": "Image description"
        },
    ]

    asyncio.run(process_websocket_action_sequence(action_sequence)) 