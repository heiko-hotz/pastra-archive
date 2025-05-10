import asyncio
import json
import websockets
import ssl
import certifi
import google.auth
from google.auth.transport.requests import Request
import traceback
import logging # Import logging
import pyaudio  # For audio playback
import base64   # For decoding audio data
# import wave     # No longer needed for WAV file operations
# import os       # No longer needed for path operations for WAV saving
from google import genai
import io # For in-memory image handling
from PIL import Image # For image processing

# Configure basic logging
# Set level to DEBUG to capture all messages; handlers can filter further if needed.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Get a logger instance

# --- Configuration for Output Modality ---
# Change to "AUDIO" to receive spoken responses and save as WAV
# Change to "TEXT" to receive textual responses
OUTPUT_MODALITY = "TEXT"  # or "AUDIO"
# -------

# --- Audio Output Constants (assumed defaults, try to verify with mimeType if possible) ---
# OUTPUT_WAV_FILENAME = "test/gemini_spoken_response.wav" # No longer saving to WAV
DEFAULT_CHANNELS = 1  # Mono
DEFAULT_SAMPLE_WIDTH_BYTES = 2  # 2 bytes = 16-bit audio (corresponds to pyaudio.paInt16)
DEFAULT_FRAMERATE = 24000  # 24kHz sample rate
# -------

GEMINI_LIVE_API_URL = "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
MODEL_NAME_ID = "gemini-2.0-flash-live-preview-04-09" # The specific model ID you are trying to use
# MODEL_NAME will be constructed dynamically using project_id

# --- Automated Initial Message ---
# AUTOMATED_INITIAL_MESSAGE = "Hello Gemini, this is an automated introductory message. Please acknowledge." # Will be replaced
AUTOMATED_MESSAGE_CONTENT_PARTS = [] # Will be populated with text and image parts
IMAGE_PATH_FOR_AUTOMATED_MESSAGE = "test/test_image.png"
# ------

# Blackboard Tool Schema (from user)
PRINT_BLACKBOARD_TOOL_SCHEMA = {
    "name": "print_blackboard",
    "description": "Blackboard for the teacher to write down the key knowledge of the current step.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "content": {"type": "STRING", "description": "The content need to be displayed on the blackboard, showed to students."},
            "explanation_is_finish": {"type": "BOOLEAN", "description": "Indicates whether the teacher's explanation is over. True means explanation is finish, False otherwise"}
        },
        "required": ["content", "explanation_is_finish"] # Assuming both are required based on typical tool use
    }
}

# Placeholder function for the new tool
def print_blackboard_tool(content, explanation_is_finish):
    """Displays content on a virtual blackboard and notes if the explanation is finished."""
    logger.info(f"Executing print_blackboard_tool: Content: ''{content}'', Explanation Finished: {explanation_is_finish}")
    # In a real application, this would update a UI or some state.
    # For now, we'll just log and return a confirmation.

    client = genai.Client(vertexai=True, project='heikohotz-genai-sa', location='us-central1')
    response = client.models.generate_content(
        model='gemini-2.0-flash-001', contents=f'If there is a formula in the text, format it as Latex. \n\n{content}'
    )
    print(response.text)


    return {"status": "success", "message": "Blackboard updated."}

# --- New function to save audio data to a WAV file ---
# def save_audio_to_wav(filename, audio_data_bytes, channels, framerate, sample_width_bytes):
#     if not audio_data_bytes:
#         logger.warning("No audio data to save.")
#         return
#     try:
#         # Ensure the directory exists
#         directory = os.path.dirname(filename)
#         if directory: # Check if directory is not an empty string (e.g. for files in current dir)
#             os.makedirs(directory, exist_ok=True)
#
#         with wave.open(filename, 'wb') as wf:
#             wf.setnchannels(channels)
#             wf.setsampwidth(sample_width_bytes)
#             wf.setframerate(framerate)
#             wf.writeframes(audio_data_bytes)
#         logger.info(f"Audio saved to {filename} ({len(audio_data_bytes)} bytes, {channels}ch, {framerate}Hz, {sample_width_bytes*8}-bit)")
#     except Exception as e:
#         logger.error(f"Error saving WAV file '{filename}': {e}", exc_info=True)
# -------

# --- Function to prepare automated message with image ---
def prepare_automated_message_with_image():
    global AUTOMATED_MESSAGE_CONTENT_PARTS
    text_part = "Hello Gemini, this is an automated introductory message. Please look at this image and describe it briefly."
    try:
        # Open image using Pillow
        img = Image.open(IMAGE_PATH_FOR_AUTOMATED_MESSAGE)
        
        # Convert to RGB if it has an alpha channel (e.g. some PNGs) to ensure JPEG compatibility
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert("RGB")

        # Resize the image (optional, but good practice for large images)
        img.thumbnail((1024, 1024)) # Max width/height of 1024, maintains aspect ratio

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg") # Save as JPEG
        image_bytes = image_io.getvalue()
        
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        AUTOMATED_MESSAGE_CONTENT_PARTS = [
            {"text": text_part},
            {"inlineData": {"mimeType": "image/jpeg", "data": encoded_image}} # Use image/jpeg
        ]
        logger.info(f"Successfully loaded, processed (to JPEG), and encoded image {IMAGE_PATH_FOR_AUTOMATED_MESSAGE} for automated message.")
    except FileNotFoundError:
        logger.error(f"Image file {IMAGE_PATH_FOR_AUTOMATED_MESSAGE} not found. Automated message will only contain text.")
        AUTOMATED_MESSAGE_CONTENT_PARTS = [{"text": text_part + " (Note: Image was intended but not found.)"}]
    except Exception as e:
        logger.error(f"Error preparing image for automated message: {e}", exc_info=True)
        AUTOMATED_MESSAGE_CONTENT_PARTS = [{"text": text_part + " (Note: Error processing intended image.)"}]
# ---

async def get_access_token():
    """Retrieves the access token and project ID for the currently authenticated account."""
    try:
        creds, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if not creds.valid:
            # Refresh the credentials if they're not valid
            auth_req = Request()
            creds.refresh(auth_req)
        logger.info(f"Successfully obtained access token and project ID: {project_id}")
        return creds.token, project_id
    except Exception as e:
        logger.error(f"Error getting access token or project ID: {e}")
        logger.debug(f"Full traceback for get_access_token error:\n{traceback.format_exc()}")
        raise

async def get_user_input(prompt_message="\nYou: "):
    """Run input() in a separate thread to avoid blocking asyncio loop."""
    return await asyncio.to_thread(input, prompt_message)

async def gemini_live_chat():
    logger.info(f"Connecting to {GEMINI_LIVE_API_URL}...")
    logger.info(f"Output Modality set to: {OUTPUT_MODALITY}")
    
    # collected_audio_bytes = bytearray() # No longer collecting bytes for saving
    # audio_received_this_turn = False # No longer tracking for saving
    generation_has_completed = False
    turn_has_completed = False

    pya = None
    output_stream = None

    try:
        if OUTPUT_MODALITY == "AUDIO":
            pya = pyaudio.PyAudio()
            output_stream = await asyncio.to_thread(
                pya.open,
                format=pyaudio.paInt16, # Corresponds to DEFAULT_SAMPLE_WIDTH_BYTES = 2
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_FRAMERATE,
                output=True
            )
            logger.info("PyAudio output stream opened.")

        bearer_token, project_id = await get_access_token()
        if not bearer_token or not project_id:
            logger.error("Failed to get bearer token or project ID. Exiting.")
            return

        # Construct the full model name
        # Assuming location is 'us-central1' from GEMINI_LIVE_API_URL and publisher is 'google'
        location = "us-central1"
        publisher = "google"
        # Use the MODEL_NAME_ID defined at the top of the file
        dynamic_model_name = f"projects/{project_id}/locations/{location}/publishers/{publisher}/models/{MODEL_NAME_ID}"
        logger.info(f"Using fully qualified model name: {dynamic_model_name}")

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            # "Content-Type": "application/json", # websockets library handles this for text frames
        }

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with websockets.connect(
            GEMINI_LIVE_API_URL,
            extra_headers=headers,
            ssl=ssl_context
        ) as websocket:
            logger.info("Connected to Vertex AI WebSocket.")

            # Read system instructions from file
            system_instructions_text = ""
            try:
                with open("test/system-instructions.txt", "r") as f:
                    system_instructions_text = f.read()
                logger.info("Successfully loaded system instructions from test/system-instructions.txt")
            except FileNotFoundError:
                logger.error("test/system-instructions.txt not found. Using a default system instruction.")
                system_instructions_text = "You are a helpful assistant." # Default fallback
            except Exception as e:
                logger.error(f"Error reading test/system-instructions.txt: {e}. Using a default system instruction.")
                system_instructions_text = "You are a helpful assistant." # Default fallback

            # 3. Update Session Setup with tools and system instruction
            setup_message = {
                "setup": {
                    "model": dynamic_model_name,
                    "generationConfig": {
                        "responseModalities": [OUTPUT_MODALITY],
                        "speechConfig": { # Add speechConfig for AUDIO output
                            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}},
                            "languageCode": "en-US"
                        }
                    },
                    "systemInstruction": { # System instruction for the tool
                        "parts": [{
                            # Modality-neutral instruction
                            "text": system_instructions_text
                        }]
                    },
                    "tools": [ # Tool declaration
                        {"functionDeclarations": [PRINT_BLACKBOARD_TOOL_SCHEMA]}
                    ]
                }
            }
            logger.info(f"Sending setup message: {json.dumps(setup_message)}")
            await websocket.send(json.dumps(setup_message))

            current_prompt_to_send = None # Initialize as None
            ready_for_next_user_input = False 

            # State flags for each turn
            setup_complete_flag = False
            expecting_reply_after_tool = False 
            tool_action_taken_in_current_exchange = False

            while True: # Main loop for conversation turns
                if not setup_complete_flag:
                    # Wait for setupComplete before sending the first prompt
                    pass # Handled by the message loop below
                elif ready_for_next_user_input:
                    user_input = await get_user_input()
                    if user_input.lower() in ["quit", "exit", "q"]:
                        logger.info("User requested to quit. Closing connection.")
                        break # Exit the while True loop, will close websocket via 'async with'
                    current_prompt_to_send = user_input
                    ready_for_next_user_input = False # Reset flag
                    # Reset flags for the new turn
                    generation_has_completed = False
                    turn_has_completed = False
                    # if OUTPUT_MODALITY == "AUDIO": # No longer needed
                    #     collected_audio_bytes = bytearray()
                    #     audio_received_this_turn = False
                    expecting_reply_after_tool = False 
                    tool_action_taken_in_current_exchange = False

                if current_prompt_to_send and setup_complete_flag: # Only send if there's a prompt and setup is done
                    user_message_parts = []
                    if isinstance(current_prompt_to_send, list): # Handles automated message with parts
                        user_message_parts = current_prompt_to_send
                    elif isinstance(current_prompt_to_send, str): # Handles regular user text input
                        user_message_parts = [{"text": current_prompt_to_send}]
                    else:
                        logger.warning(f"current_prompt_to_send is of unexpected type: {type(current_prompt_to_send)}. Skipping send.")
                        current_prompt_to_send = None # Clear to avoid re-processing

                    if user_message_parts:
                        user_message_payload = {
                            "clientContent": {
                                "turns": [{"role": "user", "parts": user_message_parts}],
                                "turnComplete": True
                            }
                        }
                        # Create a loggable summary to avoid logging full base64 data
                        loggable_parts_summary = []
                        for part in user_message_parts:
                            if "text" in part:
                                loggable_parts_summary.append({"text": part["text"]})
                            elif "inlineData" in part and isinstance(part["inlineData"], dict):
                                loggable_parts_summary.append({
                                    "inlineData": {
                                        "mimeType": part["inlineData"].get("mimeType"),
                                        "data_length": len(part["inlineData"].get("data", ""))
                                    }
                                })
                            else:
                                loggable_parts_summary.append(part) # Fallback for unknown part structure

                        loggable_payload = {
                            "clientContent": {
                                "turns": [{"role": "user", "parts": loggable_parts_summary}],
                                "turnComplete": True
                            }
                        }
                        logger.info(f"Sending user message: {json.dumps(loggable_payload)}")
                        await websocket.send(json.dumps(user_message_payload))
                    current_prompt_to_send = None # Clear after sending or if parts were empty/invalid
                
                # Non-blocking check for messages from server
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1) # Short timeout to be non-blocking
                except asyncio.TimeoutError:
                    if setup_complete_flag and not current_prompt_to_send and not (expecting_reply_after_tool or tool_action_taken_in_current_exchange) and not (generation_has_completed or turn_has_completed):
                        # If setup is done, no prompt is pending, and model isn't mid-turn, we might be ready for user input
                        if not ready_for_next_user_input: # check to avoid multiple prompts
                            # Only prompt if model's previous turn fully completed or if it's the very start after setup (and initial prompt was handled)
                            # This logic needs to be careful not to prompt while model is still generating.
                            # Let's rely on the model's turn completion flags primarily.
                            pass # Will be handled by flags after serverContent processing
                    continue # Go back to check for user input or loop again for messages
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server (during recv).")
                    break

                try:
                    message = json.loads(message_str)
                    logger.debug(f"Raw server message: {json.dumps(message, indent=2)}")

                    if "setupComplete" in message and not setup_complete_flag:
                        logger.info("Session setup complete. Sending automated initial message with image (if available).")
                        setup_complete_flag = True
                        if AUTOMATED_MESSAGE_CONTENT_PARTS: # Check if parts were prepared
                            current_prompt_to_send = AUTOMATED_MESSAGE_CONTENT_PARTS
                        else:
                            # Fallback if AUTOMATED_MESSAGE_CONTENT_PARTS is empty for some reason
                            logger.warning("AUTOMATED_MESSAGE_CONTENT_PARTS is empty. Sending a default text message.")
                            current_prompt_to_send = [{"text": "Hello Gemini, automated setup complete."}]
                        # Do not set ready_for_next_user_input = True here;
                        # it will be set after the model responds to the automated message.
                        generation_has_completed = False
                        turn_has_completed = False
                        expecting_reply_after_tool = False
                        tool_action_taken_in_current_exchange = False

                    elif "toolCall" in message:
                        tool_action_taken_in_current_exchange = True
                        raw_tool_info = message["toolCall"] # Renamed for clarity
                        logger.debug(f"Received raw toolCall object from server: {json.dumps(raw_tool_info, indent=2)}")

                        actual_tool_call_to_process = None

                        if isinstance(raw_tool_info, dict) and "functionCalls" in raw_tool_info:
                            function_calls_list = raw_tool_info["functionCalls"]
                            if isinstance(function_calls_list, list) and len(function_calls_list) > 0:
                                actual_tool_call_to_process = function_calls_list[0] # Process the first one
                                if len(function_calls_list) > 1:
                                    logger.warning(f"Multiple function calls received in a single toolCall message ({len(function_calls_list)} found). Processing only the first one: {actual_tool_call_to_process.get('name')}")
                            else:
                                logger.error(f"'functionCalls' key found but it is not a non-empty list: {function_calls_list}")
                        # Fallback for the old structure, or if functionCalls is not present
                        elif isinstance(raw_tool_info, dict) and "name" in raw_tool_info: 
                            logger.warning("Received toolCall in a direct name/args structure. Adapting.")
                            actual_tool_call_to_process = raw_tool_info
                        
                        if actual_tool_call_to_process and isinstance(actual_tool_call_to_process, dict) and "name" in actual_tool_call_to_process:
                            tool_name = actual_tool_call_to_process["name"]
                            tool_args = actual_tool_call_to_process.get("args", {})
                            tool_result = None
                            tool_error = False

                            logger.info(f"Tool call processing: {tool_name} with args: {tool_args}")

                            if tool_name == "print_blackboard":
                                try:
                                    result = print_blackboard_tool(content=tool_args.get("content", ""), explanation_is_finish=tool_args.get("explanation_is_finish", False))
                                    tool_result = result # Use the direct dictionary returned by the tool
                                except Exception as e:
                                    logger.error(f"Error executing print_blackboard_tool: {e}", exc_info=True)
                                    tool_result = {"error": str(e)}
                                    tool_error = True
                            else:
                                logger.warning(f"Unknown tool requested: {tool_name}")
                                tool_result = {"error": f"Unknown tool: {tool_name}"}
                                tool_error = True
                            
                            # tool_response_message = {
                            #     "clientContent": {
                            #         "toolResponse": {
                            #             "toolCallName": tool_name, # Echo back the tool call name
                            #             "response": tool_result
                            #         },
                            #         "turnComplete": False # Model needs to process this and then complete its turn
                            #     }
                            # }

                            # logger.info(f"Sending tool response: {json.dumps(tool_response_message)}")
                            # await websocket.send(json.dumps(tool_response_message))
                            
                            expecting_reply_after_tool = True
                            generation_has_completed = False 
                            turn_has_completed = False      

                        else:
                            logger.error(f"Unexpected toolCall structure received: {json.dumps(actual_tool_call_to_process, indent=2)}")
                            # Decide how to proceed: maybe skip and wait for user, or send an error to model if possible
                            # For now, prepare for next user input to avoid getting stuck
                            ready_for_next_user_input = True
                            expecting_reply_after_tool = False
                            # No tool_action_taken if structure is bad
                            tool_action_taken_in_current_exchange = False 

                    elif "serverContent" in message:
                        server_content = message["serverContent"]
                        # ... (handle TEXT or AUDIO output based on OUTPUT_MODALITY)
                        if OUTPUT_MODALITY == "AUDIO":
                            # ... (audio collection) ...
                            if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                                for part in server_content["modelTurn"]["parts"]:
                                    if part.get("inlineData") and part["inlineData"].get("data"):
                                        mime_type = part["inlineData"].get("mimeType", "N/A")
                                        # logger.info(f"Received audio chunk. MimeType: {mime_type}") # Can be too verbose
                                        try:
                                            b64_data = part["inlineData"]["data"]
                                            decoded_bytes = base64.b64decode(b64_data)
                                            # logger.debug(f"Decoded {len(decoded_bytes)} audio bytes.")
                                            if output_stream:
                                                await asyncio.to_thread(output_stream.write, decoded_bytes)
                                            # collected_audio_bytes.extend(decoded_bytes) # No longer collecting
                                            # audio_received_this_turn = True # No longer tracking
                                            # logger.debug(f"Total collected audio bytes now: {len(collected_audio_bytes)}")
                                        except Exception as e:
                                            logger.error(f"Error decoding or playing audio: {e}", exc_info=True)
                        elif OUTPUT_MODALITY == "TEXT":
                            if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                                for part in server_content["modelTurn"]["parts"]:
                                    if "text" in part:
                                        logger.info(f"Gemini: {part['text']}")
                        
                        if server_content.get("generationComplete"):
                            generation_has_completed = True
                            logger.info("generationComplete received")

                        if server_content.get("turnComplete"):
                            turn_has_completed = True
                            logger.info("turnComplete received")
                        
                        if generation_has_completed and turn_has_completed:
                            logger.info("Model turn and generation complete.")
                            # if OUTPUT_MODALITY == "AUDIO" and audio_received_this_turn: # Saving logic removed
                            #     logger.info(f"Attempting to save WAV. audio_received_this_turn: {audio_received_this_turn}, collected_audio_bytes length: {len(collected_audio_bytes)}")
                            #     logger.info("Saving WAV file.")
                            #     save_audio_to_wav(OUTPUT_WAV_FILENAME, collected_audio_bytes, 
                            #                       DEFAULT_CHANNELS, DEFAULT_FRAMERATE, DEFAULT_SAMPLE_WIDTH_BYTES)
                            
                            # Model's turn is done, ready for next user input
                            ready_for_next_user_input = True
                            # Flags will be reset when new user input is prepared
                            # Resetting specific flags here that indicate an active model response phase
                            expecting_reply_after_tool = False
                            tool_action_taken_in_current_exchange = False

                    elif "goAway" in message:
                        logger.info(f"Server sent GoAway: {message['goAway']}. Closing connection.")
                        break # Keep this break for server-initiated close
                    
                    elif "error" in message or ("status" in message and message.get("status", {}).get("code") != 0) :
                        logger.error(f"Received error/status from server: {json.dumps(message, indent=2)}")
                        break

                except json.JSONDecodeError as e:
                    logger.error(f"Received non-JSON message: {message_str}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing received message: {e}", exc_info=True)
                    logger.debug(f"Message causing error: {message_str}")
                    logger.debug(f"Full traceback for message processing error:\n{traceback.format_exc()}")
                    break
            
            logger.info("Exited conversational loop.")
            # if OUTPUT_MODALITY == "AUDIO" and len(collected_audio_bytes) > 0: # Final save logic removed
            #     logger.warning("Connection closed with pending audio data. Attempting to save.")
            #     # Construct path for pending audio correctly
            #     output_dir = os.path.dirname(OUTPUT_WAV_FILENAME)
            #     base_filename = os.path.basename(OUTPUT_WAV_FILENAME)
            #     final_pending_filename = os.path.join(output_dir, f"final_pending_{base_filename}")
            #     
            #     save_audio_to_wav(final_pending_filename, collected_audio_bytes, 
            #                       DEFAULT_CHANNELS, DEFAULT_FRAMERATE, DEFAULT_SAMPLE_WIDTH_BYTES)

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket connection failed with status code {e.status_code}: {e.headers.get('www-authenticate') or e}", exc_info=True)
        logger.info("This often indicates an authentication or authorization issue.")
        logger.info("Ensure your gcloud CLI is authenticated (gcloud auth application-default login) and has permissions for AI Platform.")
        logger.debug(f"Full traceback for InvalidStatusCode:\n{traceback.format_exc()}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e.code} {e.reason}", exc_info=True)
        logger.debug(f"Full traceback for ConnectionClosedError:\n{traceback.format_exc()}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.debug(f"Full traceback for unhandled error:\n{traceback.format_exc()}")
    finally:
        if output_stream:
            await asyncio.to_thread(output_stream.stop_stream)
            await asyncio.to_thread(output_stream.close)
            logger.info("PyAudio output stream closed.")
        if pya:
            await asyncio.to_thread(pya.terminate)
            logger.info("PyAudio instance terminated.")

if __name__ == "__main__":
    logger.info("Attempting to connect to Vertex AI Gemini Live API...")
    logger.info("Ensure you have run 'gcloud auth application-default login' and have the necessary permissions.")
    logger.info("You might need to install additional libraries: pip install websockets google-auth certifi pyaudio Pillow")
    
    # Prepare the automated message with image before starting the chat
    prepare_automated_message_with_image()

    # The MODEL_NAME_ID at the top of the file is what you might need to change
    # if you switch models. The project ID will be fetched automatically.
    logger.info(f"Script started. Configured output modality: {OUTPUT_MODALITY}")
    asyncio.run(gemini_live_chat())
