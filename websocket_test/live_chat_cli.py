import asyncio
import json
import websockets
import ssl
import certifi
import google.auth
from google.auth.transport.requests import Request
import traceback
import logging
import pyaudio
import base64
from google import genai
import io
from PIL import Image

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_MODALITY = "AUDIO"  # "TEXT" or "AUDIO"
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH_BYTES = 2
DEFAULT_FRAMERATE = 24000

GEMINI_LIVE_API_URL = "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
MODEL_NAME_ID = "gemini-2.0-flash-live-preview-04-09"

IMAGE_PATH_FOR_AUTOMATED_MESSAGE = "websocket_test/test_image.png"

PRINT_BLACKBOARD_TOOL_SCHEMA = {
    "name": "print_blackboard",
    "description": "Blackboard for the teacher to write down the key knowledge of the current step.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "content": {"type": "STRING", "description": "The content need to be displayed on the blackboard, showed to students."},
            "explanation_is_finish": {"type": "BOOLEAN", "description": "Indicates whether the teacher's explanation is over. True means explanation is finish, False otherwise"}
        },
        "required": ["content", "explanation_is_finish"]
    }
}

async def print_blackboard_tool(content, explanation_is_finish):
    """Displays content on a virtual blackboard and notes if the explanation is finished."""
    logger.info(f"Executing print_blackboard_tool: Content: ''{content}'', Explanation Finished: {explanation_is_finish}")

    client = genai.Client(vertexai=True, project='heikohotz-genai-sa', location='us-central1')
    response = await asyncio.to_thread(
        client.models.generate_content,
        model='gemini-2.5-flash-preview-04-17', 
        contents=f"""Your task is to process the given text.
If the text contains a mathematical formula:
1.  Identify the mathematical formula(s) in the text.
2.  Convert the identified mathematical formula(s) to LaTeX strings.
3.  For inline math (e.g., 'a plus b times c'), use single dollar signs: `$a + b \\times c$`.
4.  For display math (e.g., 'E equals m c squared'), use double dollar signs: `$$E = mc^2$$`.
5.  Replace the identified mathematical formula(s) in the original text with their LaTeX representation. Return the full text, with formulas converted. For example, if the input is 'The equation is x plus y equals z.', you should return 'The equation is $x + y = z$.'.

If the text does NOT contain any mathematical formula, you MUST return the original text exactly as provided, without any modifications or additions.

Text to process:
{content}"""
    )
    await asyncio.to_thread(print, response.text)

    return {"status": "success", "message": "Blackboard updated."}

def prepare_image_for_realtime_input(image_path):
    """Loads, processes (to JPEG), base64 encodes image, and returns a dict for mediaChunks."""
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
        logger.error(f"Image file {image_path} not found for realtimeInput.")
        return None
    except Exception as e:
        logger.error(f"Error preparing image {image_path} for realtimeInput: {e}", exc_info=True)
        return None

async def get_access_token():
    """Retrieves the access token and project ID for the currently authenticated account."""
    try:
        creds, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        if not creds.valid:
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
    
    generation_has_completed = False
    turn_has_completed = False
    initial_sequence_step = 0

    pya = None
    output_stream = None

    try:
        if OUTPUT_MODALITY == "AUDIO":
            pya = pyaudio.PyAudio()
            output_stream = await asyncio.to_thread(
                pya.open,
                format=pyaudio.paInt16,
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_FRAMERATE,
                output=True
            )
            logger.info("PyAudio output stream opened.")

        bearer_token, project_id = await get_access_token()
        if not bearer_token or not project_id:
            logger.error("Failed to get bearer token or project ID. Exiting.")
            return

        location = "us-central1"
        publisher = "google"
        dynamic_model_name = f"projects/{project_id}/locations/{location}/publishers/{publisher}/models/{MODEL_NAME_ID}"
        logger.info(f"Using fully qualified model name: {dynamic_model_name}")

        headers = {
            "Authorization": f"Bearer {bearer_token}",
        }

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with websockets.connect(
            GEMINI_LIVE_API_URL,
            extra_headers=headers,
            ssl=ssl_context
        ) as websocket:
            logger.info("Connected to Vertex AI WebSocket.")

            system_instructions_text = ""
            try:
                with open("websocket_test/system-instructions.txt", "r") as f:
                    system_instructions_text = f.read()
                logger.info("Successfully loaded system instructions from test/system-instructions.txt")
            except FileNotFoundError:
                logger.error("test/system-instructions.txt not found. Using a default system instruction.")
                system_instructions_text = "You are a helpful assistant."
            except Exception as e:
                logger.error(f"Error reading test/system-instructions.txt: {e}. Using a default system instruction.")
                system_instructions_text = "You are a helpful assistant."

            setup_message = {
                "setup": {
                    "model": dynamic_model_name,
                    "generationConfig": {
                        "responseModalities": [OUTPUT_MODALITY],
                        "speechConfig": {
                            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}},
                            "languageCode": "en-US"
                        }
                    },
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

            current_prompt_to_send = None
            ready_for_next_user_input = False 

            setup_complete_flag = False
            
            while True:
                if not setup_complete_flag:
                    pass
                elif initial_sequence_step < 5:
                    pass
                elif ready_for_next_user_input:
                    user_input = await get_user_input()
                    if user_input.lower() in ["quit", "exit", "q"]:
                        logger.info("User requested to quit. Closing connection.")
                        break
                    current_prompt_to_send = user_input
                    ready_for_next_user_input = False
                    generation_has_completed = False
                    turn_has_completed = False

                if current_prompt_to_send and setup_complete_flag and initial_sequence_step == 5:
                    user_message_parts = []
                    if isinstance(current_prompt_to_send, str):
                        user_message_parts = [{"text": current_prompt_to_send}]
                    else:
                        logger.warning(f"current_prompt_to_send is of unexpected type: {type(current_prompt_to_send)}. Skipping send.")
                        user_message_parts = []

                    if user_message_parts:
                        user_message_payload = {
                            "clientContent": {
                                "turns": [{"role": "user", "parts": user_message_parts}],
                                "turnComplete": True
                            }
                        }
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
                                loggable_parts_summary.append(part)

                        loggable_payload = {
                            "clientContent": {
                                "turns": [{"role": "user", "parts": loggable_parts_summary}],
                                "turnComplete": True
                            }
                        }
                        logger.info(f"Sending user message: {json.dumps(loggable_payload)}")
                        await websocket.send(json.dumps(user_message_payload))
                    current_prompt_to_send = None
                
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server (during recv).")
                    break

                try:
                    message = json.loads(message_str)
                    logger.debug(f"Raw server message: {json.dumps(message, indent=2)}")

                    if "setupComplete" in message and not setup_complete_flag:
                        logger.info("Session setup complete. Starting initial image and prompt sequence.")
                        setup_complete_flag = True
                        initial_sequence_step = 1

                        image_chunk = prepare_image_for_realtime_input(IMAGE_PATH_FOR_AUTOMATED_MESSAGE)
                        if image_chunk:
                            realtime_image_payload = {"realtimeInput": {"mediaChunks": [image_chunk]}}
                            loggable_realtime_payload = {
                                "realtimeInput": {
                                    "mediaChunks_count": 1,
                                    "mediaChunk_mimeType": image_chunk.get("mimeType"),
                                    "mediaChunk_data_length": len(image_chunk.get("data", ""))
                                }
                            }
                            logger.info(f"Sending initial image (realtimeInput): {json.dumps(loggable_realtime_payload)}")
                            await websocket.send(json.dumps(realtime_image_payload))
                            
                            first_prompt_text = "Here is the image."
                            first_prompt_payload = {
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": first_prompt_text}]}],
                                    "turnComplete": True
                                }
                            }
                            logger.info(f"Sending first prompt: '{first_prompt_text}'")
                            await websocket.send(json.dumps(first_prompt_payload))
                            initial_sequence_step = 2
                            generation_has_completed = False
                            turn_has_completed = False
                        else:
                            logger.error("Failed to prepare image for initial sequence. Skipping to user input.")
                            initial_sequence_step = 5
                            ready_for_next_user_input = True

                    elif "toolCall" in message:
                        raw_tool_info = message["toolCall"]
                        logger.debug(f"Received raw toolCall object from server: {json.dumps(raw_tool_info, indent=2)}")

                        actual_tool_call_to_process = None

                        if isinstance(raw_tool_info, dict) and "functionCalls" in raw_tool_info:
                            function_calls_list = raw_tool_info["functionCalls"]
                            if isinstance(function_calls_list, list) and len(function_calls_list) > 0:
                                actual_tool_call_to_process = function_calls_list[0]
                                if len(function_calls_list) > 1:
                                    logger.warning(f"Multiple function calls received in a single toolCall message ({len(function_calls_list)} found). Processing only the first one: {actual_tool_call_to_process.get('name')}")
                            else:
                                logger.error(f"'functionCalls' key found but it is not a non-empty list: {function_calls_list}")
                        elif isinstance(raw_tool_info, dict) and "name" in raw_tool_info: 
                            logger.warning("Received toolCall in a direct name/args structure. Adapting.")
                            actual_tool_call_to_process = raw_tool_info
                        
                        if actual_tool_call_to_process and isinstance(actual_tool_call_to_process, dict) and "name" in actual_tool_call_to_process:
                            tool_name = actual_tool_call_to_process["name"]
                            tool_args = actual_tool_call_to_process.get("args", {})

                            logger.info(f"Tool call processing: {tool_name} with args: {tool_args}")

                            if tool_name == "print_blackboard":
                                try:
                                    blackboard_task = asyncio.create_task(
                                        print_blackboard_tool(
                                            content=tool_args.get("content", ""), 
                                            explanation_is_finish=tool_args.get("explanation_is_finish", False)
                                        )
                                    )
                                    
                                    def callback(task):
                                        try:
                                            actual_result = task.result()
                                            logger.info(f"Blackboard task completed: {actual_result}")
                                        except Exception as e:
                                            logger.error(f"Blackboard task failed: {e}")
                                    
                                    blackboard_task.add_done_callback(callback)
                                except Exception as e:
                                    logger.error(f"Error starting print_blackboard_tool task: {e}", exc_info=True)
                            else:
                                logger.warning(f"Unknown tool requested: {tool_name}")
                            
                            generation_has_completed = False 
                            turn_has_completed = False      

                        else:
                            logger.error(f"Unexpected toolCall structure received: {json.dumps(actual_tool_call_to_process, indent=2)}")
                            ready_for_next_user_input = True

                    elif "serverContent" in message:
                        server_content = message["serverContent"]
                        if OUTPUT_MODALITY == "AUDIO":
                            if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                                for part in server_content["modelTurn"]["parts"]:
                                    if part.get("inlineData") and part["inlineData"].get("data"):
                                        mime_type = part["inlineData"].get("mimeType", "N/A")
                                        try:
                                            b64_data = part["inlineData"]["data"]
                                            decoded_bytes = base64.b64decode(b64_data)
                                            if output_stream:
                                                await asyncio.to_thread(output_stream.write, decoded_bytes)
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
                            
                            if initial_sequence_step == 2:
                                logger.info("Received response to first prompt. Sending second prompt.")
                                second_prompt_text = "Describe the image in detail."
                                second_prompt_payload = {
                                    "clientContent": {
                                        "turns": [{"role": "user", "parts": [{"text": second_prompt_text}]}],
                                        "turnComplete": True
                                    }
                                }
                                logger.info(f"Sending second prompt: '{second_prompt_text}'")
                                await websocket.send(json.dumps(second_prompt_payload))
                                initial_sequence_step = 4
                                generation_has_completed = False
                                turn_has_completed = False
                            elif initial_sequence_step == 4:
                                logger.info("Received response to second prompt. Initial sequence complete. Ready for user input.")
                                initial_sequence_step = 5
                                ready_for_next_user_input = True
                            elif initial_sequence_step == 5:
                                logger.info("Model turn and generation complete (normal operation). Ready for next user input.")
                                ready_for_next_user_input = True
                            elif initial_sequence_step < 2:
                                logger.warning(f"Generation/Turn complete received during unexpected initial_sequence_step: {initial_sequence_step}. Advancing to user input.")
                                initial_sequence_step = 5
                                ready_for_next_user_input = True

                    elif "goAway" in message:
                        logger.info(f"Server sent GoAway: {message['goAway']}. Closing connection.")
                        break
                    
                    elif "error" in message or ("status" in message and message.get("status", {}).get("code") != 0):
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
    logger.info(f"Script started. Configured output modality: {OUTPUT_MODALITY}")
    asyncio.run(gemini_live_chat())
