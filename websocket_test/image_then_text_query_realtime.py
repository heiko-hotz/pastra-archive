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
import logging # Keep basic logging for connection/error info

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_LIVE_API_URL = "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
MODEL_NAME_ID = "gemini-2.0-flash-live-preview-04-09" # Using the same model ID
IMAGE_PATH = "test/test_image.png" # User changed to test.png
OUTPUT_MODALITY = "TEXT"
# -------

async def get_access_token():
    """Retrieves the access token and project ID."""
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

def prepare_image_chunk_for_realtime_input(image_path):
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
        logger.error(f"Image file {image_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Error preparing image {image_path} for realtimeInput: {e}", exc_info=True)
        return None

async def run_image_then_text_query():
    logger.info(f"Starting query. Connecting to {GEMINI_LIVE_API_URL}...")
    
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

    # Flags for managing response for the current turn
    current_turn_generation_has_completed = False
    current_turn_has_completed = False      
    
    # States: 0=initial, 1=setup_sent, 2=inputs_sent_waiting_for_response
    current_state = 0 
    conversation_turns_completed = 0 # Tracks completed model response turns

    try:
        async with websockets.connect(
            GEMINI_LIVE_API_URL,
            extra_headers=headers,
            ssl=ssl_context
        ) as websocket:
            logger.info("Connected to WebSocket.")

            setup_message = {
                "setup": {
                    "model": dynamic_model_name,
                    "generationConfig": {"responseModalities": [OUTPUT_MODALITY]}
                }
            }
            logger.info(f"Sending setup message: {json.dumps(setup_message)}")
            await websocket.send(json.dumps(setup_message))
            current_state = 1

            image_chunk = prepare_image_chunk_for_realtime_input(IMAGE_PATH)
            if not image_chunk:
                logger.error("Could not prepare image chunk for realtimeInput. Aborting.")
                return

            # Loop for the entire conversation (e.g., two turns)
            while conversation_turns_completed < 2: # Expecting two full model responses
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for server message. Still connected.")
                    if websocket.closed:
                        logger.error("Connection closed during timeout.")
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server.")
                    break
                
                message = json.loads(message_str)
                logger.debug(f"Raw server message: {json.dumps(message, indent=2)}")

                if "setupComplete" in message and current_state == 1:
                    logger.info("Session setup complete.")
                    
                    # 1. Send Image using realtimeInput
                    realtime_image_payload = {
                        "realtimeInput": { 
                            "mediaChunks": [image_chunk] 
                        }
                    }
                    loggable_realtime_payload = {
                        "realtimeInput": {
                            "mediaChunks_count": 1,
                            "mediaChunk_mimeType": image_chunk.get("mimeType"),
                            "mediaChunk_data_length": len(image_chunk.get("data", ""))
                        }
                    }
                    logger.info(f"Sending image message using realtimeInput: {json.dumps(loggable_realtime_payload)}")
                    await websocket.send(json.dumps(realtime_image_payload))

                    # 2. Immediately send Text Prompt using clientContent
                    text_prompt = "here is an image"
                    client_text_payload = {
                        "clientContent": {
                            "turns": [{"role": "user", "parts": [{"text": text_prompt}]}],
                            "turnComplete": True # Signal end of user turn here
                        }
                    }
                    logger.info(f"Sending text prompt message using clientContent: {json.dumps(client_text_payload)}")
                    await websocket.send(json.dumps(client_text_payload))
                    
                    current_state = 2 # Moved to waiting for first response
                    logger.info(f"Image (via realtimeInput) and first text prompt (via clientContent) sent. Waiting for model response for turn {conversation_turns_completed}.")
                    # Ensure flags are reset for the first response
                    current_turn_generation_has_completed = False
                    current_turn_has_completed = False

                elif "serverContent" in message and current_state == 2:
                    server_content = message["serverContent"]
                    response_text_part = "" # For current message part
                    if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                        for part in server_content["modelTurn"]["parts"]:
                            if "text" in part:
                                response_text_part += part['text']
                    
                    if server_content.get("generationComplete", False):
                        logger.debug("Received generationComplete: true from server.")
                        current_turn_generation_has_completed = True
                    
                    if server_content.get("turnComplete", False):
                        logger.debug("Received turnComplete: true from server.")
                        current_turn_has_completed = True

                    if response_text_part:
                        # Print each part of the response as it arrives
                        print(f"Gemini (Turn {conversation_turns_completed}): {response_text_part}") 

                    if current_turn_has_completed and current_turn_generation_has_completed:
                        logger.info(f"Model response for turn {conversation_turns_completed} fully complete (generation and turn).")
                        
                        # Reset flags for the next potential turn or before sending new client content
                        current_turn_generation_has_completed = False
                        current_turn_has_completed = False
                        
                        conversation_turns_completed += 1 # Increment after processing the current turn's response

                        if conversation_turns_completed == 1: # First turn's response processed, now send second prompt
                            logger.info("First model response received. Sending second text prompt.")
                            second_text_prompt = "read what you see in the image"
                            second_client_text_payload = {
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": second_text_prompt}]}],
                                    "turnComplete": True 
                                }
                            }
                            logger.info(f"Sending second text prompt: {json.dumps(second_client_text_payload)}")
                            await websocket.send(json.dumps(second_client_text_payload))
                            logger.info(f"Second text prompt sent. Waiting for model response for turn {conversation_turns_completed}.")
                        
                        elif conversation_turns_completed == 2: # Second turn's response processed
                            logger.info("Second model response received. All conversation turns completed successfully.")
                            # Loop will terminate as conversation_turns_completed is now 2
                
                elif "error" in message or ("status" in message and message.get("status", {}).get("code") != 0):
                    logger.error(f"Received error from server: {json.dumps(message, indent=2)}")
                    break
                elif "goAway" in message:
                    logger.info(f"Server sent GoAway: {message['goAway']}. Closing connection.")
                    break
            
            if conversation_turns_completed < 2:
                logger.warning(f"Loop exited before all ({2}) conversation turns were fully received. Completed turns: {conversation_turns_completed}.")

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"WebSocket connection failed: Status {e.status_code}, Headers: {e.headers}", exc_info=True)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed unexpectedly: {e.code} {e.reason}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Script finished.")

if __name__ == "__main__":
    logger.info("Starting Gemini image and text query script...")
    logger.info("Ensure 'gcloud auth application-default login' has been run.")
    logger.info("Required libraries: websockets, google-auth, certifi, Pillow")
    
    if not IMAGE_PATH: # Basic check
        logger.error("CRITICAL: IMAGE_PATH is not set.")
        exit(1)
        
    try:
        with open(IMAGE_PATH, 'rb') as f:
            pass 
        logger.info(f"Using image: {IMAGE_PATH}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Image file '{IMAGE_PATH}' not found. Please ensure the path is correct.")
        logger.error("You might need to copy the image to the specified path, or update IMAGE_PATH in the script.")
        exit(1)

    asyncio.run(run_image_then_text_query()) 