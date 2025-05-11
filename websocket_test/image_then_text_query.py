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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GEMINI_LIVE_API_URL = "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
MODEL_NAME_ID = "gemini-2.0-flash-live-preview-04-09" # Using the same model ID
IMAGE_PATH = "websocket_test/test_image.png" # Image to upload
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

def prepare_image_part(image_path):
    """Loads, processes, and encodes the image, returning the content part."""
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_bytes = image_io.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"Successfully loaded and encoded image {image_path}")
        return {"inlineData": {"mimeType": "image/jpeg", "data": encoded_image}}
    except FileNotFoundError:
        logger.error(f"Image file {image_path} not found.")
        return None
    except Exception as e:
        logger.error(f"Error preparing image {image_path}: {e}", exc_info=True)
        return None

async def run_image_then_text_query():
    logger.info(f"Starting query. Connecting to {GEMINI_LIVE_API_URL}...")
    
    bearer_token, project_id = await get_access_token()
    if not bearer_token or not project_id:
        logger.error("Failed to get bearer token or project ID. Exiting.")
        return

    location = "us-central1" # Assuming from API URL
    publisher = "google"     # Assuming standard Google models
    dynamic_model_name = f"projects/{project_id}/locations/{location}/publishers/{publisher}/models/{MODEL_NAME_ID}"
    logger.info(f"Using model: {dynamic_model_name}")

    headers = {"Authorization": f"Bearer {bearer_token}"}
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    setup_complete_flag = False
    first_response_received = False
    second_response_received = False
    current_turn_generation_has_completed = False
    current_turn_has_completed = False
    
    # States: 0=initial, 1=setup_sent, 2=image_sent, 3=first_response_done, 4=text_prompt_sent, 5=second_response_done
    current_state = 0 

    try:
        async with websockets.connect(
            GEMINI_LIVE_API_URL,
            extra_headers=headers,
            ssl=ssl_context
        ) as websocket:
            logger.info("Connected to WebSocket.")

            # 1. Send Session Setup
            setup_message = {
                "setup": {
                    "model": dynamic_model_name,
                    "generationConfig": {"responseModalities": [OUTPUT_MODALITY]}
                }
            }
            logger.info(f"Sending setup message: {json.dumps(setup_message)}")
            await websocket.send(json.dumps(setup_message))
            current_state = 1

            image_part = prepare_image_part(IMAGE_PATH)
            if not image_part:
                logger.error("Could not prepare image. Aborting.")
                return

            while not second_response_received: # Loop until all steps are done
                try:
                    message_str = await asyncio.wait_for(websocket.recv(), timeout=60.0) # Increased timeout to 60s
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for server message. Still connected.")
                    if websocket.closed:
                        logger.error("Connection closed during timeout.")
                        break
                    continue # Continue waiting if still open
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed by server.")
                    break
                
                message = json.loads(message_str)
                logger.debug(f"Raw server message: {json.dumps(message, indent=2)}")

                if "setupComplete" in message and current_state == 1:
                    logger.info("Session setup complete.")
                    setup_complete_flag = True
                    
                    # 2. Send Image (without text)
                    image_message_payload = {
                        "clientContent": {
                            "turns": [{"role": "user", "parts": [image_part]}],
                            "turnComplete": True
                        }
                    }
                    loggable_image_payload = { # Avoid logging full base64 data
                        "clientContent": {
                            "turns": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/jpeg", "data_length": len(image_part['inlineData']['data'])}}]}],
                            "turnComplete": True
                        }
                    }
                    logger.info(f"Sending image message: {json.dumps(loggable_image_payload)}")
                    await websocket.send(json.dumps(image_message_payload))
                    current_state = 2

                elif "serverContent" in message:
                    server_content = message["serverContent"]
                    response_text = ""
                    if "modelTurn" in server_content and "parts" in server_content["modelTurn"]:
                        for part in server_content["modelTurn"]["parts"]:
                            if "text" in part:
                                response_text += part['text']
                    
                    if server_content.get("generationComplete", False):
                        logger.debug("Received generationComplete: true")
                        current_turn_generation_has_completed = True
                    
                    if server_content.get("turnComplete", False):
                        logger.debug("Received turnComplete: true")
                        current_turn_has_completed = True

                    if response_text:
                        if current_state == 2 and not first_response_received : # Response to image
                            logger.info(f"RECVD IMG RESPONSE CHUNK: {response_text}")
                            print(f"Gemini (response to image): {response_text}")
                        elif current_state == 4 and not second_response_received: # Response to text prompt
                            logger.info(f"RECVD TEXT PROMPT RESPONSE CHUNK: {response_text}")
                            print(f"Gemini (response to text prompt): {response_text}")
                        else:
                            logger.info(f"Received text, but not at expected state: {response_text}")


                    if current_turn_has_completed and current_turn_generation_has_completed:
                        if current_state == 2 and not first_response_received:
                            logger.info("First response (to image) fully complete (generation and turn).")
                            first_response_received = True
                            current_state = 3
                            
                            # Reset flags for the next turn
                            current_turn_generation_has_completed = False
                            current_turn_has_completed = False

                            # 3. Send Text Prompt
                            text_prompt = "read what you see in the image"
                            text_message_payload = {
                                "clientContent": {
                                    "turns": [{"role": "user", "parts": [{"text": text_prompt}]}],
                                    "turnComplete": True
                                }
                            }
                            logger.info(f"Sending text prompt: {json.dumps(text_message_payload)}")
                            await websocket.send(json.dumps(text_message_payload))
                            current_state = 4

                        elif current_state == 4 and not second_response_received:
                            logger.info("Second response (to text prompt) fully complete (generation and turn).")
                            second_response_received = True
                            current_state = 5
                            logger.info("All operations completed successfully.")
                            break # Exit the while loop as we're done
                        else:
                            if not (current_state == 2 and not first_response_received) and not (current_state == 4 and not second_response_received):
                                logger.debug(f"Turn/Generation complete signals received, but not in a state to process them as a full response. State: {current_state}, first_resp: {first_response_received}, sec_resp: {second_response_received}")
                            # Reset flags if we are not processing this as a full turn completion for a known step, to avoid stale true flags for next real turn.
                            current_turn_generation_has_completed = False
                            current_turn_has_completed = False

                elif "error" in message or ("status" in message and message.get("status", {}).get("code") != 0):
                    logger.error(f"Received error from server: {json.dumps(message, indent=2)}")
                    break
                elif "goAway" in message:
                    logger.info(f"Server sent GoAway: {message['goAway']}. Closing connection.")
                    break
            
            if not second_response_received:
                logger.warning("Loop exited before all operations were completed.")

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
    
    # Ensure the image path is correct or adjust as needed
    if not IMAGE_PATH or IMAGE_PATH == "test/test1.jpg": # Check if default path might need verification
        try:
            with open(IMAGE_PATH, 'rb') as f:
                pass # Just try to open it to see if it exists
            logger.info(f"Using image: {IMAGE_PATH}")
        except FileNotFoundError:
            logger.error(f"CRITICAL: Image file '{IMAGE_PATH}' not found. Please ensure the path is correct.")
            logger.error("You might need to copy 'test1.jpg' to the 'test' directory relative to this script, or update IMAGE_PATH.")
            exit(1) # Exit if the primary image is missing

    asyncio.run(run_image_then_text_query()) 