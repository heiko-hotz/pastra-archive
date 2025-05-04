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
Gemini client initialization and connection management
"""

import logging
import os
from google import genai
from google.genai.types import ( # Import necessary types
    LiveConnectConfig, 
    SpeechConfig,
    VoiceConfig,
    PrebuiltVoiceConfig,
    GenerationConfig, # Import GenerationConfig if needed for other params
    SessionResumptionConfig # Import SessionResumptionConfig
)
from config.config import MODEL, CONFIG, api_config, ConfigurationError
from typing import Optional

logger = logging.getLogger(__name__)

async def create_gemini_session(
    response_modality: str = "AUDIO", 
    enable_input_transcription: bool = False, 
    enable_output_transcription: bool = False,
    session_handle_id: Optional[str] = None # Add session_handle_id parameter
):
    """Create and initialize the Gemini client and session, optionally enabling transcriptions and session resumption."""
    try:
        # Initialize authentication
        await api_config.initialize()
        
        if api_config.use_vertex:
            # Vertex AI configuration
            location = os.getenv('VERTEX_LOCATION', 'us-central1')
            project_id = os.environ.get('PROJECT_ID')
            
            if not project_id:
                raise ConfigurationError("PROJECT_ID is required for Vertex AI")
            
            logger.info(f"Initializing Vertex AI client with location: {location}, project: {project_id}")
            
            # Initialize Vertex AI client
            client = genai.Client(
                vertexai=True,
                location=location,
                project=project_id,
                # http_options={'api_version': 'v1beta'}
            )
            logger.info(f"Vertex AI client initialized with client: {client}")
        else:
            # Development endpoint configuration
            logger.info("Initializing development endpoint client")
            
            # Initialize development client
            client = genai.Client(
                vertexai=False,
                http_options={'api_version': 'v1alpha'},
                api_key=api_config.api_key
            )
                
        # Create the session config according to the new structure
        session_config_dict = {}

        # --- Top-level parameters --- 
        session_config_dict["response_modalities"] = [response_modality]
        
        # Extract voice name from old config structure (handle potential errors)
        voice_name = "Aoede" # Default voice
        if CONFIG and 'generation_config' in CONFIG and 'speech_config' in CONFIG['generation_config']:
             # Assuming speech_config in old config was just the voice name string
             if isinstance(CONFIG['generation_config']['speech_config'], str):
                 voice_name = CONFIG['generation_config']['speech_config']
             else:
                 logger.warning("Unexpected structure for speech_config in CONFIG, using default voice.")
        else:
            logger.warning("speech_config not found in CONFIG['generation_config'], using default voice.")

        # Create SpeechConfig object
        session_config_dict["speech_config"] = SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
            # language_code="en-US" # Optional: Add language code if needed
        )

        if enable_input_transcription:
            session_config_dict["input_audio_transcription"] = {} # Needs to be an object
        if enable_output_transcription:
            session_config_dict["output_audio_transcription"] = {} # Needs to be an object

        if CONFIG and 'tools' in CONFIG:
            session_config_dict["tools"] = CONFIG['tools']

        if CONFIG and 'system_instruction' in CONFIG:
            session_config_dict["system_instruction"] = CONFIG['system_instruction']

        # --- Session Resumption ---
        if session_handle_id:
            logger.info(f"Resuming session with handle: {session_handle_id}")
            session_config_dict["session_resumption"] = SessionResumptionConfig(handle=session_handle_id)
        else:
            logger.info("Enabling session resumption to get a new handle.")
            session_config_dict["session_resumption"] = SessionResumptionConfig() # Enable to get a new handle

        # --- Nested generation_config (only for standard generation params) ---
        base_gen_config = CONFIG.get('generation_config', {}) if CONFIG else {}
        # Filter out params moved to top-level or handled specifically
        filtered_gen_config_params = {
            k: v for k, v in base_gen_config.items() 
            if k not in ['response_modalities', 'speech_config'] 
        }
        
        # Only add generation_config if it contains relevant parameters
        if filtered_gen_config_params:
             session_config_dict["generation_config"] = GenerationConfig(**filtered_gen_config_params)

        logger.info(f"Creating Gemini LiveSession with config dict: {session_config_dict}")

        session = client.aio.live.connect(
            model=MODEL,
            config=session_config_dict # Pass the structured dictionary
        )
        
        return session
        
    except ConfigurationError as e:
        logger.error(f"Configuration error while creating Gemini session: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating Gemini session: {str(e)}")
        raise 