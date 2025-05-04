/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export class GeminiAPI {
    constructor(endpoint = null) {
        this.endpoint = endpoint;
        this.ws = null;
        this.isSpeaking = false;
        // Promise to resolve when the WebSocket is open and ready signal received
        this._readyPromise = null; 
        this._resolveReady = null;
        // Assign default handlers in constructor
        this.onReady = () => {};
        this.onAudioData = () => {};
        this.onTextContent = () => {};
        this.onError = () => {};
        this.onTurnComplete = () => {};
        this.onFunctionCall = () => {};
        this.onFunctionResponse = () => {};
        this.onInterrupted = () => {};
        this.onConnect = () => {}; 
        // Don't connect immediately
        // this.connect(); 
    }

    connect() {
        // Prevent multiple connections
        if (this.ws && this.ws.readyState < 2) { // CONNECTING or OPEN
            console.log('WebSocket connection already exists or is in progress.');
            return this._readyPromise; // Return existing promise
        }
        
        console.log('Initializing GeminiAPI with endpoint:', this.endpoint);
        this.ws = new WebSocket(this.endpoint);
        
        // Setup the ready promise
        this._readyPromise = new Promise((resolve) => {
            this._resolveReady = resolve;
        });
        
        this.setupWebSocket();
        return this._readyPromise; // Return the promise
    }

    setupWebSocket() {
        this.ws.onopen = () => {
            console.log('WebSocket connection is opening... (onopen)');
            // Don't call onReady or onConnect here directly
            // Wait for the server's ready signal
        };

        this.ws.onmessage = async (event) => {
            try {
                let response;
                if (event.data instanceof Blob) {
                    console.log('Received blob data, converting to text...');
                    const responseText = await event.data.text();
                    response = JSON.parse(responseText);
                } else {
                    response = JSON.parse(event.data);
                }
                
                console.log('WebSocket Response:', response);

                if (response.type === 'error') {
                    console.error('Server error:', response.data);
                    this.onError(response.data);
                    return;
                }

                if (response.ready) {
                    console.log('Received ready signal from server');
                    this.onReady(); // Call the user-defined onReady handler
                    if(this._resolveReady) {
                        this._resolveReady(); // Resolve the internal ready promise
                        this._resolveReady = null; // Prevent multiple resolves
                    }
                    // Send initial setup message AFTER ready signal
                    console.log('[gemini-api.js] About to call this.onConnect()...');
                    this.onConnect(); 
                    return;
                }

                if (response.type === 'interrupted') {
                    console.log('Response interrupted:', response.data);
                    this.isSpeaking = false;
                    this.onInterrupted(response.data);
                } else if (response.type === 'audio') {
                    console.log('Received audio data');
                    this.onAudioData(response.data);
                } else if (response.type === 'text') {
                    console.log('Received text content:', response.data);
                    this.onTextContent(response.data);
                } else if (response.type === 'turn_complete') {
                    console.log('Turn complete');
                    this.onTurnComplete();
                } else if (response.type === 'function_call') {
                    console.log('Received function call:', response.data);
                    this.onFunctionCall(response.data);
                } else if (response.type === 'function_response') {
                    console.log('Received function response:', response.data);
                    this.onFunctionResponse(response.data);
                } else {
                    console.log('Received unknown message type:', response);
                }
            } catch (error) {
                console.error('Error parsing response:', error);
                console.error('Raw response data:', event.data);
                this.onError({
                    message: 'Error parsing response: ' + error.message,
                    error_type: 'client_error'
                });
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
            this.onError({
                message: 'Connection error occurred',
                action: 'Please check your internet connection and try again',
                error_type: 'websocket_error'
            });
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket connection closed:', {
                code: event.code,
                reason: event.reason,
                wasClean: event.wasClean
            });
            
            // Only show error if it wasn't a clean close
            if (!event.wasClean) {
                this.onError({
                    message: 'Connection was interrupted',
                    action: 'Please refresh the page to reconnect',
                    error_type: 'connection_closed'
                });
            }
        };
    }

    sendAudioChunk(base64Audio) {
        console.log('Sending audio chunk...');
        this.sendMessage({
            type: 'audio',
            data: base64Audio
        });
    }

    sendImage(base64Image) {
        console.log('Sending image data...');
        this.sendMessage({
            type: 'image',
            data: base64Image
        });
    }

    sendEndMessage() {
        console.log('Sending end message');
        this.sendMessage({
            type: 'end'
        });
    }

    sendTextMessage(text) {
        console.log('Sending text message');
        this.sendMessage({
            type: 'text',
            data: text
        });
    }

    sendSetupMessage(setupData) {
        console.log('Sending setup message:', setupData);
        this.sendMessage({
            type: 'setup',
            data: setupData
        });
    }

    sendMessage(message) {
        if (this.ws.readyState === WebSocket.OPEN) {
            console.log('Sending message:', {
                type: message.type,
                dataLength: message.data ? message.data.length : 0
            });
            this.ws.send(JSON.stringify(message));
        } else {
            const states = {
                0: 'CONNECTING',
                1: 'OPEN',
                2: 'CLOSING',
                3: 'CLOSED'
            };
            console.error('WebSocket is not open. Current state:', states[this.ws.readyState]);
            this.onError(`WebSocket is not ready (State: ${states[this.ws.readyState]}). Please try again.`);
        }
    }

    // Method to explicitly wait for connection readiness
    async ensureReady() {
        if (!this.ws || this.ws.readyState >= 2) { // CLOSING or CLOSED
           await this.connect(); // Reconnect if closed or not initialized
        }
        console.log('Waiting for API to be ready...');
        await this._readyPromise;
        console.log('API is ready.');
    }

    async ensureConnected() {
        console.log('Ensuring WebSocket connection...');
        if (this.ws.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                console.error('Connection timeout after 5000ms');
                reject(new Error('Connection timeout'));
            }, 5000);

            const onOpen = () => {
                console.log('WebSocket connection established');
                clearTimeout(timeout);
                this.ws.removeEventListener('open', onOpen);
                this.ws.removeEventListener('error', onError);
                resolve();
            };

            const onError = (error) => {
                console.error('WebSocket connection failed:', error);
                clearTimeout(timeout);
                this.ws.removeEventListener('open', onOpen);
                this.ws.removeEventListener('error', onError);
                reject(error);
            };

            this.ws.addEventListener('open', onOpen);
            this.ws.addEventListener('error', onError);
        });
    }
}