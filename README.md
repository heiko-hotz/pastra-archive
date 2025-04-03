# Project Pastra 🍝 - DEMO BRANCH: Meeting Recall

**❗ Note:** This branch demonstrates a specific feature: **filtering past meetings by date and location** to recall details like attendee names. The core functionality remains based on the main Project Pastra implementation.

![Project Pastra Banner](assets/project_pastra.png)

**Talk to AI like never before! Project Pastra is a real-time, multimodal chat application showcasing the power of Google's Gemini 2.0 Flash (experimental) Live API.**

Think "Star Trek computer" interaction – speak naturally, show your webcam, share your screen, and get instant, streamed audio responses. Pastra brings this futuristic experience to your devices today.

This project builds upon the concepts from the [Gemini Multimodal Live API Developer Guide](https://github.com/heiko-hotz/gemini-multimodal-live-dev-guide) with a focus on a more production-ready setup and enhanced features.

**ℹ️ For general project setup, deployment instructions, and architecture details, please refer to the [README in the `main` branch](https://github.com/heiko-hotz/project-pastra-v2/blob/main/README.md).**

## ✨ Key Features

*   **🎤 Real-time Voice:** Natural, low-latency voice conversations.
*   **👁️ Multimodal Input:** Combines voice, text, webcam video, and screen sharing.
*   **🔊 Streamed Audio Output:** Hear responses instantly as they are generated.
*   **↩️ Interruptible:** Talk over the AI, just like a real conversation.
*   **🛠️ Integrated Tools:** Ask about the weather or check your calendar (via Cloud Functions - see `main` branch for weather tool example).
*   **📅 *New in Demo:* Meeting Recall:** Filter past meetings by date range and location to find specific information (e.g., "Who did I meet at Cafe X last month?"). This demo uses the example `calendar-tools` Cloud Function ([`cloud-functions/calendar-tools/get-past-appointments-tool/main.py`](./cloud-functions/calendar-tools/get-past-appointments-tool/main.py)).
*   **📱 Responsive UI:** Includes both a development interface and a mobile-optimized view.
*   **☁️ Cloud Ready:** Designed for easy deployment to Google Cloud Run (see `main` branch README for details).

<!-- Optional: Add a GIF/Video Demo Here -->
<!-- ![Demo GIF](assets/pastra-demo.gif) -->

---

## 🎯 Demo Scenario: Meeting Recall

This branch specifically showcases how Project Pastra could be extended to help users recall information from past interactions, using the `calendar-tools` Cloud Function ([`cloud-functions/calendar-tools/get-past-appointments-tool/main.py`](./cloud-functions/calendar-tools/get-past-appointments-tool/main.py)) as an example implementation for accessing and filtering meeting data from a hypothetical source (e.g., Google Calendar).

**Example Use Case:** Imagine asking, *"What was the name of the person I met a couple of months ago in Cafe X?"*

This demo branch includes the backend logic and potential UI elements (though UI implementation might vary) to:

1.  Parse the user's request for key details (date range approximation, location).
2.  Call the `calendar-tools` Cloud Function (`get-past-appointments-tool`) to query a hypothetical data source of past meetings/interactions based on the parsed details.
3.  Filter the results based on the identified criteria within the function.
4.  Return the relevant information (e.g., the name of the person) to the user via the chat interface.

This demonstrates how the core conversational AI can be augmented with specific tools and data sources to solve targeted problems.

*(Note: The actual data source connection and exact implementation details for meeting recall within the `calendar-tools` function are specific to this demo branch and serve as an illustration. You would need to adapt it to connect to your actual calendar or meeting data source. The main project branch demonstrates a weather tool.)*

---

*(Reminder: General project setup, deployment, full architecture, troubleshooting, and license information can be found in the [README of the `main` branch](https://github.com/heiko-hotz/project-pastra-v2/blob/main/README.md).)*