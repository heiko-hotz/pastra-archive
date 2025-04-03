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

import os
import json
from datetime import datetime, timedelta, timezone, date

from google.cloud import secretmanager
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# import traceback # Optional for deeper debugging

# Cache the client globally to potentially reuse connections
secret_client = secretmanager.SecretManagerServiceClient()

def get_secret(secret_id):
    """Get secret from Secret Manager."""
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', os.environ.get('PROJECT_ID', 'heikohotz-genai-sa'))
    if not project_id:
        try:
            import google.auth
            _, project_id = google.auth.default()
        except Exception:
             raise ValueError("Could not determine Google Cloud project ID. Set GOOGLE_CLOUD_PROJECT environment variable.")

    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    try:
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error accessing secret {secret_id}: {e}")
        raise RuntimeError(f"Failed to access secret: {secret_id}") from e

def format_datetime_for_api(dt_object):
    """Formats a datetime object into the ISO string format required by the API."""
    return dt_object.isoformat()

def parse_event_time(event_time_dict):
    """Parses event time, handling both 'dateTime' and 'date' (all-day)."""
    if 'dateTime' in event_time_dict:
        dt_obj = datetime.fromisoformat(event_time_dict['dateTime'].replace('Z', '+00:00'))
        return dt_obj.strftime('%Y-%m-%d'), dt_obj.strftime('%H:%M:%S %Z'), False
    elif 'date' in event_time_dict:
        return event_time_dict['date'], "All-day", True
    else:
        return "Unknown", "Unknown", False

# --- Cloud Function Entry Point ---
def get_past_appointments(request):
    """
    Cloud Function to retrieve past Google Calendar events within a date range,
    optionally filtered by location.

    Query Parameters:
        start_date (str, optional): Start date (YYYY-MM-DD). Defaults to 14 days ago.
        end_date (str, optional): End date (YYYY-MM-DD). Defaults to today.
        location (str, optional): Location string to filter by (case-insensitive contains).
        calendar_id (str, optional): Specific calendar ID. Defaults to CALENDAR_ID env var.
        max_results (int, optional): Max events to return *before* location filtering. Defaults to 50.

    Returns:
        JSON response with list of events or error message.
    """
    try:
        # --- 1. Get Calendar ID ---
        calendar_id = request.args.get('calendar_id') or os.environ.get('CALENDAR_ID')
        if not calendar_id:
            return json.dumps({'error': 'Calendar ID is required via parameter or CALENDAR_ID env var.'}), 400, {'Content-Type': 'application/json'}

        # --- 1b. Get Optional Location Filter ---
        location_filter = request.args.get('location') # Will be None if not provided
        if location_filter:
             location_filter = location_filter.strip().lower() # Normalize for comparison
             print(f"Applying location filter: '{location_filter}'")

        # --- 2. Determine Date Range & Validate ---
        today = datetime.now(timezone.utc).date()
        default_start_date = today - timedelta(days=14)

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        # Fetch potentially more results if filtering, adjust max_results if needed, or keep simple
        max_results = int(request.args.get('max_results', 50 if not location_filter else 100)) # Fetch more if filtering

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else default_start_date
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else today

            if start_date > end_date:
                 raise ValueError("Start date cannot be after end date.")

            time_min_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
            time_max_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

            time_min = format_datetime_for_api(time_min_dt)
            time_max = format_datetime_for_api(time_max_dt)

        except ValueError as e:
            return json.dumps({'error': f'Invalid date format or range: {e}. Use YYYY-MM-DD.'}), 400, {'Content-Type': 'application/json'}

        # --- 3. Authenticate ---
        try:
            service_account_key_str = get_secret('CALENDAR_SERVICE_ACCOUNT')
            service_account_key = json.loads(service_account_key_str)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_key,
                scopes=['https://www.googleapis.com/auth/calendar.readonly']
            )
            service = build('calendar', 'v3', credentials=credentials)
        except Exception as e:
            print(f"Authentication error: {e}")
            return json.dumps({'error': f'Authentication failed: {e}'}), 500, {'Content-Type': 'application/json'}

        # --- 4. Query Google Calendar API ---
        print(f"Querying Calendar ID: {calendar_id}")
        print(f"Querying Range: {time_min} to {time_max}")

        all_events_in_range = []
        page_token = None
        while True: # Handle pagination if fetching more for filtering
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results, # Use adjusted max_results
                singleEvents=True,
                orderBy='startTime',
                pageToken=page_token # Add pagination token
            ).execute()

            fetched_items = events_result.get('items', [])
            all_events_in_range.extend(fetched_items)
            print(f"Fetched {len(fetched_items)} events page.")

            page_token = events_result.get('nextPageToken')
            if not page_token or len(all_events_in_range) >= max_results: # Stop if no more pages or we hit limit
                 break

        print(f"Found {len(all_events_in_range)} total events in range.")

        # --- 5. Filter by Location (if requested) ---
        if location_filter:
            filtered_events = []
            for event in all_events_in_range:
                event_location = event.get('location', '').lower()
                # Simple 'contains' check, case-insensitive
                if location_filter in event_location:
                    filtered_events.append(event)
            events_to_format = filtered_events
            print(f"Filtered down to {len(events_to_format)} events matching location '{location_filter}'.")
        else:
            # No location filter applied, format all fetched events
            events_to_format = all_events_in_range

        # --- 6. Process and Format Events ---
        formatted_events = []
        for event in events_to_format:
            start_date_fmt, start_time_fmt, _ = parse_event_time(event.get('start', {}))
            end_date_fmt, end_time_fmt, _ = parse_event_time(event.get('end', {}))

            attendees = [
                att.get('email')
                for att in event.get('attendees', [])
                if att.get('email') and not att.get('resource', False)
            ]
            service_account_email = service_account_key.get('client_email')
            if service_account_email in attendees:
                 attendees.remove(service_account_email)

            formatted_events.append({
                'title': event.get('summary', 'No Title Provided'),
                'start_date': start_date_fmt,
                'start_time': start_time_fmt,
                'end_date': end_date_fmt,
                'end_time': end_time_fmt,
                'location': event.get('location', 'No Location Provided'),
                # 'attendees': attendees,
                # 'description': event.get('description', 'No Description Provided')
            })

        # --- 7. Prepare and Return JSON Response ---
        response_data = {
            "requested_range": {
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            },
            "filter_applied": {
                 "location": location_filter if location_filter else "None"
            },
            "events_found": len(formatted_events),
            "events": formatted_events
        }
        return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

    except HttpError as error:
        print(f'An API error occurred: {error}')
        error_details = json.loads(error.content).get('error', {})
        return json.dumps({'error': f'Google Calendar API error: {error_details.get("message", str(error))}'}), getattr(error, 'resp', {}).get('status', 500), {'Content-Type': 'application/json'}
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        # traceback.print_exc()
        return json.dumps({'error': f'An unexpected error occurred: {str(e)}'}), 500, {'Content-Type': 'application/json'}

# Example of how you might simulate a request for local testing (requires ADC setup)
if __name__ == '__main__':
    # Set environment variable for testing if needed
    os.environ['CALENDAR_ID'] = '3a0e2cb00ae76004ed65b48a355eccaf81f8f42036d69ff6c65fa06a9f2a5cbb@group.calendar.google.com' # Or your specific calendar ID
    # Set GOOGLE_APPLICATION_CREDENTIALS locally to your service account key file

    class MockRequest:
        def __init__(self, args):
            self.args = args

    # Test with default range
    print("--- Testing Default Range (Last 14 days) ---")
    req_default = MockRequest({})
    resp, code, _ = get_past_appointments(req_default)
    print(f"Status Code: {code}")
    print(json.dumps(json.loads(resp), indent=2))

    # Test with specific range
    print("\n--- Testing Specific Range ---")
    # Adjust dates as needed for your calendar
    start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d')
    end = (datetime.now(timezone.utc) - timedelta(days=15)).strftime('%Y-%m-%d')
    req_specific = MockRequest({'start_date': start, 'end_date': end})
    resp, code, _ = get_past_appointments(req_specific)
    print(f"Status Code: {code}")
    print(json.dumps(json.loads(resp), indent=2))