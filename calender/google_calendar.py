import os
import datetime
import pytz
import logging
import json
import difflib
from typing import Optional

from langchain_core.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build, Resource
from google.api_core.exceptions import NotFound
from google.cloud import secretmanager

import config
from config import APPOINTMENT_DURATION_MINUTES, TIMEZONE, PROPERTIES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Calendar API Setup ---
SCOPES = ["https://www.googleapis.com/auth/calendar"]
# The ID of the secret in Google Secret Manager containing the OAuth token JSON.
OAUTH_TOKEN_SECRET_ID = "oauth-token-rayyan-a-mahammed-gmail-comwfvef"#os.getenv("OAUTH_TOKEN_SECRET_ID")

# --- Caching ---
_calendar_service: Optional[Resource] = None


def get_property_by_id(property_id: str) -> Optional[dict]:
    """Helper function to find a property by its ID."""
    for prop in PROPERTIES:
        if prop['id'] == property_id:
            return prop
    return None


def _load_credentials_from_secret_manager() -> Optional[Credentials]:
    """Loads Google OAuth credentials from Google Secret Manager."""
    if not OAUTH_TOKEN_SECRET_ID or not config.GCP_PROJECT_ID:
        logging.info("Secret Manager environment variables not set. Skipping.")
        return None
    
    try:
        logging.info(f"Attempting to load OAuth token from Secret Manager: '{OAUTH_TOKEN_SECRET_ID}'")
        client = secretmanager.SecretManagerServiceClient(credentials=config.GOOGLE_CREDENTIALS)
        secret_name = f"projects/{config.GCP_PROJECT_ID}/secrets/{OAUTH_TOKEN_SECRET_ID}/versions/latest"
        response = client.access_secret_version(request={"name": secret_name})
        token_data = json.loads(response.payload.data.decode("UTF-8"))
        
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        
        # Refresh the token if it's expired
        if creds.expired and creds.refresh_token:
            logging.info("Credentials from Secret Manager are expired. Refreshing...")
            creds.refresh(Request())
            # Note: In a complete solution, you might want to save the refreshed token back to Secret Manager.
            # This is omitted for simplicity but is a key production consideration.
            
        logging.info("Successfully loaded and validated credentials from Secret Manager.")
        return creds
    except NotFound:
        logging.error(f"Secret '{OAUTH_TOKEN_SECRET_ID}' not found in project '{config.GCP_PROJECT_ID}'.")
        return None
    except Exception as e:
        logging.error(f"Failed to load or refresh token from Secret Manager: {e}", exc_info=True)
        return None


def _load_credentials_from_local_file() -> Optional[Credentials]:
    """Loads Google OAuth credentials from a local 'token.json' file for development."""
    if not os.path.exists("token.json"):
        logging.info("'token.json' not found. Skipping local file credential loading.")
        return None
        
    try:
        logging.info("Loading credentials from local 'token.json' file.")
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        
        # Refresh the token if it's expired
        if creds.expired and creds.refresh_token:
            logging.info("Local credentials expired. Refreshing token.")
            creds.refresh(Request())
            # Save the refreshed credentials back to token.json for subsequent local runs
            with open("token.json", "w") as token:
                token.write(creds.to_json())
            logging.info("Refreshed token saved to 'token.json'.")

        return creds
    except Exception as e:
        logging.error(f"Failed to load or refresh token from 'token.json': {e}", exc_info=True)
        return None


def get_calendar_service() -> Resource:
    """
    Initializes and returns an authorized Google Calendar service instance.

    This function orchestrates credential loading with the following priority:
    1. In-memory cache (`_calendar_service`).
    2. Google Secret Manager (for production environments).
    3. Local `token.json` file (for local development).

    Raises:
        Exception: If no valid credentials can be found or created.
    """
    global _calendar_service
    if _calendar_service:
        return _calendar_service

    creds = _load_credentials_from_secret_manager()
    logging.info(f"Credentials from Secret Manager: {creds.valid if creds else 'None'}")
    logging.info(f"Credentials from Secret Manager: {creds.valid if creds else 'None'}")

    if not creds:
        logging.warning("Could not load valid credentials from Secret Manager. Falling back to local file.")
        creds = _load_credentials_from_local_file()

    if not creds:
        logging.critical("FATAL: No valid Google Calendar credentials found.")
        raise Exception("Could not authenticate with Google Calendar. Please run the OAuth flow.")

    try:
        service = build("calendar", "v3", credentials=creds)
        _calendar_service = service
        logging.info("Google Calendar service initialized successfully.")
        return service
    except Exception as e:
        logging.error(f"Failed to build Google Calendar service: {e}", exc_info=True)
        raise


@tool
def find_available_slots(date_str: str) -> str:
    """
    Use this tool to find available appointment slots on a specific day.

    You MUST ask the user for a specific date before using this tool.

    Args:
        date_str: The date to check for availability in 'YYYY-MM-DD' format.

    This tool is the ONLY way to check for appointment availability.
    Do not guess or suggest times without calling this tool first.
    It returns a string with available 30-minute slots for the given day.
    """
    logging.info(f"Tool 'find_available_slots' invoked for date: {date_str}")
    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)

        # 1. Parse input and define search range for the day in the correct timezone
        try:
            # Handle both 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS' inputs gracefully
            search_date = datetime.datetime.fromisoformat(date_str.split(' ')[0]).date()
        except ValueError:
            logging.error(f"Invalid date format received: '{date_str}'")
            return "Error: Invalid date format. Please ask the user for a date in 'YYYY-MM-DD' format."

        day_start = tz.localize(datetime.datetime.combine(search_date, datetime.time(9, 0)))
        day_end = tz.localize(datetime.datetime.combine(search_date, datetime.time(17, 0)))
        
        # 2. Get all busy slots from the calendar for that day
        now = datetime.datetime.now(tz)
        time_min = max(day_start, now) # Don't search for slots in the past

        logging.info(f"Searching for free slots on {date_str} between {time_min.isoformat()} and {day_end.isoformat()}.")

        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        busy_slots = events_result.get('items', [])
        logging.info(f"Found {len(busy_slots)} busy slots in the calendar for {date_str}.")

        # 3. Generate all potential slots for the day
        potential_slots = []
        slot = time_min
        # Align start time to the next 15 or 30-minute mark
        if slot.minute not in [0, 15, 30, 45]:
            slot += datetime.timedelta(minutes=15 - slot.minute % 15)

        while slot < day_end:
            potential_slots.append(slot)
            slot += datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

        # 4. Filter out slots that overlap with busy periods
        available_slots = []
        for potential_slot in potential_slots:
            slot_end = potential_slot + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
            is_free = True
            for event in busy_slots:
                event_start = datetime.datetime.fromisoformat(event['start'].get('dateTime')).astimezone(tz)
                event_end = datetime.datetime.fromisoformat(event['end'].get('dateTime')).astimezone(tz)
                
                # Check for overlap: (StartA <= EndB) and (EndA >= StartB)
                if potential_slot < event_end and slot_end > event_start:
                    is_free = False
                    break
            
            if is_free:
                available_slots.append(potential_slot.strftime('%H:%M'))

        # 5. Format and return the result
        if not available_slots:
            logging.warning(f"No available slots found for {date_str}.")
            return f"I'm sorry, but there are no available slots on {date_str}. Would you like to check another date?"

        result_str = f"On {date_str}, the following times are available: {', '.join(available_slots)}."
        logging.info(f"Found available slots for {date_str}: {result_str}")
        return result_str
        
    except Exception as e:
        logging.error(f"Error in find_available_slots: {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to find available slots. Could you please try another date?"

@tool
def book_appointment(datetime_str: str, full_name: str, property_id: str) -> str:
    """
    Use this tool to book a property visit appointment.

    This is the final step in the booking process.
    You MUST have the user's full name, the desired datetime_str,
    and the property_id BEFORE calling this tool.

    Args:
        datetime_str: The appointment time in 'YYYY-MM-DD HH:MM' format (24-hour clock),
                      which MUST be one of the slots provided by 'find_available_slots'.
        full_name: The full name of the person booking the visit.
        property_id: The ID of the property they want to visit (e.g., 'PV001').
    """
    logging.info(f"Tool 'book_appointment' invoked with: datetime='{datetime_str}', name='{full_name}', property_id='{property_id}'")

    property_details = get_property_by_id(property_id)
    if not property_details:
        logging.error(f"Invalid property_id '{property_id}' passed to book_appointment.")
        return f"Error: I couldn't find a property with the ID '{property_id}'. Please confirm the property ID."

    property_name = property_details.get('address', 'Unknown Property')
    email = "mahammed.rayyan.a@gmail.com"


    try:
        service = get_calendar_service()
        tz = pytz.timezone(TIMEZONE)
        start_time = tz.localize(datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M"))
        end_time = start_time + datetime.timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
    except ValueError:
        logging.error(f"Invalid datetime format received: '{datetime_str}'")
        return "Error: Invalid datetime format. Please use 'YYYY-MM-DD HH:MM'."

    event = {
        'summary': f'Property Visit: {property_name} for {full_name}',
        'description': f'Booked by AI Assistant "Sky".\nClient Name: {full_name}\nClient Email: {email}\nProperty ID: {property_id}',
        'start': {'dateTime': start_time.isoformat(), 'timeZone': TIMEZONE},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': TIMEZONE},
        'attendees': [{'email': email}],
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60}, # 1 day before
                {'method': 'popup', 'minutes': 60},      # 1 hour before
            ],
        },
        'conferenceData': {
            'createRequest': {
                'requestId': f'visit-{property_id}-{start_time.timestamp()}',
                'conferenceSolutionKey': {'type': 'hangoutsMeet'}
            }
        }
    }

    try:
        created_event = service.events().insert(calendarId='primary', body=event, sendUpdates='all', conferenceDataVersion=1).execute()
        hangout_link = created_event.get('hangoutLink', 'Not available')
        logging.info(f"Successfully created event with ID: {created_event.get('id')}. Meet link: {hangout_link}")
        return f"Success! The appointment for {full_name} at {datetime_str} for the property at {property_name} has been booked. A calendar invite with a Google Meet link has been sent to your email."
    except Exception as e:
        logging.error(f"Failed to create calendar event: {e}", exc_info=True)
        return "Sorry, I was unable to book the appointment. There was an error with the calendar service."


@tool
def get_property_details(location: str) -> str:
    """
    Use this tool to get details of a property based on its location.
    
    Args:
        location: The location or address of the property to search for.
    """
    logging.info(f"Tool 'get_property_details' invoked for location: {location}")
    location_lower = location.lower()
    found_properties = []
    
    for prop in PROPERTIES:
        address_lower = prop['address'].lower()
        
        # 1. Exact substring match
        if location_lower in address_lower:
            found_properties.append(prop)
            continue
            
        # 2. Fuzzy match against address parts (to handle STT errors)
        # Split address into words/parts to check similarity against the search location
        # Replace commas with spaces to handle "City, Country" format
        parts = address_lower.replace(',', ' ').split()
        
        for part in parts:
            # Calculate similarity ratio (0.0 to 1.0)
            similarity = difflib.SequenceMatcher(None, location_lower, part).ratio()
            if similarity >= 0.7: # 70% similarity threshold
                found_properties.append(prop)
                break
            
    if not found_properties:
        return f"I'm sorry, I couldn't find any properties in {location}."
        
    if len(found_properties) == 1:
        prop = found_properties[0]
        return json.dumps(prop)
        
    return json.dumps(found_properties)