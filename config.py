import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from a .env file if it exists.
# This is great for local development.
load_dotenv()

# --- Logging Configuration ---
# A centralized place for logging setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Google Cloud & Gemini Configuration ---
# Using os.getenv() is the standard way to access environment variables.
# It returns None if the variable is not found, which is handled gracefully.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Specific models for different tasks
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-1.5-flash-latest")

# Configuration for the real-time audio stream
GEMINI_AUDIO_SAMPLE_RATE = int(os.getenv("GEMINI_AUDIO_SAMPLE_RATE", 16000))

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# --- Google Calendar & OAuth Configuration ---
# The ID of the secret in Google Secret Manager containing the user's OAuth token.
# This is the recommended way to store user tokens in production.
OAUTH_TOKEN_SECRET_ID = os.getenv("OAUTH_TOKEN_SECRET_ID")

# Scopes define the level of access the application is requesting.
# It's best practice to request only the scopes you need.
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
]

# --- Business Logic & Agent Configuration ---
YOUR_BUSINESS_NAME: str = "Prestige Properties"
APPOINTMENT_DURATION_MINUTES: int = 30
TIMEZONE: str = os.getenv("TIMEZONE", "Asia/Kolkata")  # IANA Time Zone format (e.g., "America/New_York", "Europe/London")
OUTBOUND_CALL_INTERVAL_SECONDS: int = int(os.getenv("OUTBOUND_CALL_INTERVAL_SECONDS", 15))

# --- Static Knowledge Base for the Agent ---
# This list of dictionaries serves as the agent's knowledge about available properties.
# It's injected into the system prompt.
PROPERTIES: List[dict] = [
    {
        "id": "PV001",
        "address": "bellandur, Bangalore, India",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 50000000,
        "features": "A beautiful family home with a large backyard and a newly renovated, modern kitchen.",
    },
    {
        "id": "PV002",
        "address": "whitefield, Bangalore, India",
        "bedrooms": 4,
        "bathrooms": 3.5,
        "price": 75000000,
        "features": "A spacious luxury home featuring a private pool, home theater, and a three-car garage.",
    },
    {
        "id": "PC001",
        "address": "indiranagar, Bangalore, India",
        "bedrooms": 2,
        "bathrooms": 2,
        "price": 32000000,
        "features": "A modern downtown condo with stunning city views, a rooftop terrace, and 24-hour concierge service.",
    }
]

# --- Validation ---
# A simple function to check for the presence of critical environment variables.
# This helps catch configuration errors early.
def validate_config():
    """Validates that essential configuration variables are set."""
    logging.info("Validating configuration...")
    required_variables = {
        "GCP_PROJECT_ID": GCP_PROJECT_ID,
        "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
        "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
        "TWILIO_PHONE_NUMBER": TWILIO_PHONE_NUMBER,
    }

    missing_vars = [key for key, value in required_variables.items() if not value]

    if missing_vars:
        error_message = f"Missing critical environment variables: {', '.join(missing_vars)}"
        logging.critical(error_message)
        raise ValueError(error_message)

    # This variable is only required for the OAuth flow, not for the main app to run
    if not os.getenv("GOOGLE_OAUTH_WEB_CLIENT_SECRETS"):
        logging.warning(
            "GOOGLE_OAUTH_WEB_CLIENT_SECRETS is not set. "
            "The /auth and /oauth2callback endpoints will not function."
        )

    logging.info("Configuration validated successfully.")

# Run validation when the module is imported.
validate_config()

