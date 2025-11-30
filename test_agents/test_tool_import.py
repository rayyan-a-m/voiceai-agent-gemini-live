
from calender.google_calendar import find_available_slots, book_appointment

print(f"Type of find_available_slots: {type(find_available_slots)}")
print(f"Dir of find_available_slots: {dir(find_available_slots)}")

try:
    # Try calling it as a function
    # It might expect a dict if it's a langchain tool
    print("Trying to call as function...")
    # We won't actually run it fully to avoid API calls if possible, but we can catch the error
    # Or just check the type.
except Exception as e:
    print(f"Error calling as function: {e}")
