import json
import requests
import os
from config import GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY is missing. Set it in your .env file.")
    # Try to get it from env directly if config failed (though config does getenv)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY is definitely missing.")

# Try v1beta first
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_API_KEY}"
resp = requests.get(url, timeout=30)
print(f"Fetching models from {url}...")
print("HTTP", resp.status_code)

if not resp.ok:
    print(resp.text)
    raise SystemExit(1)

data = resp.json()
models = data.get("models", [])

# Print concise, useful info
rows = []
for m in models:
    supported_methods = m.get("supportedGenerationMethods", [])
    rows.append({
        "name": m.get("name"),
        "displayName": m.get("displayName"),
        "supported": supported_methods,
        "supports_bidi": "bidiGenerateContent" in supported_methods
    })

rows.sort(key=lambda r: r["name"] or "")
print(json.dumps(rows, indent=2))

# Filter for bidiGenerateContent
print("\nModels supporting bidiGenerateContent:")
for r in rows:
    if r["supports_bidi"]:
        print(f"- {r['name']}")
