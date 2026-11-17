
import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.environ["OPENWEBUI_URL"].rstrip("/")
api_key = os.environ["OPENWEBUI_API_KEY"]

response = requests.get(
    f"{base_url}/api/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30
)

response.raise_for_status()

for model in response.json().get("data", []):
    print("Name:", model.get("name"))
    print("ID:  ", model.get("id"))
    print()
