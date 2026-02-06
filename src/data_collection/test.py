from google import genai
from dotenv import load_dotenv
import os

# Load env vars from project root
load_dotenv(dotenv_path='../../.env')

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set (check .env path and contents)")

client = genai.Client(api_key=api_key)

for m in client.models.list():
    print(m.name, m.supported_generation_methods)