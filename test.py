import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
# ensure API key is clean
key = os.getenv("GEMINI_API_KEY")
if not key:
    raise RuntimeError("GEMINI_API_KEY not set")
key = key.strip('"\'')
# also set GOOGLE_API_KEY for new SDK
os.environ["GOOGLE_API_KEY"] = key

print("using api key", key[:10] + "...")

print("genai attributes:", [a for a in dir(genai) if not a.startswith('_')])
# configure may not exist; check existence
if hasattr(genai, 'configure'):
    print("genai.configure is available")
    genai.configure(api_key=key)
else:
    print("genai.configure not available, skipping configure")

print("Available models for your API key:")
client = genai.ModelsClient()
for m in client.list_models():
    methods = getattr(m, 'supported_generation_methods', None)
    print(f" - {m.name} methods={methods}")
