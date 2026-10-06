import os
from dotenv import load_dotenv
load_dotenv(".env")
from google import genai
api_key = os.environ.get("LLM_API_KEY")
client = genai.Client(api_key=api_key)
print("Available models:")
for m in client.models.list():
    if "gemini" in m.name:
        print(m.name)
