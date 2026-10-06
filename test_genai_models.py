import os
from dotenv import load_dotenv
load_dotenv(".env")
from google import genai
api_key = os.environ.get("LLM_API_KEY")
client = genai.Client(api_key=api_key, http_options={'timeout': 45000.0})

for m in ["gemini-3.5-flash", "gemini-2.5-flash"]:
    print(f"Testing {m}...")
    try:
        resp = client.models.generate_content(model=m, contents="Hello")
        print("Success!")
    except Exception as e:
        print("Failed:", str(e))
