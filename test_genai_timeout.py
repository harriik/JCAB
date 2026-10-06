import os
from dotenv import load_dotenv
load_dotenv(".env")

from google import genai
import time

api_key = os.environ.get("LLM_API_KEY")
model = "gemini-1.5-flash"

print("Test 1: no http_options")
client1 = genai.Client(api_key=api_key)
t0 = time.time()
try:
    resp1 = client1.models.generate_content(model=model, contents="Hello")
    print("Success in", time.time()-t0)
except Exception as e:
    print("Failed in", time.time()-t0, str(e))

print("Test 2: timeout 45000")
client2 = genai.Client(api_key=api_key, http_options={'timeout': 45000.0})
t0 = time.time()
try:
    resp2 = client2.models.generate_content(model=model, contents="Hello")
    print("Success in", time.time()-t0)
except Exception as e:
    print("Failed in", time.time()-t0, str(e))

print("Test 3: timeout 45.0")
client3 = genai.Client(api_key=api_key, http_options={'timeout': 45.0})
t0 = time.time()
try:
    resp3 = client3.models.generate_content(model=model, contents="Hello")
    print("Success in", time.time()-t0)
except Exception as e:
    print("Failed in", time.time()-t0, str(e))
