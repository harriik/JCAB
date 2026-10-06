import os
import sys
import socket
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
load_dotenv(".env")

def test_dns():
    print("Testing DNS resolution for generativelanguage.googleapis.com...")
    try:
        socket.gethostbyname("generativelanguage.googleapis.com")
        print("DNS PASS")
        return True
    except socket.gaierror as e:
        print(f"DNS FAILURE: {e}")
        return False

def test_https():
    print("\nTesting HTTPS connectivity to generativelanguage.googleapis.com...")
    try:
        response = urllib.request.urlopen("https://generativelanguage.googleapis.com", timeout=10)
        print("HTTPS PASS")
        return True
    except urllib.error.HTTPError as e:
        # Expected since we aren't sending a valid request, but means we reached the server
        print(f"HTTPS PASS (HTTP {e.code})")
        return True
    except urllib.error.URLError as e:
        print(f"NETWORK FAILURE: {e.reason}")
        return False
    except Exception as e:
        print(f"NETWORK FAILURE: {e}")
        return False

def test_api():
    print("\nTesting Gemini API generation...")
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("API INITIALIZATION FAILED: No LLM_API_KEY found")
        return False
        
    model_name = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
    print(f"Using model: {model_name}")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key, http_options={'timeout': 15000.0})
        
        response = client.models.generate_content(
            model=model_name,
            contents="Hello, reply with only the word OK."
        )
        print(f"API Response: {response.text.strip()}")
        print("\nGEMINI CONNECTION PASS")
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "503" in err_str or "502" in err_str or "504" in err_str or "unavailable" in err_str or "high demand" in err_str:
            print("\nGEMINI SERVER CAPACITY")
        elif "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            print("\nGEMINI QUOTA/RATE LIMIT")
        elif "timeout" in err_str or "timed out" in err_str or "connection" in err_str:
            print(f"\nNETWORK FAILURE: {e}")
        else:
            print(f"\nAPI FAILURE: {e}")
        return False

def main():
    print("==================================================")
    print("      GEMINI CONNECTIVITY DIAGNOSTIC")
    print("==================================================")
    
    if not test_dns():
        return
    if not test_https():
        return
    test_api()

if __name__ == "__main__":
    main()
