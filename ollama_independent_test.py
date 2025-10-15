# test_ollama_simple.py
import requests
import json
import time

def test_ollama():
    print("Testing Ollama with simple prompt...")
    
    start = time.time()
    
    payload = {
        "model": "llama3.2",
        "prompt": "What is 2+2? Answer in one sentence.",
        "stream": False,
        "options": {
            "num_ctx": 512,  # Small context
            "temperature": 0.3
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response received in {elapsed:.2f} seconds")
            print(f"Answer: {result['response']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Timeout after 30 seconds - Ollama not responding properly")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama()