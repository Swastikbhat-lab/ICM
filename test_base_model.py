"""
Test script for Hyperbolic BASE model (completions endpoint)
"""
import requests
import json

API_KEY = "sk_live_lkl-Jd7_GUiIrBTeXKsODwfAAK29ev7Meu6qZzETuGa2rOWCoe0aVsYqs-3F0bVJE"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Test completions endpoint (for base models)
print("="*50)
print("Testing COMPLETIONS endpoint (for base models)")
print("="*50)

base_models = [
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Llama-3.1-405B",
    "meta-llama/Meta-Llama-3.1-405B-Base",
    "meta-llama/Llama-3.1-405B-Base",
    "meta-llama/Llama-3.1-405B-BASE",
]

for model in base_models:
    print(f"\nTesting: {model}")
    
    # Try completions endpoint
    data = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    
    try:
        response = requests.post(
            "https://api.hyperbolic.xyz/v1/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"  Completions endpoint: Status {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  SUCCESS! Response: {result['choices'][0]['text'][:50]}")
        else:
            try:
                err = response.json()
                print(f"  Error: {str(err)[:100]}")
            except:
                print(f"  Error: {response.text[:100]}")
                
    except Exception as e:
        print(f"  Exception: {e}")

# Also test chat endpoint with base model name (just in case)
print("\n" + "="*50)
print("Testing CHAT endpoint with base model names")
print("="*50)

for model in base_models:
    print(f"\nTesting: {model}")
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    
    try:
        response = requests.post(
            "https://api.hyperbolic.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"  Chat endpoint: Status {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  SUCCESS! Response: {result['choices'][0]['message']['content'][:50]}")
        else:
            try:
                err = response.json()
                print(f"  Error: {str(err)[:80]}")
            except:
                print(f"  Error: {response.text[:80]}")
                
    except Exception as e:
        print(f"  Exception: {e}")

print("\n\nDone!")
