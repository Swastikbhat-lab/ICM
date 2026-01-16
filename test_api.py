"""
Test script to find working Hyperbolic models
"""
import requests
import json

API_KEY = "sk_live_lkl-Jd7_GUiIrBTeXKsODwfAAK29ev7Meu6qZzETuGa2rOWCoe0aVsYqs-3F0bVJE"

# All possible model name formats to test
models_to_test = [
    # 405B variants
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct", 
    "meta-llama/Meta-Llama-3.1-405B",
    "meta-llama/Llama-3.1-405B",
    # 70B variants (more likely to work)
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B",
    # 3.3 70B
    "meta-llama/Llama-3.3-70B-Instruct",
    # 8B (smallest, most likely to work)
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

print("Testing Hyperbolic API with different model names...\n")

working_models = []

for model in models_to_test:
    print(f"Testing: {model}...", end=" ")
    
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
        
        if response.status_code == 200:
            print("WORKS!")
            working_models.append(model)
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"   Response: {content[:50]}")
        else:
            print(f"Error {response.status_code}")
            # Show error detail for first few failures
            if len(working_models) == 0:
                try:
                    err = response.json()
                    print(f"   Detail: {str(err)[:100]}")
                except:
                    print(f"   Detail: {response.text[:100]}")
        
    except Exception as e:
        print(f"Exception: {e}")

print("\n" + "="*50)
print("WORKING MODELS:")
print("="*50)
for m in working_models:
    print(f"  - {m}")

if not working_models:
    print("  No models worked! Check your API key and account balance.")
