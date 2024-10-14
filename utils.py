import os
import sys
import json
import requests
import time
from typing import Optional, List, Literal

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

Model = Literal["gpt-4", "gpt-3.5-turbo", "o1-mini", "gpt-4o"]

# Set the API key and base URL
api_key = YOUR_OpenAI_API_KEY
# openai completion API
api_base = "https://api.openai.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def get_chat(prompt: str, model: Model, seed: int, temperature: float = 0.4, max_tokens: int = 1000, stop_strs: Optional[List[str]] = None, is_batched: bool = False, debug: bool = False):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    data = {
        "model": model,
        "messages": messages,
        # "temperature": temperature,
        # "max_tokens": max_tokens
    }
    
    if seed is not None:
        data["seed"] = seed
    
    if stop_strs:
        data["stop"] = stop_strs

    try:
        response = requests.post(api_base, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            if debug:
                print(result)
            return result['choices'][0]['message']['content']
        else:
            print(f"Response error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    print(f"Using API key ending with: {api_key[-4:]}")

    prompt = "Hello! Could you tell me what is the transformer?"
    response = get_chat(prompt, "o1-mini", seed=None, debug=True)
    
    if response:
        print(response)
    else:
        print("Failed to get a response.")
