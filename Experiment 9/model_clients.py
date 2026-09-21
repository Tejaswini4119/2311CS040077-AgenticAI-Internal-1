"""
Module containing API clients for calling different Large Language Models.
Measures latency, token usage, and handles basic errors.
"""
import time
import os
import json

# Try to import SDKs, handling cases where they might not be installed yet
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

def call_openai(prompt: str, model_name: str) -> dict:
    """Calls OpenAI API and returns standard metrics."""
    if not OpenAI:
        return {"error": "openai library not installed", "response_text": "", "latency_ms": 0, "token_count": 0}
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not found", "response_text": "", "latency_ms": 0, "token_count": 0}

    client = OpenAI(api_key=api_key)
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = int((time.time() - start_time) * 1000)
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        return {
            "response_text": content,
            "latency_ms": latency_ms,
            "token_count": tokens,
            "error": None
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "response_text": "",
            "latency_ms": latency_ms,
            "token_count": 0,
            "error": str(e)
        }

def call_google(prompt: str, model_name: str) -> dict:
    """Calls Google Generative AI API and returns standard metrics."""
    if not genai:
        return {"error": "google-generativeai library not installed", "response_text": "", "latency_ms": 0, "token_count": 0}
        
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found", "response_text": "", "latency_ms": 0, "token_count": 0}

    genai.configure(api_key=api_key)
    start_time = time.time()
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Gemini token counting requires a separate call usually, rough estimation here if needed
        # Or if available in response.usage_metadata
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.total_token_count
            
        return {
            "response_text": response.text,
            "latency_ms": latency_ms,
            "token_count": tokens,
            "error": None
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "response_text": "",
            "latency_ms": latency_ms,
            "token_count": 0,
            "error": str(e)
        }

def call_anthropic(prompt: str, model_name: str) -> dict:
    """Calls Anthropic API and returns standard metrics."""
    if not Anthropic:
        return {"error": "anthropic library not installed", "response_text": "", "latency_ms": 0, "token_count": 0}
        
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not found", "response_text": "", "latency_ms": 0, "token_count": 0}

    client = Anthropic(api_key=api_key)
    start_time = time.time()
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
        
        return {
            "response_text": content,
            "latency_ms": latency_ms,
            "token_count": tokens,
            "error": None
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "response_text": "",
            "latency_ms": latency_ms,
            "token_count": 0,
            "error": str(e)
        }

def get_model_client(model_name: str):
    """Factory function to route to the correct client based on model name."""
    if model_name.startswith("gpt"):
        return call_openai
    elif model_name.startswith("gemini"):
        return call_google
    elif model_name.startswith("claude"):
        return call_anthropic
    else:
        raise ValueError(f"Unknown model prefix for: {model_name}")
