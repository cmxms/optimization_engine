import time
from openai import OpenAI

def call_llm_with_retry(client, messages, temperature=0.3, max_tokens=2000, retries=3, timeout=120):
    """
    Calls the LLM client with exponential backoff and a generous timeout.
    """
    last_error = None
    for attempt in range(retries):
        try:
            from vram_manager import vram
            response = client.chat.completions.create(
                model=vram.get_current_model_key(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            print(f"  [LLM] Attempt {attempt+1}/{retries} failed: {e}. Retrying in {2**(attempt+1)}s...")
            time.sleep(2**(attempt+1))
            
    print(f"  [LLM] All {retries} retries failed. Last error: {last_error}")
    return None
