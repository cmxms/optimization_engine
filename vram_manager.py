"""
VRAM Manager for Optimization Engine
Handles automatic model loading and unloading between agent steps using the
official LM Studio Python SDK. This ensures only ONE model occupies VRAM at
a time, which is critical for 8GB VRAM systems.

Usage:
    manager = VRAMManager()
    client = manager.load_agent("critic")       # Unloads prev, loads DeepSeek-Coder
    # ... run critic ...
    client = manager.load_agent("developer")    # Unloads DeepSeek, loads Qwen
    # ... run quant developer ...
    manager.unload_all()                        # Clean up at end of pipeline
"""

import time
from openai import OpenAI
from config import config

# Global placeholder for lazy loading
lms = None

# Agent → Model Mapping
# Update model keys to match the exact model filenames in your LM Studio library.
AGENT_MODEL_MAP = {
    "critic":    "deepseek-ai/deepseek-coder-v2-lite-instruct",  # 7B — Code logic audit
    "developer": "qwen/qwen2.5-coder-7b-instruct",              # 7B — Structured JSON extraction
    "strategist": "mistralai/mistral-nemo-instruct-2407",      # 12B — Narrative synthesis (fits in 8GB Q4)
}

class VRAMManager:
    def __init__(self):
        self._current_handle = None
        self._current_agent = None
        self._current_resolved_key = None

    def _ensure_lms(self):
        global lms
        if lms is None:
            import lmstudio
            lms = lmstudio

    def _unload_current(self):
        """Explicitly unload the currently active model to free VRAM."""
        if self._current_handle is not None:
            try:
                print(f"  [VRAM] Unloading '{self._current_agent}' model...")
                self._current_handle.unload()
                self._current_handle = None
                self._current_agent = None
                self._current_resolved_key = None
                # Brief pause to let VRAM fully clear before next load
                time.sleep(2)
                print("  [VRAM] Model unloaded successfully.")
            except Exception as e:
                print(f"  [VRAM] Warning: Could not unload model cleanly: {e}")
                self._current_handle = None
                self._current_agent = None

    def load_agent(self, agent_name: str) -> OpenAI:
        """
        Unloads current model, finds the best matching downloaded model, and loads it.
        """
        if not config.use_llm:
            return OpenAI(base_url=config.llm_base_url, api_key="not-used")

        # 1. Determine search keyword
        search_kw = ""
        if agent_name == "critic": search_kw = "deepseek"
        elif agent_name == "developer": search_kw = "qwen"
        elif agent_name == "strategist": search_kw = "mistral"
        
        # 2. Search local library for the best match
        print(f"  [VRAM] Searching local library for '{search_kw}' model...")
        try:
            self._ensure_lms()
            downloaded = lms.list_downloaded_models("llm")
            # Filter for keyword and prefer Q4/Q3 if multiple exist
            matches = [m.model_key for m in downloaded if search_kw.lower() in m.model_key.lower()]
            
            if not matches:
                print(f"  [VRAM] ERROR: No models found matching '{search_kw}'.")
                print(f"  [VRAM] Please download a {search_kw} model in LM Studio first.")
                raise FileNotFoundError(f"No {search_kw} model found.")
            
            # Simple heuristic: use the first match (usually newest or best name match)
            model_key = matches[0]
            print(f"  [VRAM] Auto-detected best match: {model_key}")
        except Exception as e:
            print(f"  [VRAM] Error accessing LM Studio library: {e}")
            raise

        # 3. Reload logic
        if self._current_agent == agent_name and self._current_handle is not None:
            print(f"  [VRAM] '{agent_name}' model already loaded. Skipping reload.")
        else:
            self._unload_current()
            print(f"  [VRAM] Loading '{agent_name}' model...")
            try:
                self._ensure_lms()
                # Request a larger context window (16k) to handle big strategies
                # Using 'n_ctx' as the standard parameter for context size
                self._current_handle = lms.llm(model_key, config={"n_ctx": 16384})
                self._current_agent = agent_name
                self._current_resolved_key = model_key
                print(f"  [VRAM] Success. Model {model_key} loaded (n_ctx=16k).")
            except Exception as e:
                print(f"  [VRAM] CRITICAL: Failed to load {model_key}: {e}")
                raise

        # Step 3: Return an OpenAI-compatible client for inference
        # (optimization_engine agents use the openai library for completions)
        return OpenAI(
            base_url=config.llm_base_url,
            api_key="lm-studio"
        )

    def get_current_model_key(self) -> str | None:
        """Returns the LM Studio model key for the currently loaded agent."""
        return self._current_resolved_key

    def unload_all(self):
        """Call this at the end of the pipeline to release all VRAM."""
        self._unload_current()
        print("  [VRAM] All models unloaded. VRAM released.")


# Singleton instance used by all agents
vram = VRAMManager()
