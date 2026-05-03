import os
import json
from openai import OpenAI
from config import config

PROFILE_SCHEMA_INSTRUCTIONS = """
The Strategy Profile JSON must follow this exact schema:

```json
{
  "strategy_name": "String name of the strategy",
  "profile_version": 1,
  
  "indicators": {
    "indicator_id_1": {"fn": "NAME_FROM_CATALOG", "params": {"length": "parameter_name_from_pine"}},
    ...
  },
  
  "entry_logic": {
    "long": [
      {"type": "price_above_stack", "stack": ["indicator_id_1", "indicator_id_2"]},
      {"type": "regime_filter", "prior_trend": "bear", "min_bars": "parameter_name", "window": "parameter_name", "enabled_by": "parameter_name"},
      {"type": "fresh_cross", "direction": "above", "within_bars": "parameter_name"},
      {"type": "candle_streak", "source": "indicator_id", "direction": "green", "min": "parameter_name"},
      {"type": "wick_filter", "source": "indicator_id", "side": "upper", "enabled_by": "parameter_name"}
    ],
    "short": [ ... ]
  },
  
  "optimizable_parameters": [ ... array of parameters ... ]
}
```

Available INDICATOR_CATALOG names:
EMA, SMA, WMA, VWAP, DONCHIAN, RSI, MACD, STOCH, ATR, OBV, VOL_MA, BODY_PCT, WICK_RATIO, TREND_STREAK, BARS_SINCE, CROSS_ABOVE, CROSS_BELOW, TRAMA, HA
"""

class StrategyProfile:
    @staticmethod
    def generate_profile(pine_text: str, market_snapshot: dict, base_recipe: dict, client: OpenAI = None, project_dir: str = None, use_cached: bool = False) -> dict:
        profile_path = os.path.join(project_dir, "strategy_profile.json") if project_dir else None
        
        if use_cached and profile_path and os.path.exists(profile_path):
            print(f"[OptiEngine - Profile] Loading cached strategy profile from {profile_path}")
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[OptiEngine - Profile] Failed to load cached profile: {e}. Regenerating...")
                
        print("[OptiEngine - Profile] Generating dynamic strategy profile via LLM...")
        
        if client is None:
            client = OpenAI(base_url=config.llm_base_url, api_key="lm-studio")
            
        minified_lines = [line.split('//')[0].rstrip() for line in pine_text.split('\n') if line.split('//')[0].rstrip().strip()]
        minified_pine = '\n'.join(minified_lines)
        
        prompt = f"""
You are the Quant Developer agent for Optimization Engine.
I have statically parsed the following Pine Script and extracted its parameters.
Your task is to:
1. Refine the `min` and `max` bounds for these parameters based on your domain knowledge.
2. For any parameter with `"role": "unclassified"`, assign it a role: "signal", "risk", "execution", "display", or "session".
3. Map the strategy's logic onto my available indicator building blocks and output a Strategy Profile JSON.

{PROFILE_SCHEMA_INSTRUCTIONS}

Current Market Context:
- Regime: {market_snapshot.get('regime', 'Unknown')}
- Breadth: {market_snapshot.get('breadth', 'Unknown')}

Base Recipe (from static analysis):
```json
{json.dumps(base_recipe, indent=2)}
```

Here is the minified Pine Script for context:
```pine
{minified_pine}
```

Output ONLY valid JSON.
"""
        try:
            from llm_utils import call_llm_with_retry
            content = call_llm_with_retry(
                client, 
                messages=[
                    {"role": "system", "content": "You are a highly structured quantitative developer. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            if not content:
                return base_recipe

            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.endswith("```"):
                content = content[:-3]
                
            profile = json.loads(content)
            
            # Save profile
            if profile_path:
                with open(profile_path, 'w') as f:
                    json.dump(profile, f, indent=2)
                print(f"[OptiEngine - Profile] Saved generated profile to {profile_path}")
                
            return profile
            
        except Exception as e:
            print(f"[OptiEngine - Profile] Error generating profile: {e}")
            return base_recipe
