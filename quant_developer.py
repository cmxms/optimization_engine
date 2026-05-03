import json
import re
from openai import OpenAI
from config import config
from pine_parser import PineParser

class QuantDeveloper:
    def __init__(self, pine_text: str, market_snapshot: dict = None):
        self.pine_text = pine_text
        self.market_snapshot = market_snapshot or {}
        # Minify script to save context
        minified_lines = []
        for line in pine_text.split("\n"):
            clean_line = line.split('//')[0].rstrip()
            if clean_line.strip():
                minified_lines.append(clean_line)
        self.minified_pine = '\n'.join(minified_lines)

    def extract_recipe(self, client: OpenAI = None, project_dir: str = None, use_cached: bool = False) -> dict:
        # Use the static parser to guarantee 100% parameter coverage
        print("[OptiEngine - Developer] Statically extracting full strategy recipe via PineParser...")
        parser = PineParser(self.pine_text)
        base_recipe = parser.extract_recipe()
        print(f"  -> Extracted {len(base_recipe['optimizable_parameters'])} parameters.")

        if not config.use_llm:
            return base_recipe

        print("[OptiEngine - Developer] Enhancing recipe via LLM and generating Strategy Profile...")
        from strategy_profile import StrategyProfile
        enhanced_recipe = StrategyProfile.generate_profile(
            pine_text=self.pine_text,
            market_snapshot=self.market_snapshot,
            base_recipe=base_recipe,
            client=client,
            project_dir=project_dir,
            use_cached=use_cached
        )
        
        # Safety check: Ensure the LLM didn't drop parameters
        if len(enhanced_recipe.get('optimizable_parameters', [])) < len(base_recipe['optimizable_parameters']):
            print("[OptiEngine - Developer] LLM dropped parameters. Falling back to static recipe.")
            return base_recipe
            
        # Clamp LLM bounds to safe ranges (no more than 50% expansion)
        for param in enhanced_recipe.get('optimizable_parameters', []):
            try:
                original = next(p for p in base_recipe['optimizable_parameters'] if p['name'] == param['name'])
                if param.get('max', 0) > original.get('max', 0) * 1.5:
                    print(f"  -> Clamping unsafe LLM bound for {param['name']}: {param.get('max')} -> {original.get('max')}")
                    param['max'] = original['max']
            except StopIteration:
                pass
                
        return enhanced_recipe

def run_developer(project_dir, snapshot=None, client=None):
    import os, glob
    pine_path = glob.glob(os.path.join(project_dir, "*.pine"))[0]
    with open(pine_path, "r") as f: 
        pine_text = f.read()
    dev = QuantDeveloper(pine_text, snapshot)
    recipe = dev.extract_recipe(client=client, project_dir=project_dir)
    return pine_text, recipe
