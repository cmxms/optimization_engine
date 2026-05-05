from ir_builder import build_strategy_ir

class QuantDeveloper:
    def __init__(self, pine_text: str, market_snapshot: dict = None):
        self.pine_text = pine_text
        self.market_snapshot = market_snapshot or {}

    def extract_recipe(self) -> dict:
        print("[OptiEngine - Developer] Extracting Intermediate Representation (IR) via IRBuilder...")
        ir = build_strategy_ir(self.pine_text)
        print(f"  -> Extracted {len(ir['optimizable_parameters'])} parameters and {len(ir['filters'])} filters.")
        return ir

def run_developer(pine_text, snapshot=None):
    dev = QuantDeveloper(pine_text, snapshot)
    recipe = dev.extract_recipe()
    return recipe
