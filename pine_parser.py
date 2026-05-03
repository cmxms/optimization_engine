import re

class PineParser:
    def __init__(self, pine_text: str):
        self.pine_text = pine_text

    def _classify_display_params(self, name: str, line: str) -> str:
        """
        Regex pre-filter to identify display/cosmetic parameters.
        Returns 'display', 'risk', or 'signal'.
        """
        name_lower = name.lower()
        line_lower = line.lower()
        
        # Explicit display types
        if any(t in line_lower for t in ("input.color", "input.session", "input.source", "input.string")):
            return "display"

        # Generic display patterns
        if name_lower.startswith(('show_', 'hide_', 'plot_', 'color_', 'label_')) or \
           any(s in name_lower for s in ('_color', '_table', '_line', '_dot', '_bg', '_debug')):
            return "display"

        # Execution guards — these filter trades rather than improve exit mechanics.
        # Never optimize these in Tier 1 (TV signal) mode.
        if any(e in name_lower for e in (
            "slippage", "multiplier", "commission", "order_type",
            "min_bars", "daily_loss", "max_trades", "block_same", "use_daily", "use_max"
        )):
            return "execution"

        # Pure risk/exit mechanics — stop placement, target, trailing
        if any(r in name_lower for r in ("stop", "target", "trail", "risk", "sl_ticks", "tp_ticks")):
            return "risk"
            
        return "signal"  # Default for all other numerical/bool inputs

    def extract_recipe(self) -> dict:
        """
        Parses Pine Script to extract input variables and their bounds.
        Returns a recipe dictionary compatible with QuantEngine.
        """
        parameters = []
        
        # Split text into lines to process line by line
        lines = self.pine_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Ignore comments
            if line.startswith('//'):
                continue
            
            # Look for input declarations: var_name = input.type(...) or var_name = input(...)
            # This regex captures the variable name, the input type (int, float, bool), and the rest of the arguments
            match = re.match(r'^([a-zA-Z0-9_]+)\s*=\s*input\.(int|float|bool)\((.*)\)', line)
            
            if match:
                var_name = match.group(1)
                var_type = match.group(2)
                args_str = match.group(3)
                
                # Extract default value. It's usually the first argument.
                # E.g., 10, "TRAMA Fast" -> 10
                # true, "Use regime filter" -> true
                
                # Try to extract default value
                default_val_match = re.match(r'^\s*([^,]+)', args_str)
                if not default_val_match:
                    continue
                
                default_val_str = default_val_match.group(1).strip()
                
                param_dict = {
                    "name": var_name,
                    "type": var_type,
                    "role": self._classify_display_params(var_name, line)
                }
                
                if var_type == 'bool':
                    param_dict['default'] = default_val_str.lower() == 'true'
                    # For bools, we can just say min 0, max 1 in optuna or handle specially
                    # It's better to let optuna suggest categorical [True, False]
                    # But for now, returning type bool is enough
                elif var_type in ('int', 'float'):
                    if var_type == 'int':
                        try:
                            default_val = int(default_val_str)
                        except ValueError:
                            continue # Could be a variable reference, skip
                    else:
                        try:
                            default_val = float(default_val_str)
                        except ValueError:
                            continue
                            
                    param_dict['default'] = default_val
                    
                    # Extract minval
                    minval_match = re.search(r'minval\s*=\s*([-\d.]+)', args_str)
                    if minval_match:
                        param_dict['min'] = int(minval_match.group(1)) if var_type == 'int' else float(minval_match.group(1))
                    else:
                        # Smart defaults if minval is missing
                        if var_type == 'int':
                            if "len" in var_name.lower() or "lookback" in var_name.lower():
                                param_dict['min'] = max(1, default_val // 2)
                            else:
                                param_dict['min'] = max(0, int(default_val * 0.5))
                        else:
                            param_dict['min'] = default_val * 0.5
                            
                    # Extract maxval
                    maxval_match = re.search(r'maxval\s*=\s*([-\d.]+)', args_str)
                    if maxval_match:
                        param_dict['max'] = int(maxval_match.group(1)) if var_type == 'int' else float(maxval_match.group(1))
                    else:
                        # Smart defaults if maxval is missing
                        if var_type == 'int':
                            if "len" in var_name.lower():
                                param_dict['max'] = max(10, default_val * 2)
                            else:
                                param_dict['max'] = int(default_val * 1.5) if default_val > 0 else 10
                        else:
                            param_dict['max'] = default_val * 1.5 if default_val > 0 else 1.0
                            
                parameters.append(param_dict)

        # Detect Archetype based on keywords in the script
        archetype = "UNKNOWN"
        pine_lower = self.pine_text.lower()
        if "trama" in pine_lower and "ha_open" in pine_lower:
            archetype = "TRAMA_HA_MOMENTUM"
        
        return {
            "strategy_name": "Parsed Strategy",
            "archetype": archetype,
            "optimizable_parameters": parameters
        }

