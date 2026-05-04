import re
import json

class IRBuilder:
    def __init__(self, pine_text: str):
        self.pine_text = pine_text
        self.lines = pine_text.split('\n')

    def _classify_display_params(self, name: str, line: str) -> str:
        """
        Regex pre-filter to identify parameter roles.
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

        # Pure risk/exit mechanics
        if any(r in name_lower for r in ("stop", "target", "trail", "risk", "sl_ticks", "tp_ticks")):
            return "risk"
            
        return "signal"  # Default for all other numerical/bool inputs

    def extract_parameters(self) -> list:
        """Extracts parameters with safe default bounds."""
        parameters = []
        for line in self.lines:
            line = line.strip()
            if line.startswith('//'):
                continue
            
            match = re.match(r'^([a-zA-Z0-9_]+)\s*=\s*input\.(int|float|bool|string)\((.*)\)', line)
            if match:
                var_name, var_type, args_str = match.groups()
                
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
                elif var_type == 'string':
                    param_dict['default'] = default_val_str.strip("'\"")
                    param_dict['role'] = 'display' # Ensure strings aren't passed to optuna by default
                elif var_type in ('int', 'float'):
                    try:
                        default_val = int(default_val_str) if var_type == 'int' else float(default_val_str)
                    except ValueError:
                        continue
                            
                    param_dict['default'] = default_val
                    
                    minval_match = re.search(r'minval\s*=\s*([-\d.]+)', args_str)
                    if minval_match:
                        param_dict['min'] = int(minval_match.group(1)) if var_type == 'int' else float(minval_match.group(1))
                    else:
                        param_dict['min'] = max(1, int(default_val * 0.5)) if var_type == 'int' else default_val * 0.5
                            
                    maxval_match = re.search(r'maxval\s*=\s*([-\d.]+)', args_str)
                    if maxval_match:
                        param_dict['max'] = int(maxval_match.group(1)) if var_type == 'int' else float(maxval_match.group(1))
                    else:
                        param_dict['max'] = int(default_val * 1.5) if var_type == 'int' else default_val * 1.5
                            
                parameters.append(param_dict)
        return parameters

    def detect_filters(self) -> list:
        """
        Scans the Pine Script text to identify which filters are active.
        """
        filters = []
        pine_lower = self.pine_text.lower()
        
        if "trade_eth" in pine_lower or "session" in pine_lower:
            filters.append({
                "type": "session_window", 
                "start_hour": 9.0, 
                "end_hour": 16.0, 
                "controlled_by": "trade_eth"
            })
            
        if "tdv_vol_ma_len" in pine_lower:
            filters.append({
                "type": "volume_gate", 
                "style": "tdv", 
                "params": ["tdv_vol_ma_len", "tdv_smoothBars", "tdv_min_body_pct"]
            })
            
        if "require_single_wick" in pine_lower:
            filters.append({
                "type": "wick_quality", 
                "params": ["require_single_wick"]
            })
            
        if "use_sweep_filter" in pine_lower:
            filters.append({
                "type": "session_sweep",
                "params": ["use_sweep_filter", "sweep_lookback"]
            })

        return filters

    def build_ir(self) -> dict:
        signal_logic = {
            "tier": 2,
            "indicators": {},
            "entry_logic": {"long": [], "short": []}
        }
        
        pine_lower = self.pine_text.lower()
        if "rsilookback" in pine_lower or "shortlength" in pine_lower:
            signal_logic["indicators"]["rsi_exh"] = {
                "fn": "RSI_EXHAUSTION_SIGNALS",
                "params": {
                    "rsiLen": "rsiLen",
                    "rsiSmaLength": "rsiSmaLength",
                    "rsiObLevel": "rsiObLevel",
                    "rsiOsLevel": "rsiOsLevel",
                    "rsiLookback": "rsiLookback",
                    "threshold": "threshold",
                    "shortLength": "shortLength",
                    "longLength": "longLength",
                    "cooldown": "signalCooldown",
                    "smoothType": "smoothType",
                    "formula": "formula",
                    "shortSmoothingLength": "shortSmoothingLength",
                    "longSmoothingLength": "longSmoothingLength",
                    "average_ma_len": "average_ma_len"
                }
            }
            signal_logic["entry_logic"]["long"].append({"type": "boolean_series", "source": "rsi_exh", "column": "buy"})
            signal_logic["entry_logic"]["short"].append({"type": "boolean_series", "source": "rsi_exh", "column": "sell"})
            
        elif "trama_fast_len" in pine_lower:
            signal_logic["indicators"]["trama_ha"] = {
                "fn": "TRAMA_HA_SIGNALS",
                "params": {
                    "trama_fast_len": "trama_fast_len",
                    "trama_med_len": "trama_med_len",
                    "trama_slow_len": "trama_slow_len",
                    "use_regime_filter": "use_regime_filter",
                    "cross_lookback": "cross_lookback",
                    "prior_regime_bars": "prior_regime_bars",
                    "prior_regime_window": "prior_regime_window",
                    "ha_stack_min": "ha_stack_min",
                    "require_rising_high": "require_rising_high",
                    "require_falling_low": "require_falling_low"
                }
            }
            signal_logic["entry_logic"]["long"].append({"type": "boolean_series", "source": "trama_ha", "column": "buy"})
            signal_logic["entry_logic"]["short"].append({"type": "boolean_series", "source": "trama_ha", "column": "sell"})

        return {
            "strategy_name": "Parsed Strategy",
            "version": 1,
            "optimizable_parameters": self.extract_parameters(),
            "filters": self.detect_filters(),
            "signal_logic": signal_logic
        }

def build_strategy_ir(pine_text: str) -> dict:
    builder = IRBuilder(pine_text)
    return builder.build_ir()
