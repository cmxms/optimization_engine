import re
import json

# All indicator function names currently available in the Python library.
# When the IR Builder finds a Pine indicator NOT in this set, it flags it
# for the Pine Transpiler agent.
from indicator_lib import INDICATOR_CATALOG
import re

_KNOWN_INDICATOR_NAMES = {k.lower() for k in INDICATOR_CATALOG.keys()} | {
    # Pine built-ins that map trivially and don't need explicit catalog entries
    'ta.ema', 'ta.sma', 'ta.rma', 'ta.wma', 'ta.rsi', 'ta.atr', 'ta.macd',
    'ta.stoch', 'ta.vwap', 'ta.highest', 'ta.lowest', 'ta.barssince',
    'ta.crossover', 'ta.crossunder', 'ta.change', 'ta.valuewhen',
    'math.max', 'math.min', 'math.abs', 'math.pow', 'math.sum',
    'ta.bbands', 'ta.pivothigh', 'ta.pivotlow', 'ta.median',
    'request.security', 'ticker.new', 'str.tostring', 'str.format',
    'timeframe.isintraday', 'timeframe.isdaily', 'syminfo.mintick',
    'syminfo.prefix', 'syminfo.ticker', 'barstate.isconfirmed',
    'barstate.islast', 'time', 'ticker', 'request', 'ta', 'math', 'strategy',
    'input', 'plot', 'plotshape', 'plotchar', 'plotcandle', 'plotbar',
    'bgcolor', 'alert', 'alertcondition', 'label', 'line', 'box', 'table',
    'array', 'matrix', 'map', 'runtime', 'syminfo', 'timeframe', 'barstate',
    'color.new', 'color.rgb', 'color.from_gradient', 'color.aqua', 'color.lime',
    'color.red', 'color.orange', 'color.purple', 'color.fuchsia', 'color.white',
    'color.black', 'color.gray', 'color.silver', 'color.navy', 'color.blue',
    'color.teal', 'color.green', 'color.yellow', 'color.maroon'
}

# Pine input types that are purely display — never passed to Optuna
_DISPLAY_TYPES = {'input.color', 'input.session', 'input.source'}
_DISPLAY_NAME_PREFIXES = ('show_', 'hide_', 'plot_', 'color_', 'label_', 'real_')
_DISPLAY_NAME_SUFFIXES = ('_color', '_table', '_line', '_dot', '_bg', '_debug', '_dots', '_lines')

# Pine var-type keywords that indicate stateful (persistent) variables
_VAR_PATTERN = re.compile(r'^\s*var\s+(?:int|float|bool|line|label|string)\s+(\w+)', re.MULTILINE)

# Pine input() extraction pattern
_INPUT_PATTERN = re.compile(
    r'^([a-zA-Z0-9_]+)\s*=\s*input\.(int|float|bool|string|color|session|source)\((.*)$',
    re.MULTILINE,
)

# Patterns for known indicator function calls in Pine
_INDICATOR_CALL_PATTERN = re.compile(
    r'\b(?:ta|math)\.[a-zA-Z_]+\s*\(|'    # built-ins: ta.xxx(), math.xxx()
    r'\b([A-Z][A-Z0-9_]{2,})\s*\(',        # ALL_CAPS custom indicators
)


class IRBuilder:
    def __init__(self, pine_text: str):
        self.pine_text = pine_text
        self.lines = pine_text.split('\n')

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------

    def _classify_param(self, name: str, input_type: str, line: str) -> str:
        """Tags each input() parameter with its role."""
        name_lower = name.lower()
        line_lower = line.lower()

        if input_type in ('color', 'session', 'source'):
            return 'display'
        if any(line_lower.startswith(t) for t in _DISPLAY_TYPES):
            return 'display'
        if name_lower.startswith(_DISPLAY_NAME_PREFIXES):
            return 'display'
        if any(name_lower.endswith(s) for s in _DISPLAY_NAME_SUFFIXES):
            return 'display'

        if any(r in name_lower for r in ('stop', 'target', 'trail', 'risk', 'sl_', 'tp_')):
            return 'risk'

        return 'signal'

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def extract_parameters(self) -> list:
        """Extracts all input() declarations with min/max/default bounds."""
        parameters = []
        for match in _INPUT_PATTERN.finditer(self.pine_text):
            var_name, var_type, args_str = match.groups()
            line = match.group(0)

            # Skip commented lines
            line_start = self.pine_text.rfind('\n', 0, match.start()) + 1
            if self.pine_text[line_start:match.start()].strip().startswith('//'):
                continue

            param = {
                'name': var_name,
                'type': var_type,
                'role': self._classify_param(var_name, var_type, line),
            }

            if var_type == 'bool':
                default_match = re.match(r'^\s*(true|false)', args_str, re.IGNORECASE)
                param['default'] = default_match.group(1).lower() == 'true' if default_match else True

            elif var_type == 'string':
                default_match = re.match(r'^\s*["\']([^"\']*)["\']', args_str)
                param['default'] = default_match.group(1) if default_match else ''
                param['role'] = 'display'  # strings are never optimizable

            elif var_type in ('int', 'float'):
                default_match = re.match(r'^\s*([-\d.]+)', args_str)
                if not default_match:
                    continue
                try:
                    default_val = int(default_match.group(1)) if var_type == 'int' else float(default_match.group(1))
                except ValueError:
                    continue

                param['default'] = default_val

                minval_match = re.search(r'minval\s*=\s*([-\d.]+)', args_str)
                param['min'] = (
                    int(minval_match.group(1)) if var_type == 'int' else float(minval_match.group(1))
                ) if minval_match else max(1, int(default_val * 0.5)) if var_type == 'int' else default_val * 0.5

                maxval_match = re.search(r'maxval\s*=\s*([-\d.]+)', args_str)
                param['max'] = (
                    int(maxval_match.group(1)) if var_type == 'int' else float(maxval_match.group(1))
                ) if maxval_match else int(default_val * 1.5) if var_type == 'int' else default_val * 1.5

            elif var_type in ('color', 'session', 'source'):
                param['default'] = None
                param['role'] = 'display'

            parameters.append(param)

        return parameters

    # ------------------------------------------------------------------
    # Filter detection
    # ------------------------------------------------------------------

    def detect_filters(self) -> list:
        """Scans Pine Script for known filter blocks."""
        filters = []
        pine_lower = self.pine_text.lower()

        if 'trade_eth' in pine_lower or 'am_session' in pine_lower:
            filters.append({
                'type': 'session_window',
                'start_hour': 9.0,
                'end_hour': 16.0,
                'controlled_by': 'trade_eth',
            })

        if 'tdv_vol_ma_len' in pine_lower:
            filters.append({
                'type': 'volume_gate',
                'style': 'tdv',
                'params': ['tdv_vol_ma_len', 'tdv_smoothBars', 'tdv_min_body_pct'],
            })

        if 'require_single_wick' in pine_lower:
            filters.append({
                'type': 'wick_quality',
                'params': ['require_single_wick'],
            })

        if 'use_sweep_filter' in pine_lower:
            filters.append({
                'type': 'session_sweep',
                'params': ['use_sweep_filter', 'sweep_lookback'],
            })

        return filters

    # ------------------------------------------------------------------
    # Signal logic detection
    # ------------------------------------------------------------------

    def detect_signal_logic(self) -> dict:
        """Maps known Pine signal patterns to Python INDICATOR_CATALOG entries."""
        signal_logic = {
            'tier': 2,
            'indicators': {},
            'entry_logic': {'long': [], 'short': []},
        }
        pine_lower = self.pine_text.lower()

        if 'rsilookback' in pine_lower or 'shortlength' in pine_lower:
            signal_logic['indicators']['rsi_exh'] = {
                'fn': 'RSI_EXHAUSTION_SIGNALS',
                'params': {
                    'rsiLen': 'rsiLen', 'rsiSmaLength': 'rsiSmaLength',
                    'rsiObLevel': 'rsiObLevel', 'rsiOsLevel': 'rsiOsLevel',
                    'rsiLookback': 'rsiLookback', 'threshold': 'threshold',
                    'shortLength': 'shortLength', 'longLength': 'longLength',
                    'cooldown': 'signalCooldown', 'smoothType': 'smoothType',
                    'formula': 'formula', 'shortSmoothingLength': 'shortSmoothingLength',
                    'longSmoothingLength': 'longSmoothingLength', 'average_ma_len': 'average_ma_len',
                },
            }
            signal_logic['entry_logic']['long'].append({'type': 'boolean_series', 'source': 'rsi_exh', 'column': 'buy'})
            signal_logic['entry_logic']['short'].append({'type': 'boolean_series', 'source': 'rsi_exh', 'column': 'sell'})

        elif 'trama_fast_len' in pine_lower or 'trama_med_len' in pine_lower:
            signal_logic['indicators']['trama_ha'] = {
                'fn': 'TRAMA_HA_SIGNALS',
                'params': {
                    'trama_fast_len': 'trama_fast_len', 'trama_med_len': 'trama_med_len',
                    'trama_slow_len': 'trama_slow_len', 'use_regime_filter': 'use_regime_filter',
                    'cross_lookback': 'cross_lookback', 'prior_regime_bars': 'prior_regime_bars',
                    'prior_regime_window': 'prior_regime_window', 'ha_stack_min': 'ha_stack_min',
                    'require_rising_high': 'require_rising_high', 'require_falling_low': 'require_falling_low',
                },
            }
            signal_logic['entry_logic']['long'].append({'type': 'boolean_series', 'source': 'trama_ha', 'column': 'buy'})
            signal_logic['entry_logic']['short'].append({'type': 'boolean_series', 'source': 'trama_ha', 'column': 'sell'})

        return signal_logic

    # ------------------------------------------------------------------
    # Unknown indicator detection
    # ------------------------------------------------------------------

    def detect_unknown_indicators(self) -> list[str]:
        """
        Scans the Pine Script for indicator function calls that have no Python equivalent.
        Ignores strings and known Pine built-ins.
        """
        unknown = []
        
        # Strip strings and comments before searching to avoid false positives (like "ETH" in a label)
        clean_text = re.sub(r'//.*', '', self.pine_text) # Remove comments
        clean_text = re.sub(r'"[^"]*"', '""', clean_text) # Remove double-quote strings
        clean_text = re.sub(r"'[^']*'", "''", clean_text) # Remove single-quote strings

        # Look for function calls: name(
        for match in re.finditer(r'\b([a-zA-Z0-9_.]+)\s*\(', clean_text):
            raw_name = match.group(1)
            full_name = raw_name.lower()
            
            # If it's a dotted call (ta.sma), check if we know it
            if '.' in full_name:
                prefix = full_name.split('.')[0]
                if prefix in _KNOWN_INDICATOR_NAMES or full_name in _KNOWN_INDICATOR_NAMES:
                    continue
                unknown.append(raw_name)
                continue
            
            # If it's a standalone call, only flag if it's ALL_CAPS (custom indicator convention)
            # or if it's a known ta/math but somehow not dotted (unlikely in Pine v5+)
            if raw_name.isupper() and len(raw_name) >= 3:
                if full_name not in _KNOWN_INDICATOR_NAMES:
                    unknown.append(raw_name)

        if unknown:
            print(f"  [Developer] DEBUG: Detected potential unknown indicators: {list(set(unknown))}")
        
        return list(set(unknown))

    # ------------------------------------------------------------------
    # Stateful var extraction
    # ------------------------------------------------------------------

    def detect_stateful_vars(self) -> list[str]:
        """
        Extracts all Pine 'var' variable names from the script.
        These require special handling in Python (recursive initialization).
        """
        return _VAR_PATTERN.findall(self.pine_text)

    # ------------------------------------------------------------------
    # Build IR
    # ------------------------------------------------------------------

    def build_ir(self) -> dict:
        """Constructs the full Intermediate Representation (IR) recipe."""
        params = self.extract_parameters()
        filters = self.detect_filters()
        signal_logic = self.detect_signal_logic()
        unknown = self.detect_unknown_indicators()
        stateful = self.detect_stateful_vars()

        return {
            'strategy_name': self._extract_strategy_name(),
            'version': 2,
            'optimizable_parameters': params,
            'filters': filters,
            'signal_logic': signal_logic,
            # New fields for the overhaul:
            'unknown_indicators': unknown,
            'stateful_vars': stateful,
            'parity_required': True,  # Always true — optimization never runs unverified
        }

    def _extract_strategy_name(self) -> str:
        """Extracts the strategy title from the Pine Script header."""
        match = re.search(r'strategy\s*\(\s*["\']([^"\']+)["\']', self.pine_text)
        return match.group(1) if match else 'Parsed Strategy'


def build_strategy_ir(pine_text: str) -> dict:
    """Entry point for IR building."""
    builder = IRBuilder(pine_text)
    return builder.build_ir()
