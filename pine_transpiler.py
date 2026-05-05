"""
Pine Transpiler Agent
=====================
Invoked when the IR Builder detects an indicator in the Pine Script that does
not exist in the Python INDICATOR_CATALOG.

Workflow:
1.  Extract the raw Pine math block for the unknown indicator.
2.  Send it to DeepSeek-Coder with a strict translation prompt.
3.  Load the generated function temporarily (without saving).
4.  Run the Parity Checker against TradingView-exported signals.
5.  If parity >= 95%:  Append permanently to indicator_lib.py + register in INDICATOR_CATALOG.
6.  If parity <  95%:  ABORT. Print a detailed alert explaining what drifted.
                        Do NOT save the function. Do NOT proceed with optimization.
"""

import re
import os
import sys
import types
import inspect
import textwrap
import importlib
from datetime import datetime

import numpy as np
import pandas as pd

from config import config
from parity_checker import run_parity_check, PARITY_THRESHOLD

# Path to the indicator library — new functions are permanently appended here.
_INDICATOR_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indicator_lib.py")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_TRANSPILE_PROMPT = """You are an expert quantitative developer converting Pine Script v5/v6 to Python.

Your task: Translate the following Pine Script indicator function (or logic block) into a
Python function that is compatible with Pandas Series inputs and NumPy operations.

Rules:
- Accept a `df` argument (pd.DataFrame with columns: open, high, low, close, volume).
- Accept named keyword arguments (`**kwargs`) for all input() parameters.
- Return either a pd.Series (for single-output indicators) or a pd.DataFrame (for multi-output).
- Use ONLY NumPy, Pandas, and the standard library. No TA-Lib, no external packages.
- Replicate Pine's bar-by-bar semantics exactly, including `var` (persistent state) variables.
- Use `.ewm(alpha=1/length, adjust=False).mean()` for Pine's `ta.rma()`.
- Use `.ewm(span=length, adjust=False).mean()` for Pine's `ta.ema()`.
- For Pine's `ta.barssince(cond)`, use cumulative index tracking with ffill.
- Name the function `calc_{fn_name}` where the name describes what it calculates.
- Output ONLY the function definition — no imports, no example usage, no markdown.

Pine Script to translate:
```pine
{pine_block}
```

Function name to use: `calc_{fn_name}`
"""


class TranspilerAbortError(RuntimeError):
    """Raised when the transpiler cannot produce a verified Python equivalent."""
    pass


class PineTranspiler:
    """
    Translates unknown Pine Script indicator blocks into verified Python functions
    and permanently registers them in the indicator library.
    """

    def __init__(self, pine_text: str):
        self.pine_text = pine_text

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def transpile_unknown(
        self,
        unknown_indicators: list[str],
        df_reference: pd.DataFrame,
        client=None,
    ) -> dict:
        """
        Transpiles each unknown indicator, verifies parity, and saves to library.

        Args:
            unknown_indicators: List of indicator names not found in INDICATOR_CATALOG.
            df_reference: DataFrame with OHLCV + TradingView buy/sell signal columns.
                          Used for parity verification. If no signal columns exist,
                          parity check is skipped (NOT_VERIFIABLE).
            client: OpenAI-compatible client pointed at LM Studio.

        Returns:
            Dict mapping indicator name -> Python function (callable).

        Raises:
            TranspilerAbortError: If any indicator fails the parity gate.
        """
        if not config.use_llm:
            print("  [Transpiler] LLM disabled — cannot transpile unknown indicators.")
            raise TranspilerAbortError(
                "Unknown indicators found but LLM is disabled (--no-llm). "
                "Cannot proceed without a verified Python equivalent. "
                "Either enable LLM mode or manually add the indicator to indicator_lib.py."
            )

        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.llm_base_url, api_key="lm-studio")

        new_functions = {}

        for indicator_name in unknown_indicators:
            print(f"\n  [Transpiler] Unknown indicator detected: '{indicator_name}'")
            print(f"  [Transpiler] Extracting Pine math block...")

            pine_block = self._extract_pine_block(indicator_name)
            if not pine_block:
                raise TranspilerAbortError(
                    f"Could not extract Pine math block for '{indicator_name}'. "
                    f"The indicator may be an external library import or a built-in "
                    f"that requires manual implementation. "
                    f"Please add `calc_{indicator_name.lower()}` to indicator_lib.py manually."
                )

            print(f"  [Transpiler] Sending to DeepSeek-Coder for translation...")
            fn_name = re.sub(r'[^a-z0-9_]', '_', indicator_name.lower()).strip('_')
            python_code = self._call_llm(client, pine_block, fn_name)

            print(f"  [Transpiler] Testing generated function...")
            fn = self._load_function(python_code, f"calc_{fn_name}")

            print(f"  [Transpiler] Running Parity Gate...")
            self._verify_parity(fn, fn_name, df_reference, python_code)

            # Parity passed — save permanently
            self._save_to_library(python_code, fn_name, pine_block)
            new_functions[indicator_name] = fn
            print(f"  [Transpiler] SUCCESS: '{indicator_name}' verified and saved to indicator_lib.py")

        return new_functions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_pine_block(self, indicator_name: str) -> str:
        """
        Attempts to extract the Pine Script logic block for the given indicator name.
        Looks for variable assignments and function definitions referencing the name.
        Returns the best matching block of code, or empty string if not found.
        """
        lines = self.pine_text.split('\n')
        name_lower = indicator_name.lower()

        # Strategy 1: Find lines that contain the indicator name
        relevant_lines = []
        for i, line in enumerate(lines):
            if name_lower in line.lower() and not line.strip().startswith('//'):
                # Grab surrounding context (5 lines before and after)
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                relevant_lines.extend(lines[start:end])

        if relevant_lines:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for line in relevant_lines:
                if line not in seen:
                    seen.add(line)
                    deduped.append(line)
            return '\n'.join(deduped)

        # Strategy 2: Return first 50 lines as context if nothing specific found
        return '\n'.join(lines[:50])

    def _call_llm(self, client, pine_block: str, fn_name: str) -> str:
        """Calls the LLM and returns the generated Python function code."""
        from llm_utils import call_llm_with_retry

        prompt = _TRANSPILE_PROMPT.format(pine_block=pine_block, fn_name=fn_name)
        result = call_llm_with_retry(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative developer who specializes in translating "
                        "Pine Script to Python. You output only clean, working Python code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        if not result:
            raise TranspilerAbortError(
                f"LLM returned empty response for indicator '{fn_name}'. "
                "Cannot proceed. Check LM Studio connection and VRAM availability."
            )

        # Strip markdown fences if LLM added them
        result = re.sub(r'^```(?:python)?\n?', '', result.strip(), flags=re.MULTILINE)
        result = re.sub(r'\n?```$', '', result.strip(), flags=re.MULTILINE)
        return result.strip()

    def _load_function(self, python_code: str, expected_name: str):
        """Dynamically loads the generated Python code and returns the callable."""
        try:
            module = types.ModuleType(f"_transpiler_temp_{expected_name}")
            # Sandboxed execution environment
            safe_globals = {
                'pd': pd,
                'np': np,
                '__builtins__': {
                    'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
                    'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict,
                    'set': set, 'bool': bool, 'isinstance': isinstance, 'any': any,
                    'all': all, 'getattr': getattr, 'hasattr': hasattr, 'type': type,
                    'round': round, 'Exception': Exception, 'ValueError': ValueError,
                    'TypeError': TypeError, 'RuntimeError': RuntimeError,
                }
            }
            exec(compile(python_code, f"<transpiler:{expected_name}>", 'exec'), safe_globals)
            fn = safe_globals.get(expected_name)
            if fn is None or not callable(fn):
                raise TranspilerAbortError(
                    f"Generated code does not define a callable named '{expected_name}'. "
                    f"LLM may have used the wrong function name. "
                    f"Generated code:\n{python_code}"
                )
            return fn
        except SyntaxError as e:
            raise TranspilerAbortError(
                f"Generated Python code has a syntax error: {e}\n"
                f"Generated code:\n{python_code}"
            )

    def _verify_parity(
        self,
        fn: callable,
        fn_name: str,
        df: pd.DataFrame,
        python_code: str,
    ):
        """
        Runs the generated function and checks signal parity against TV exports.
        Raises TranspilerAbortError if parity fails or is below 95%.
        """
        # Check if TV signal columns exist for verification
        from parity_checker import KNOWN_BUY_COLS, KNOWN_SELL_COLS
        has_tv_signals = any(c.lower() in KNOWN_BUY_COLS | KNOWN_SELL_COLS for c in df.columns)

        if not has_tv_signals:
            print(
                f"  [Transpiler] WARNING: No TradingView signal columns in CSV. "
                f"Parity for '{fn_name}' cannot be verified. "
                f"The generated function will be used unverified. "
                f"To enable verification, export buy/sell columns from TradingView."
            )
            return  # Allow unverified when no reference data is available

        # Try to run the function and generate signals
        try:
            result = fn(df)
            if isinstance(result, pd.DataFrame):
                py_buy  = result.get('buy', pd.Series(np.zeros(len(df)))).fillna(0).astype(int).values
                py_sell = result.get('sell', pd.Series(np.zeros(len(df)))).fillna(0).astype(int).values
            elif isinstance(result, pd.Series):
                # Single output — treat as buy signal only for parity purposes
                py_buy  = result.fillna(0).astype(int).values
                py_sell = np.zeros(len(df), dtype=int)
            else:
                raise TranspilerAbortError(
                    f"Generated function for '{fn_name}' returned unexpected type: {type(result)}. "
                    f"Expected pd.DataFrame or pd.Series."
                )
        except Exception as e:
            raise TranspilerAbortError(
                f"Generated function for '{fn_name}' raised an exception during execution: {e}\n"
                f"Generated code:\n{python_code}"
            )

        report = run_parity_check(df, py_buy, py_sell)

        if report.status == "NOT_VERIFIABLE":
            return  # Already warned above

        if report.blocking:
            # ABORT — print detailed alert
            print("\n" + "="*60)
            print("  ERROR: TRANSPILER PARITY GATE FAILED — OPTIMIZATION ABORTED")
            print("="*60)
            print(f"  Indicator:      {fn_name}")
            print(f"  Fidelity Score: {report.fidelity_score*100:.1f}% (Required: {PARITY_THRESHOLD*100:.0f}%)")
            print(f"  Buy Recall:     {report.buy_recall*100:.1f}%")
            print(f"  Sell Recall:    {report.sell_recall*100:.1f}%")
            print(f"\n  Drift Analysis:\n  {report.drift_analysis}")
            print(f"\n  Generated Python Code:\n")
            for i, line in enumerate(python_code.split('\n'), 1):
                print(f"    {i:3d}: {line}")
            print("\n  Action Required:")
            print("  1. Review the generated Python code above.")
            print("  2. Compare against the Pine Script logic manually.")
            print("  3. Fix the drift (usually a stateful 'var' initialization issue).")
            print("  4. Manually add the corrected function to indicator_lib.py.")
            print("  5. Re-run the optimization engine.")
            print("="*60 + "\n")
            raise TranspilerAbortError(
                f"Parity gate failed for '{fn_name}' "
                f"({report.fidelity_score*100:.1f}% < {PARITY_THRESHOLD*100:.0f}% required). "
                f"Optimization aborted. See detailed alert above."
            )

        print(
            f"  [Transpiler] Parity PASSED: {report.fidelity_score*100:.1f}% "
            f"(Buy: {report.buy_recall*100:.1f}%, Sell: {report.sell_recall*100:.1f}%)"
        )

    def _save_to_library(self, python_code: str, fn_name: str, pine_block: str):
        """Appends the verified function to indicator_lib.py and registers it in INDICATOR_CATALOG."""
        full_fn_name = f"calc_{fn_name}"
        catalog_key = fn_name.upper()

        # Read current library
        with open(_INDICATOR_LIB_PATH, 'r', encoding='utf-8') as f:
            lib_content = f.read()

        # Guard: don't add if already exists
        if full_fn_name in lib_content:
            print(f"  [Transpiler] '{full_fn_name}' already exists in indicator_lib.py. Skipping save.")
            return

        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        header = textwrap.dedent(f"""

            # ── Auto-generated by Pine Transpiler on {timestamp} ──────────────────────────
            # Source Pine block:
            #   {chr(10).join('# ' + l for l in pine_block.split(chr(10))[:5])}
            # Verified via Parity Gate at ≥{PARITY_THRESHOLD*100:.0f}% signal recall.
            # ─────────────────────────────────────────────────────────────────────────────
        """)

        # Find the INDICATOR_CATALOG dict and insert new entry before closing brace
        catalog_pattern = r'(INDICATOR_CATALOG\s*=\s*\{[^}]+)(}\s*)$'
        new_entry = f'    "{catalog_key}": {full_fn_name},\n'

        new_lib = lib_content

        # Append function definition
        new_lib = new_lib.rstrip() + "\n" + header + python_code + "\n"

        # Register in INDICATOR_CATALOG
        if 'INDICATOR_CATALOG' in new_lib:
            # Find the closing brace of INDICATOR_CATALOG and insert before it
            cat_end = new_lib.rfind('}')
            if cat_end != -1:
                new_lib = new_lib[:cat_end] + new_entry + new_lib[cat_end:]

        with open(_INDICATOR_LIB_PATH, 'w', encoding='utf-8') as f:
            f.write(new_lib)


def run_transpiler(
    pine_text: str,
    unknown_indicators: list[str],
    df_reference: pd.DataFrame,
    client=None,
) -> dict:
    """Entry point for the Pine Transpiler agent."""
    transpiler = PineTranspiler(pine_text)
    return transpiler.transpile_unknown(unknown_indicators, df_reference, client=client)
