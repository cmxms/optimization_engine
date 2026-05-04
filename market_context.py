"""
Market Context Agent (Optimization Engine v2)
Uses MCP tools via direct import to enrich the backtesting environment with
current macro regime and market breadth data. This prevents strategies
from being evaluated in a vacuum.
"""

import asyncio
import sys
import os

# Add the parent directory to sys.path so we can import from mcp_tools
# Assuming optimization_engine is at AI HUB/optimization_engine and mcp-server is at AI HUB/mcp-server
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
mcp_server_path = os.path.join(parent_dir, "mcp-server", "src")
if mcp_server_path not in sys.path:
    sys.path.append(mcp_server_path)

try:
    from mcp_tools.momentum.macro_regime import detect_macro_regime
    from mcp_tools.momentum.breadth import analyze_breadth
except ImportError as e:
    print(f"[OptiEngine - Context] Error importing MCP tools: {e}")
    detect_macro_regime = None
    analyze_breadth = None

def get_market_snapshot() -> dict:
    """
    Fetches the current macro regime and market breadth synchronously.
    Returns a dictionary of the results.
    """
    if detect_macro_regime is None or analyze_breadth is None:
        return {
            "status": "error",
            "message": "MCP tools not available. Run without market context."
        }
        
    print("  -> Fetching macro regime and market breadth from MCP tools...")

    async def _fetch_all():
        regime_task = asyncio.create_task(detect_macro_regime())
        breadth_task = asyncio.create_task(analyze_breadth())
        return await asyncio.wait_for(asyncio.gather(regime_task, breadth_task), timeout=10.0)

    try:
        # Run the async tasks synchronously with a timeout
        results = asyncio.run(_fetch_all())
        regime_res, breadth_res = results

        snapshot = {
            "status": "success",
            "regime": regime_res.data.get("summary", "Unknown Regime") if regime_res.success else "Regime Fetch Failed",
            "breadth": breadth_res.data.get("summary", "Unknown Breadth") if breadth_res.success else "Breadth Fetch Failed"
        }
        
        # Add some specific data points if available
        if regime_res.success:
            snapshot["regime_label"] = regime_res.data.get("regime", "Unknown")
        if breadth_res.success:
            snapshot["breadth_score"] = breadth_res.data.get("score", 0)
            snapshot["breadth_verdict"] = breadth_res.data.get("verdict", "Unknown")

        print(f"  -> {snapshot['regime']}")
        print(f"  -> {snapshot['breadth']}")
        
        return snapshot
        
    except asyncio.TimeoutError:
        print("[OptiEngine - Context] Market context fetch timed out after 10s. Proceeding without context.")
        return {
            "status": "timeout",
            "message": "Market context fetch timed out."
        }
    except Exception as e:
        print(f"[OptiEngine - Context] Error during snapshot fetch: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Test execution
    res = get_market_snapshot()
    import json
    print(json.dumps(res, indent=2))
