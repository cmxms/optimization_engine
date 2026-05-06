import argparse
import os
import sys
import glob

from rag import seed_pine_bug_knowledge
from pine_critic import run_critic
from pine_transpiler import run_transpiler, TranspilerAbortError
from quant_engine import run_quant
from strategist import run_strategist
from artifact_writer import write_artifacts
from failure_analyst import run_failure_analyst
from config import config
from vram_manager import vram
from market_context import get_market_snapshot
from quant_developer import run_developer
from validation_ingestor import run_ingestor
from catfish import run_catfish
from quant_operator import run_operator
from lane_manager import orchestrator
import shutil
import re


def main():
    """
    Main entry point for the Optimization Engine pipeline.
    Orchestrates the multi-agent workflow via the Urithiru LaneManager.
    """
    parser = argparse.ArgumentParser(description="Optimization Engine - Multi-Agent Quant Lab")
    parser.add_argument("--project-dir", type=str, required=False, help="Path to the Project directory (Optional if using Inbox)")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna optimization trials")
    parser.add_argument("--no-llm", action="store_true", help="Force disable LLM")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests before starting")
    
    args = parser.parse_args()

    if args.test:
        print("\n[Test Mode] Running internal unit tests...")
        import subprocess
        result = subprocess.run(["python", "-m", "pytest", "tests/test_math.py", "-v"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("FAILED: Tests failed. Aborting optimization for safety.")
            sys.exit(1)
        print("PASSED: All tests passed. Proceeding with pipeline...\n")

    if args.no_llm:
        config.use_llm = False

    print("==================================================")
    print("Optimization Engine - Initializing (Urithiru Mode)")
    print("==================================================")

    # Step 0: Ingest TV Validations
    run_ingestor()

    # Step 1: Seed RAG
    print("\n[Step 1/8] Seeding RAG knowledge base...")
    seed_pine_bug_knowledge()

    # Inbox Routing or Project Dir
    project_dir = args.project_dir
    if not project_dir:
        inbox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inbox")
        os.makedirs(inbox_dir, exist_ok=True)
        pine_files = glob.glob(os.path.join(inbox_dir, "*.pine")) + glob.glob(os.path.join(inbox_dir, "*.txt"))
        if not pine_files:
            print("Error: No --project-dir provided and no Pine script found in data/inbox/")
            sys.exit(1)
            
        pine_file = pine_files[0]
        strategy_name = "New_Strategy"
        with open(pine_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'strategy(' in line:
                    match = re.search(r'strategy\((?:title\s*=\s*)?["\']([^"\']+)["\']', line)
                    if match:
                        strategy_name = match.group(1).replace(" ", "_").replace("/", "_").replace("\\", "_")
                        break
                        
        project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "projects", strategy_name)
        os.makedirs(project_dir, exist_ok=True)
        
        print(f"\n  -> [Inbox Route] Moving files to {project_dir}")
        dest_pine = os.path.join(project_dir, "strategy.pine")
        if os.path.exists(dest_pine):
            os.remove(dest_pine)
        shutil.move(pine_file, dest_pine)
        
        for csv_f in glob.glob(os.path.join(inbox_dir, "*.csv")):
            dest_csv = os.path.join(project_dir, os.path.basename(csv_f))
            if os.path.exists(dest_csv):
                os.remove(dest_csv)
            shutil.move(csv_f, dest_csv)

    pine_path = os.path.join(project_dir, "strategy.pine")
    
    if not os.path.exists(pine_path):
        candidates = glob.glob(os.path.join(project_dir, "*.pine")) + glob.glob(os.path.join(project_dir, "*.txt"))
        candidates = [c for c in candidates if "Strategy_Dossier" not in c and "output" not in c]
        if candidates:
            pine_path = candidates[0]
            print(f"  -> Auto-detected Pine script: {os.path.basename(pine_path)}")
        else:
            print(f"Error: No Pine script found in {project_dir}")
            sys.exit(1)

    with open(pine_path, 'r', encoding='utf-8') as f:
        pine_text = f.read()

    config.output_dir = os.path.join(project_dir, "output")
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 2: Context Fetching
    print("\n[Step 2/8] Fetching Market Context...")
    snapshot = get_market_snapshot()

    # Step 3: Developer Agent (Navani)
    print("\n[Step 3/9] Agent 1: Navani (Developer) extracting strategy recipe...")
    recipe = run_developer(pine_text, snapshot=snapshot)
    unknown_indicators = recipe.get("unknown_indicators", [])
    stateful_vars = recipe.get("stateful_vars", [])
    if unknown_indicators:
        print(f"  -> WARNING: {len(unknown_indicators)} unknown indicator(s) detected: {unknown_indicators}")
    if stateful_vars:
        print(f"  -> Stateful vars detected: {stateful_vars}")

    # Step 4: Logic Critic (Jasnah)
    print("\n[Step 4/9] Agent 2: Jasnah (Critic) auditing Pine Script...")
    try:
        critic_report = orchestrator.run_lane("critic", run_critic, pine_text)
        print(f"  -> Found {len(critic_report.issues)} issues. Repaint Risk: {critic_report.repaint_risk_score}/10")
    except Exception as e:
        print(f"  [OptiEngine - Jasnah] WARNING: LLM audit failed: {e}")
        print("  -> Proceeding with static analysis only.")
        from pine_critic import run_critic as _run_critic_internal
        critic_report = _run_critic_internal(pine_text, client=None)

    # Step 4.5: Auto-Fixer
    from auto_fixer import apply_auto_fixes
    pine_text = apply_auto_fixes(pine_text, critic_report)

    # Step 4b: Transpiler (Renarin)
    if unknown_indicators:
        print(f"\n[Step 4b/9] Agent 2b: Renarin (Transpiler) handling unknown indicator(s)...")
        import pandas as pd, glob as _glob
        csv_files = _glob.glob(os.path.join(project_dir, "*.csv"))
        df_ref = None
        for csv_f in csv_files:
            if 'output' in csv_f: continue
            try:
                df_ref = pd.read_csv(csv_f)
                df_ref.columns = [c.lower() for c in df_ref.columns]
                break
            except Exception: continue

        if df_ref is None:
            print("  -> ERROR: No CSV data found for parity verification.")
            sys.exit(1)

        try:
            # Transpiler uses Renarin lane (Wit persona)
            orchestrator.run_lane("transpiler", run_transpiler, pine_text, unknown_indicators, df_ref)
            print(f"  -> All unknown indicators transpiled. Reloading indicator catalog...")
            import importlib, indicator_lib
            importlib.reload(indicator_lib)
        except TranspilerAbortError as e:
            print(f"\n  TRANSPILER ABORTED: {e}")
            sys.exit(1)

    # Step 5: Quant Engine (Pattern)
    print(f"\n[Step 5/9] Agent 3: Pattern (Quant) running optimization...")
    quant_report = run_quant(project_dir, recipe, n_trials=args.trials)
    
    # Step 6: Catfish (Shallan)
    print("\n[Step 6/9] Agent 4: Shallan (Catfish) finding flaws...")
    dissent = orchestrator.run_lane("catfish", run_catfish, pine_text, critic_report, quant_report)
    print(f"  -> Catfish Dissent Generated.")

    # Step 7: Operator (Adolin)
    print("\n[Step 7/9] Agent 5: Adolin (Operator) production audit...")
    audit = orchestrator.run_lane("operator", run_operator, pine_text)
    print(f"  -> {audit}")

    # Step 8: Strategist Agent (Dalinar)
    print("\n[Step 8/9] Agent 6: Dalinar (Strategist) synthesis...")
    verdict = orchestrator.run_lane("strategist", run_strategist, critic_report, quant_report, snapshot=snapshot, dissent=dissent)
    print(f"  -> Final Verdict: {verdict.verdict} (Confidence: {verdict.confidence_pct}%)")

    # Step 9: Artifacts & Self-Healing
    print("\n[Step 9/9] Generating Artifacts...")
    write_artifacts(pine_text, critic_report, quant_report, verdict, dissent=dissent)
    
    if config.use_llm:
        print("\n[Failure Analyst] Running Dalinar post-mortem...")
        orchestrator.run_lane("strategist", run_failure_analyst, pine_text, quant_report, critic_report)

    # Cleanup
    orchestrator.unload_all()
    print("\nOptimization Engine Pipeline Complete.")


if __name__ == "__main__":
    main()
