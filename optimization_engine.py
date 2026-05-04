import argparse
import os
import sys
import glob

from rag import seed_pine_bug_knowledge
from pine_critic import run_critic
from quant_engine import run_quant
from strategist import run_strategist
from artifact_writer import write_artifacts
from failure_analyst import run_failure_analyst
from config import config
from vram_manager import vram
from market_context import get_market_snapshot
from quant_developer import run_developer

def main():
    """
    Main entry point for the Optimization Engine pipeline.
    Orchestrates the multi-agent workflow:
    1. Context Fetching
    2. Developer (Parsing/Profiling)
    3. Logic Critic (Audit)
    4. Quant Engine (Backtest/Optuna)
    5. Strategist (Synthesis)
    6. Artifact Generation & Self-Healing
    """
    parser = argparse.ArgumentParser(description="Optimization Engine - Multi-Agent Quant Lab")
    parser.add_argument("--project-dir", type=str, required=True, help="Path to the Project directory")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna optimization trials")
    parser.add_argument("--no-llm", action="store_true", help="Force disable LLM")
    parser.add_argument("--test", action="store_true", help="Run internal unit tests before starting")
    
    args = parser.parse_args()

    if args.test:
        print("\n[Test Mode] Running internal unit tests...")
        import subprocess
        result = subprocess.run(["python", "-m", "pytest", "tests/test_math.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("FAILED: Tests failed. Aborting optimization for safety.")
            sys.exit(1)
        print("PASSED: All tests passed. Proceeding with pipeline...\n")

    if args.no_llm:
        config.use_llm = False

    print("==================================================")
    print("Optimization Engine - Initializing")
    print("==================================================")

    # Step 1: Seed RAG
    print("\n[Step 1/7] Seeding RAG knowledge base...")
    seed_pine_bug_knowledge()

    # Find Pine Script
    project_dir = args.project_dir
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
    print("\n[Step 2/7] Fetching Market Context...")
    snapshot = get_market_snapshot()

    # Step 3: Developer Agent
    print("\n[Step 3/7] Agent 1: Developer extracting strategy recipe...")
    recipe = run_developer(pine_text, snapshot=snapshot)

    # Step 4: Logic Critic
    print("\n[Step 4/7] Agent 2: Logic Critic auditing Pine Script...")
    critic_client = vram.load_agent("critic")
    critic_report = run_critic(pine_text, client=critic_client)
    print(f"  -> Found {len(critic_report.issues)} issues. Repaint Risk: {critic_report.repaint_risk_score}/10")

    # Step 5: Quant Engine
    print(f"\n[Step 5/7] Agent 3: Quant Engine running backtest...")
    quant_report = run_quant(args.project_dir, recipe, n_trials=args.trials)
    
    # Step 6: Strategist Agent
    print("\n[Step 6/7] Agent 4: Strategist synthesizing results...")
    verdict = run_strategist(critic_report, quant_report, snapshot=snapshot, client=vram.load_agent("strategist"))
    print(f"  -> Final Verdict: {verdict.verdict} (Confidence: {verdict.confidence_pct}%)")

    # Step 7: Artifacts & Self-Healing
    print("\n[Step 7/7] Generating Artifacts & Analysis...")
    write_artifacts(pine_text, critic_report, quant_report, verdict)
    
    if config.use_llm:
        run_failure_analyst(pine_text, quant_report, critic_report, client=vram.load_agent("strategist"))

    # Cleanup
    vram.unload_all()
    print("\nOptimization Engine Pipeline Complete.")

if __name__ == "__main__":
    main()
