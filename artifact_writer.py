import os
import re
import json
from datetime import datetime
from config import config

def write_artifacts(pine_text, critic_report, quant_report, verdict):
    # Ensure output dir exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. Write Strategy_Dossier.md
    dossier_path = os.path.join(config.output_dir, "Strategy_Dossier.md")
    
    dossier_content = f"""# Strategy Dossier: {quant_report.recipe.get('strategy_name', 'Strategy Analysis')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## [OptiEngine Verdict: {verdict.verdict}]
**Confidence:** {verdict.confidence_pct}%
**Rationale:** {verdict.rationale}

---

## 1. Logic Critic Audit
**Repaint Risk Score:** {critic_report.repaint_risk_score}/10

### Identified Issues
"""
    for issue in critic_report.issues:
        dossier_content += f"- **[{issue.severity}]** Line {getattr(issue, 'line_number', 'N/A')}: {issue.description} (Fix: {issue.suggested_fix})\n"

    dossier_content += f"""
---

## 🧬 Translation Fidelity
"""
    pr = getattr(quant_report, "parity_report", None)
    if pr and pr.available:
        dossier_content += f"""Pine Signal Export: ✅ Detected
Fidelity Score: {pr.fidelity_score * 100:.1f}%  (Long: {pr.buy_recall * 100:.1f}% | Short: {pr.sell_recall * 100:.1f}%)
Status: {pr.status}
"""
    else:
        dossier_content += "Pine Signal Export: ❌ Not Detected (Cannot Verify Parity)\n"

    dossier_content += f"""
---

## 2. Quant Engine Performance (Optuna Optimized)
*Backtest executed on Out-of-Sample (OOS) data (last 30% of history).*

- **In-Sample Sharpe (IS):** {quant_report.in_sample_sharpe:.2f}
- **Out-of-Sample Sharpe (OOS):** {quant_report.oos_sharpe:.2f}
- **OOS Sortino:** {getattr(quant_report, 'sortino', 0.0):.2f}
- **Max Drawdown:** {getattr(quant_report, 'max_drawdown', 0.0) * 100:.2f}%
- **Trade Count (OOS):** {quant_report.trade_count}
- **Win Rate (OOS):** {quant_report.win_rate * 100:.1f}%

### Best Parameters Found:
"""
    for k, v in quant_report.best_params.items():
        dossier_content += f"- `{k}`: {v}\n"

    warning_str = "⚠️ YES (Unmodeled Slippage)" if getattr(quant_report, "execution_realism_warning", False) else "✅ None"
    
    dossier_content += f"""
---

## 🏛️ Historical Regime Performance"""
    for r, perf in quant_report.regime_performance.items():
        dossier_content += f"\n- **{r}**: {perf['trades']} trades | Win Rate: {perf['win_rate']:.1f}% | Avg PnL: {perf['avg_pnl']:.2f}%"

    dossier_content += f"""
---

## ⚙️ Execution Assumptions
| Parameter           | Value            |
|---------------------|------------------|
| Order Type          | Market           |
| Entry Slippage      | {getattr(quant_report, 'slippage_ticks_used', 0)} ticks |
| Commission (RT)     | ${getattr(quant_report, 'commission_per_rt', 0.0):.2f} |
| Realism Warning     | {warning_str} |
"""

    with open(dossier_path, 'w', encoding='utf-8') as f:
        f.write(dossier_content)
        
    # 2. Write Refined_Strategy.pine
    refined_path = os.path.join(config.output_dir, "Refined_Strategy.pine")
    refined_text = pine_text
    
    # Inject parameters via Regex
    if quant_report.data_found and verdict.verdict in ("GO", "CONDITIONAL"):
        for k, v in quant_report.best_params.items():
            if isinstance(v, bool):
                # Matches: variable = input.bool(true, ...
                v_str = "true" if v else "false"
                pattern = rf'({k}\s*=\s*input\.bool\()(true|false)'
                replacement = rf'\g<1>{v_str}'
            else:
                # Matches: variable = input.int(OLD_VAL, ...
                pattern = rf'({k}\s*=\s*input\.(?:int|float)\()([-\d]+(?:\.\d+)?)'
                replacement = rf'\g<1>{v}'
            refined_text = re.sub(pattern, replacement, refined_text, flags=re.IGNORECASE)
    
    # Fix lookahead_on -> lookahead_off
    refined_text = refined_text.replace("barmerge.lookahead_on", "barmerge.lookahead_off // [OptiEngine: FIXED REPAINTING]")
        
    # Add Header
    header = f"// Optimization v1 — Optimized {datetime.now().strftime('%Y-%m-%d')}\n"
    if verdict.verdict == "NO-GO":
        header += "// [OptiEngine VERDICT: NO-GO] See dossier for required changes. Original parameters retained.\n"
        
    with open(refined_path, 'w', encoding='utf-8') as f:
        f.write(header + refined_text)
        
    print(f"[OptiEngine - Artifacts] Dossier written to {dossier_path}")
    print(f"[OptiEngine - Artifacts] Refined script written to {refined_path}")
