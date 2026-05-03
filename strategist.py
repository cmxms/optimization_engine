from dataclasses import dataclass, field
from config import config
from pine_critic import CriticReport
from quant_engine import QuantReport
from openai import OpenAI
import json

@dataclass
class StrategistVerdict:
    """Stores the final Go/No-Go decision from the Strategist agent."""
    verdict: str = "UNKNOWN"
    confidence_pct: int = 0
    rationale: str = ""
    required_code_changes: list[str] = field(default_factory=list)
    market_context: str = ""

class Strategist:
    """Synthesizes logic and quantitative findings into a final strategy assessment."""
    def __init__(self, critic_report: CriticReport, quant_report: QuantReport, market_snapshot: dict = None):
        self.critic_report = critic_report
        self.quant_report = quant_report
        self.market_snapshot = market_snapshot or {}
        self.verdict = StrategistVerdict()

    def generate_rule_based_verdict(self):
        # Determine Verdict
        sortino_metric = getattr(self.quant_report, 'sortino', self.quant_report.oos_sharpe)
        if self.critic_report.repaint_risk_score >= 7 or (self.quant_report.data_found and self.quant_report.overfitting_risk == "HIGH" and sortino_metric < 0.3):
            self.verdict.verdict = "NO-GO"
            self.verdict.confidence_pct = 95
            self.verdict.rationale = "High repaint risk or severe overfitting detected in Out-Of-Sample data."
        elif self.quant_report.data_found and self.quant_report.wfa_consistency_score >= 75 and sortino_metric >= 0.8 and self.critic_report.repaint_risk_score < 3:
            self.verdict.verdict = "GO"
            self.verdict.confidence_pct = 85
            self.verdict.rationale = "Strong OOS Sortino, clean static analysis, and highly consistent across Walk-Forward folds."
        else:
            self.verdict.verdict = "CONDITIONAL"
            self.verdict.confidence_pct = 60
            self.verdict.rationale = "Marginal performance or inconsistent across time periods. Proceed with caution."

        # Propagate required code changes from Critic
        for issue in self.critic_report.issues:
            if issue.severity in ("CRITICAL", "WARNING"):
                self.verdict.required_code_changes.append(f"Line {issue.line_number}: {issue.suggested_fix}")

    def run_llm_synthesis(self, client: OpenAI = None):
        if not config.use_llm:
            return

        print("[OptiEngine - Strategist] Running LLM synthesis (Mistral-Small-24B)...")
        regime_str = ""
        for r, perf in self.quant_report.regime_performance.items():
            regime_str += f"\n- {r}: {perf['trades']} trades, {perf['win_rate']:.1f}% WR, {perf['avg_pnl']:.2f}% avg PnL"
        if not regime_str:
            regime_str = "\n- None"

        try:
            if client is None:
                client = OpenAI(base_url=config.llm_base_url, api_key="lm-studio")

            prompt = f"""
You are the Strategist agent for Optimization Engine.
Synthesize the findings from the Logic Critic and the Quant Engine to provide a final strategy assessment.

Critic Report:
- Repaint Risk: {self.critic_report.repaint_risk_score}/10
- Issues: {json.dumps([{'sev': i.severity, 'desc': i.description} for i in self.critic_report.issues])}

Quant Report (5-Fold Walk Forward Analysis):
- Data Found: {self.quant_report.data_found}
- Avg IS Sharpe: {self.quant_report.in_sample_sharpe:.2f}
- Avg OOS Sharpe: {self.quant_report.oos_sharpe:.2f}
- OOS Sortino: {getattr(self.quant_report, 'sortino', 0.0):.2f}
- Max Drawdown: {getattr(self.quant_report, 'max_drawdown', 0.0) * 100:.2f}%
- WFA Consistency Score: {self.quant_report.wfa_consistency_score:.0f}% of periods were profitable
- Overfitting Risk: {self.quant_report.overfitting_risk}

Historical Regime Performance:{regime_str}

Monte Carlo Stress Test (10,000 Simulations):
- 95th Percentile Max Drawdown: {self.quant_report.mc_max_dd_95 * 100:.2f}%
- Luck Factor: {self.quant_report.mc_luck_factor:.0f}th percentile (Lower is better; 99th means we got extremely lucky vs median path)

Current Market Context (REAL-TIME):
- Regime: {self.market_snapshot.get('regime', 'Unknown')}
- Breadth: {self.market_snapshot.get('breadth', 'Unknown')}

Based on this, perform a deep market reality check. 
Contrast the technical backtest performance against the structural logic integrity AND the current real-time market regime. 
Output a 3-sentence narrative explaining if this strategy would survive a 'Black Swan' event or if it's merely 'curve-fitted' to the noise, and how it might perform in the CURRENT market context.
"""
            from llm_utils import call_llm_with_retry
            llm_narrative = call_llm_with_retry(
                client, 
                messages=[
                    {"role": "system", "content": "You are a lead quantitative strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            if llm_narrative:
                self.verdict.rationale += f"\n\nLLM Synthesis:\n{llm_narrative}"
        except Exception as e:
            print(f"[OptiEngine - Strategist] LLM synthesis failed: {e}")

    def analyze(self, client: OpenAI = None) -> StrategistVerdict:
        self.generate_rule_based_verdict()
        self.run_llm_synthesis(client=client)
        return self.verdict

def run_strategist(critic_report: CriticReport, quant_report: QuantReport, snapshot: dict = None, client: OpenAI = None) -> StrategistVerdict:
    strategist = Strategist(critic_report, quant_report, market_snapshot=snapshot)
    return strategist.analyze(client=client)
