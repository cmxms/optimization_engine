import json
from llm_utils import call_llm_with_retry

class CatfishAgent:
    """
    Catfish Agent (Shallan persona) - The Devil's Advocate.
    Challenges the findings of the Critic and Quant engines to prevent groupthink.
    """
    def __init__(self, pine_text: str, client=None):
        self.pine_text = pine_text
        self.client = client

    def run_dissent(self, critic_report, quant_report) -> str:
        """
        Generates a dissenting analysis of the current strategy.
        """
        print("[OptiEngine - Catfish] Finding the 'Hard Truth' and alternative angles...")
        
        system_prompt = (
            "You are Shallan (The Catfish). Your role is to be the Devil's Advocate. "
            "You are brilliant, cynical, and deeply skeptical of 'perfect' backtests. "
            "Your goal is to find why this strategy will fail in the real world, even if the numbers look good. "
            "Look for: overfitting, regime mismatch, execution impossibilities, and logic holes. "
            "Be direct, slightly sarcastic, and don't hold back. Discard the noise."
        )
        
        user_prompt = (
            f"Critic Report Issues: {len(critic_report.issues)}\n"
            f"In-Sample Sharpe: {quant_report.in_sample_sharpe:.2f}\n"
            f"Out-of-Sample Sharpe: {quant_report.oos_sharpe:.2f}\n"
            f"Overfitting Risk: {quant_report.overfitting_risk}\n"
            f"Win Rate: {quant_report.win_rate*100:.1f}%\n\n"
            f"Here is the Pine Script code:\n\n{self.pine_text}\n\n"
            "Analyze these results. Tell Michael why he's about to lose money. "
            "Find the angle nobody else is looking at."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if not self.client:
            return "No LLM client available for Catfish dissent."

        dissent = call_llm_with_retry(self.client, messages, temperature=0.7)
        return dissent

def run_catfish(pine_text, critic_report, quant_report, client=None):
    agent = CatfishAgent(pine_text, client)
    return agent.run_dissent(critic_report, quant_report)
