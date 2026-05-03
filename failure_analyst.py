import json
import sqlite3
import os
from openai import OpenAI
from config import config
from rag import ALPHA_DB_PATH

class FailureAnalyst:
    def __init__(self, pine_text: str, quant_report: dict, critic_report: dict):
        self.pine_text = pine_text
        self.quant_report = quant_report
        self.critic_report = critic_report

    def run_analysis(self, client: OpenAI = None):
        if not config.use_llm:
            return
            
        # Only run if Critic liked it but Quant hated it (The "Time-Travel/Overfit" Paradox)
        if self.critic_report.get('repaint_risk_score', 0) > 4:
            return # Critic already caught it
        
        oos_sharpe = self.quant_report.get('oos_sharpe', 0.0)
        is_sharpe = self.quant_report.get('in_sample_sharpe', 0.0)
        
        # Criteria for "Unexpected Failure"
        if oos_sharpe > 0.5:
            return # Not a failure
            
        print("[OptiEngine - Failure Analyst] Analyzing unexpected backtest failure...")
        
        prompt = f"""
You are the Failure Analyst for Optimization Engine.
A trading strategy passed static logic analysis (Logic Critic) but failed miserably in the Quant Engine backtest.

Logic Critic Score: {self.critic_report.get('repaint_risk_score')}/10 (Low is good)
Quant IS Sharpe: {is_sharpe:.2f}
Quant OOS Sharpe: {oos_sharpe:.2f}

Pine Script Excerpt (First 100 lines):
{self.pine_text[:2000]}

TASK:
Identify WHY this strategy likely failed. Common reasons:
1. "Ghost" indicator repainting (not caught by regex).
2. Extreme overfitting to a specific market regime.
3. Sub-minute execution issues.

Write a ONE-SENTENCE rule describing this failure pattern to be added to our Knowledge Base.
Example: "Strategies relying on X indicator without Y confirmation tend to overfit in low-volatility regimes."

Output format: JSON
{{
  "failure_pattern": "The one sentence rule",
  "tags": "comma, separated, tags"
}}
"""
        try:
            if client is None:
                client = OpenAI(base_url=config.llm_base_url, api_key="lm-studio")
            
            from vram_manager import vram
            response = client.chat.completions.create(
                model=vram.get_current_model_key(),
                messages=[
                    {"role": "system", "content": "You are a quantitative failure analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            res = json.loads(response.choices[0].message.content)
            self._save_to_knowledge(res['failure_pattern'], res['tags'])
            print(f"  -> New failure pattern logged: {res['failure_pattern']}")
            
        except Exception as e:
            print(f"[OptiEngine - Failure Analyst] Analysis failed: {e}")

    def _save_to_knowledge(self, pattern: str, tags: str):
        conn = sqlite3.connect(ALPHA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO knowledge (category, content, tags, source, verified) VALUES (?, ?, ?, ?, ?)",
            ('self_healing_rule', pattern, tags, 'Failure_Analyst', 0)
        )
        conn.commit()
        conn.close()

def run_failure_analyst(pine_text, quant_report, critic_report, client=None):
    # Convert reports to dict if they are objects
    q_dict = quant_report.__dict__ if hasattr(quant_report, '__dict__') else quant_report
    c_dict = critic_report.__dict__ if hasattr(critic_report, '__dict__') else critic_report
    
    analyst = FailureAnalyst(pine_text, q_dict, c_dict)
    analyst.run_analysis(client=client)
