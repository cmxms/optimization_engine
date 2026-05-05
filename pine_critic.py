import re
from dataclasses import dataclass, field
from config import config
from rag import query_pine_bugs
from openai import OpenAI

@dataclass
class CriticIssue:
    severity: str
    line_number: int | None
    description: str
    suggested_fix: str

@dataclass
class CriticReport:
    issues: list[CriticIssue] = field(default_factory=list)
    repaint_risk_score: int = 0

    def add_issue(self, severity: str, line_number: int | None, description: str, fix: str):
        self.issues.append(CriticIssue(severity, line_number, description, fix))
        if severity == "CRITICAL" and "repaint" in description.lower():
            self.repaint_risk_score += 5
        elif severity == "WARNING" and "lookahead" in description.lower():
            self.repaint_risk_score += 2

class PineCritic:
    def __init__(self, pine_text: str):
        self.pine_text = pine_text
        self.lines = pine_text.split("\n")
        self.report = CriticReport()

    def run_static_analysis(self):
        """
        Hard-coded regex rule engine. Flags fatal Pine Script errors instantly
        so the LLM doesn't have to guess.
        """
        script_text = self.pine_text.lower()
        
        # 1. Global Strategy/Indicator checks
        if "calc_on_every_tick=true" in script_text or "calc_on_every_tick = true" in script_text:
            self.report.add_issue(
                "CRITICAL", None,
                "calc_on_every_tick=true causes extreme intra-bar repainting differences between backtests and live trading.",
                "Remove calc_on_every_tick or set it to false."
            )

        # 2. Line-by-Line Regex Checks
        for i, line in enumerate(self.lines):
            line_num = i + 1
            line_lower = line.lower()
            
            # Skip comments
            if line.strip().startswith("//"):
                continue

            # Check: Explicit lookahead_on (Guaranteed Repaint)
            if "barmerge.lookahead_on" in line_lower:
                self.report.add_issue(
                    "CRITICAL", line_num, 
                    "Uses lookahead_on, which literally looks into the future. 100% repaint risk.",
                    "Change to barmerge.lookahead_off or remove entirely."
                )
            
            # Check: request.security missing lookahead
            if "request.security" in line_lower:
                if "lookahead" not in line_lower:
                    self.report.add_issue(
                        "WARNING", line_num,
                        "request.security() called without explicit lookahead. Defaults to ON in v3/v4, causing repaints.",
                        "Add explicit lookahead=barmerge.lookahead_off."
                    )
                # Check for using current resolution data inside security
                if re.search(r'request\.security\([^,]*,[^,]*,[^,]*close[^,]*\)', line_lower):
                    self.report.add_issue(
                        "WARNING", line_num,
                        "Passing 'close' to request.security instead of 'close[1]' can cause repainting on the current bar.",
                        "Pass historical series like close[1] into the security call."
                    )
            
            # Check: Alert/Entry without barstate confirmation
            if re.search(r'\b(alert|strategy\.entry|strategy\.close|strategy\.exit)\b', line_lower):
                # If the script doesn't check barstate ANYWHERE near this, it's dangerous
                if "barstate.isconfirmed" not in script_text and "barstate.isrealtime" not in script_text:
                    self.report.add_issue(
                        "WARNING", line_num,
                        "Executes trades or alerts without checking barstate.isconfirmed. Will fire multiple times intra-bar.",
                        "Wrap execution logic in 'if barstate.isconfirmed'."
                    )
                    
            # Check: Division by zero risk
            if re.search(r'/\s*[a-zA-Z_]', line) and "nz(" not in line_lower and "math.max(" not in line_lower:
                # Naive check: division by a variable without nz() or max() protection
                self.report.add_issue(
                    "INFO", line_num,
                    "Potential division by zero if the denominator variable equals 0.",
                    "Wrap denominator in math.max(var, 0.0001) or nz(var, 1)."
                )

    def run_llm_analysis(self, client: OpenAI = None):
        if not config.use_llm:
            return

        print("[OptiEngine - Critic] Running LLM analysis (DeepSeek-Coder)...")
        try:
            if client is None:
                client = OpenAI(base_url=config.llm_base_url, api_key="lm-studio")
            
            # Targeted Snippet Approach: Find risk zones and preserve line numbers
            risk_keywords = ["security", "entry", "exit", "close", "var", "=>"]
            included_line_indices = set()
            
            # Find relevant lines and surrounding context
            for i, line in enumerate(self.lines):
                line_lower = line.lower()
                # Exclude purely visual lines from triggering a zone
                if any(x in line_lower for x in ['plot', 'table.', 'line.', 'label.', 'bgcolor', 'fill', 'input.']):
                    continue
                
                if any(keyword in line_lower for keyword in risk_keywords):
                    start = max(0, i - 2)
                    end = min(len(self.lines), i + 3)
                    for j in range(start, end):
                        included_line_indices.add(j)
            
            # Reconstruct the snippet preserving order and line numbers
            minified_lines = []
            sorted_indices = sorted(list(included_line_indices))
            
            for idx in sorted_indices:
                clean_line = self.lines[idx].split('//')[0].rstrip()
                if clean_line.strip():
                    minified_lines.append(f"{idx+1}: {clean_line}")
                    
            if not minified_lines:
                minified_pine_text = "No logic zones detected."
            else:
                minified_pine_text = '\n'.join(minified_lines)

            prompt = f"""
Audit this Pine Script for repainting and logic errors.
Logic Only:
```pine
{minified_pine_text}
```

Identify critical execution or repainting issues.
Output as a list: Severity, Line, Description, Fix.
"""
            from llm_utils import call_llm_with_retry
            llm_output = call_llm_with_retry(
                client, 
                messages=[
                    {"role": "system", "content": "You are a quantitative developer auditing Pine Script."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            if llm_output:
                # Add LLM findings as a single meta-issue for now
                self.report.add_issue(
                    "INFO", None,
                    "LLM Analysis Complete. Details attached.",
                    llm_output
                )
        except Exception as e:
            print(f"[OptiEngine - Critic] LLM analysis failed: {e}")

    def analyze(self, client: OpenAI = None) -> CriticReport:
        self.run_static_analysis()
        self.run_llm_analysis(client=client)
        
        # Cap repaint risk
        self.report.repaint_risk_score = min(self.report.repaint_risk_score, 10)
        return self.report

def run_critic(pine_text: str, client: OpenAI = None) -> CriticReport:
    critic = PineCritic(pine_text)
    return critic.analyze(client=client)
