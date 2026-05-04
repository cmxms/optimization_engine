import re

def apply_auto_fixes(pine_text, critic_report):
    """
    Applies safe, regex-based micro-patches to the Pine Script based on Critic findings.
    Currently supports:
    - Division by zero on simple variable denominators.
    """
    if not critic_report or not hasattr(critic_report, 'issues') or not critic_report.issues:
        return pine_text
        
    lines = pine_text.split('\n')
    patched_count = 0
    
    for issue in critic_report.issues:
        # Check if it's a division by zero issue
        desc = issue.description.lower()
        if "division by zero" in desc or "denominator" in desc:
            # Extract line number
            if hasattr(issue, 'line_number') and issue.line_number is not None:
                try:
                    line_idx = int(issue.line_number) - 1
                except ValueError:
                    match = re.search(r'Line (\d+)', issue.description, re.IGNORECASE)
                    if match:
                        line_idx = int(match.group(1)) - 1
                    else:
                        continue
            else:
                match = re.search(r'Line (\d+)', issue.description, re.IGNORECASE)
                if match:
                    line_idx = int(match.group(1)) - 1
                else:
                    continue
                
            if 0 <= line_idx < len(lines):
                    original_line = lines[line_idx]
                    
                    # Safe regex: Only matches division by a single variable (e.g. " / varname ")
                    # Doesn't touch complex expressions like " / (a + b) "
                    # Replaces with " / math.max(varname, 0.000001) "
                    def repl(m):
                        var_name = m.group(1)
                        # Don't double-wrap if it's already wrapped or is a number
                        if 'math.max' in original_line or var_name.isnumeric():
                            return m.group(0)
                        return f" / math.max({var_name}, 0.000001)"
                        
                    # Match slash, optional spaces, then a valid Pine variable name not followed by a dot or paren
                    new_line = re.sub(r'/\s*([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*[\.\(\[])', repl, original_line)
                    
                    if new_line != original_line:
                        lines[line_idx] = new_line
                        patched_count += 1
                        print(f"  [Auto-Fixer] Patched Line {line_idx + 1}: Division by zero guard applied.")
                        
    if patched_count > 0:
        print(f"  [Auto-Fixer] Successfully applied {patched_count} structural patches.")
        
    return '\n'.join(lines)
