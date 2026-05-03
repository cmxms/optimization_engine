from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class ParityReport:
    available: bool = False
    fidelity_score: float = 0.0
    buy_recall: float = 0.0
    sell_recall: float = 0.0
    status: str = "NOT_VERIFIABLE"

KNOWN_BUY_COLS = {'buy', 'long', 'entry_long', 'signal_buy', 'bull'}
KNOWN_SELL_COLS = {'sell', 'short', 'entry_short', 'signal_sell', 'bear'}

def run_parity_check(df: pd.DataFrame, python_buy: np.ndarray, python_sell: np.ndarray) -> ParityReport:
    report = ParityReport()
    
    # Attempt to detect Pine-exported signal columns
    pine_buy_col = next((c for c in df.columns if c.lower() in KNOWN_BUY_COLS), None)
    pine_sell_col = next((c for c in df.columns if c.lower() in KNOWN_SELL_COLS), None)
    
    if not pine_buy_col and not pine_sell_col:
        return report
        
    report.available = True
    
    pine_buy = df[pine_buy_col].fillna(0).astype(int).values if pine_buy_col else np.zeros(len(df))
    pine_sell = df[pine_sell_col].fillna(0).astype(int).values if pine_sell_col else np.zeros(len(df))
    
    pine_buy_bars = (pine_buy == 1)
    pine_sell_bars = (pine_sell == 1)
    
    # Calculate recall (did Python catch Pine's signals?)
    if pine_buy_bars.sum() > 0:
        report.buy_recall = float((python_buy[pine_buy_bars] == 1).mean())
    else:
        report.buy_recall = 1.0 # No signals to miss
        
    if pine_sell_bars.sum() > 0:
        report.sell_recall = float((python_sell[pine_sell_bars] == 1).mean())
    else:
        report.sell_recall = 1.0
        
    report.fidelity_score = (report.buy_recall + report.sell_recall) / 2
    
    if report.fidelity_score >= 0.90:
        report.status = "PASS"
    elif report.fidelity_score >= 0.75:
        report.status = "WARN"
    else:
        report.status = "CRITICAL"
        
    return report
