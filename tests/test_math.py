import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_engine import QuantEngine

def test_sortino_calculation():
    engine = QuantEngine(".")
    # Test with 10% gain, 5% loss repeated
    trades = np.array([0.1, -0.05, 0.1, -0.05, 0.1, -0.05, 0.1, -0.05, 0.1, -0.05, 
                       0.1, -0.05, 0.1, -0.05, 0.1, -0.05, 0.1, -0.05, 0.1, -0.05])
    
    score = engine._compute_utility(trades, 80, 300)
    assert score > 0
    
    # Test with zero trades
    assert engine._compute_utility(np.array([]), 80, 300) == -1.0

def test_utility_rr_penalty():
    engine = QuantEngine(".")
    trades = np.array([0.01] * 25) # 25 winning trades
    
    # Good RR (300/80 = 3.75)
    score_good = engine._compute_utility(trades, 80, 300)
    
    # Bad RR (80/300 = 0.26)
    score_bad = engine._compute_utility(trades, 300, 80)
    
    assert score_good > score_bad

def test_monte_carlo_stability():
    engine = QuantEngine(".")
    trades = np.array([0.01, -0.01, 0.02, -0.005, 0.015, -0.01, 0.01, -0.005, 0.02, -0.01])
    
    dd, luck = engine.run_monte_carlo(trades, simulations=100)
    
    assert 0 <= dd <= 1.0
    assert 0 <= luck <= 100

def test_regime_tagging_structure():
    from regime_tagger import tag_regimes
    df = pd.DataFrame({
        'close': np.linspace(100, 200, 100),
        'high': np.linspace(101, 201, 100),
        'low': np.linspace(99, 199, 100)
    })
    
    df_tagged = tag_regimes(df)
    assert 'regime' in df_tagged.columns
    assert df_tagged['regime'].iloc[-1] in ['TRENDING_UP', 'TRENDING_DN', 'HIGH_VOL', 'CHOP']
