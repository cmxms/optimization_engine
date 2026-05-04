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
    # Test with 10% gain, 5% loss repeated (30 trades to pass the 25-trade min penalty)
    trades = np.array([0.1, -0.05] * 15)
    
    score = engine._compute_utility(trades, 80, 300)
    assert score > 0
    
    # Test with zero trades
    assert engine._compute_utility(np.array([]), 80, 300) == -1.0

def test_utility_rr_penalty():
    engine = QuantEngine(".")
    trades = np.array([0.01] * 30) # 30 winning trades (to pass 25-trade min)
    
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

def test_parity_gate_logic():
    from parity_checker import run_parity_check, PARITY_THRESHOLD
    
    # Create dummy data
    df = pd.DataFrame({
        'buy': [1, 0, 1, 0, 0],
        'sell': [0, 1, 0, 1, 0]
    })
    
    # 1. Test Perfect Match (100%)
    py_buy = np.array([True, False, True, False, False])
    py_sell = np.array([False, True, False, True, False])
    report = run_parity_check(df, py_buy, py_sell)
    assert report.status == "PASS"
    assert report.blocking == False
    assert report.fidelity_score == 1.0

    # 2. Test Total Mismatch (0%)
    py_buy = np.array([False, False, False, False, False])
    py_sell = np.array([False, False, False, False, False])
    report = run_parity_check(df, py_buy, py_sell)
    assert report.status == "FAIL"
    assert report.blocking == True
    assert report.fidelity_score == 0.0

    # 3. Test Partial Mismatch (below 95% threshold)
    # Pine has 2 buys. We match 1. Recall = 50%.
    py_buy = np.array([True, False, False, False, False])
    py_sell = np.array([False, True, False, True, False])
    report = run_parity_check(df, py_buy, py_sell)
    assert report.blocking == True
    assert report.fidelity_score < PARITY_THRESHOLD

def test_tdv_filter_math():
    from indicator_lib import calc_tdv_locked_state
    
    # Create a 20-bar dataset
    # First 10 bars: Bullish (Close near High)
    # Next 10 bars: Bearish (Close near Low)
    df = pd.DataFrame({
        'open':  [100]*20,
        'high':  [110]*20,
        'low':   [90]*20,
        'close': [108]*10 + [92]*10, # 10 bars near high, 10 bars near low
        'volume': [1000]*20
    })
    
    # Set internal TRAMA gate to False to isolate volume logic
    res = calc_tdv_locked_state(df, tdv_vol_ma_len=2, tdv_smoothBars=2, tdv_use_trama_gate=False)
    
    # Check that it starts bullish (locked_state = 1)
    assert res['tdv_pos'].iloc[0] == True
    
    # Check that it flips to bearish after smooth_bars (2) of bearish data
    # Bearish starts at index 10. Should flip at index 11 (2 bars later).
    assert res['tdv_neg'].iloc[11] == True
    assert res['turned_bear'].iloc[11] == True
