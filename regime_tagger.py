import pandas as pd
import numpy as np

def tag_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags historical price data with market regimes (HIGH_VOL, LOW_VOL, TRENDING_UP, TRENDING_DOWN, CHOPPY).
    Returns the DataFrame with a new 'regime' column.
    """
    if 'regime' in df.columns:
        return df
        
    df_out = df.copy()
    
    # Calculate True Range and ATR for volatility
    high_low = df_out['high'] - df_out['low']
    high_close = (df_out['high'] - df_out['close'].shift()).abs()
    low_close = (df_out['low'] - df_out['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()
    atr_pct = atr / df_out['close']
    
    # Volatility percentiles based on expanding or rolling window
    # Using expanding to avoid future leak, though for purely historical bucketing, full distribution is also fine.
    # We use a 500-bar rolling window to adapt to local volatility environments.
    atr_rolling_pct_75 = atr_pct.rolling(window=500, min_periods=20).quantile(0.75)
    atr_rolling_pct_25 = atr_pct.rolling(window=500, min_periods=20).quantile(0.25)
    
    # Calculate Trend with adaptive SMA length based on data frequency
    sma_window = 50
    diff_window = 5
    if 'time' in df_out.columns:
        try:
            ts = pd.to_datetime(df_out['time'], unit='s', errors='coerce')
            diffs = ts.diff().dropna().dt.total_seconds()
            if not diffs.empty:
                interval = diffs.mode()[0]
                if interval <= 15:    # e.g., 15-second bars
                    sma_window = 200  # ~50 minutes
                    diff_window = 20
                elif interval <= 30:  # e.g., 30-second bars
                    sma_window = 100
                    diff_window = 10
        except Exception:
            pass
            
    sma_trend = df_out['close'].rolling(window=sma_window).mean()
    slope = sma_trend.diff(diff_window)
    
    # Regime Logic
    conditions = [
        (df_out['close'] > sma_trend) & (slope > 0),
        (df_out['close'] < sma_trend) & (slope < 0),
        (atr_pct > atr_rolling_pct_75),
        (atr_pct < atr_rolling_pct_25)
    ]
    choices = ['TRENDING_UP', 'TRENDING_DOWN', 'HIGH_VOL', 'LOW_VOL']
    
    # Order of conditions matters. We check Trend first, then Volatility.
    # Actually, the user plan specifies Volatility first might be better, but "TRENDING" vs "CHOPPY" is standard.
    # Let's apply conditions: if highly volatile, call it HIGH_VOL regardless of trend?
    # The plan: "Neither trending condition met, volatility mid-range -> CHOPPY".
    df_out['regime'] = np.select(conditions, choices, default='CHOPPY')
    
    # Drop first 50 bars as they don't have enough data for SMA and will default to CHOPPY
    return df_out
