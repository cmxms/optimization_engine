import pandas as pd
import numpy as np

def calc_trama(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Calculates Trend Regularity Adaptive Moving Average (TRAMA)"""
    # Vectorized pre-calculations
    hh = np.maximum(np.sign(high.rolling(length).max().diff()), 0)
    ll = np.maximum(np.sign(low.rolling(length).min().diff() * -1), 0)
    trig = np.maximum(hh, ll)
    tc = np.power(trig.rolling(length).mean(), 2).fillna(0).values
    
    close_arr = close.values
    trama = np.zeros(len(close_arr))
    
    if len(close_arr) > 0:
        trama[0] = close_arr[0]
        # Core recursive loop (Optimized for minimal overhead)
        for i in range(1, len(close_arr)):
            trama[i] = trama[i-1] + tc[i] * (close_arr[i] - trama[i-1])
            
    return pd.Series(trama, index=close.index)

def calc_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Heikin Ashi Open, High, Low, Close (Vectorized)"""
    ha_df = pd.DataFrame(index=df.index)
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HA Open calculation: H[i] = (H[i-1] + C[i-1]) / 2
    # This is an EMA with alpha=0.5 on the shifted HA Close
    ha_open = ha_close.shift(1).copy()
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    # Using EWM to solve the recursive relationship: y[i] = (1-alpha)*y[i-1] + alpha*x[i]
    # Here alpha=0.5, so y[i] = 0.5*y[i-1] + 0.5*x[i]
    ha_open = ha_open.ewm(alpha=0.5, adjust=False).mean()
    
    ha_df['ha_open'] = ha_open
    ha_df['ha_close'] = ha_close
    ha_df['ha_high'] = np.maximum(df['high'], np.maximum(ha_df['ha_open'], ha_df['ha_close']))
    ha_df['ha_low'] = np.minimum(df['low'], np.minimum(ha_df['ha_open'], ha_df['ha_close']))
    
    return ha_df

# Trend Indicators
def calc_ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False).mean()

def calc_sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(window=length).mean()

def calc_wma(close: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return close.rolling(length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def calc_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    typ_price = (high + low + close) / 3
    # Rolling VWAP as a generic proxy
    return (typ_price * volume).rolling(length).sum() / volume.rolling(length).sum()

def calc_donchian(high: pd.Series, low: pd.Series, length: int) -> pd.DataFrame:
    upper = high.rolling(length).max()
    lower = low.rolling(length).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({'upper': upper, 'lower': lower, 'mid': mid})

# Oscillators
def calc_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    macd = calc_ema(close, fast) - calc_ema(close, slow)
    sig = calc_ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({'macd': macd, 'signal': sig, 'hist': hist})

def calc_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int) -> pd.DataFrame:
    ll = low.rolling(window=k).min()
    hh = high.rolling(window=k).max()
    stoch_k = 100 * ((close - ll) / (hh - ll))
    stoch_d = stoch_k.rolling(window=d).mean()
    return pd.DataFrame({'k': stoch_k, 'd': stoch_d})

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

# Volume
def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (volume * direction).fillna(0).cumsum()

def calc_volume_ma(volume: pd.Series, length: int) -> pd.Series:
    return volume.rolling(length).mean()

# Candle Patterns
def calc_candle_body_pct(open: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    range_val = high - low
    body = (close - open).abs()
    return pd.Series(np.where(range_val > 0, (body / range_val) * 100, 0), index=close.index)

def calc_wick_ratio(open: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    body_top = np.maximum(open, close)
    body_bottom = np.minimum(open, close)
    upper_wick = high - body_top
    lower_wick = body_bottom - low
    return pd.DataFrame({'upper_wick': upper_wick, 'lower_wick': lower_wick})

# Regime / Structural
def calc_trend_streak(condition_series: pd.Series) -> pd.Series:
    streak = np.zeros(len(condition_series))
    cond_arr = condition_series.values
    for i in range(1, len(cond_arr)):
        if cond_arr[i]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        else:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
    return pd.Series(streak, index=condition_series.index)

def calc_bars_since(condition_series: pd.Series) -> pd.Series:
    bars = np.zeros(len(condition_series))
    cond_arr = condition_series.values
    count = 0
    found = False
    for i in range(len(cond_arr)):
        if cond_arr[i]:
            count = 0
            found = True
        elif found:
            count += 1
        bars[i] = count if found else 999999
    return pd.Series(bars, index=condition_series.index)

def calc_cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def calc_cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

INDICATOR_CATALOG = {
    "EMA": calc_ema,
    "SMA": calc_sma,
    "WMA": calc_wma,
    "VWAP": calc_vwap,
    "DONCHIAN": calc_donchian,
    "RSI": calc_rsi,
    "MACD": calc_macd,
    "STOCH": calc_stoch,
    "ATR": calc_atr,
    "OBV": calc_obv,
    "VOL_MA": calc_volume_ma,
    "BODY_PCT": calc_candle_body_pct,
    "WICK_RATIO": calc_wick_ratio,
    "TREND_STREAK": calc_trend_streak,
    "BARS_SINCE": calc_bars_since,
    "CROSS_ABOVE": calc_cross_above,
    "CROSS_BELOW": calc_cross_below,
    "TRAMA": calc_trama,
    "HA": calc_heikin_ashi
}
