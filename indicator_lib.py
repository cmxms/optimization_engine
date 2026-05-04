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

def calc_hma(close: pd.Series, length: int) -> pd.Series:
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wmaf = calc_wma(close, half_length)
    wmas = calc_wma(close, length)
    return calc_wma(wmaf * 2 - wmas, sqrt_length)

def calc_rma(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(alpha=1/length, adjust=False).mean()

def get_ma(source: pd.Series, ma_type: str, length: int, volume: pd.Series = None) -> pd.Series:
    t = ma_type.lower()
    if length <= 1: return source
    if t == 'sma': return calc_sma(source, length)
    elif t == 'ema': return calc_ema(source, length)
    elif t == 'hma': return calc_hma(source, length)
    elif t == 'rma': return calc_rma(source, length)
    elif t == 'wma': return calc_wma(source, length)
    elif t == 'vwma': return (source * volume).rolling(length).sum() / volume.rolling(length).sum() if volume is not None else calc_sma(source, length)
    return source

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

def calc_rsi_exhaustion_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    rsi_len = int(kwargs.get("rsiLen", 14))
    rsi_sma_len = int(kwargs.get("rsiSmaLength", 14))
    rsi_ob = int(kwargs.get("rsiObLevel", 68))
    rsi_os = int(kwargs.get("rsiOsLevel", 32))
    rsi_lookback = int(kwargs.get("rsiLookback", 5))
    threshold = int(kwargs.get("threshold", 30))
    short_len = int(kwargs.get("shortLength", 21))
    long_len = int(kwargs.get("longLength", 112))
    cooldown = int(kwargs.get("cooldown", 30))
    smooth_type = kwargs.get("smoothType", "ema")
    formula = kwargs.get("formula", "Standard (2 Period)")
    short_smooth_len = int(kwargs.get("shortSmoothingLength", 7))
    long_smooth_len = int(kwargs.get("longSmoothingLength", 3))
    avg_ma_len = int(kwargs.get("average_ma_len", 3))
    
    close = df['close']
    vol = df.get('volume')
    rsi = calc_rsi(close, rsi_len)
    rsi_sma = calc_sma(rsi, rsi_sma_len)
    
    def percent_r(length):
        highest = close.rolling(length).max()
        lowest = close.rolling(length).min()
        rng = (highest - lowest).clip(lower=1e-10)
        return 100.0 * (close - highest) / rng
        
    s_pr = percent_r(short_len)
    l_pr = percent_r(long_len)
    
    if short_smooth_len > 1:
        s_pr = get_ma(s_pr, smooth_type, short_smooth_len, vol)
    if long_smooth_len > 1:
        l_pr = get_ma(l_pr, smooth_type, long_smooth_len, vol)
        
    if formula == "Average":
        avg_pr = (s_pr + l_pr) / 2
        if avg_ma_len > 1:
            avg_pr = get_ma(avg_pr, smooth_type, avg_ma_len, vol)
        overbought = avg_pr >= -threshold
        oversold = avg_pr <= -100 + threshold
    else:
        overbought = (s_pr >= -threshold) & (l_pr >= -threshold)
        oversold = (s_pr <= -100 + threshold) & (l_pr <= -100 + threshold)
    
    rsi_cross_above = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))
    rsi_cross_below = (rsi < rsi_sma) & (rsi.shift(1) >= rsi_sma.shift(1))
    
    was_os = (rsi <= rsi_os).rolling(rsi_lookback + 1, min_periods=1).max().astype(bool)
    was_ob = (rsi >= rsi_ob).rolling(rsi_lookback + 1, min_periods=1).max().astype(bool)
    
    raw_long = (rsi_cross_above & was_os & oversold).values
    raw_short = (rsi_cross_below & was_ob & overbought).values
    
    n = len(df)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    last_long_bar = -(cooldown + 1)
    last_short_bar = -(cooldown + 1)
    for i in range(n):
        if raw_long[i] and (i - last_long_bar) >= cooldown:
            buy[i] = True
            last_long_bar = i
        if raw_short[i] and (i - last_short_bar) >= cooldown:
            sell[i] = True
            last_short_bar = i
            
    return pd.DataFrame({'buy': buy, 'sell': sell}, index=df.index)

def calc_trama_ha_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    t_f = int(kwargs.get("trama_fast_len", 13))
    t_m = int(kwargs.get("trama_med_len", 28))
    t_s = int(kwargs.get("trama_slow_len", 40))
    tf = calc_trama(df['high'], df['low'], df['close'], t_f)
    tm = calc_trama(df['high'], df['low'], df['close'], t_m)
    ts = calc_trama(df['high'], df['low'], df['close'], t_s)
    
    ha = calc_heikin_ashi(df)
    ha_close = ha['ha_close']
    
    above = (ha_close > tf) & (ha_close > tm) & (ha_close > ts)
    below = (ha_close < tf) & (ha_close < tm) & (ha_close < ts)
    
    use_regime = kwargs.get("use_regime_filter", False)
    cross_lb = int(kwargs.get("cross_lookback", 4))
    p_bars = int(kwargs.get("prior_regime_bars", 3))
    p_window = int(kwargs.get("prior_regime_window", 20))
    
    n = len(df)
    above_np = above.values
    below_np = below.values
    regime_streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        if above_np[i]:
            regime_streak[i] = regime_streak[i-1] + 1 if regime_streak[i-1] > 0 else 1
        elif below_np[i]:
            regime_streak[i] = regime_streak[i-1] - 1 if regime_streak[i-1] < 0 else -1
        else:
            regime_streak[i] = regime_streak[i-1]
            
    regime_streak_s = pd.Series(regime_streak, index=df.index)
    regime_streak_prev = regime_streak_s.shift(1).fillna(0)
    
    was_clearly_bear = regime_streak_prev.rolling(p_window).min() <= -p_bars
    was_clearly_bull = regime_streak_prev.rolling(p_window).max() >= p_bars
    
    cross_to_bull_simple = above & ~above.shift(1).fillna(False)
    cross_to_bear_simple = below & ~below.shift(1).fillna(False)
    
    true_cross_to_bull = (cross_to_bull_simple & was_clearly_bear) if use_regime else cross_to_bull_simple
    true_cross_to_bear = (cross_to_bear_simple & was_clearly_bull) if use_regime else cross_to_bear_simple
    
    def bars_since(condition):
        idx = np.arange(len(condition))
        last_true = pd.Series(np.where(condition, idx, np.nan), index=condition.index).ffill()
        return (idx - last_true).fillna(999)
        
    bars_since_rev_bull = bars_since(true_cross_to_bull)
    bars_since_rev_bear = bars_since(true_cross_to_bear)
    
    fresh_reversal_long = bars_since_rev_bull <= cross_lb
    fresh_reversal_short = bars_since_rev_bear <= cross_lb
    
    ha_green = (ha['ha_close'] > ha['ha_open']).astype(int)
    ha_red = (ha['ha_close'] < ha['ha_open']).astype(int)
    
    ha_stack_min = int(kwargs.get("ha_stack_min", 2))
    
    l_s_streak = ha_green.groupby(ha_green.ne(ha_green.shift()).cumsum()).cumsum()
    s_s_streak = ha_red.groupby(ha_red.ne(ha_red.shift()).cumsum()).cumsum()
    
    req_rising = kwargs.get("require_rising_high", False)
    req_falling = kwargs.get("require_falling_low", False)
    
    rising_highs = (ha['ha_high'].diff(1) >= 0).rolling(max(1, ha_stack_min - 1)).min().fillna(0).astype(bool)
    falling_lows = (ha['ha_low'].diff(1) <= 0).rolling(max(1, ha_stack_min - 1)).min().fillna(0).astype(bool)
        
    l_s = (l_s_streak >= ha_stack_min) & (rising_highs if req_rising else True)
    s_s = (s_s_streak >= ha_stack_min) & (falling_lows if req_falling else True)
    
    buy = fresh_reversal_long & l_s & above
    sell = fresh_reversal_short & s_s & below
    return pd.DataFrame({'buy': buy, 'sell': sell}, index=df.index)


def calc_stateful_streak(condition_a: pd.Series, condition_b: pd.Series) -> pd.Series:
    """
    Generic Python replica of Pine's persistent streak pattern:
        var int streak = 0
        streak := condition_a ? (streak[1] > 0 ? streak[1]+1 : 1)
                : condition_b ? (streak[1] < 0 ? streak[1]-1 : -1)
                : streak[1]

    Returns a signed integer Series:
        > 0  →  consecutive bars where condition_a was True
        < 0  →  consecutive bars where condition_b was True
        0    →  neither condition active yet
    """
    n = len(condition_a)
    a_arr = condition_a.values
    b_arr = condition_b.values
    streak = np.zeros(n, dtype=int)
    for i in range(1, n):
        if a_arr[i]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif b_arr[i]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = streak[i - 1]
    return pd.Series(streak, index=condition_a.index)


def calc_tdv_locked_state(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Exact Python replica of the TD Volume (TDV) locked_state logic from the TD NQ Bot.

    Pine's stateful variables replicated:
        var int locked_state = 1
        var bool tdv_turned_bullish = false
        var bool tdv_turned_bearish = false

    The key subtlety: state only changes on barstate.isconfirmed (bar close).
    Weak-body candles penalize state changes (reset state_streak to 0).

    Returns DataFrame with columns:
        tdv_pos     : bool — True when locked_state == 1 (bullish)
        tdv_neg     : bool — True when locked_state == 0 (bearish)
        turned_bull : bool — True on the exact bar the state flipped to bullish
        turned_bear : bool — True on the exact bar the state flipped to bearish
    """
    vol_ma_len   = max(1, int(kwargs.get('tdv_vol_ma_len', 12)))
    smooth_bars  = max(1, int(kwargs.get('tdv_smoothBars', 4)))
    min_body_pct = float(kwargs.get('tdv_min_body_pct', 20))
    use_trama_gate = bool(kwargs.get('tdv_use_trama_gate', True))
    trama_len    = int(kwargs.get('trama_med_len', 20))  # internal TRAMA(20)

    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    open_ = df['open'].values if 'open' in df.columns else close
    vol   = df['volume'].values
    n     = len(df)

    # Candle metrics
    candle_range = high - low
    body_pct = np.where(candle_range > 0, np.abs(close - open_) / candle_range * 100.0, 0.0)
    is_weak  = body_pct < min_body_pct

    # Buy/sell volume decomposition
    bvol = np.where(candle_range > 0, vol * (close - low) / candle_range, vol / 2)
    svol = np.where(candle_range > 0, vol * (high - close) / candle_range, vol / 2)

    bvol_sum = pd.Series(bvol).rolling(vol_ma_len, min_periods=1).sum().values
    svol_sum = pd.Series(svol).rolling(vol_ma_len, min_periods=1).sum().values
    raw_buy  = (bvol_sum > svol_sum).astype(int)

    # Optional TRAMA gate
    if use_trama_gate:
        trama_vals = calc_trama(
            df['high'], df['low'], df['close'], trama_len
        ).values
        price_above_trama = (close > trama_vals).astype(int)
        raw_buy = raw_buy & price_above_trama

    # State streak + weak-body penalty (mirrors Pine's state_streak logic)
    sig     = raw_buy.copy()
    streak  = np.zeros(n, dtype=int)
    locked  = np.ones(n, dtype=int)  # locked_state starts as 1 (bullish)

    for j in range(1, n):
        # Weak body penalises a potential state change
        if is_weak[j] and sig[j] != sig[j - 1]:
            sig[j] = sig[j - 1]

        if sig[j] == sig[j - 1]:
            streak[j] = streak[j - 1] + 1
        else:
            streak[j] = 1

        # Acceleration shortcut: strong accel allows immediate flip (required_bars = 1)
        # We approximate this conservatively — always require smooth_bars.
        required = smooth_bars

        if streak[j] >= required:
            locked[j] = sig[j]
        else:
            locked[j] = locked[j - 1]

    tdv_pos     = pd.Series(locked == 1, index=df.index)
    tdv_neg     = pd.Series(locked == 0, index=df.index)
    turned_bull = tdv_pos & ~tdv_pos.shift(1).fillna(False)
    turned_bear = tdv_neg & ~tdv_neg.shift(1).fillna(False)

    return pd.DataFrame({
        'tdv_pos':     tdv_pos,
        'tdv_neg':     tdv_neg,
        'turned_bull': turned_bull,
        'turned_bear': turned_bear,
    }, index=df.index)


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
    "STATEFUL_STREAK": calc_stateful_streak,
    "BARS_SINCE": calc_bars_since,
    "CROSS_ABOVE": calc_cross_above,
    "CROSS_BELOW": calc_cross_below,
    "TRAMA": calc_trama,
    "HA": calc_heikin_ashi,
    "RSI_EXHAUSTION_SIGNALS": calc_rsi_exhaustion_signals,
    "TRAMA_HA_SIGNALS": calc_trama_ha_signals,
    "TDV_LOCKED_STATE": calc_tdv_locked_state,
}
