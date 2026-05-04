import pandas as pd
import numpy as np

def compile_filters(df: pd.DataFrame, params: dict, active_filters: list) -> dict:
    """
    Takes a list of filter definitions from the IR and generates boolean masks.
    Returns a dictionary of filter masks that the quant engine can cleanly apply.
    """
    n = len(df)
    masks = {
        'in_window': np.ones(n, dtype=bool),
        'tdv_pos': np.ones(n, dtype=bool),
        'tdv_neg': np.ones(n, dtype=bool),
        'wick_long_ok': np.ones(n, dtype=bool),
        'wick_short_ok': np.ones(n, dtype=bool),
        'sweep_long_ok': np.ones(n, dtype=bool),
        'sweep_short_ok': np.ones(n, dtype=bool)
    }

    open_ = df['open'].values if 'open' in df.columns else df['close'].values
    high, low, close = df['high'].values, df['low'].values, df['close'].values

    for f in active_filters:
        f_type = f.get("type")
        
        if f_type == "session_window":
            trade_eth = params.get(f.get("controlled_by", "trade_eth"), False)
            if not trade_eth and 'time' in df.columns:
                try:
                    ts = pd.to_datetime(df['time'].values, unit='s', utc=True)
                    ts_et = ts.tz_convert('America/New_York')
                    hours = ts_et.hour + ts_et.minute / 60.0
                    masks['in_window'] = (hours >= f.get("start_hour", 9.0)) & (hours < f.get("end_hour", 16.0))
                except Exception:
                    pass

        elif f_type == "volume_gate" and f.get("style") == "tdv":
            vol_ma_len = max(1, int(params.get('tdv_vol_ma_len', 12)))
            smooth_bars = max(1, int(params.get('tdv_smoothBars', 4)))
            min_body_pct = params.get('tdv_min_body_pct', 20)

            candle_range = high - low
            body_pct = np.where(candle_range > 0, np.abs(close - open_) / candle_range * 100.0, 0.0)
            is_weak = body_pct < min_body_pct

            bvol = np.where(candle_range > 0, df['volume'].values * (close - low) / candle_range, df['volume'].values / 2)
            svol = np.where(candle_range > 0, df['volume'].values * (high - close) / candle_range, df['volume'].values / 2)

            bvol_sum = pd.Series(bvol).rolling(vol_ma_len, min_periods=1).sum().values
            svol_sum = pd.Series(svol).rolling(vol_ma_len, min_periods=1).sum().values
            raw_buy = bvol_sum > svol_sum

            sig = raw_buy.astype(int)
            locked = np.ones(n, dtype=int)
            streak = np.zeros(n, dtype=int)
            for j in range(1, n):
                if is_weak[j] and sig[j] != sig[j-1]:
                    sig[j] = sig[j-1]
                if sig[j] == sig[j-1]:
                    streak[j] = streak[j-1] + 1
                else:
                    streak[j] = 1
                locked[j] = sig[j] if streak[j] >= smooth_bars else locked[j-1]

            masks['tdv_pos'] = locked == 1
            masks['tdv_neg'] = locked == 0

        elif f_type == "wick_quality":
            require_single_wick = params.get(f.get("params", ["require_single_wick"])[0], True)
            if require_single_wick and 'open' in df.columns:
                try:
                    from indicator_lib import calc_heikin_ashi
                    ha = calc_heikin_ashi(df)
                    ha_body = (ha['ha_close'] - ha['ha_open']).abs().values
                    ha_upper = (ha['ha_high'] - np.maximum(ha['ha_open'].values, ha['ha_close'].values)).values
                    ha_lower = (np.minimum(ha['ha_open'].values, ha['ha_close'].values) - ha['ha_low'].values).values
                    masks['wick_long_ok'] = ha_upper <= ha_body
                    masks['wick_short_ok'] = ha_lower <= ha_body
                except Exception:
                    pass

        elif f_type == "session_sweep":
            use_sweep = params.get(f.get("params", ["use_sweep_filter", "sweep_lookback"])[0], False)
            sweep_lookback = int(params.get(f.get("params", ["use_sweep_filter", "sweep_lookback"])[1], 20))
            
            if use_sweep and 'time' in df.columns:
                try:
                    ts = pd.to_datetime(df['time'].values, unit='s', utc=True).tz_convert('America/New_York')
                    hours = ts.hour + ts.minute / 60.0
                    
                    in_asia = (hours >= 19.0) | (hours < 2.0)
                    in_lon = (hours >= 2.0) & (hours < 8.0)
                    in_nyam = (hours >= 9.0) & (hours < 16.0)
                    in_nypm = (hours >= 13.5) & (hours < 15.0)
                    
                    asia_h, asia_l, lon_h, lon_l, nyam_h, nyam_l, nypm_h, nypm_l = [np.nan]*8
                    ar_h, ar_l, lr_h, lr_l, amr_h, amr_l, pmr_h, pmr_l = [np.nan]*8
                    
                    bull_sweep = np.zeros(n, dtype=bool)
                    bear_sweep = np.zeros(n, dtype=bool)
                    
                    for i in range(1, n):
                        if in_asia[i] and not in_asia[i-1]: ar_h, ar_l = high[i], low[i]
                        elif in_asia[i]: ar_h, ar_l = max(ar_h, high[i]) if not np.isnan(ar_h) else high[i], min(ar_l, low[i]) if not np.isnan(ar_l) else low[i]
                        if in_asia[i-1] and not in_asia[i]: asia_h, asia_l = ar_h, ar_l
                            
                        if in_lon[i] and not in_lon[i-1]: lr_h, lr_l = high[i], low[i]
                        elif in_lon[i]: lr_h, lr_l = max(lr_h, high[i]) if not np.isnan(lr_h) else high[i], min(lr_l, low[i]) if not np.isnan(lr_l) else low[i]
                        if in_lon[i-1] and not in_lon[i]: lon_h, lon_l = lr_h, lr_l
                            
                        if in_nyam[i] and not in_nyam[i-1]: amr_h, amr_l = high[i], low[i]
                        elif in_nyam[i]: amr_h, amr_l = max(amr_h, high[i]) if not np.isnan(amr_h) else high[i], min(amr_l, low[i]) if not np.isnan(amr_l) else low[i]
                        if in_nyam[i-1] and not in_nyam[i]: nyam_h, nyam_l = amr_h, amr_l
                            
                        if in_nypm[i] and not in_nypm[i-1]: pmr_h, pmr_l = high[i], low[i]
                        elif in_nypm[i]: pmr_h, pmr_l = max(pmr_h, high[i]) if not np.isnan(pmr_h) else high[i], min(pmr_l, low[i]) if not np.isnan(pmr_l) else low[i]
                        if in_nypm[i-1] and not in_nypm[i]: nypm_h, nypm_l = pmr_h, pmr_l
                            
                        c, l, h_p = close[i], low[i], high[i]
                        is_bull, is_bear = False, False
                        
                        if not np.isnan(asia_l) and l <= asia_l and c > asia_l: is_bull = True
                        if not np.isnan(lon_l) and l <= lon_l and c > lon_l: is_bull = True
                        if not np.isnan(nyam_l) and l <= nyam_l and c > nyam_l: is_bull = True
                        if not np.isnan(nypm_l) and l <= nypm_l and c > nypm_l: is_bull = True
                        
                        if not np.isnan(asia_h) and h_p >= asia_h and c < asia_h: is_bear = True
                        if not np.isnan(lon_h) and h_p >= lon_h and c < lon_h: is_bear = True
                        if not np.isnan(nyam_h) and h_p >= nyam_h and c < nyam_h: is_bear = True
                        if not np.isnan(nypm_h) and h_p >= nypm_h and c < nypm_h: is_bear = True
                        
                        bull_sweep[i] = is_bull
                        bear_sweep[i] = is_bear
                        
                    def bars_since(cond):
                        idx = np.arange(len(cond))
                        last_true = pd.Series(np.where(cond, idx, np.nan)).ffill()
                        return (idx - last_true).fillna(999).values
                        
                    masks['sweep_long_ok'] = bars_since(bull_sweep) <= sweep_lookback
                    masks['sweep_short_ok'] = bars_since(bear_sweep) <= sweep_lookback
                except Exception:
                    pass

    return masks
