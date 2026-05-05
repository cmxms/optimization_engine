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
                    
                    def get_session_levels(in_sess):
                        enters = in_sess & ~in_sess.shift(1, fill_value=False)
                        exits = ~in_sess & in_sess.shift(1, fill_value=False)
                        session_id = enters.cumsum()
                        completed_id = exits.cumsum()
                        
                        s_high = pd.Series(np.where(in_sess, high, np.nan))
                        s_low = pd.Series(np.where(in_sess, low, np.nan))
                        
                        s_max = s_high.groupby(session_id).max()
                        s_min = s_low.groupby(session_id).min()
                        
                        return completed_id.map(s_max).values, completed_id.map(s_min).values

                    asia_h, asia_l = get_session_levels(pd.Series(in_asia))
                    lon_h, lon_l = get_session_levels(pd.Series(in_lon))
                    nyam_h, nyam_l = get_session_levels(pd.Series(in_nyam))
                    nypm_h, nypm_l = get_session_levels(pd.Series(in_nypm))

                    bull_sweep = (
                        (~np.isnan(asia_l) & (low <= asia_l) & (close > asia_l)) |
                        (~np.isnan(lon_l) & (low <= lon_l) & (close > lon_l)) |
                        (~np.isnan(nyam_l) & (low <= nyam_l) & (close > nyam_l)) |
                        (~np.isnan(nypm_l) & (low <= nypm_l) & (close > nypm_l))
                    )

                    bear_sweep = (
                        (~np.isnan(asia_h) & (high >= asia_h) & (close < asia_h)) |
                        (~np.isnan(lon_h) & (high >= lon_h) & (close < lon_h)) |
                        (~np.isnan(nyam_h) & (high >= nyam_h) & (close < nyam_h)) |
                        (~np.isnan(nypm_h) & (high >= nypm_h) & (close < nypm_h))
                    )
                        
                    def bars_since(cond):
                        idx = np.arange(len(cond))
                        last_true = pd.Series(np.where(cond, idx, np.nan)).ffill()
                        return (idx - last_true).fillna(999).values
                        
                    masks['sweep_long_ok'] = bars_since(bull_sweep) <= sweep_lookback
                    masks['sweep_short_ok'] = bars_since(bear_sweep) <= sweep_lookback
                except Exception:
                    pass

    return masks
