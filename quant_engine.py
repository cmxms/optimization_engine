import os
import glob
import json
import pandas as pd
import numpy as np
import optuna
from dataclasses import dataclass, field
from indicator_lib import calc_trama, calc_heikin_ashi
from rag import log_failed_backtest
from tqdm import tqdm
from functools import lru_cache
from parity_checker import run_parity_check, ParityReport
from regime_tagger import tag_regimes

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class QuantReport:
    """Data structure for storing optimization and backtest results."""
    data_found: bool = False
    recipe: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    in_sample_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    overfitting_risk: str = "UNKNOWN"
    trade_count: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    wfa_consistency_score: float = 0.0
    wfa_fold_sharpes: list[float] = field(default_factory=list)
    mc_max_dd_95: float = 0.0
    mc_luck_factor: float = 0.0
    # Execution Realism
    slippage_ticks_used: int = 0
    commission_per_rt: float = 0.0
    execution_realism_warning: bool = False
    # Utility Objective
    sortino: float = 0.0
    utility_score: float = 0.0
    # Parity Report
    parity_report: ParityReport = field(default_factory=ParityReport)
    # Regime Report
    regime_performance: dict = field(default_factory=dict)

class QuantEngine:
    """Core optimization engine for testing strategy recipes against historical data."""
    
    def __init__(self, project_dir: str):
        """
        Initializes the engine and loads project-specific configurations.
        
        Args:
            project_dir: Path to the directory containing strategy assets and data.
        """
        self.project_dir = project_dir
        self.df = None
        self.report = QuantReport()
        self.recipe = {}
        self._indicator_cache = {}
        
        # Default Market Settings (NQ)
        self.tick_value = 5.0
        self.tick_size = 0.25
        self.bar_interval_seconds = None
        self.order_type = "market"
        self.slippage_ticks = 1
        self.stop_slippage_ticks = 2
        self.commission_per_side = 2.05
        self.contract_multiplier = 1
        self.has_slippage_config = False
        
        self._load_project_config()

    def _load_project_config(self):
        """Internal helper to load project_config.json if available."""
        config_path = os.path.join(self.project_dir, "project_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.tick_value = cfg.get("tick_value", 5.0)
                    self.tick_size = cfg.get("tick_size", 0.25)
                    self.bar_interval_seconds = cfg.get("bar_interval_seconds")
                    self.order_type = cfg.get("order_type", "market")
                    self.slippage_ticks = cfg.get("slippage_ticks", 1)
                    self.stop_slippage_ticks = cfg.get("stop_slippage_ticks", 2)
                    self.commission_per_side = cfg.get("commission_per_side", 2.05)
                    self.contract_multiplier = cfg.get("contract_multiplier", 1)
                    if "slippage_ticks" in cfg:
                        self.has_slippage_config = True
            except Exception as e:
                print(f"  [Quant] Warning: Failed to parse project_config.json: {e}")

    def set_recipe(self, recipe: dict):
        """Registers a strategy recipe for optimization."""
        self.recipe = recipe
        self.report.recipe = recipe

    def load_data(self) -> bool:
        """
        Detects and loads the primary CSV dataset from the project directory.
        Performs regime tagging and interval detection.
        """
        csv_files = glob.glob(os.path.join(self.project_dir, "*.csv"))
        if not csv_files:
            return False

        valid_file = None
        for f in csv_files:
            if "output" in f: continue
            try:
                cols = pd.read_csv(f, nrows=0).columns
                if 'close' in cols or 'Close' in cols:
                    valid_file = f
                    break
            except: continue
        
        if not valid_file: return False
        
        print(f"  [Quant] Loading data from {os.path.basename(valid_file)}...")
        self.df = pd.read_csv(valid_file)
        self.df.columns = [c.lower() for c in self.df.columns]
        
        if 'volume' not in self.df.columns:
            self.df['volume'] = 1.0
            
        self.df = tag_regimes(self.df)
        self._detect_interval_and_warn()
        
        self.report.slippage_ticks_used = self.slippage_ticks
        self.report.commission_per_rt = self.commission_per_side * 2
        self.report.data_found = True
        return True

    def _detect_interval_and_warn(self):
        """Warns the user if sub-minute intervals are used without slippage modeling."""
        if 'time' in self.df.columns:
            try:
                ts = pd.to_datetime(self.df['time'], unit='s', errors='coerce')
                diffs = ts.diff().dropna().dt.total_seconds()
                if not diffs.empty:
                    detected_interval = diffs.mode()[0]
                    if detected_interval <= 60 and (not self.has_slippage_config or self.slippage_ticks == 0):
                        print("⚠️  WARNING: Sub-minute bars detected with no slippage configured.")
                        print("   For HFT strategies, unmodeled slippage can invalidate results entirely.")
                        self.report.execution_realism_warning = True
            except: pass

    def generate_signals(self, df: pd.DataFrame, params: dict):
        """
        Two-tier signal generation — strategy agnostic.

        Tier 1 (preferred): Use pre-exported TradingView 'buy'/'sell' columns when
                            present in the data CSV.  These are the actual Pine Script
                            signals — 100% fidelity, no reconstruction needed.
        Tier 2 (fallback):  Reconstruct signals via strategy profile or archetype.
                            Used when no TV export is available.
        """
        buy_col  = next((c for c in df.columns if c.lower() in ('buy', 'buy_signal', 'long')), None)
        sell_col = next((c for c in df.columns if c.lower() in ('sell', 'sell_signal', 'short')), None)
        if buy_col and sell_col:
            return df[buy_col].fillna(0).astype(bool).values, df[sell_col].fillna(0).astype(bool).values

        # Tier 2
        profile = self.recipe.get("profile")
        if profile is not None:
            return self._generate_profile_signals(df, params, profile)
        return self._generate_fallback_signals(df, params)

    def _generate_profile_signals(self, df: pd.DataFrame, params: dict, profile: dict):
        """Internal helper for profile-driven signal generation."""
        import inspect
        from indicator_lib import INDICATOR_CATALOG
        computed = {}
        for name, spec in profile.get("indicators", {}).items():
            fn_name = spec.get("fn")
            fn = INDICATOR_CATALOG.get(fn_name)
            if not fn: continue
            
            resolved_params = {k: (params.get(v, v) if isinstance(v, str) else v) for k, v in spec.get("params", {}).items()}
            sig = inspect.signature(fn)
            kwargs = {p_name: (df if p_name == 'df' else df[p_name]) for p_name in sig.parameters if p_name == 'df' or p_name in df.columns}
            kwargs.update(resolved_params)
            computed[name] = fn(**kwargs)
            
        buy_signal = self._evaluate_conditions(df, params, computed, profile.get("entry_logic", {}).get("long", []))
        short_signal = self._evaluate_conditions(df, params, computed, profile.get("entry_logic", {}).get("short", []))
        return buy_signal.values, short_signal.values

    def _generate_fallback_signals(self, df: pd.DataFrame, params: dict):
        """Hardcoded archetype logic for legacy or manual strategies."""
        archetype = self.recipe.get("archetype", "UNKNOWN")
        if archetype == "TRAMA_HA_MOMENTUM":
            return self._logic_trama_ha(df, params)
        if archetype == "RSI_PERCENTR_EXHAUSTION":
            return self._logic_rsi_percentr(df, params)
        # Auto-detect: if RSI column exists, try RSI/PercentR exhaustion logic
        if 'rsi' in df.columns:
            return self._logic_rsi_percentr(df, params)
        # Default simple momentum fallback
        return (df['close'] > df['close'].shift(1)).values, (df['close'] < df['close'].shift(1)).values

    def _logic_rsi_percentr(self, df: pd.DataFrame, params: dict):
        """
        Reconstructs the RSI + Williams %R Exhaustion signal (fully vectorized).
        Mirrors Pine Script CM NQ strategy:
          - Long:  RSI crosses above SMA AND was recently oversold (within rsiLookback bars)
                   AND %R dual-timeframe oversold
          - Short: RSI crosses below SMA AND was recently overbought AND %R overbought
        """
        from indicator_lib import calc_rsi, calc_sma

        rsi_len      = int(params.get("rsiLen",        14))
        rsi_sma_len  = int(params.get("rsiSmaLength",  14))
        rsi_ob       = int(params.get("rsiObLevel",    68))
        rsi_os       = int(params.get("rsiOsLevel",    32))
        rsi_lookback = int(params.get("rsiLookback",   5))
        threshold    = int(params.get("threshold",     30))
        short_len    = int(params.get("shortLength",   21))
        long_len     = int(params.get("longLength",    112))
        cooldown     = int(params.get("signalCooldown", 30))

        close = df['close']

        # Use pre-exported RSI if available (Tier 1.5 — partial export)
        if 'rsi' in df.columns and df['rsi'].notna().sum() > 100:
            rsi = df['rsi'].ffill()
        else:
            rsi = calc_rsi(close, rsi_len)

        if 'rsi-based ma' in df.columns and df['rsi-based ma'].notna().sum() > 100:
            rsi_sma = df['rsi-based ma'].ffill()
        else:
            rsi_sma = calc_sma(rsi, rsi_sma_len)

        # Williams %R (short and long) — fully vectorized
        def percent_r(length):
            highest = close.rolling(length).max()
            lowest  = close.rolling(length).min()
            rng = (highest - lowest).clip(lower=1e-10)
            return 100.0 * (close - highest) / rng

        s_pr = percent_r(short_len)
        l_pr = percent_r(long_len)

        overbought = (s_pr >= -threshold) & (l_pr >= -threshold)
        oversold   = (s_pr <= -100 + threshold) & (l_pr <= -100 + threshold)

        # RSI crossovers — vectorized
        rsi_cross_above = (rsi > rsi_sma) & (rsi.shift(1) <= rsi_sma.shift(1))
        rsi_cross_below = (rsi < rsi_sma) & (rsi.shift(1) >= rsi_sma.shift(1))

        # "Was recently OB/OS within N bars" — rolling max on boolean = any() equivalent
        was_os = (rsi <= rsi_os).rolling(rsi_lookback + 1, min_periods=1).max().astype(bool)
        was_ob = (rsi >= rsi_ob).rolling(rsi_lookback + 1, min_periods=1).max().astype(bool)

        raw_long  = (rsi_cross_above & was_os & oversold).values
        raw_short = (rsi_cross_below & was_ob & overbought).values

        # Apply cooldown — vectorized via cumsum grouping
        n = len(df)
        buy  = np.zeros(n, dtype=bool)
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

        return buy, sell




    def _evaluate_conditions(self, df: pd.DataFrame, params: dict, computed: dict, conditions: list) -> pd.Series:
        """Evaluates a stack of conditions (indicators, price action) into a boolean series."""
        signal = pd.Series(True, index=df.index)
        for cond in conditions:
            c_type = cond.get("type")
            if c_type == "price_above_stack":
                for ind in cond.get("stack", []): signal &= (df['close'] > computed[ind])
            elif c_type == "price_below_stack":
                for ind in cond.get("stack", []): signal &= (df['close'] < computed[ind])
            elif c_type == "candle_streak":
                direction = cond.get("direction", "green")
                ha_df = computed.get(cond.get("source"))
                if ha_df is None: continue
                is_dir = (ha_df['ha_close'] > ha_df['ha_open']) if direction == "green" else (ha_df['ha_close'] < ha_df['ha_open'])
                streak = is_dir.astype(int).groupby(is_dir.ne(is_dir.shift()).cumsum()).cumsum()
                signal &= (streak >= int(params.get(cond.get("min", ""), 1)))
            elif c_type == "wick_filter":
                if not params.get(cond.get("enabled_by", ""), True): continue
                ha_df = computed.get(cond.get("source"))
                if ha_df is None: continue
                ha_body = (ha_df['ha_close'] - ha_df['ha_open']).abs()
                wick = (ha_df['ha_high'] - np.maximum(ha_df['ha_open'], ha_df['ha_close'])) if cond.get("side") == "upper" else (np.minimum(ha_df['ha_open'], ha_df['ha_close']) - ha_df['ha_low'])
                signal &= (wick <= ha_body)
        return signal

    def run_backtest(self, df: pd.DataFrame, buy: np.ndarray, short: np.ndarray, params: dict) -> dict:
        """
        Pure risk/exit simulator — intentionally strategy-agnostic.

        No indicator logic, no session filters, no volume gates live here.
        Entry signals arrive fully-formed from generate_signals().
        This layer handles only: trailing stops, fixed stops/targets, slippage,
        commissions, cooldown bars, and same-bar reversal guards.
        """
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        regimes = df['regime'].values if 'regime' in df.columns else np.array(['UNKNOWN'] * len(df))

        stop_ticks        = params.get("stop_ticks", 80)
        target_ticks      = params.get("target_ticks", 300)
        use_trail         = params.get("use_trail", True)
        trail_act         = params.get("trail_act", 20)
        trail_off         = params.get("trail_off", 4)
        min_bars_between  = params.get("min_bars_between", 3)
        allow_long        = params.get("allow_long", True)
        allow_short       = params.get("allow_short", True)
        block_same_bar_rev = params.get("block_same_bar_rev", True)

        sl_dist        = stop_ticks   * self.tick_size
        tp_dist        = target_ticks * self.tick_size
        act_dist       = trail_act    * self.tick_size
        off_dist       = trail_off    * self.tick_size
        slip_dist      = self.slippage_ticks      * self.tick_size
        stop_slip_dist = self.stop_slippage_ticks * self.tick_size

        trades, trade_regimes = [], []
        in_trade      = False
        trade_dir     = 0
        entry_p       = sl_p = tp_p = 0.0
        trail_active  = False
        last_exit     = -999
        last_exit_dir = 0
        entry_bar     = 0

        for i in range(len(close)):
            # ── Exit ─────────────────────────────────────────────────────────
            if in_trade:
                exit_pnl, triggered = 0.0, False
                if trade_dir == 1:
                    if use_trail and not trail_active and high[i] >= entry_p + act_dist:
                        trail_active, sl_p = True, high[i] - off_dist
                    if trail_active:
                        sl_p = max(sl_p, high[i] - off_dist)
                    if low[i] <= sl_p:
                        exit_pnl, triggered = (sl_p - stop_slip_dist) - entry_p, True
                    elif high[i] >= tp_p:
                        exit_pnl, triggered = tp_p - entry_p, True
                else:
                    if use_trail and not trail_active and low[i] <= entry_p - act_dist:
                        trail_active, sl_p = True, low[i] + off_dist
                    if trail_active:
                        sl_p = min(sl_p, low[i] + off_dist)
                    if high[i] >= sl_p:
                        exit_pnl, triggered = entry_p - (sl_p + stop_slip_dist), True
                    elif low[i] <= tp_p:
                        exit_pnl, triggered = entry_p - tp_p, True

                if triggered:
                    commission_rt = (self.commission_per_side * 2) / (entry_p * self.contract_multiplier)
                    trades.append(exit_pnl / entry_p - commission_rt)
                    trade_regimes.append(regimes[entry_bar])
                    last_exit_dir = trade_dir
                    in_trade      = False
                    last_exit     = i
                    continue

            # ── Entry (execution guards only — signals already filtered) ──────
            if not in_trade and (i - last_exit >= min_bars_between):
                same_bar  = (i == last_exit) and block_same_bar_rev
                can_long  = allow_long  and bool(buy[i])  and not (same_bar and last_exit_dir == 1)
                can_short = allow_short and bool(short[i]) and not (same_bar and last_exit_dir == -1)

                if can_long:
                    entry_p = close[i] + slip_dist if self.order_type != "limit" else close[i]
                    in_trade, trade_dir, trail_active = True, 1, False
                elif can_short:
                    entry_p = close[i] - slip_dist if self.order_type != "limit" else close[i]
                    in_trade, trade_dir, trail_active = True, -1, False

                if in_trade:
                    sl_p      = entry_p - sl_dist if trade_dir == 1 else entry_p + sl_dist
                    tp_p      = entry_p + tp_dist if trade_dir == 1 else entry_p - tp_dist
                    entry_bar = i

        if not trades:
            return {'sharpe': 0.0, 'wr': 0, 'count': 0, 'trades': np.array([]), 'regimes': []}
        t_arr  = np.array(trades)
        sharpe = (np.mean(t_arr) / np.std(t_arr)) * np.sqrt(252) if np.std(t_arr) > 0 else 0
        return {'sharpe': sharpe, 'wr': np.mean(t_arr > 0), 'count': len(t_arr), 'trades': t_arr, 'regimes': trade_regimes}

    def _precompute_filters(self, df, params):
        """
        Pre-computes all Pine Script entry filter arrays for a given df slice and param set.
        Filters: session window, TDV volume gate, HA wick quality, same-bar reversal.
        These are computed once per backtest call and applied in the simulation loop.
        """
        n = len(df)
        open_ = df['open'].values if 'open' in df.columns else df['close'].values
        high, low, close = df['high'].values, df['low'].values, df['close'].values

        # --- Filter 1: Session Time Window ---
        in_window = np.ones(n, dtype=bool)
        trade_eth = params.get('trade_eth', False)
        if not trade_eth and 'time' in df.columns:
            try:
                ts = pd.to_datetime(df['time'].values, unit='s', utc=True)
                ts_et = ts.tz_convert('America/New_York')
                hours = ts_et.hour + ts_et.minute / 60.0
                in_window = (hours >= 9.0) & (hours < 16.0)
            except Exception:
                pass  # If timezone conversion fails, allow all bars

        # --- Filter 2: TDV Volume Gate State Machine ---
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

        # Stateful locked state machine (mirrors Pine Script TDV logic)
        sig = raw_buy.astype(int)
        locked = np.ones(n, dtype=int)
        streak = np.zeros(n, dtype=int)
        for j in range(1, n):
            if is_weak[j] and sig[j] != sig[j-1]:
                sig[j] = sig[j-1]   # penalize weak-body state change: revert signal
                
            if sig[j] == sig[j-1]:
                streak[j] = streak[j-1] + 1
            else:
                streak[j] = 1
                
            locked[j] = sig[j] if streak[j] >= smooth_bars else locked[j-1]

        tdv_pos = locked == 1   # volume bullish
        tdv_neg = locked == 0   # volume bearish

        # --- Filter 3: HA Wick Quality Filter ---
        wick_long_ok = np.ones(n, dtype=bool)
        wick_short_ok = np.ones(n, dtype=bool)
        require_single_wick = params.get('require_single_wick', True)
        if require_single_wick and 'open' in df.columns:
            try:
                ha = calc_heikin_ashi(df)
                ha_body = (ha['ha_close'] - ha['ha_open']).abs().values
                ha_upper = (ha['ha_high'] - np.maximum(ha['ha_open'].values, ha['ha_close'].values)).values
                ha_lower = (np.minimum(ha['ha_open'].values, ha['ha_close'].values) - ha['ha_low'].values).values
                wick_long_ok = ha_upper <= ha_body
                wick_short_ok = ha_lower <= ha_body
            except Exception:
                pass

        # --- Filter 4: Session Sweep Filter ---
        use_sweep = params.get('use_sweep_filter', False)
        sweep_lookback = int(params.get('sweep_lookback', 20))
        sweep_long_ok = np.ones(n, dtype=bool)
        sweep_short_ok = np.ones(n, dtype=bool)
        
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
                    is_bull = False
                    is_bear = False
                    
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
                    
                sweep_long_ok = bars_since(bull_sweep) <= sweep_lookback
                sweep_short_ok = bars_since(bear_sweep) <= sweep_lookback
                
            except Exception as e:
                pass

        return {
            'in_window': in_window,
            'tdv_pos': tdv_pos,
            'tdv_neg': tdv_neg,
            'wick_long_ok': wick_long_ok,
            'wick_short_ok': wick_short_ok,
            'sweep_long_ok': sweep_long_ok,
            'sweep_short_ok': sweep_short_ok,
        }

    def run_backtest(self, df: pd.DataFrame, buy: np.ndarray, short: np.ndarray, params: dict) -> dict:
        """
        Simulates strategy execution with full Pine Script entry filter parity.
        Applies: session window, TDV volume gate, HA wick filter, same-bar reversal block,
        trailing stops, slippage, and commissions.
        """
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        regimes = df['regime'].values if 'regime' in df.columns else np.array(['UNKNOWN'] * len(df))

        # Strategy Parameters
        stop_ticks      = params.get("stop_ticks", 80)
        target_ticks    = params.get("target_ticks", 300)
        use_trail       = params.get("use_trail", True)
        trail_act       = params.get("trail_act", 20)
        trail_off       = params.get("trail_off", 4)
        min_bars_between = params.get("min_bars_between", 3)
        allow_long      = params.get("allow_long", True)
        allow_short     = params.get("allow_short", True)
        block_same_bar_rev = params.get("block_same_bar_rev", True)

        # Scaling
        sl_dist        = stop_ticks   * self.tick_size
        tp_dist        = target_ticks * self.tick_size
        act_dist       = trail_act    * self.tick_size
        off_dist       = trail_off    * self.tick_size
        slip_dist      = self.slippage_ticks       * self.tick_size
        stop_slip_dist = self.stop_slippage_ticks  * self.tick_size

        # Pre-compute all entry filters (the 5 missing Pine Script conditions)
        filt = self._precompute_filters(df, params)
        in_window     = filt['in_window']
        tdv_pos       = filt['tdv_pos']
        tdv_neg       = filt['tdv_neg']
        wick_long_ok  = filt['wick_long_ok']
        wick_short_ok = filt['wick_short_ok']
        sweep_long_ok = filt.get('sweep_long_ok', np.ones(len(df), dtype=bool))
        sweep_short_ok = filt.get('sweep_short_ok', np.ones(len(df), dtype=bool))

        trades, trade_regimes = [], []
        in_trade = False
        trade_dir = 0
        entry_p = sl_p = tp_p = 0.0
        trail_active = False
        last_exit = -999
        last_exit_dir = 0   # tracks direction of last closed trade for same-bar block
        entry_bar = 0

        for i in range(len(close)):
            # --- Exit Logic ---
            if in_trade:
                exit_pnl, triggered = 0.0, False
                if trade_dir == 1:
                    if use_trail and not trail_active and high[i] >= entry_p + act_dist:
                        trail_active, sl_p = True, high[i] - off_dist
                    if trail_active:
                        sl_p = max(sl_p, high[i] - off_dist)
                    if low[i] <= sl_p:
                        exit_pnl, triggered = (sl_p - stop_slip_dist) - entry_p, True
                    elif high[i] >= tp_p:
                        exit_pnl, triggered = tp_p - entry_p, True
                else:
                    if use_trail and not trail_active and low[i] <= entry_p - act_dist:
                        trail_active, sl_p = True, low[i] + off_dist
                    if trail_active:
                        sl_p = min(sl_p, low[i] + off_dist)
                    if high[i] >= sl_p:
                        exit_pnl, triggered = entry_p - (sl_p + stop_slip_dist), True
                    elif low[i] <= tp_p:
                        exit_pnl, triggered = entry_p - tp_p, True

                if triggered:
                    commission_rt = (self.commission_per_side * 2) / (entry_p * self.contract_multiplier)
                    trades.append(exit_pnl / entry_p - commission_rt)
                    trade_regimes.append(regimes[entry_bar])
                    last_exit_dir = trade_dir
                    in_trade = False
                    last_exit = i
                    continue

            # --- Entry Logic (Full Pine Script filter stack) ---
            if not in_trade and (i - last_exit >= min_bars_between):
                # Filter 4: Same-bar reversal block (mirrors Pine's block_same_bar_rev)
                same_bar = (i == last_exit) and block_same_bar_rev
                can_long  = (allow_long  and bool(buy[i])
                             and in_window[i] and tdv_pos[i] and wick_long_ok[i] and sweep_long_ok[i]
                             and not (same_bar and last_exit_dir == 1))
                can_short = (allow_short and bool(short[i])
                             and in_window[i] and tdv_neg[i] and wick_short_ok[i] and sweep_short_ok[i]
                             and not (same_bar and last_exit_dir == -1))

                if can_long:
                    entry_p = close[i] + slip_dist if self.order_type != "limit" else close[i]
                    in_trade, trade_dir, trail_active = True, 1, False
                elif can_short:
                    entry_p = close[i] - slip_dist if self.order_type != "limit" else close[i]
                    in_trade, trade_dir, trail_active = True, -1, False

                if in_trade:
                    sl_p = entry_p - sl_dist if trade_dir == 1 else entry_p + sl_dist
                    tp_p = entry_p + tp_dist if trade_dir == 1 else entry_p - tp_dist
                    entry_bar = i

        if not trades:
            return {'sharpe': 0.0, 'wr': 0, 'count': 0, 'trades': np.array([]), 'regimes': []}
        t_arr = np.array(trades)
        sharpe = (np.mean(t_arr) / np.std(t_arr)) * np.sqrt(252) if np.std(t_arr) > 0 else 0
        return {'sharpe': sharpe, 'wr': np.mean(t_arr > 0), 'count': len(t_arr), 'trades': t_arr, 'regimes': trade_regimes}

    def _compute_utility(self, trades: np.ndarray, stop_ticks: int, target_ticks: int) -> float:
        """
        Multi-metric utility score — designed to resist cherry-picking.

        Four components:
          1. Sortino ratio     — risk-adjusted return (CAPPED at 5.0)
          2. Drawdown factor   — penalizes large equity swings
          3. R:R bonus         — rewards stop:target ratios >= 3:1
          4. Frequency factor  — penalizes configurations that drastically
                                 reduce trade count (the cherry-picking guard)

        The Sortino cap is the critical fix. Without it, configs that produce
        zero losing trades in-sample yield Sortino = infinity, causing Optuna
        to pathologically hunt for ha_stack_min=8 style parameter combos that
        cherry-pick a handful of perfect trades rather than finding a robust edge.

        A Sortino of 5.0 is already exceptional by institutional standards.
        Anything above that in-sample is overfitting, not alpha.
        """
        if len(trades) < 30: return -1.0  # Raised from 20 for statistical significance

        mean_ret = np.mean(trades)
        downside  = np.std(trades[trades < 0]) if np.any(trades < 0) else 1e-9
        if downside == 0: downside = 1e-9

        # ── 1. Sortino (capped) ─────────────────────────────────────────────
        sortino = min((mean_ret / downside) * np.sqrt(252), 5.0)

        # ── 2. Max Drawdown ─────────────────────────────────────────────────
        cum  = np.cumprod(1 + trades)
        peak = np.maximum.accumulate(cum)
        max_dd = np.max((peak - cum) / peak)

        # ── 3. R:R Bonus ────────────────────────────────────────────────────
        rr_ratio = target_ticks / max(stop_ticks, 1)
        rr_bonus = min(rr_ratio / 3.0, 1.0)   # full bonus at 3:1 or better

        # ── 4. Frequency Factor (anti-cherry-pick guard) ────────────────────
        # Logarithmic scale so marginal trades still matter:
        #   30 trades → 0.63 | 60 → 0.80 | 100 → 0.87 | 200 → 1.0
        frequency_factor = min(np.log(len(trades)) / np.log(200), 1.0)

        return sortino * (1.0 - max_dd) * rr_bonus * frequency_factor


    def optimize(self, n_trials: int = 30):
        """
        Main optimization entry point. Runs 5-Fold Walk Forward Analysis.
        Fully strategy-agnostic — optimizes all signal + risk parameters
        for any Pine Script project.
        """
        all_params_def = self.recipe.get("optimizable_parameters", [])

        # Detect signal tier — affects parity check only
        buy_col  = next((c for c in self.df.columns if c.lower() in ('buy', 'buy_signal', 'long')), None)
        sell_col = next((c for c in self.df.columns if c.lower() in ('sell', 'sell_signal', 'short')), None)
        using_tv_signals = bool(buy_col and sell_col)
        if using_tv_signals:
            print("  [Quant] Tier 1: TradingView signal export detected.")
        else:
            print("  [Quant] Tier 2: No TV export. Optimizing all signal + risk parameters.")

        # 1. Parity Check
        default_params = {pr['name']: pr.get('default') for pr in all_params_def}
        b_def, s_def = self.generate_signals(self.df, default_params)
        self.report.parity_report = run_parity_check(self.df, b_def, s_def)

        # 2. Walk Forward Splits
        k_folds = 5
        chunk_size = len(self.df) // k_folds
        if chunk_size < 100: k_folds = 2

        fold_oos_sharpes, fold_is_sharpes, all_oos_trades, all_oos_regimes = [], [], [], []
        final_best = {}
        all_fold_params = []

        # 3. Optimization Loop
        for i in range(k_folds - 1):
            is_end = (i + 1) * chunk_size
            oos_end = (i + 2) * chunk_size if i < k_folds - 2 else len(self.df)
            df_is, df_oos = self.df.iloc[:is_end], self.df.iloc[is_end:oos_end]

            # Optimize all signal + risk params
            optimizable_params = [p for p in all_params_def if p.get('role') in ('signal', 'risk')]
            
            best_fold_params, best_fold_val = self._run_optuna_fold(df_is, all_params_def, optimizable_params, n_trials)
            all_fold_params.append(best_fold_params)
            fold_is_sharpes.append(best_fold_val)

            # OOS Validation
            b_o, s_o = self.generate_signals(df_oos, best_fold_params)
            res = self.run_backtest(df_oos, b_o, s_o, best_fold_params)
            fold_oos_sharpes.append(res['sharpe'])
            
            if res['count'] > 0:
                all_oos_trades.extend(res['trades'].tolist())
                all_oos_regimes.extend(res.get('regimes', []))
            
            # Log failures to RAG
            if fold_is_sharpes[-1] > 1.0 and res['sharpe'] < fold_is_sharpes[-1] * 0.3:
                log_failed_backtest(os.path.basename(self.project_dir), best_fold_params, res['sharpe'], "Severe OOS Degradation", {"is": fold_is_sharpes[-1], "fold": i})

        # Aggregate final params (median across folds)
        import statistics
        if all_fold_params:
            for k in all_fold_params[0].keys():
                vals = [fp[k] for fp in all_fold_params]
                if isinstance(vals[0], bool):
                    final_best[k] = sum(vals) > len(vals) / 2
                else:
                    try:
                        final_best[k] = type(vals[0])(statistics.median(vals))
                    except:
                        final_best[k] = vals[0]

        # 4. Final Report Aggregation
        return self._finalize_report(fold_is_sharpes, fold_oos_sharpes, all_oos_trades, all_oos_regimes, final_best)

    def _run_optuna_fold(self, df_is, all_params, opt_params, n_trials):
        """Helper to run a single Optuna study for a fold."""
        
        def objective(trial):
            p = {pr['name']: pr.get('default') for pr in all_params}
            for pr in opt_params:
                name = pr['name']
                min_val = pr.get('min', 0)
                max_val = pr.get('max', 100)
                
                if pr['type'] == 'int': p[name] = trial.suggest_int(name, min_val, max_val)
                elif pr['type'] == 'float': p[name] = trial.suggest_float(name, float(min_val), float(max_val))
                elif pr['type'] == 'bool': p[name] = trial.suggest_categorical(name, [True, False])
            
            b, s = self.generate_signals(df_is, p)
            res = self.run_backtest(df_is, b, s, p)
            return self._compute_utility(res['trades'], p.get("stop_ticks", 80), p.get("target_ticks", 300))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study.best_value

    def _finalize_report(self, is_vals, oos_vals, oos_trades, oos_regimes, best_params):
        """Helper to compute final performance metrics for the report."""
        rep = self.report
        rep.best_params = best_params
        rep.oos_sharpe = np.mean(oos_vals) if oos_vals else 0.0
        rep.in_sample_sharpe = np.mean(is_vals) if is_vals else 0.0
        rep.trade_count = len(oos_trades)
        rep.win_rate = float(np.mean(np.array(oos_trades) > 0)) if oos_trades else 0.0
        
        if oos_trades:
            oos_trades_arr = np.array(oos_trades)
            dd, luck = self.run_monte_carlo(oos_trades_arr)
            rep.mc_max_dd_95, rep.mc_luck_factor = dd, luck
            
            mean_ret = np.mean(oos_trades_arr)
            downside = np.std(oos_trades_arr[oos_trades_arr < 0]) if np.any(oos_trades_arr < 0) else 1e-9
            rep.sortino = (mean_ret / downside) * np.sqrt(252)
            
            cum = np.cumprod(1 + oos_trades_arr)
            peak = np.maximum.accumulate(cum)
            rep.max_drawdown = float(np.max((peak - cum) / peak))
            rep.utility_score = self._compute_utility(oos_trades_arr, best_params.get("stop_ticks", 80), best_params.get("target_ticks", 300))
            
            # Regime Bucketing
            df_r = pd.DataFrame({'pnl': oos_trades, 'regime': oos_regimes})
            rep.regime_performance = {r: {'trades': len(df_r[df_r.regime==r]), 'win_rate': float((df_r[df_r.regime==r].pnl > 0).mean())*100, 'avg_pnl': float(df_r[df_r.regime==r].pnl.mean())*100} for r in df_r.regime.unique()}
            
        rep.overfitting_risk = "LOW" if rep.oos_sharpe > rep.in_sample_sharpe * 0.5 else "HIGH"
        rep.wfa_consistency_score = (sum(1 for s in oos_vals if s > 0) / len(oos_vals)) * 100 if oos_vals else 0
        return rep

    def run_monte_carlo(self, trades: np.ndarray, simulations: int = 1000):
        """Simulates 1000 paths of the trade sequence to find 95% Var Drawdown."""
        if len(trades) < 5: return 0.0, 0.0
        sim_max_dds, sim_returns = [], []
        actual_cum = np.prod(1 + trades) - 1
        for _ in range(simulations):
            s_t = np.random.choice(trades, size=len(trades), replace=True)
            c_r = np.cumprod(1 + s_t)
            pk = np.maximum.accumulate(c_r)
            sim_max_dds.append(np.max((pk - c_r) / pk))
            sim_returns.append(c_r[-1] - 1)
        return float(np.percentile(sim_max_dds, 95)), float(np.sum(np.array(sim_returns) <= actual_cum) / simulations * 100)

    def _logic_trama_ha(self, df: pd.DataFrame, params: dict):
        """Hardcoded TRAMA + HA logic with Regime Filter and HA Rising/Falling logic."""
        from indicator_lib import calc_trama, calc_heikin_ashi
        t_f = int(params.get("trama_fast_len", 13))
        t_m = int(params.get("trama_med_len", 28))
        t_s = int(params.get("trama_slow_len", 40))
        tf = calc_trama(df['high'], df['low'], df['close'], t_f)
        tm = calc_trama(df['high'], df['low'], df['close'], t_m)
        ts = calc_trama(df['high'], df['low'], df['close'], t_s)
        
        ha = calc_heikin_ashi(df)
        ha_close = ha['ha_close']
        
        above = (ha_close > tf) & (ha_close > tm) & (ha_close > ts)
        below = (ha_close < tf) & (ha_close < tm) & (ha_close < ts)
        
        use_regime = params.get("use_regime_filter", False)
        cross_lb = int(params.get("cross_lookback", 4))
        p_bars = int(params.get("prior_regime_bars", 3))
        p_window = int(params.get("prior_regime_window", 20))
        
        # Calculate regime streak (+1 if above, -1 if below, else keep previous)
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
        
        ha_stack_min = int(params.get("ha_stack_min", 2))
        
        l_s_streak = ha_green.groupby(ha_green.ne(ha_green.shift()).cumsum()).cumsum()
        s_s_streak = ha_red.groupby(ha_red.ne(ha_red.shift()).cumsum()).cumsum()
        
        req_rising = params.get("require_rising_high", False)
        req_falling = params.get("require_falling_low", False)
        
        rising_highs = (ha['ha_high'].diff(1) >= 0).rolling(max(1, ha_stack_min - 1)).min().fillna(0).astype(bool)
        falling_lows = (ha['ha_low'].diff(1) <= 0).rolling(max(1, ha_stack_min - 1)).min().fillna(0).astype(bool)
            
        l_s = (l_s_streak >= ha_stack_min) & (rising_highs if req_rising else True)
        s_s = (s_s_streak >= ha_stack_min) & (falling_lows if req_falling else True)
        
        return (fresh_reversal_long & l_s & above).values, (fresh_reversal_short & s_s & below).values

def run_quant(project_dir, recipe, n_trials=30):
    """Entry point for the quant engine."""
    engine = QuantEngine(project_dir)
    engine.set_recipe(recipe)
    if not engine.load_data():
        return engine.report
    return engine.optimize(n_trials=n_trials)
