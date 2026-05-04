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
from filter_compiler import compile_filters

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
        Generates buy/sell signals using the Python indicator library (always Tier 2).

        TradingView-exported signal columns (buy/sell) are intentionally NOT used here.
        They are used exclusively by the Parity Checker to verify that the Python math
        matches the Pine Script math before optimization begins.

        If the IR recipe has no recognized signal_logic mapping, returns empty signals.
        """
        profile = self.recipe.get("signal_logic")
        if profile and profile.get("indicators"):
            return self._generate_profile_signals(df, params, profile)

        # No signal logic in IR — return empty signals
        # This should not normally happen after the overhaul: the IR Builder should
        # always produce a signal_logic block. If this fires, check ir_builder.py.
        print("  [Quant] WARNING: No signal_logic in recipe. Returning empty signals.")
        return np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)

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


    def _evaluate_conditions(self, df: pd.DataFrame, params: dict, computed: dict, conditions: list) -> pd.Series:
        """Evaluates a stack of conditions (indicators, price action) into a boolean series."""
        signal = pd.Series(True, index=df.index)
        for cond in conditions:
            c_type = cond.get("type")
            if c_type == "boolean_series":
                src = computed.get(cond.get("source"))
                if src is not None:
                    col = cond.get("column")
                    if col and hasattr(src, "columns") and col in src.columns:
                        signal &= src[col].astype(bool)
                    else:
                        signal &= src.astype(bool)
            elif c_type == "price_above_stack":
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



    def run_backtest(self, df: pd.DataFrame, buy: np.ndarray, short: np.ndarray, params: dict, filter_masks: dict = None) -> dict:
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

        if filter_masks is None:
            n = len(df)
            filter_masks = {k: np.ones(n, dtype=bool) for k in ['in_window', 'tdv_pos', 'tdv_neg', 'wick_long_ok', 'wick_short_ok', 'sweep_long_ok', 'sweep_short_ok']}

        in_window     = filter_masks.get('in_window', np.ones(len(df), dtype=bool))
        tdv_pos       = filter_masks.get('tdv_pos', np.ones(len(df), dtype=bool))
        tdv_neg       = filter_masks.get('tdv_neg', np.ones(len(df), dtype=bool))
        wick_long_ok  = filter_masks.get('wick_long_ok', np.ones(len(df), dtype=bool))
        wick_short_ok = filter_masks.get('wick_short_ok', np.ones(len(df), dtype=bool))
        sweep_long_ok = filter_masks.get('sweep_long_ok', np.ones(len(df), dtype=bool))
        sweep_short_ok = filter_masks.get('sweep_short_ok', np.ones(len(df), dtype=bool))

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

        Full-variable optimization: ALL non-display parameters are optimized every run.
        The Parity Gate is enforced before Optuna starts — if parity fails (blocking=True),
        optimization aborts immediately with a clear error.
        """
        all_params_def = self.recipe.get("optimizable_parameters", [])

        # --- Parity Gate ---
        # Generate signals with default params to run the parity check.
        default_params = {pr['name']: pr.get('default') for pr in all_params_def}
        b_def, s_def = self.generate_signals(self.df, default_params)
        self.report.parity_report = run_parity_check(self.df, b_def, s_def)

        if self.report.parity_report.blocking:
            print("\n" + "="*60)
            print("  ERROR: PARITY GATE FAILED — OPTIMIZATION ABORTED")
            print("="*60)
            print(f"  Fidelity: {self.report.parity_report.fidelity_score*100:.1f}% (Required: 95%)")
            print(f"  Analysis: {self.report.parity_report.drift_analysis}")
            print("="*60 + "\n")
            import sys
            sys.exit(1)

        if self.report.parity_report.available:
            print(f"  [Quant] Parity Gate: PASSED ({self.report.parity_report.fidelity_score*100:.1f}%)")
        else:
            print("  [Quant] Parity Gate: NOT_VERIFIABLE (no TV signal export in CSV — proceeding unverified)")

        # --- Full-Variable Optimization ---
        # Optimize ALL non-display parameters — signal, risk, and filter roles.
        # Display params (colors, labels, session strings) are never passed to Optuna.
        optimizable_params = [p for p in all_params_def if p.get('role') != 'display']
        print(f"  [Quant] Optimizing {len(optimizable_params)} parameters "
              f"({len([p for p in optimizable_params if p.get('role')=='signal'])} signal, "
              f"{len([p for p in optimizable_params if p.get('role')=='risk'])} risk, "
              f"{len([p for p in optimizable_params if p.get('role') not in ('signal','risk')])} filter).")

        # 5-Fold Walk-Forward splits
        k_folds = 5
        chunk_size = len(self.df) // k_folds
        if chunk_size < 100:
            k_folds = 2

        fold_oos_sharpes, fold_is_sharpes, all_oos_trades, all_oos_regimes = [], [], [], []
        final_best = {}
        all_fold_params = []

        for i in range(k_folds - 1):
            is_end  = (i + 1) * chunk_size
            oos_end = (i + 2) * chunk_size if i < k_folds - 2 else len(self.df)
            df_is, df_oos = self.df.iloc[:is_end], self.df.iloc[is_end:oos_end]

            best_fold_params, best_fold_val = self._run_optuna_fold(
                df_is, all_params_def, optimizable_params, n_trials
            )
            all_fold_params.append(best_fold_params)
            fold_is_sharpes.append(best_fold_val)

            # OOS Validation
            b_o, s_o = self.generate_signals(df_oos, best_fold_params)
            f_o = compile_filters(df_oos, best_fold_params, self.recipe.get("filters", []))
            res = self.run_backtest(df_oos, b_o, s_o, best_fold_params, filter_masks=f_o)
            fold_oos_sharpes.append(res['sharpe'])

            if res['count'] > 0:
                all_oos_trades.extend(res['trades'].tolist())
                all_oos_regimes.extend(res.get('regimes', []))

            # Log severe OOS degradation to RAG
            if fold_is_sharpes[-1] > 1.0 and res['sharpe'] < fold_is_sharpes[-1] * 0.3:
                log_failed_backtest(
                    os.path.basename(self.project_dir), best_fold_params,
                    res['sharpe'], "Severe OOS Degradation",
                    {"is": fold_is_sharpes[-1], "fold": i},
                )

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
                    except Exception:
                        final_best[k] = vals[0]

        return self._finalize_report(
            fold_is_sharpes, fold_oos_sharpes, all_oos_trades, all_oos_regimes, final_best
        )

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
                elif pr['type'] == 'string': p[name] = pr.get('default')
            
            b, s = self.generate_signals(df_is, p)
            f_masks = compile_filters(df_is, p, self.recipe.get("filters", []))
            res = self.run_backtest(df_is, b, s, p, filter_masks=f_masks)
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



def run_quant(project_dir, recipe, n_trials=30):
    """Entry point for the quant engine."""
    engine = QuantEngine(project_dir)
    engine.set_recipe(recipe)
    if not engine.load_data():
        return engine.report
    return engine.optimize(n_trials=n_trials)
