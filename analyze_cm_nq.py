"""
CM NQ Strategy Analyzer
========================
Uses actual TradingView trade results (with MFE/MAE) to:
1. Report baseline performance statistics
2. Simulate different stop/target combinations on the actual trade history
3. Run Monte Carlo stress testing on the best configuration
"""

import csv
import numpy as np
import pandas as pd
from itertools import product

TICK_VALUE = 5.0       # $5 per tick for NQ
TICK_SIZE  = 0.25      # 0.25 points per tick
COMMISSION = 4.10      # round-trip commission per trade

# ── Load Trades ───────────────────────────────────────────────────────────────
def load_trades(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    df = pd.DataFrame(rows)
    exits = df[df['Type'].str.lower().str.contains('exit')].copy()

    exits['pnl']  = pd.to_numeric(exits['Net P&L USD'],            errors='coerce')
    exits['mfe']  = pd.to_numeric(exits['Favorable excursion USD'], errors='coerce')
    exits['mae']  = pd.to_numeric(exits['Adverse excursion USD'],   errors='coerce').abs()
    exits['dir']  = exits['Type'].str.lower().str.contains('long').map({True: 1, False: -1})
    exits = exits.dropna(subset=['pnl', 'mfe', 'mae'])
    return exits.reset_index(drop=True)

# ── Simulate Exit Parameters on Actual Trade History ────────────────────────
def simulate_exits(exits, tp_ticks, sl_ticks):
    """
    Uses MFE/MAE to re-simulate outcomes with different stop/target.
    - If MAE >= sl_dist  → stopped out at -sl_dist - commission
    - Elif MFE >= tp_dist → target hit at +tp_dist - commission
    - Else               → original exit (held to close) - commission
    """
    tp_dist = tp_ticks * TICK_SIZE * 20   # NQ point value: 1pt = $20
    sl_dist = sl_ticks * TICK_SIZE * 20

    pnls = []
    for _, row in exits.iterrows():
        mae = row['mae']
        mfe = row['mfe']
        orig = row['pnl']

        if mae >= sl_dist:
            pnl = -sl_dist - COMMISSION
        elif mfe >= tp_dist:
            pnl = tp_dist - COMMISSION
        else:
            pnl = orig - COMMISSION  # original exit, adjust for commission
        pnls.append(pnl)
    return np.array(pnls)

# ── Stats ─────────────────────────────────────────────────────────────────────
def calc_stats(pnls):
    total = len(pnls)
    if total == 0:
        return {}
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    net    = pnls.sum()
    wr     = len(wins) / total * 100
    pf     = wins.sum() / abs(losses.sum()) if len(losses) > 0 else float('inf')
    avg_w  = wins.mean() if len(wins) > 0 else 0
    avg_l  = losses.mean() if len(losses) > 0 else 0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    rets = pnls / 10000  # normalize to % of $10k capital
    mean_r = np.mean(rets)
    down_r = np.std(rets[rets < 0]) if np.any(rets < 0) else 1e-9
    sortino = (mean_r / down_r) * np.sqrt(252) if down_r > 0 else 0

    return {
        'trades': total, 'wr': wr, 'net': net, 'pf': pf,
        'avg_win': avg_w, 'avg_loss': avg_l, 'max_dd': max_dd, 'sortino': sortino
    }

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def monte_carlo(pnls, simulations=1000):
    sim_max_dds = []
    sim_finals  = []
    actual_final = pnls.sum()
    for _ in range(simulations):
        shuffled = np.random.choice(pnls, size=len(pnls), replace=True)
        cum = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cum)
        sim_max_dds.append((peak - cum).max())
        sim_finals.append(cum[-1])
    p95_dd   = np.percentile(sim_max_dds, 95)
    luck_pct = np.mean(np.array(sim_finals) <= actual_final) * 100
    return p95_dd, luck_pct

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    path = 'data/projects/CM NQ/NQ_Futures_CME_MINI_NQ1!_2026-05-03.csv'
    exits = load_trades(path)

    # ── Baseline (original TradingView results) ──
    orig_pnls = exits['pnl'].values - COMMISSION
    base = calc_stats(orig_pnls)
    print("=" * 65)
    print("CM NQ STRATEGY — BASELINE (TradingView results)")
    print("=" * 65)
    print(f"  Trades:     {base['trades']:,}")
    print(f"  Win Rate:   {base['wr']:.1f}%")
    print(f"  Net P&L:    ${base['net']:,.0f}")
    print(f"  Profit Fac: {base['pf']:.3f}")
    print(f"  Avg Win:    ${base['avg_win']:.0f}")
    print(f"  Avg Loss:   ${base['avg_loss']:.0f}")
    print(f"  Max DD:     ${base['max_dd']:,.0f}")
    print(f"  Sortino:    {base['sortino']:.2f}")

    # ── Grid Search: TP × SL ──
    tp_range = [50, 75, 100, 125, 140, 175, 200, 250]  # ticks
    sl_range = [30, 50, 75, 100, 125, 150]              # ticks

    print(f"\n{'='*65}")
    print("STOP/TARGET GRID SEARCH (simulated on actual MFE/MAE data)")
    print(f"{'='*65}")
    print(f"{'TP':>5} {'SL':>5} | {'Trades':>7} {'WR':>7} {'Net P&L':>12} {'PF':>7} {'MaxDD':>10} {'Sortino':>9}")
    print("-" * 65)

    best_score = -999
    best_config = None

    for tp, sl in product(tp_range, sl_range):
        if tp <= sl:
            continue
        pnls = simulate_exits(exits, tp, sl)
        s = calc_stats(pnls)
        if s['trades'] < 10:
            continue
        # Score = Sortino × PF × (1 - DD_pct)
        dd_pct = s['max_dd'] / 10000
        score = s['sortino'] * min(s['pf'], 5.0) * (1 - dd_pct)
        if score > best_score:
            best_score = score
            best_config = (tp, sl, s, pnls)
        print(f"{tp:>5} {sl:>5} | {s['trades']:>7,} {s['wr']:>6.1f}% {s['net']:>12,.0f} {s['pf']:>7.3f} {s['max_dd']:>10,.0f} {s['sortino']:>9.2f}")

    # ── Best Config Deep Dive ──
    if best_config:
        tp, sl, s, pnls = best_config
        print(f"\n{'='*65}")
        print(f"BEST CONFIG: TP={tp} ticks | SL={sl} ticks  (Score: {best_score:.3f})")
        print(f"{'='*65}")
        for k, v in s.items():
            if isinstance(v, float): print(f"  {k}: {v:.2f}")
            else: print(f"  {k}: {v:,}")

        # ── Monte Carlo ──
        print(f"\n{'='*65}")
        print("MONTE CARLO STRESS TEST (1,000 simulations on best config)")
        print(f"{'='*65}")
        p95_dd, luck = monte_carlo(pnls)
        print(f"  95th percentile Max Drawdown: ${p95_dd:,.0f}")
        print(f"  Luck Factor (median rank):    {luck:.0f}th percentile")
        if luck < 50:
            print("  >> Performance is ABOVE median expectation (skill-driven)")
        else:
            print("  >> Performance is near/below median (may include luck)")
