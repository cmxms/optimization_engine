import csv
from datetime import datetime


def analyze(path, label):
    exits = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Type', '').startswith('Exit'):
                try:
                    exits.append(float(row['Net P&L USD']))
                except:
                    pass
    total = len(exits)
    if total == 0:
        print(label + ': No trades found')
        return
    wins = [p for p in exits if p > 0]
    losses = [p for p in exits if p < 0]
    net = sum(exits)
    wr = len(wins) / total * 100
    pf = sum(wins) / abs(sum(losses)) if losses else float('inf')
    cum = 0; peak = 0; max_dd = 0
    for p in exits:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if max_dd < dd: max_dd = dd
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    max_cl = 0; cur = 0
    for p in exits:
        if p < 0:
            cur += 1; max_cl = max(max_cl, cur)
        else:
            cur = 0
    print(label)
    print(f"  Trades: {total} | WR: {wr:.1f}% | Net: ${net:,.0f} | PF: {pf:.3f} | MaxDD: ${max_dd:,.0f} | AvgW: ${avg_w:.0f} | AvgL: ${avg_l:.0f} | MaxConsecL: {max_cl}")


def monthly_breakdown(path, label):
    monthly = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Type', '').startswith('Exit'):
                try:
                    raw = row['Date and time'].strip()
                    dt = datetime.strptime(raw, '%Y-%m-%d %H:%M')
                    key = dt.strftime('%Y-%m')
                    pnl = float(row['Net P&L USD'])
                    if key not in monthly:
                        monthly[key] = {'pnl': 0, 'w': 0, 't': 0}
                    monthly[key]['pnl'] += pnl
                    monthly[key]['t'] += 1
                    if pnl > 0:
                        monthly[key]['w'] += 1
                except:
                    pass
    print(f"\n=== {label} Monthly Breakdown ===")
    for k in sorted(monthly.keys()):
        m = monthly[k]
        wr = m['w'] / m['t'] * 100 if m['t'] else 0
        pnl = m['pnl']
        trades = m['t']
        print(f"  {k}: {trades:3d} trades | WR {wr:5.1f}% | P&L ${pnl:>10,.0f}")


print("=" * 65)
print("FULL VERSION COMPARISON")
print("=" * 65)
analyze('data/Test Results/V5 Results/TD_NQ_Bot_v5_CME_MINI_NQ1!_2026-05-03.csv', 'V5 (Best so far)')
analyze('data/Test Results/V8 Results/TD_NQ_Bot_v8_CME_MINI_NQ1!_2026-05-03.csv', 'V8 (Latest - with fixes)')
analyze('data/projects/TD_NQ_Bot/TD_NQ_Bot_TEST_CME_MINI_NQ1!_2026-05-03.csv', 'Original Baseline')

monthly_breakdown('data/Test Results/V5 Results/TD_NQ_Bot_v5_CME_MINI_NQ1!_2026-05-03.csv', 'V5')
monthly_breakdown('data/Test Results/V8 Results/TD_NQ_Bot_v8_CME_MINI_NQ1!_2026-05-03.csv', 'V8')
