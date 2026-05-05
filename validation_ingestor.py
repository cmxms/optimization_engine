import os
import glob
import sqlite3
import pandas as pd
from rag import LABS_DB_PATH, ROOT_DIR, init_optimization_db

def run_ingestor():
    init_optimization_db()
    print("\n[Step 0/8] Checking for TV validations in project output folders...")
    conn = sqlite3.connect(LABS_DB_PATH)
    cursor = conn.cursor()
    
    projects_dir = os.path.join(ROOT_DIR, "data", "projects")
    csv_files = glob.glob(os.path.join(projects_dir, "*", "output", "*.csv"))
    
    ingested_count = 0
    for csv_file in csv_files:
        # Check if we already ingested this exact file (use filename + size as a rough hash, or just filename)
        filename = os.path.basename(csv_file)
        strategy_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
        
        # We can store the filename in a "source" column or check if we already have it.
        # Let's add a quick check if this filename was already ingested today.
        # To be safe, we'll check if any record exists with this strategy name and run_date matching today.
        # Actually, simpler: we'll check if the DB already has this file in the 'source' column.
        # Since I didn't add 'source' to validated_runs, I'll alter it quickly if missing.
        try:
            cursor.execute("ALTER TABLE validated_runs ADD COLUMN source_csv TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass # Column already exists
            
        cursor.execute("SELECT COUNT(*) FROM validated_runs WHERE source_csv = ?", (filename,))
        if cursor.fetchone()[0] > 0:
            continue # Already ingested
            
        try:
            df = pd.read_csv(csv_file)
            
            # Simple TV CSV detection
            pnl_cols = df.filter(regex=r'Net P&L', axis=1).columns
            if 'Trade #' in df.columns and len(pnl_cols) > 0:
                pnl_col = pnl_cols[0]
                # TV Export format
                # Filter for Exit trades
                exits = df[df['Type'].str.contains('Exit', case=False, na=False)]
                tv_trades = len(exits)
                
                if tv_trades > 0:
                    wins = exits[exits[pnl_col] > 0]
                    tv_win_rate = (len(wins) / tv_trades) * 100.0
                    tv_net_pnl = exits[pnl_col].sum()
                    
                    cursor.execute("""
                        INSERT INTO validated_runs (strategy_name, tv_trades, tv_win_rate, tv_net_pnl, source_csv)
                        VALUES (?, ?, ?, ?, ?)
                    """, (strategy_name, tv_trades, tv_win_rate, tv_net_pnl, filename))
                    conn.commit()
                    print(f"  -> Ingested TV validation for {strategy_name}: {tv_trades} trades, {tv_win_rate:.1f}% WR, ${tv_net_pnl:,.2f} P&L")
                    ingested_count += 1
        except Exception as e:
            print(f"  -> Failed to ingest {filename}: {e}")
            
    conn.close()
    if ingested_count == 0:
        print("  -> No new TV validations found.")

if __name__ == "__main__":
    run_ingestor()
