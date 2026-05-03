import sqlite3
import os
import json

# Use the directory containing this file as the root for the engine
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ALPHA_DB_PATH = os.path.join(DATA_DIR, "alpha_library.db")
LABS_DB_PATH = os.path.join(DATA_DIR, "internal_labs.db")

def init_optimization_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Alpha DB (Knowledge)
    conn = sqlite3.connect(ALPHA_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            source TEXT,
            verified INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    
    # Labs DB (Operational)
    conn = sqlite3.connect(LABS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def seed_pine_bug_knowledge():
    """Seeds the knowledge base with common Pine Script anti-patterns if they don't exist."""
    init_optimization_db()
    conn = sqlite3.connect(ALPHA_DB_PATH)
    cursor = conn.cursor()
    
    # Check if already seeded
    cursor.execute("SELECT COUNT(*) FROM knowledge WHERE category = 'pine_bugs'")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    bugs = [
        {
            "content": "Using lookahead=barmerge.lookahead_on in request.security for historical checks causes repainting because it looks into the future. Always use lookahead=barmerge.lookahead_off for live trading signals.",
            "tags": "pine, repainting, security, lookahead"
        },
        {
            "content": "Executing buy/sell signals without ensuring barstate.isconfirmed == true can lead to repainting during the real-time bar.",
            "tags": "pine, repainting, barstate"
        },
        {
            "content": "When pulling data like ADD or VIX via request.security on a sub-minute timeframe (e.g., 30s), ensure that the data provider updates frequently enough, otherwise stale data may trigger false signals.",
            "tags": "pine, mtf, data_feed"
        }
    ]
    
    for bug in bugs:
        cursor.execute(
            "INSERT INTO knowledge (category, content, tags, source) VALUES (?, ?, ?, ?)",
            ('pine_bugs', bug["content"], bug["tags"], 'optimization_engine_Seed')
        )
        
    conn.commit()
    conn.close()

def log_failed_backtest(ticker: str, params: dict, sharpe: float, reason: str, metrics: dict = None):
    init_optimization_db()
    conn = sqlite3.connect(LABS_DB_PATH)
    cursor = conn.cursor()
    
    data = {
        "params": params,
        "sharpe": sharpe,
        "reason": reason,
        "metrics": metrics or {}
    }
    content = json.dumps(data)
    
    cursor.execute(
        "INSERT INTO labs (category, content, tags, source) VALUES (?, ?, ?, ?)",
        ('failed_backtest', content, f"{ticker}, backtest, failure", 'Optimization_Quant')
    )
    
    conn.commit()
    conn.close()

def query_pine_bugs(tag: str = "pine") -> list[str]:
    init_optimization_db()
    conn = sqlite3.connect(ALPHA_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT content FROM knowledge WHERE category = 'pine_bugs' AND tags LIKE ?",
        (f"%{tag}%",)
    )
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results

def query_prior_optimizations(ticker: str) -> list[str]:
    init_optimization_db()
    conn = sqlite3.connect(LABS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT content FROM labs WHERE category = 'failed_backtest' AND tags LIKE ?",
        (f"%{ticker}%",)
    )
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    return results
