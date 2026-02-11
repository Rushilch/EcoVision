import sqlite3

DB_NAME = "ecovision.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT,
        module TEXT,
        timestamp TEXT,
        value REAL
    )
    """)

    conn.commit()
    conn.close()

def save_prediction(city, module, timestamp, value):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (city, module, timestamp, value) VALUES (?, ?, ?, ?)",
        (city, module, timestamp, value)
    )
    conn.commit()
    conn.close()
