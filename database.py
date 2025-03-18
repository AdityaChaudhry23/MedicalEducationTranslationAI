import sqlite3

# Connect to SQLite (Creates 'mededai.db' if it doesn’t exist)
conn = sqlite3.connect("mededai.db")
cursor = conn.cursor()

# ✅ Enable Foreign Keys for proper relationships
cursor.execute("PRAGMA foreign_keys = ON;")

# ✅ Create Users Table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        mother_tongue TEXT NOT NULL
    )
''')

# ✅ Create Slide Deck Table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS slide_deck (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        input_word TEXT NOT NULL,
        input_word_translation TEXT NOT NULL,
        meaning_english TEXT NOT NULL,
        meaning_translation TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
''')

# ✅ Create History Table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        input TEXT NOT NULL,
        elaboration TEXT NOT NULL,
        output TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
''')

# Commit & Close Connection
conn.commit()
conn.close()

print("✅ SQLite Database Created Successfully!")
