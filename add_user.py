import sqlite3

def add_user(username, password, mother_tongue):
    """Adds a new user to the SQLite database."""
    conn = sqlite3.connect("mededai.db")
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO users (username, password, mother_tongue) VALUES (?, ?, ?)", 
                       (username, password, mother_tongue))
        conn.commit()
        print(f"✅ User '{username}' added successfully!")
    except sqlite3.IntegrityError:
        print(f"⚠️ Username '{username}' already exists!")

    conn.close()


