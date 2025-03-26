from tinydb import TinyDB, Query
import os

# Ensure DB folder exists
os.makedirs("db", exist_ok=True)

db = TinyDB("db/users.json")
User = Query()

def register_user(username):
    if not db.search(User.username == username):
        db.insert({"username": username, "history": []})
        return True
    return False

def user_exists(username):
    return db.contains(User.username == username)

def add_history(username, entry):
    if not user_exists(username):
        register_user(username)

    user_data = db.search(User.username == username)[0]
    updated_history = user_data["history"] + [entry]
    db.update({"history": updated_history}, User.username == username)

def get_history(username):
    if user_exists(username):
        return db.search(User.username == username)[0]["history"]
    return []

def clear_history(username):
    if user_exists(username):
        db.update({"history": []}, User.username == username)
