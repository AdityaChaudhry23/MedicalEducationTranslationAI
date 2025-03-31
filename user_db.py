from tinydb import TinyDB, Query
import os

# Ensure database directory exists
os.makedirs("db", exist_ok=True)

# Initialize databases
users_db = TinyDB("db/users.json")
history_db = TinyDB("db/history.json")
slides_db = TinyDB("db/slides.json")

User = Query()

# -------------------------------------
# 🔐 AUTHENTICATION
# -------------------------------------

def register_user(user_id: str, name: str, password: str, mother_tongue: str):
    if not users_db.contains(User.id == user_id):
        users_db.insert({
            "id": user_id,
            "name": name,
            "password": password,  # 🔐 (In production, use hashing!)
            "mother_tongue": mother_tongue
        })

def authenticate(user_id: str, password: str):
    user = users_db.get(User.id == user_id)
    if user and user["password"] == password:
        return user
    return None

# -------------------------------------
# 📘 HISTORY
# -------------------------------------

def add_history(user_id: str, input_text: str, elaboration: str, output: str):
    history_db.insert({
        "user_id": user_id,
        "input": input_text,
        "elaboration": elaboration,
        "output": output
    })

def get_user_history(user_id: str):
    return history_db.search(Query().user_id == user_id)

# -------------------------------------
# 📚 SLIDES / DICTIONARY
# -------------------------------------

def add_slide(user_id: str, input_text: str, input_translated: str, elaboration: str, translated_elaboration: str):
    slides_db.insert({
        "user_id": user_id,
        "input": input_text,
        "input_translated": input_translated,
        "elaboration": elaboration,
        "translated_elaboration": translated_elaboration
    })

def get_user_slides(user_id: str):
    return slides_db.search(Query().user_id == user_id)
# -------------------------------------
# 👤 USER FETCH
# -------------------------------------

def get_user(user_id: str):
    return users_db.get(User.id == user_id)