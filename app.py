import json
import os

# User database file
USER_DB = "user.json"

# Function to load users from JSON
def load_users():
    try:
        with open(USER_DB, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Authenticate user (Password visible)
def authenticate_user():
    users = load_users()
    username = input("Enter Student ID: ")
    password = input("Enter Password: ")  # Now the password is visible

    if username in users and users[username]["Password"] == password:
        print("\n✅ Authentication Successful!")
        return users[username]
    else:
        print("\n❌ Invalid credentials! Please try again.")
        return None

# Main CLI
def main():
    print("\n🎓 Welcome to Medical Education Translation AI 🎓")
    
    # Authenticate user
    user_data = None
    while user_data is None:
        user_data = authenticate_user()

    mother_tongue = user_data["Mother Tongue"]
    print(f"\n🌍 Your default translation language is set to: {mother_tongue}\n")

    # User Menu
    while True:
        print("\n📌 Select an option:")
        print("1️⃣ Elaborate & Translate (Main Program)")
        print("2️⃣ Create Slide Deck (Save Translations)")
        print("3️⃣ Exit")

        choice = input("\nEnter choice (1/2/3): ")

        if choice == "1":
            print("\nLaunching Elaboration & Translation...\n")
            os.system("python3 main.py")
        elif choice == "2":
            print("\nLaunching Slide Deck Maker...\n")
            os.system("python3 slidedeck.py")
        elif choice == "3":
            print("\n👋 Exiting... Thank you!")
            break
        else:
            print("\n❌ Invalid choice! Please select a valid option.")

if __name__ == "__main__":
    main()
