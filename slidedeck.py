import json
from elaborate import elaborate_text
from translate import translate_to_hindi

# 📌 File to store vocabulary deck
SLIDEDECK_FILE = "slidedeck.json"

def save_to_slidedeck(word):
    """
    Saves the given medical term, its Hindi translation, elaboration, and elaboration translation to slidedeck.json.
    """
    print(f"\n📖 Processing: {word}")

    # 🔹 Step 1: Translate the Word
    word_translation = translate_to_hindi(word)
    print(f"🌍 Translation: {word} → {word_translation}")

    # 🔹 Step 2: Elaborate the Word
    elaboration = elaborate_text(word)
    print(f"🩺 Elaboration: {elaboration}")

    # 🔹 Step 3: Translate the Elaboration
    elaboration_translation = translate_to_hindi(elaboration)
    print(f"🇮🇳 Hindi Meaning: {elaboration_translation}")

    # 🔹 Step 4: Store Data
    entry = {
        "input_word": word,
        "input_word_translation": word_translation,
        "meaning": elaboration,
        "meaning_translation": elaboration_translation
    }

    # Load existing data
    try:
        with open(SLIDEDECK_FILE, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Append new entry
    existing_data.append(entry)

    # Save updated data
    with open(SLIDEDECK_FILE, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4, ensure_ascii=False)

    print("\n✅ Saved to slidedeck.json!")

# 🔹 Run standalone
if __name__ == "__main__":
    user_input = input("Enter a medical term: ")
    save_to_slidedeck(user_input)