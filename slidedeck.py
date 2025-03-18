import json
import torch  # 🔹 Added torch import
from elaborate import elaborate_text
from translate import batch_translate, initialize_model_and_tokenizer
from IndicTransToolkit.processor import IndicProcessor

# 📌 File to store vocabulary deck
SLIDEDECK_FILE = "slidedeck.json"

# 📌 Define available language mapping
INDIAN_LANGUAGES = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

# 📌 Default configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None  # Adjust for "4-bit" or "8-bit" if needed
SRC_LANG = "eng_Latn"  # English is fixed as source language

# Initialize Model & Tokenizer
ckpt_dir = "/workspace/MedicalEducationTranslationAI/Models/indictrans2"
tokenizer, model = initialize_model_and_tokenizer(ckpt_dir, quantization)
ip = IndicProcessor(inference=True)

def save_to_slidedeck(word, output_language):
    """
    Saves the given medical term, its translation, elaboration, and elaboration translation to slidedeck.json.
    """
    print(f"\n📖 Processing: {word}")

    # 🔹 Validate output language
    if output_language not in INDIAN_LANGUAGES:
        print(f"⚠️ Error: {output_language} is not supported. Please choose a valid language.")
        return

    # 🔹 Define target language code
    tgt_lang = INDIAN_LANGUAGES[output_language]

    # 🔹 Step 1: Translate the Word
    word_translation = batch_translate([word], SRC_LANG, tgt_lang, model, tokenizer, ip)[0]
    print(f"🌍 Translation ({output_language}): {word} → {word_translation}")

    # 🔹 Step 2: Elaborate the Word
    elaboration = elaborate_text(word)
    print(f"🩺 Elaboration: {elaboration}")

    # 🔹 Step 3: Translate the Elaboration
    elaboration_translation = batch_translate([elaboration], SRC_LANG, tgt_lang, model, tokenizer, ip)[0]
    print(f"🌍 Meaning Translation ({output_language}): {elaboration_translation}")

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
    output_language = input("Select Output Language: ")
    save_to_slidedeck(user_input, output_language)
