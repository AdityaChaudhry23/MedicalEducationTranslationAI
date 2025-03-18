import json
from elaborate import elaborate_text
from translate import batch_translate, initialize_model_and_tokenizer
from IndicTransToolkit.processor import IndicProcessor

# Language Mapping Dictionary
Indian_Languages = {
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
    "Urdu": "urd_Arab",
}

# User inputs
inputPrompt = "Describe the function of the liver in simple terms."
output_language = "Tamil"

# Validate language input
if output_language not in Indian_Languages:
    print("Invalid language choice. Defaulting to Hindi.")
    output_language = "Hindi"

# Get language codes
src_lang = "eng_Latn"
tgt_lang = Indian_Languages[output_language]

# 📌 Dictionary to store results
input_translate_output = {
    "input": inputPrompt,
    "elaboration": "",
    "output": ""
}

# 🔹 Step 1: Elaborate Medical Text (MedAlpaca)
print("\n🔍 Elaborating text using MedAlpaca-7B...")
elaborated_text = elaborate_text(inputPrompt)
input_translate_output["elaboration"] = elaborated_text
print("\n🩺 Simplified Explanation:\n", elaborated_text)

# 🔹 Step 2: Initialize Translation Model & Tokenizer
print("\n🚀 Initializing translation model...")
model_path = "/workspace/MedicalEducationTranslationAI/Models/indictrans2"
tokenizer, model = initialize_model_and_tokenizer(model_path, None)  # No quantization for translation model

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# 🔹 Step 3: Translate to selected language
print(f"\n🌐 Translating to {output_language} using IndicTrans2...")
translated_output = batch_translate([elaborated_text], src_lang, tgt_lang, model, tokenizer, ip)
input_translate_output["output"] = translated_output[0]
print(f"\n🌍 {output_language} Translation:\n", translated_output[0])

# 🔹 Step 4: Append Output to File
output_filename = "outputs.txt"

# Load existing data if the file is not empty
try:
    with open(output_filename, "r") as file:
        existing_data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    existing_data = []  # Start with an empty array if file doesn't exist or is empty

# Append new result
existing_data.append(input_translate_output)

# Write updated data back to file
with open(output_filename, "w") as file:
    json.dump(existing_data, file, indent=4, ensure_ascii=False)

print("\n✅ Results saved to `outputs.txt`!")

# Clean up memory
del tokenizer, model, ip
