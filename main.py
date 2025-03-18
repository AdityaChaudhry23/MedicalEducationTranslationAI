import json
from elaborate import elaborate_text
from translate import translate_to_hindi

# 📝 Hardcoded Input (MVP)
inputPrompt = "Explain the difference between ECG and EEG"

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

# 🔹 Step 2: Translate to Hindi (IndicTrans2)
print("\n🌐 Translating to Hindi using IndicTrans2...")
hindi_translation = translate_to_hindi(elaborated_text)
input_translate_output["output"] = hindi_translation
print("\n🇮🇳 Final Hindi Translation:\n", hindi_translation)

# 🔹 Step 3: Append Output to File
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
