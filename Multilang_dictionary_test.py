# Dictionary_Test.py

import time
import os
import json
import torch
from Functions.pipelines import Dictionary_Pipeline
from comet import download_model, load_from_checkpoint

# ⚙️ Improve GPU matrix performance
torch.set_float32_matmul_precision('high')

# 📥 Download COMET model once
print("🔁 Loading COMET model...")
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
print("✅ COMET model loaded!\n")

# 🔤 Languages supported by both IndicTrans2 and COMET
EVALUABLE_LANGUAGES = [
    "Assamese", "Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
    "Marathi", "Nepali", "Odia", "Punjabi", "Sanskrit", "Sindhi (Arabic)",
    "Sindhi (Devanagari)", "Tamil", "Telugu", "Urdu"
]

# 🔍 Test terms
test_words = [
    "Hypertension", "Diabetes", "Asthma", "Anemia", "Inflammation",
    "Antibiotic", "Cardiovascular", "Kidney", "Immune", "Vaccination"
]

# 📁 Prepare output folder
os.makedirs("Tests/Comet_Dictionary", exist_ok=True)

for lang in EVALUABLE_LANGUAGES:
    print(f"\n🌐 Testing language: {lang}")
    lang_results = []
    comet_batch = []

    for word in test_words:
        print(f"📘 Processing: {word} → {lang}")
        start_time = time.time()
        try:
            result = Dictionary_Pipeline(word, lang)
            end_time = time.time()

            lang_results.append({
                "input_word": result["input_word"],
                "input_word_translation": result["input_word_translation"],
                "meaning": result["meaning"],
                "meaning_translation": result["meaning_translation"],
                "time_taken_sec": round(end_time - start_time, 2)
            })

            comet_batch.append({
                "src": result["meaning"],
                "mt": result["meaning_translation"]
            })

        except Exception as e:
            end_time = time.time()
            err_msg = str(e)

            lang_results.append({
                "input_word": word,
                "input_word_translation": "[Error]",
                "meaning": "[Error]",
                "meaning_translation": err_msg,
                "time_taken_sec": round(end_time - start_time, 2),
                "comet_score": "N/A"
            })

            comet_batch.append({
                "src": "[Error]",
                "mt": err_msg
            })

    # 🧠 COMET scoring
    print(f"🧠 Scoring with COMET for {lang}...")
    prediction = model.predict(comet_batch, batch_size=8, gpus=1)

    # 🔍 Extract scores
    comet_scores = prediction["scores"] if isinstance(prediction, dict) else prediction[0]

    for i, score in enumerate(comet_scores):
        try:
            lang_results[i]["comet_score"] = round(float(score), 4)
        except (ValueError, TypeError):
            lang_results[i]["comet_score"] = "N/A"
            print(f"⚠️ COMET score invalid for index {i}: {score}")

    # 💾 Save file per language
    with open(f"Tests/Comet_Dictionary/{lang}.json", "w", encoding="utf-8") as f:
        json.dump(lang_results, f, ensure_ascii=False, indent=2)

print("\n✅ All dictionary tests completed and saved to Tests/Comet_Dictionary/")
