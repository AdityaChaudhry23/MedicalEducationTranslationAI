# comet_elaboration_multilang_test.py

import time
import os
import json
import torch
from Functions.pipelines import Explanation_Pipeline
from comet import download_model, load_from_checkpoint

# ⚙️ Improve GPU performance for matmul
torch.set_float32_matmul_precision('high')

# 🔁 Load COMET model
print("🔁 Loading COMET model...")
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
print("✅ COMET model loaded!\n")

# 🌍 Languages COMET can evaluate
EVALUABLE_LANGUAGES = [
    "Assamese", "Bengali", "English", "Gujarati", "Hindi", "Kannada",
    "Malayalam", "Marathi", "Nepali", "Odia", "Punjabi", "Sanskrit",
    "Sindhi (Arabic)", "Sindhi (Devanagari)", "Tamil", "Telugu", "Urdu"
]

# 🧪 Sample medical questions
TEST_QUESTIONS = [
    "Describe Cardiac Arrest in simple terms.",
    "Describe the function of the liver in simple terms.",
    "Explain the role of the nervous system in the human body.",
    "What is the importance of alveoli in respiration?",
    "Describe how the heart pumps blood in the body.",
    "What is the role of the kidneys in waste filtration?",
    "How does the immune system protect the body?",
    "Explain the digestive process in humans.",
    "What happens during an asthma attack?",
    "How do vaccines prevent diseases?"
]

# 📁 Output directory
base_output_dir = "Tests/Comet"
os.makedirs(base_output_dir, exist_ok=True)

# 🚀 Iterate through languages
for language in EVALUABLE_LANGUAGES:
    print(f"\n🌐 Evaluating language: {language}")

    lang_results = []
    comet_batch = []

    for question in TEST_QUESTIONS:
        print(f"\n🔹 Processing: {question}")
        start_time = time.time()

        try:
            result = Explanation_Pipeline(question, language)
            end_time = time.time()

            lang_results.append({
                "input": result["input"],
                "elaboration": result["elaboration"],
                "translation": result["output"],
                "time_taken_sec": round(end_time - start_time, 2)
            })

            comet_batch.append({
                "src": result["elaboration"],
                "mt": result["output"]
            })

        except Exception as e:
            err_msg = str(e)
            lang_results.append({
                "input": question,
                "elaboration": "[Error]",
                "translation": err_msg,
                "comet_score": "N/A",
                "time_taken_sec": round(time.time() - start_time, 2)
            })
            comet_batch.append({"src": "[Error]", "mt": err_msg})

    # 🧠 COMET Scoring
    print(f"\n🧠 Scoring translations with COMET for {language}...")
    prediction = model.predict(comet_batch, batch_size=8, gpus=1)

    comet_scores = prediction.get("scores") if isinstance(prediction, dict) else prediction[0]

    for i, score in enumerate(comet_scores):
        try:
            lang_results[i]["comet_score"] = round(float(score), 4)
        except Exception:
            lang_results[i]["comet_score"] = "N/A"
            print(f"⚠️ Invalid COMET score at index {i} for {language}: {score}")

    # 💾 Save results to JSON
    output_path = os.path.join(base_output_dir, f"{language}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lang_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved results for {language} → {output_path}")

print("\n🎉 All language evaluations completed!")
