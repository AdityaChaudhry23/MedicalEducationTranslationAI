# Elaboration_test.py

import time
import os
import json
import torch
from Functions.pipelines import Explanation_Pipeline
from comet import download_model, load_from_checkpoint

# ⚙️ Optional: Set precision trade-off to improve matmul performance on GPU
torch.set_float32_matmul_precision('high')  # Options: 'highest', 'high', 'medium'

# 🔁 Load COMET model
print("🔁 Loading COMET model...")
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
print("✅ COMET model loaded!\n")

# 🧪 Sample questions
test_inputs = [
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

target_language = "Hindi"
results = []
comet_batch = []

print("🚀 Running Explanation_Pipeline tests...\n")

for input_text in test_inputs:
    print(f"🔹 Processing: {input_text}")
    start_time = time.time()

    try:
        result = Explanation_Pipeline(input_text, target_language)
        end_time = time.time()

        results.append({
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
        end_time = time.time()
        err_msg = str(e)
        results.append({
            "input": input_text,
            "elaboration": "[Error]",
            "translation": err_msg,
            "comet_score": "N/A",
            "time_taken_sec": round(end_time - start_time, 2)
        })
        comet_batch.append({
            "src": "[Error]",
            "mt": err_msg
        })

# 🧠 COMET scoring
# 🧠 COMET scoring
print("\n🧠 Scoring translations with COMET...")
prediction = model.predict(comet_batch, batch_size=8, gpus=1)

# Extract individual scores from the dict
if isinstance(prediction, dict) and "scores" in prediction:
    comet_scores = prediction["scores"]
else:
    comet_scores = prediction[0] if isinstance(prediction, tuple) else prediction

# 🔁 Attach scores
for i, score in enumerate(comet_scores):
    try:
        results[i]["comet_score"] = round(float(score), 4)
    except (ValueError, TypeError):
        results[i]["comet_score"] = "N/A"
        print(f"⚠️ COMET score for index {i} is invalid: {score}")


# 💾 Save results
os.makedirs("Tests", exist_ok=True)
with open("Tests/Output.txt", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False, indent=2))
        f.write("\n\n")

print("\n✅ All test cases processed and saved to Tests/Output.txt")
