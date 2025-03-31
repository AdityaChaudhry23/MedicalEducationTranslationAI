import time
import os
import json
from Functions.pipelines import Dictionary_Pipeline

# 🔹 Test medical terms
test_words = [
    "Hypertension",
    "Diabetes",
    "Asthma",
    "Anemia",
    "Inflammation",
    "Antibiotic",
    "Cardiovascular",
    "Kidney",
    "Immune",
    "Vaccination"
]

# 🔸 Output storage
results = []

# 🌐 Language for translation
target_language = "Hindi"

# 🧪 Run test
print("🔍 Running Dictionary_Pipeline tests...\n")

for word in test_words:
    print(f"📘 Processing: {word}")
    start_time = time.time()
    try:
        result = Dictionary_Pipeline(word, target_language)
        end_time = time.time()

        results.append({
            "input_word": result["input_word"],
            "input_word_translation": result["input_word_translation"],
            "meaning": result["meaning"],
            "meaning_translation": result["meaning_translation"],
            "time_taken_sec": round(end_time - start_time, 2)
        })

    except Exception as e:
        end_time = time.time()
        results.append({
            "input_word": word,
            "input_word_translation": "[Error]",
            "meaning": "[Error]",
            "meaning_translation": str(e),
            "time_taken_sec": round(end_time - start_time, 2)
        })

# 📁 Ensure directory exists
os.makedirs("Tests", exist_ok=True)

# 💾 Save results to Output.txt
with open("Tests/Dictionary_Output.txt", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False, indent=2))
        f.write("\n\n")

print("\n✅ Dictionary test complete. Results saved to Tests/Dictionary_Output.txt")
