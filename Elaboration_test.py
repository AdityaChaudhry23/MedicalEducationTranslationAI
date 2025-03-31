import time
import os
import json
from Functions.pipelines import Explanation_Pipeline

# 🔹 Test medical questions (like the ones you shared)
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

# 🔸 Output storage
results = []

# 🌐 Language for translation
target_language = "Hindi"

# 🧪 Run test
print("🔍 Running Explanation_Pipeline tests...\n")

for input_text in test_inputs:
    print(f"📝 Processing: {input_text}")
    start_time = time.time()
    try:
        result = Explanation_Pipeline(input_text, target_language)
        end_time = time.time()

        results.append({
            "input": result["input"],
            "elaboration": result["elaboration"],
            "output": result["output"],
            "time_taken_sec": round(end_time - start_time, 2)
        })

    except Exception as e:
        end_time = time.time()
        results.append({
            "input": input_text,
            "elaboration": "[Error]",
            "output": str(e),
            "time_taken_sec": round(end_time - start_time, 2)
        })

# 📁 Ensure directory exists
os.makedirs("Tests", exist_ok=True)

# 💾 Save results to Output.txt
with open("Tests/Output.txt", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False, indent=2))
        f.write("\n\n")

print("\n✅ Test complete. Results saved to Tests/Output.txt")
