#!/bin/bash

# 🔹 Set directory
MODEL_DIR="Models/cometkiwi"
MODEL_NAME="Unbabel/wmt23-cometkiwi-da-xxl"

# 🔸 Ensure Models directory exists
mkdir -p "$MODEL_DIR"

# 🔹 Activate Python environment if needed
# source MedEdAIVenv/bin/activate  # Uncomment if you use a virtual env

# 🔸 Run the download via Python
python3 - <<END
from comet import download_model

print("⬇️  Downloading COMET model: $MODEL_NAME ...")
download_model("$MODEL_NAME", saving_directory="$MODEL_DIR")
print("✅ Model downloaded to $MODEL_DIR")
END
