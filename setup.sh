#!/bin/bash

# 🚀 Script to Set Up Models & IndicTransToolkit for MedicalEducationTranslationAI

echo "==========================================="
echo " 🏥 MedicalEducationTranslationAI Setup  "
echo "==========================================="

# 1️⃣ Ensure the script is executed in the correct directory
PROJECT_DIR="/workspace/MedicalEducationTranslationAI"
cd "$PROJECT_DIR" || { echo "❌ ERROR: Project directory not found! Exiting..."; exit 1; }

# 2️⃣ Activate the Existing Virtual Environment
echo "🔹 Activating Virtual Environment..."
if [ -d "MedEdAIVenv" ]; then
    source MedEdAIVenv/bin/activate
else
    echo "❌ ERROR: Virtual environment 'MedEdAIVenv' not found! Please create it first."
    exit 1
fi

# 3️⃣ Upgrade Pip
echo "🔹 Upgrading pip..."
pip install --upgrade pip

# 4️⃣ Install Dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "🔹 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ WARNING: requirements.txt not found! Skipping dependency installation."
fi

# 5️⃣ Install Additional Necessary Packages
echo "🔹 Installing Hugging Face CLI, bitsandbytes..."
pip install huggingface_hub bitsandbytes

# 6️⃣ Log into Hugging Face CLI (Requires API Key)
echo "🔹 Logging into Hugging Face (You will need an API token)..."
huggingface-cli login

# 7️⃣ Create Models Directory (if not exists)
MODELS_DIR="$PROJECT_DIR/Models"
mkdir -p "$MODELS_DIR"

# 8️⃣ Download MedAlpaca-7B Model (Skip if already exists)
if [ ! -d "$MODELS_DIR/medalpaca-7b" ]; then
    echo "🔹 Downloading MedAlpaca-7B..."
    huggingface-cli download medalpaca/medalpaca-7b --local-dir "$MODELS_DIR/medalpaca-7b"
else
    echo "✅ MedAlpaca-7B already exists. Skipping download."
fi

# 9️⃣ Download IndicTrans2 Model (Skip if already exists)
if [ ! -d "$MODELS_DIR/indictrans2" ]; then
    echo "🔹 Downloading IndicTrans2..."
    huggingface-cli download ai4bharat/indictrans2-en-indic-1B --local-dir "$MODELS_DIR/indictrans2"
else
    echo "✅ IndicTrans2 already exists. Skipping download."
fi

# 🔟 Clone & Install IndicTransToolkit (Skip if already cloned)
INDIC_TRANS_DIR="$PROJECT_DIR/IndicTransToolkit"
if [ ! -d "$INDIC_TRANS_DIR" ]; then
    echo "🔹 Cloning and installing IndicTransToolkit..."
    git clone https://github.com/VarunGumma/IndicTransToolkit.git "$INDIC_TRANS_DIR"
    cd "$INDIC_TRANS_DIR" || exit
    pip install --editable ./
    cd "$PROJECT_DIR" || exit
else
    echo "✅ IndicTransToolkit already exists. Skipping cloning."
fi

# 1️⃣1️⃣ Export PYTHONPATH to include IndicTransToolkit
if ! grep -q "IndicTransToolkit" ~/.bashrc; then
    echo "export PYTHONPATH=$INDIC_TRANS_DIR:\$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc
    echo "✅ PYTHONPATH updated!"
else
    echo "✅ PYTHONPATH already set!"
fi

# ✅ Setup Complete
echo "==========================================="
echo " ✅ Setup Complete! Ready to Run the AI!  "
echo "==========================================="
