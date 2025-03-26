import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/workspace/MedicalEducationTranslationAI/Models/medalpaca-7b"

# Quantization settings
qconfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=qconfig,
    low_cpu_mem_usage=True
)

# Text generation pipeline
pl = pipeline("text-generation", model=model, tokenizer=tokenizer)

def elaborate(text: str) -> str:
    """
    Uses MedAlpaca to simplify or elaborate a medical term, question, or paragraph.

    Args:
        text (str): Medical term, sentence, or paragraph to elaborate.

    Returns:
        str: Simplified explanation in plain English.
    """
    prompt = f"""You are a medical expert. Explain the following medical text in a simple and easy-to-understand way:
    
    Medical Text: {text}
    
    Explanation:
    """

    response = pl(
        prompt,
        max_length=400,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    return response[0]["generated_text"].replace(prompt, "").strip()
