import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Define local model path
model_path = "/workspace/MedicalEducationTranslationAI/Models/medalpaca-7b"

# Define quantization settings (8-bit)
qconfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=qconfig,
    low_cpu_mem_usage=True
)

# ⚠️ Fix: Remove `device=0` from the pipeline to avoid conflict with `accelerate`
pl = pipeline("text-generation", model=model, tokenizer=tokenizer)

def elaborate_text(text):
    """ Uses MedAlpaca to elaborate medical text in simple English """
    prompt = f"""You are a medical expert. Explain the following medical term in a simple and easy-to-understand way:
    
    Medical Term: {text}
    
    Explanation:
    """

    response = pl(
        prompt,
        max_length=400,  # Increased length for better elaboration
        num_return_sequences=1,
        temperature=0.7,  # More varied responses
        top_p=0.9,  # Ensures diverse explanations
        repetition_penalty=1.1,  # Reduces repetitive outputs
    )

    return response[0]["generated_text"].replace(prompt, "").strip()

# If running standalone
if __name__ == "__main__":
    user_input = input("Enter medical term or text: ")
    output = elaborate_text(user_input)
    print("\n🩺 Simplified Explanation:\n", output)
