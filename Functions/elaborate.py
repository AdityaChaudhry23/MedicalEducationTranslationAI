import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Constants
MODEL_PATH = "/workspace/MedicalEducationTranslationAI/Models/medalpaca-7b"

# Quant config for 8-bit loading
qconfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# Load model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=qconfig,
        low_cpu_mem_usage=True,
        device_map="auto"  # Important for bitsandbytes
    )

    # Fix pad_token_id warning if missing
    if model.generation_config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = tokenizer.eos_token_id

    # DO NOT pass `device=` for accelerate
    pl = pipeline("text-generation", model=model, tokenizer=tokenizer)

except Exception as e:
    print(f"❌ Failed to load MedAlpaca model: {e}")
    pl = None

def elaborate(text: str) -> str:
    if not pl:
        return "[Elaboration failed: Model not loaded]"

    prompt = f"""You are a medical expert. Explain the following medical text in a simple and easy-to-understand way:
    
    Medical Text: {text}
    
    Explanation:
    """

    try:
        response = pl(
            prompt,
            max_length=400,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        return response[0]["generated_text"].replace(prompt, "").strip()

    except torch.cuda.OutOfMemoryError:
        print("❌ CUDA OOM in elaborate() — freeing memory...")
        torch.cuda.empty_cache()
        return "[Elaboration failed: Out of Memory]"

    except Exception as e:
        print(f"⚠️ Error in elaborate(): {e}")
        return "[Elaboration failed: Unexpected Error]"

    finally:
        torch.cuda.empty_cache()
