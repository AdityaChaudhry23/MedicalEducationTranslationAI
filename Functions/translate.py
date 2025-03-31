import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor  # type: ignore

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_LANG = "eng_Latn"
MODEL_PATH = "/workspace/MedicalEducationTranslationAI/Models/indictrans2"

# Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()
processor = IndicProcessor(inference=True)

# Language mapping
LANGUAGE_CODE_MAP = {
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", "Dogri": "doi_Deva", "English": "eng_Latn",
    "Gujarati": "guj_Gujr", "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva", "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng", "Manipuri (Meitei)": "mni_Mtei", "Marathi": "mar_Deva", "Nepali": "npi_Deva",
    "Odia": "ory_Orya", "Punjabi": "pan_Guru", "Sanskrit": "san_Deva", "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab", "Sindhi (Devanagari)": "snd_Deva", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"🔍 GPU Memory — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def translate(text: str, target_language: str) -> str:
    if not text.strip():
        return ""

    if target_language not in LANGUAGE_CODE_MAP:
        return f"[Translation Error: Unsupported language '{target_language}']"

    tgt_lang_code = LANGUAGE_CODE_MAP[target_language]

    try:
        # Preprocessing
        processed = processor.preprocess_batch([text], src_lang=SRC_LANG, tgt_lang=tgt_lang_code)

        # Tokenization
        inputs = tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            return_attention_mask=True
        ).to(DEVICE)

        # Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=3,
                num_return_sequences=1,
                use_cache=True
            )

        # Decoding
        with tokenizer.as_target_tokenizer():
            decoded = tokenizer.batch_decode(
                outputs.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        # Postprocessing
        return processor.postprocess_batch(decoded, lang=tgt_lang_code)[0]

    except torch.cuda.OutOfMemoryError:
        print("❌ CUDA Out of Memory: Releasing cache...")
        torch.cuda.empty_cache()
        return "[Translation Failed: Out of Memory]"

    except Exception as e:
        print(f"⚠️ Unexpected Error: {e}")
        return "[Translation Failed: Unexpected Error]"

    finally:
        torch.cuda.empty_cache()
        print_gpu_memory()
