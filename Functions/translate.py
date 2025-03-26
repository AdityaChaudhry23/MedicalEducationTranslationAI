import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor # type: ignore

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC_LANG = "eng_Latn"
BATCH_SIZE = 4

# Model path
MODEL_PATH = "/workspace/MedicalEducationTranslationAI/Models/indictrans2"

# Load model & tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()
processor = IndicProcessor(inference=True)

# Language mapping: user-friendly name → IndicTrans2 tag
LANGUAGE_CODE_MAP = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}


def translate(text: str, target_language: str) -> str:
    """
    Translates English text to the specified target language (name-based).

    Args:
        text (str): English sentence to translate.
        target_language (str): Human-readable name of target language (e.g., 'Hindi', 'Odia').

    Returns:
        str: Translated sentence.
    """
    if target_language not in LANGUAGE_CODE_MAP:
        raise ValueError(f"Unsupported language: {target_language}. Choose from: {list(LANGUAGE_CODE_MAP.keys())}")

    target_lang_code = LANGUAGE_CODE_MAP[target_language]

    # Preprocess input
    batch = [text]
    batch = processor.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=target_lang_code)

    # Tokenize
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
            use_cache=True,
        )

    # Decode & postprocess
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            outputs.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    result = processor.postprocess_batch(decoded, lang=target_lang_code)
    return result[0]
