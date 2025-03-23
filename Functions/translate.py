import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from IndicTransToolkit.processor import IndicProcessor

# Configuration
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None  # Set to "4-bit" or "8-bit" if needed for quantization

# This dictionary maps user-friendly language names to IndicTrans2 codes.
INDIAN_LANGUAGES = {
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
    "Urdu": "urd_Arab",
}


def initialize_model_and_tokenizer(ckpt_dir, quantization_mode=None):
    """
    Loads the tokenizer + model from local or remote checkpoint.
    Allows optional 4-bit or 8-bit quantization. 
    Also patches missing 'vocab_size' to avoid config errors with new Transformers.
    """
    if quantization_mode == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_mode == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    # --- Patch missing vocab_size in IndicTransConfig ---
    if not hasattr(model.config, "vocab_size"):
        print("Patching missing 'vocab_size' on model.config...")
        # You can derive it either from tokenizer length or model input embeddings
        model.config.vocab_size = model.get_input_embeddings().weight.shape[0]

    # If no quantization, move model to GPU (if available) and half precision
    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """
    Translates a list of sentences in batches from src_lang -> tgt_lang 
    using the provided model, tokenizer, and IndicProcessor instance.
    """
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch (adding special tokens, placeholders, etc.)
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess 
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


def translate_text(english_text, output_language="Hindi"):
    """
    Generic function to translate the given English text into an Indian language. 
    By default, translates to Hindi. 
    You can call: translate_text("Hello World", "Tamil"), etc.
    """
    # Validate the requested output language
    if output_language not in INDIAN_LANGUAGES:
        valid_langs = ", ".join(INDIAN_LANGUAGES.keys())
        raise ValueError(
            f"Unsupported language '{output_language}'. "
            f"Supported languages: {valid_langs}"
        )

    src_lang = INDIAN_LANGUAGES["English"]  # "eng_Latn"
    tgt_lang = INDIAN_LANGUAGES[output_language]

    # Path to your local IndicTrans2 checkpoint
    en_indic_ckpt_dir = "/workspace/MedicalEducationTranslationAI/Functions/Models/indictrans2"

    # Load model & tokenizer with optional quantization
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
        en_indic_ckpt_dir, quantization
    )

    # Create the IndicProcessor
    ip = IndicProcessor(inference=True)

    # Translate a single text 
    translations = batch_translate([english_text], src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

    # Cleanup GPU memory if desired
    del en_indic_tokenizer, en_indic_model

    return translations[0]


if __name__ == "__main__":
    text = input("Enter English text for translation: ")
    # Example: default is Hindi, or you can pass another language
    result = translate_text(text, "Hindi")
    print("\nTranslation (English -> Hindi):\n", result)
