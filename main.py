import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# Load model and tokenizer only once
MODEL_PATH = "/workspace/MedicalEducationTranslationAI/Models/indictrans2"  # Update if needed
SRC_LANG = "eng_Latn"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()
processor = IndicProcessor(inference=True)


def translate(sentence: str, target_lang: str) -> str:
    """Translates a single English sentence to the given target language."""
    batch = [sentence]
    batch = processor.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=target_lang)

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
            use_cache=True,
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            output.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    final_output = processor.postprocess_batch(decoded, lang=target_lang)
    return final_output[0]


# Example usage
if __name__ == "__main__":
    en_sents = [
        "When I was young, I used to go to the park every day.",
        "He has many old books, which he inherited from his ancestors.",
        "I can't figure out how to solve my problem.",
        "She is very hardworking and intelligent, which is why she got all the good marks.",
        "We watched a new movie last week, which was very inspiring.",
        "If you had met me at that time, we would have gone out to eat.",
        "She went to the market with her sister to buy a new sari.",
        "Raj told me that he is going to his grandmother's house next month.",
        "All the kids were having fun at the party and were eating lots of sweets.",
        "My friend has invited me to his birthday party, and I will give him a gift.",
    ]

    tgt_lang = "ory_Orya"  # Change to another language tag if needed

    results = []

    for sent in en_sents:
        translated = translate(sent, tgt_lang)
        results.append(f"EN: {sent}\n{tgt_lang}: {translated}\n")
        print(f"\nEN: {sent}\n{tgt_lang}: {translated}")

    # 💾 Save to text file
    with open("translations.txt", "w", encoding="utf-8") as f:
        f.writelines(results)
