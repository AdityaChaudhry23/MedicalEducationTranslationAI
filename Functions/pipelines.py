from Functions.elaborate import elaborate
from Functions.translate import translate


def Explanation_Pipeline(input_text: str, output_language: str) -> dict:
    """
    Explains the medical input in simple English, then translates the explanation into the desired output language.
    """
    explanation = elaborate(input_text)
    translated_output = translate(explanation, output_language)

    return {
        "input": input_text,
        "elaboration": explanation,
        "output": translated_output
    }


def Dictionary_Pipeline(input_word: str, output_language: str) -> dict:
    """
    Translates the input word to the desired language, elaborates on the word in English,
    and translates the explanation as well.
    """
    translated_word = translate(input_word, output_language)
    explanation = elaborate(input_word)
    translated_explanation = translate(explanation, output_language)

    return {
        "input_word": input_word,
        "input_word_translation": translated_word,
        "meaning": explanation,
        "meaning_translation": translated_explanation
    }
