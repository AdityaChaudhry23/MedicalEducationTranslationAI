# test_translation.py

from translate import translate

# Simple English -> Hindi test
print("Testing translation from English to Hindi...")
input_text = "Hello, how are you?"
translation = translate(input_text, "Hindi")
print(f"English: {input_text}\nHindi: {translation}")
