import gradio as gr
import json
from main import elaborate_text
from translate import translate_to_hindi
from slidedeck import save_to_slidedeck

# Load users from users.json
with open("users.json", "r") as file:
    users = json.load(file)

# Language Mapping
INDIAN_LANGUAGES = {
    "Hindi": "hin_Deva",
    "Malayalam": "mal_Mlym",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Bengali": "ben_Beng",
    "Punjabi": "pan_Guru",
    "Kannada": "kan_Knda",
    "Odia": "ory_Orya",
    "Urdu": "urd_Arab",
}

# User authentication function
def login(username, password):
    for user in users:
        if user["UserName"] == username and user["Password"] == password:
            return f"✅ Login successful! Welcome, {username}. Your default language is {user['Mother Tongue']}.", user["Mother Tongue"]
    return "❌ Invalid credentials. Please try again.", None

# Elaboration & Translation function
def elaborate_and_translate(term, language):
    if language not in INDIAN_LANGUAGES:
        return "⚠️ Selected language not supported.", "", ""

    # Elaborate the term
    elaboration = elaborate_text(term)

    # Translate elaboration
    target_lang = INDIAN_LANGUAGES[language]
    translated_output = translate_to_hindi(elaboration)  # Modify to use `target_lang`

    return elaboration, translated_output

# Save to Slide Deck
def save_word_to_slidedeck(word):
    save_to_slidedeck(word)
    return f"✅ '{word}' saved to Slide Deck!"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 Medical Education Translation AI")

    with gr.Tab("🔐 Login"):
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        login_status = gr.Label()
        language_dropdown = gr.Dropdown(list(INDIAN_LANGUAGES.keys()), label="Select Language", visible=False)

        def handle_login(user, pwd):
            status, lang = login(user, pwd)
            return status, gr.update(visible=True, value=lang) if lang else gr.update(visible=False)

        login_button.click(handle_login, inputs=[username, password], outputs=[login_status, language_dropdown])

    with gr.Tab("🩺 Elaborate & Translate"):
        input_term = gr.Textbox(label="Enter Medical Term")
        output_lang = gr.Dropdown(list(INDIAN_LANGUAGES.keys()), label="Select Output Language")
        elaboration = gr.Textbox(label="Elaborated Explanation", interactive=False)
        translation = gr.Textbox(label="Translated Output", interactive=False)
        translate_button = gr.Button("Translate")

        translate_button.click(elaborate_and_translate, inputs=[input_term, output_lang], outputs=[elaboration, translation])

    with gr.Tab("📖 Slide Deck"):
        slide_term = gr.Textbox(label="Enter Word to Save")
        slide_status = gr.Label()
        save_button = gr.Button("Save to Slide Deck")
        
        save_button.click(save_word_to_slidedeck, inputs=[slide_term], outputs=[slide_status])

# Run the app
demo.launch()
