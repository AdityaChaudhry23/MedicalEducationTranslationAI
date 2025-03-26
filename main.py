from Functions.pipelines import Explanation_Pipeline, Dictionary_Pipeline
from Functions.languages import LANGUAGE_CODE_MAP
import gradio as gr

LANGUAGES = list(LANGUAGE_CODE_MAP.keys())

def run_explanation(input_text, output_language):
    return Explanation_Pipeline(input_text, output_language)

def run_dictionary(input_word, output_language):
    return Dictionary_Pipeline(input_word, output_language)

explanation_interface = gr.Interface(
    fn=run_explanation,
    inputs=[
        gr.Textbox(lines=4, label="Medical Term or Paragraph in English"),
        gr.Dropdown(choices=LANGUAGES, label="Target Language")
    ],
    outputs=gr.JSON(label="Explanation Output"),
    title="🩺 Medical Explanation Translator",
    description="Explain a medical concept in plain English and translate it to an Indic language."
)

dictionary_interface = gr.Interface(
    fn=run_dictionary,
    inputs=[
        gr.Textbox(label="Medical Term (in English)"),
        gr.Dropdown(choices=LANGUAGES, label="Target Language")
    ],
    outputs=gr.JSON(label="Dictionary Output"),
    title="📘 Medical Dictionary Translator",
    description="Translate a medical term and its meaning into your preferred Indic language."
)

demo = gr.TabbedInterface(
    interface_list=[explanation_interface, dictionary_interface],
    tab_names=["Explanation Pipeline", "Dictionary Pipeline"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
