import gradio as gr
from Functions.pipelines import Explanation_Pipeline, Dictionary_Pipeline
from Functions.languages import get_language_list

LANGUAGES = get_language_list()

# 🩺 Explanation Pipeline
explanation_history = []

def run_explanation(input_text, output_language):
    # Status update
    status = "🔄 Processing..."

    # Run pipeline
    result = Explanation_Pipeline(input_text, output_language)
    explanation_history.append(result)

    # Format output
    output_block = f"**Input:** {result['input']}\n\n" \
                   f"**Elaboration:** {result['elaboration']}\n\n" \
                   f"**Translated Output:** {result['output']}"

    # Format history
    history_list = [
        f"- **Input:** {r['input']}\n  **→ Elaboration:** {r['elaboration']}\n  **→ Translation:** {r['output']}"
        for r in explanation_history
    ]

    return "✅ Done!", output_block, "\n\n".join(history_list)

# 📘 Dictionary Pipeline
dictionary_history = []

def run_dictionary(input_word, output_language):
    # Status update
    status = "🔄 Processing..."

    # Run pipeline
    result = Dictionary_Pipeline(input_word, output_language)
    dictionary_history.append(result)

    # Format output
    output_block = f"**Input Word:** {result['input_word']}\n\n" \
                   f"**Translated Word:** {result['input_word_translation']}\n\n" \
                   f"**Meaning:** {result['meaning']}\n\n" \
                   f"**Translated Meaning:** {result['meaning_translation']}"

    # Format history
    history_list = [
        f"- **{r['input_word']}** → {r['input_word_translation']}\n  {r['meaning_translation']}"
        for r in dictionary_history
    ]

    return "✅ Done!", output_block, "\n\n".join(history_list)

# Interface: Explanation
explanation_interface = gr.Interface(
    fn=run_explanation,
    inputs=[
        gr.Textbox(lines=4, label="Medical Term or Paragraph in English"),
        gr.Dropdown(choices=LANGUAGES, label="Target Language", value="Hindi")
    ],
    outputs=[
        gr.Markdown(value="🔄 Waiting...", label="Status"),
        gr.Markdown(label="Result"),
        gr.Markdown(label="Session History")
    ],
    title="🩺 Medical Explanation Translator",
    description="Explain a medical concept in plain English and translate it to your language."
)

# Interface: Dictionary
dictionary_interface = gr.Interface(
    fn=run_dictionary,
    inputs=[
        gr.Textbox(label="Medical Term (in English)"),
        gr.Dropdown(choices=LANGUAGES, label="Target Language", value="Hindi")
    ],
    outputs=[
        gr.Markdown(value="🔄 Waiting...", label="Status"),
        gr.Markdown(label="Result"),
        gr.Markdown(label="Session History")
    ],
    title="📘 Medical Dictionary Translator",
    description="Translate a medical term and its meaning into your preferred Indic language."
)

# Combined app
demo = gr.TabbedInterface(
    interface_list=[explanation_interface, dictionary_interface],
    tab_names=["Explanation Pipeline", "Dictionary Pipeline"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
