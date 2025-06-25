import gradio as gr
from llm_handler import LLMHandler
from voice_input import VoiceInput
from pdf_reader import PDFReader
import os

# Instancias de modelos y utilidades
deepseek = LLMHandler("deepseek-r1:7b")
gemma = LLMHandler("gemma3:4b")
speech = VoiceInput()
pdf_reader = PDFReader()

# Función principal del chatbot
def chat_interface(user_input, chat_history, model_choice, pdf_text):
    if not user_input:
        return chat_history, "", None, pdf_text

    model = deepseek if model_choice == "DeepSeek" else gemma
    context = f"[INFO DEL PDF]\n{pdf_text}\n\n" if pdf_text else ""
    full_input = context + user_input

    response = model.chat(full_input)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, "", None, pdf_text

# Transcripción de voz
def audio_to_text(audio_path):
    if audio_path is None or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "[ERROR: Audio inválido]"
    return speech.transcribe(audio_path)

# Procesar el PDF al cargarlo
def process_pdf(file):
    if file is None:
        return ""
    extracted = pdf_reader.extract_text(file)
    print(f"📄 Texto extraído del PDF:\n{extracted[:500]}...")
    return extracted

# UI
with gr.Blocks() as ui:
    gr.Markdown("# 🤖 VChatbot")
    gr.Markdown("Chat por voz o texto con DeepSeek o Gemma 3.")

    with gr.Row():
        model_selector = gr.Radio(["DeepSeek", "Gemma 3"], label="🔍 Elige el modelo", value="DeepSeek")

    chatbot = gr.Chatbot(label="💬 Conversación", type="messages")
    state = gr.State([])
    pdf_state = gr.State("")

    with gr.Row():
        text_input = gr.Textbox(placeholder="Escribe un mensaje...", label="Entrada de texto", lines=2)
        send_btn = gr.Button("Enviar")

    audio_input = gr.Audio(type="filepath", label="🎙️ Graba tu voz", interactive=True)
    pdf_upload = gr.File(label="📄 Sube un PDF", file_types=[".pdf"])

    # Acciones al enviar input
    send_btn.click(
        chat_interface,
        inputs=[text_input, state, model_selector, pdf_state],
        outputs=[chatbot, text_input, audio_input, pdf_state],
    )

    text_input.submit(
        chat_interface,
        inputs=[text_input, state, model_selector, pdf_state],
        outputs=[chatbot, text_input, audio_input, pdf_state],
    )

    audio_input.change(audio_to_text, inputs=audio_input, outputs=text_input)

    pdf_upload.change(
        process_pdf,
        inputs=pdf_upload,
        outputs=pdf_state,
    )

ui.launch()
