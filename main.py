import gradio as gr
from llm_handler import LLMHandler
from voice_input import VoiceInput
import os

# Instancias
deepseek = LLMHandler("deepseek-r1:7b")
gemma = LLMHandler("gemma3:4b")
speech = VoiceInput()

# Procesar entrada de texto y enviar al modelo
def chat_interface(user_input, chat_history, model_choice):
    if not user_input:
        return chat_history, "", None

    model = deepseek if model_choice == "DeepSeek" else gemma
    response = model.chat(user_input)

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return chat_history, "", None  # Limpiar campo de texto y audio

# Procesar audio -> transcribir -> enviar al input de texto
def audio_to_text(audio_path):
    if audio_path is None or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "[ERROR: Audio inválido]"
    return speech.transcribe(audio_path)

# UI
with gr.Blocks() as ui:
    gr.Markdown("# 🤖 VChatbot")
    gr.Markdown("Chat por voz o texto con DeepSeek o Gemma 3")

    with gr.Row():
        model_selector = gr.Radio(["DeepSeek", "Gemma 3"], label="🔍 Elige el modelo", value="DeepSeek")

    chatbot = gr.Chatbot(label="💬 Conversación", type="messages")
    state = gr.State([])

    with gr.Row():
        text_input = gr.Textbox(placeholder="Escribe un mensaje...", label="Entrada de texto", lines=2)
        send_btn = gr.Button("Enviar")

    audio_input = gr.Audio(type="filepath", label="🎙️ O graba tu voz", interactive=True)

    # Al hacer click en "Enviar" o presionar Enter
    send_btn.click(chat_interface, inputs=[text_input, state, model_selector], outputs=[chatbot, text_input, audio_input])
    text_input.submit(chat_interface, inputs=[text_input, state, model_selector], outputs=[chatbot, text_input, audio_input])

    # Cuando se graba audio, se transcribe y se coloca en el campo de texto
    audio_input.change(audio_to_text, inputs=audio_input, outputs=text_input)

ui.launch()
