import gradio as gr
from llm_handler import LLMHandler
from voice_input import VoiceInput
import os

# Cargar modelos públicos sin necesidad de autenticación
gpt2 = LLMHandler("gpt2")
distilgpt2 = LLMHandler("distilgpt2")
speech = VoiceInput()

def process(audio_path, model_choice):

    if audio_path is None:
        return "❌ No se recibió audio.", None

    print(f"📁 Archivo de audio recibido (ruta): {audio_path}")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "❌ El archivo no existe o está vacío.", None

    text = speech.transcribe(audio_path)
    if not text or text.startswith("[ERROR"):
        return f"🗣️ Transcripción: {text}\n\n❌ No se pudo obtener texto para responder.", None

    if model_choice == "GPT-2":
        response = gpt2.chat(text)
    else:
        response = distilgpt2.chat(text)

    return (
        f"🗣️ Transcripción: {text}\n\n🤖 {model_choice} responde:\n{response}",
        audio_path
    )

ui = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="🎙️ Graba tu voz o sube un archivo"),
        gr.Radio(["GPT-2", "DistilGPT-2"], label="Modelo a usar")
    ],
    outputs=[
        "text",
        gr.Audio(label="🔊 Reproducir audio grabado")
    ],
    title="🤖 Chatbot con Voz + LLMs",
    description="Graba tu voz o sube un .wav; elige GPT-2 o DistilGPT-2 y verás la transcripción + respuesta."
)

ui.launch()
