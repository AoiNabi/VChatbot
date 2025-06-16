import gradio as gr
from llm_handler import LLMHandler
from voice_input import VoiceInput
import os

# Solo modelo DeepSeek a través de Ollama
deepseek = LLMHandler("deepseek")
speech = VoiceInput()

def process(audio_path):
    if audio_path is None:
        return "❌ No se recibió audio.", None

    print(f"📁 Archivo de audio recibido (ruta): {audio_path}")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return "❌ El archivo no existe o está vacío.", None

    text = speech.transcribe(audio_path)
    if not text or text.startswith("[ERROR"):
        return f"🗣️ Transcripción: {text}\n\n❌ No se pudo obtener texto para responder.", None

    response = deepseek.chat(text)

    return (
        f"🗣️ Transcripción: {text}\n\n🤖 DeepSeek responde:\n{response}",
        audio_path
    )

ui = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="🎙️ Graba tu voz o sube un archivo")
    ],
    outputs=[
        "text",
        gr.Audio(label="🔊 Reproducir audio grabado")
    ],
    title="🤖 Chatbot con Voz + DeepSeek",
    description="Graba tu voz o sube un .wav; el modelo DeepSeek responderá con texto generado."
)

ui.launch()
