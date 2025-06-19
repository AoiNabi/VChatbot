import gradio as gr
from llm_handler import LLMHandler
from voice_input import VoiceInput
import os

# Instanciar ambos modelos
deepseek = LLMHandler("deepseek-r1:7b")
gemma = LLMHandler("gemma3:4b")
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

    # Elegir modelo
    if model_choice == "DeepSeek":
        response = deepseek.chat(text)
    else:
        response = gemma.chat(text)

    return (
        f"🗣️ Transcripción: {text}\n\n🤖 {model_choice} responde:\n{response}",
        audio_path
    )

ui = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="🎙️ Graba tu voz o sube un archivo"),
        gr.Radio(["DeepSeek", "Gemma 3"], label="Modelo a usar")
    ],
    outputs=[
        "text",
        gr.Audio(label="🔊 Reproducir audio grabado")
    ],
    title="🤖 Vchatbot",
    description="Graba tu voz o sube un archivo de audio (.wav); elige entre DeepSeek o Gemma 3 para obtener una respuesta generada por IA."
)

ui.launch()
