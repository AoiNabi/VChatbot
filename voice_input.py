import whisper
import os

class VoiceInput:
    def __init__(self):
        print("🧠 Cargando modelo Whisper...")
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path="audio.wav"):
        if not os.path.exists(audio_path):
            print(f"❌ Archivo de audio no encontrado: {audio_path}")
            return "[ERROR: Audio no encontrado]"

        try:
            print(f"🎧 Transcribiendo: {audio_path}")
            result = self.model.transcribe(audio_path)
            print(f"📜 Texto transcrito: {result['text']}")
            return result["text"].strip()
        except Exception as e:
            print(f"⚠️ Error en transcripción: {e}")
            return f"[ERROR EN TRANSCRIPCIÓN: {e}]"
