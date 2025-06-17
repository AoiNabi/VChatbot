import re
import requests

class LLMHandler:
    def __init__(self, model_name):
        modelos_permitidos = ["deepseek-r1:7b", "gemma3:4b"]
        if model_name not in modelos_permitidos:
            raise ValueError(f"Este handler solo soporta: {modelos_permitidos}")
        self.model_name = model_name
        print(f"🧠 Conectando con modelo {self.model_name} vía Ollama...")

    def chat(self, prompt):
        try:
            system_prompt = (
                "You are an AI assistant that helps the user. "
                "Always reply with respect and kindness. "
                "Detect the language of the user's message and respond in that same language. "
                "Respond exclusively using words from that language, unless you need to mention proper names of people, companies, or brands in a foreign language."
            )
            full_prompt = system_prompt + prompt

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )

            raw_output = response.json()["response"].strip()

            cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

            return cleaned_output

        except Exception as e:
            print(f"❌ Error al conectar con Ollama: {e}")
            return f"[ERROR: {e}]"
