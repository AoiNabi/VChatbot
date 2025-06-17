import re
import requests

class LLMHandler:
    def __init__(self, model_name):
        if model_name != "deepseek":
            raise ValueError("Este handler solo soporta el modelo 'deepseek'")
        self.model_name = model_name
        print("🧠 Conectando con modelo DeepSeek vía Ollama...")

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
                    "model": "deepseek-r1:7b",
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
