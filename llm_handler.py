import requests

class LLMHandler:
    def __init__(self, model_name):
        if model_name != "deepseek":
            raise ValueError("Este handler solo soporta el modelo 'deepseek'")
        self.model_name = model_name
        print("🧠 Conectando con modelo DeepSeek vía Ollama...")

    def chat(self, prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()["response"].strip()
        except Exception as e:
            print(f"❌ Error al conectar con Ollama: {e}")
            return f"[ERROR: {e}]"
