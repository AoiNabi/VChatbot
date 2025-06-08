from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class LLMHandler:
    def __init__(self, model_name):
        print(f"Cargando modelo: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def chat(self, prompt):
        result = self.generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        return result[0]["generated_text"].replace(prompt, "").strip()
