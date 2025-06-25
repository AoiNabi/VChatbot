import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

class ImageGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-1"):
        print(f"🧠 Cargando modelo de generación de imágenes: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cargar modelo
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt, image_path="generated_image.png"):
        if not prompt:
            return None

        print(f"🎨 Generando imagen para: '{prompt}'")
        image = self.pipe(prompt).images[0]
        image.save(image_path)
        return image_path
