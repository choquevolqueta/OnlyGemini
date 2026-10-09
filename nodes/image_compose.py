from google.genai import types
from PIL import Image

from ..utils import get_client, pil_to_part, response_to_images, tensor_to_pil


class GeminiOrchestrate:
    """Compone múltiples imágenes limpias en una escena final usando Gemini.

    Cuándo usar: Tienes 2+ imágenes (fondo, sujetos, objetos) que quieres
    combinar en una sola escena coherente. Cada imagen debe estar limpia
    (sin objetos no deseados) — usa OG LaMa Refill primero.

    Ejemplo: Tienes un fondo de pista de nieve (sin cabaña), una pareja
    esquiando, y unas chaquetas rojas. Conectas las 3 imágenes, escribes
    "couple skiing on a snowy slope wearing bright red jackets, sunny day",
    y Gemini genera la escena compuesta.

    Parámetros:
    - prompt: Describe la escena final. Sé específico sobre posiciones,
      iluminación, y estilo. "couple on the left, red jackets prominent"
      es mejor que "couple skiing".
    - model: flash es rápido, pro es mejor calidad para composiciones
      complejas.
    - temperature (0.5): Bajo = resultado predecible, Alto = más variado.
    - ref_image_1 a ref_image_5: Elementos opcionales para incluir en la
      escena (sujetos, objetos, ropa, etc.). Conecta lo que quieras.

    Tips: Las imágenes se redimensionan automáticamente al tamaño del
    background, así que puedes conectar referencias de cualquier
    resolución.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Describe la escena final. Sé específico sobre posiciones y estilo.",
                    },
                ),
                "api_key": ("STRING", {"default": "", "tooltip": "API key o nombre de variable de entorno"}),
                "model": (["gemini-3.1-flash-image", "gemini-3-pro-image"], {"default": "gemini-3.1-flash-image"}),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"],
                    {"default": "16:9"},
                ),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "orchestrate"
    CATEGORY = "image/generation"

    def orchestrate(
        self,
        background,
        prompt,
        api_key,
        model,
        aspect_ratio,
        resolution,
        temperature,
        top_p,
        seed,
        system_instruction="",
        **kwargs,
    ):
        client = get_client(api_key)

        refs = []
        for i in range(1, 6):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                refs.append(img)

        context = "Compose a scene using these reference images:\n- Image 1: background/environment\n"
        for i in range(len(refs)):
            context += f"- Image {i + 2}: element to include (reference {i + 1})\n"
        context += f"\nUser description: {prompt}"

        bg_pil = tensor_to_pil(background)

        parts = [types.Part.from_text(text=context)]
        parts.append(pil_to_part(bg_pil))
        for img in refs:
            parts.append(pil_to_part(tensor_to_pil(img).resize(bg_pil.size, Image.LANCZOS)))

        img_config = {"aspect_ratio": aspect_ratio}
        if "gemini-3-pro" in model:
            img_config["image_size"] = resolution

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            top_p=top_p,
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(**img_config),
            system_instruction=system_instruction.strip() or None,
        )

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        return (response_to_images(response),)


NODE_CLASS_MAPPINGS = {"GeminiOrchestrate": GeminiOrchestrate}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiOrchestrate": "OG Gemini Orchestrate"}
