from google.genai import types

from ..utils import get_client, response_to_images, tensor_to_part


class NanoBananaGen:
    """Genera o edita imágenes usando Gemini (Nano Banana).

    Cuándo usar: Quieres crear una imagen nueva desde un prompt, o editar
    una imagen existente describiendo el cambio.

    Ejemplo (generación): Conectas solo el prompt "a cat wearing sunglasses
    on a beach", y el nodo genera la imagen.

    Ejemplo (edición): Conectas una foto + prompt "make the sky sunset orange",
    y el nodo edita solo el cielo manteniendo el resto.

    Parámetros:
    - prompt: Describe la imagen o el cambio. Sé específico para mejores
      resultados. En edición, describe solo lo que cambia.
    - model: flash-image es rápido. pro-image es mejor calidad.
    - temperature (0.5): Bajo = resultado predecible, Alto = más variado.
    - ref_image_1 a ref_image_5: Imágenes de referencia opcionales para
      guiar el estilo o contenido.

    Tips: Este nodo es mejor para cambios generales en la imagen completa.
    Para relleno localizado con máscara, usa OG LaMa Refill.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": "", "tooltip": "Gemini API key or env variable name"}),
                "model": (
                    ["gemini-3.1-flash-image", "gemini-3-pro-image", "gemini-3.1-flash-lite-image"],
                    {"default": "gemini-3.1-flash-image"},
                ),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    {"default": "1:1"},
                ),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "google_search": ("BOOLEAN", {"default": False}),
                "image": (
                    "IMAGE",
                    {"tooltip": "Optional reference image for editing. Leave empty for pure generation."},
                ),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"

    def generate(
        self,
        api_key,
        model,
        prompt,
        aspect_ratio,
        resolution,
        temperature,
        top_p,
        seed,
        system_instruction="",
        google_search=False,
        image=None,
        **kwargs,
    ):
        client = get_client(api_key)

        parts = []
        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))
        if image is not None:
            parts.append(tensor_to_part(image))
        for i in range(1, 6):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                parts.append(tensor_to_part(img))
        if not parts:
            raise ValueError("At least one image or prompt required.")

        tools = None
        if google_search and "gemini-2.5" not in model and "lite" not in model:
            tools = [types.Tool(google_search=types.GoogleSearch())]

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
            tools=tools,
        )

        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        return (response_to_images(response),)


NODE_CLASS_MAPPINGS = {"NanoBananaGen": NanoBananaGen}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaGen": "OG NanoBanana Generate"}
