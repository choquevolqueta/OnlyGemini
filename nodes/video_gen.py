import torch
from google.genai import types

from ..utils import decode_audio, decode_video, get_client, poll_operation, tensor_to_api_image


class VeoGenerate:
    """Genera videos con Gemini Veo. Modos: text-to-video, first+last, ingredients, extend.

    Cuándo usar: Quieres crear un video desde cero, entre dos frames,
    usando imágenes de referencia, o extender un video anterior.

    Ejemplo (text-to-video): Solo prompt + duración → video nuevo.
    Ejemplo (first+last): Conectas first_image + last_image → transición.
    Ejemplo (ingredients): Conectas reference_images → estilo guía.
    Ejemplo (extend): Conectas video_uri de un Veo previo → continúa.

    Parámetros:
    - prompt: Describe la acción o transición del video.
    - model: generate es calidad completa, fast es rápido, lite es más rápido.
    - duration_seconds (8): Duración del video. 4-8 segundos.
    - resolution: 720p, 1080p, o 4k.
    - seed: solo para compatibilidad de workflows guardados; NO se envía a la API
      (Gemini Developer API no soporta seed de video).

    Modos (detectados automáticamente por inputs conectados):
    - Sin image/last_image/reference_images/video_uri → text-to-video
    - first_image + last_image → first+last (transición entre frames)
    - reference_images → ingredients (imágenes de referencia)
    - video_uri → extend (continuar video anterior)

    Tips: La calidad del prompt es clave. Sé específico sobre movimientos,
    iluminación, y estilo. Para first+last, usa imágenes con composición
    similar para transiciones suaves.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "model": (
                    ["veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-preview"],
                    {"default": "veo-3.1-generate-preview"},
                ),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 8, "min": 4, "max": 8, "step": 1}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "api_key": ("STRING", {"default": "", "tooltip": "Gemini API key or env variable name"}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE", {"tooltip": "First frame image for first+last mode."}),
                "last_image": ("IMAGE", {"tooltip": "Last frame image. Connect with first_image for first+last mode."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference images for ingredients mode. Up to 3."}),
                "video_uri": ("STRING", {"default": "", "tooltip": "URI from previous Veo output. For extend mode."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "video_uri")
    FUNCTION = "generate"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, False, False)

    def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        duration_seconds,
        resolution,
        api_key,
        seed=None,
        negative_prompt=None,
        image=None,
        last_image=None,
        reference_images=None,
        video_uri="",
    ):
        client = get_client(api_key)

        config = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
            number_of_videos=1,
        )
        if negative_prompt and negative_prompt.strip():
            config.negative_prompt = negative_prompt.strip()

        source_kwargs = {"prompt": prompt}

        if video_uri and video_uri.strip().startswith(("http://", "https://")):
            video_obj = types.Video(uri=video_uri.strip())
            source_kwargs["video"] = video_obj
        elif last_image is not None and image is not None:
            config.last_frame = tensor_to_api_image(last_image)
            source_kwargs["image"] = tensor_to_api_image(image)
        elif image is not None:
            source_kwargs["image"] = tensor_to_api_image(image)

        if reference_images is not None:
            ref_imgs = []
            for i in range(min(len(reference_images), 3)):
                ref_imgs.append(
                    types.VideoGenerationReferenceImage(
                        image=tensor_to_api_image(reference_images[i]),
                        reference_type=types.VideoGenerationReferenceType.ASSET,
                    )
                )
            config.reference_images = ref_imgs

        op = client.models.generate_videos(
            model=model,
            source=types.GenerateVideosSource(**source_kwargs),
            config=config,
        )
        video_obj = poll_operation(client, op)
        frames, video_bytes = decode_video(client, video_obj)
        audio = decode_audio(video_bytes)
        if not frames:
            raise Exception("Failed to decode video frames")
        return ([torch.cat(frames, dim=0)], audio, video_obj.uri)


NODE_CLASS_MAPPINGS = {"VeoGenerate": VeoGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"VeoGenerate": "OG Veo Generate"}
