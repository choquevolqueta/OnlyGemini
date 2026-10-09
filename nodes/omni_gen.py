import io

import torch

from ..utils import (
    decode_audio,
    get_client,
    interaction_video_bytes,
    tensor_to_omni_image,
    upload_file,
    video_bytes_to_frames,
    video_to_mp4_bytes,
)

MAX_INPUT_SECONDS = 10.0


def build_omni_prompt(prompt, has_video, n_refs, has_first, has_last):
    lines = []
    if has_video:
        lines.append(
            "The provided video is the source to edit, tagged <VIDEO_0>. "
            "Restyle it while preserving its camera movement, timing, and subject motion."
        )
    if n_refs:
        tags = " ".join(f"<IMAGE_REF_{i}>" for i in range(n_refs))
        lines.append(
            f"Provided aesthetic references: {tags}. Use them only as style/subject references; "
            "do not use them as literal frames or as the source video."
        )
    if has_first:
        lines.append("The first provided image is the starting frame.")
    if has_last:
        lines.append("The last provided image is the ending frame.")
    if prompt and prompt.strip():
        lines.append(prompt.strip())
    return "\n".join(lines)


class OmniFlashGen:
    """Genera y edita videos con Gemini Omni Flash (Interactions API).

    Cuándo usar: Tienes un video base (p. ej. previz de Blender: movimiento de
    cámara y de personajes) y quieres reestilizarlo/reescribirlo guiándote por
    imágenes de referencia estéticas.

    Ejemplo (tu flujo): conecta el VIDEO de Blender + 4 imágenes de referencia +
    un prompt de estilo, task=edit. El nodo sube el video, lo edita y devuelve
    los frames con el movimiento aproximado del original.

    Modos (task; 'auto' deja que el modelo lo infiera):
    - edit: edita/restyle un video de entrada (requiere VIDEO).
    - text_to_video, image_to_video, reference_to_video.

    Parámetros:
    - prompt: qué quieres (estilo/acción). Describe el movimiento deseado.
    - model: gemini-omni-1.1-flash (GA) o gemini-omni-flash-preview.
    - resolution: 720p nativo; 1080p/4k son reescalados.
    - seed: se envía a la API (Omni sí lo soporta); -1 = lo deja al modelo.

    Límites: el video subido para editar debe durar ≤10 s (si dura más se
    recorta a los primeros 10 s). Salida 3-10 s. La edición de videos subidos
    no está disponible en EEA/Suiza/UK. Se bloquea material con personas
    reconocibles o menores. El resultado NO es frame-a-frame: el modelo
    regenera guiándose por el movimiento del video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        img = {"tooltip": "Imagen de referencia estética."}
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    ["gemini-omni-1.1-flash", "gemini-omni-flash-preview"],
                    {"default": "gemini-omni-1.1-flash"},
                ),
                "task": (
                    ["auto", "edit", "text_to_video", "image_to_video", "reference_to_video"],
                    {"default": "auto"},
                ),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "resolution": (["720p", "1080p", "4k"], {"default": "720p"}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "api_key": ("STRING", {"default": "", "tooltip": "Gemini API key or env variable name"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Video base a editar/reestilizar."}),
                "ref_image_1": ("IMAGE", img),
                "ref_image_2": ("IMAGE", img),
                "ref_image_3": ("IMAGE", img),
                "ref_image_4": ("IMAGE", img),
                "ref_image_5": ("IMAGE", img),
                "ref_image_6": ("IMAGE", img),
                "first_frame": ("IMAGE", {"tooltip": "Primer fotograma (interpolación)."}),
                "last_frame": ("IMAGE", {"tooltip": "Último fotograma (interpolación)."}),
                "previous_interaction_id": (
                    "STRING",
                    {"default": "", "tooltip": "ID de una edición previa de Omni para continuar (multiturno)."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "interaction_id")
    FUNCTION = "generate"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, False, False)

    def generate(
        self,
        prompt,
        model,
        task,
        aspect_ratio,
        resolution,
        seed,
        api_key,
        video=None,
        ref_image_1=None,
        ref_image_2=None,
        ref_image_3=None,
        ref_image_4=None,
        ref_image_5=None,
        ref_image_6=None,
        first_frame=None,
        last_frame=None,
        previous_interaction_id="",
    ):
        client = get_client(api_key)

        if task == "edit" and video is None:
            raise ValueError("task='edit' requires a video input.")
        if (
            video is None
            and not prompt.strip()
            and all(t is None for t in (ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6))
            and first_frame is None
            and last_frame is None
        ):
            raise ValueError("Provide a prompt, a video, or at least one reference image.")

        inputs = []

        if video is not None:
            duration = video.get_duration()
            if duration > MAX_INPUT_SECONDS:
                trimmed = video.as_trimmed(start_time=0, duration=MAX_INPUT_SECONDS)
                if trimmed is not None:
                    print(f"[OG Omni] input video is {duration:.1f}s; trimming to {MAX_INPUT_SECONDS:.0f}s")
                    video = trimmed
            data = video_to_mp4_bytes(video)
            uploaded = upload_file(client, data, mime_type="video/mp4")
            inputs.append({"type": "video", "uri": uploaded.uri, "mime_type": "video/mp4"})

        refs = [
            t for t in (ref_image_1, ref_image_2, ref_image_3, ref_image_4, ref_image_5, ref_image_6) if t is not None
        ]
        for ref in refs:
            inputs.append(tensor_to_omni_image(ref))
        if first_frame is not None:
            inputs.append(tensor_to_omni_image(first_frame))
        if last_frame is not None:
            inputs.append(tensor_to_omni_image(last_frame))

        text = build_omni_prompt(prompt, video is not None, len(refs), first_frame is not None, last_frame is not None)
        if text.strip():
            inputs.append({"type": "text", "text": text})

        response_format = {"type": "video", "aspect_ratio": aspect_ratio, "resolution": resolution, "delivery": "uri"}
        generation_config = {"video_config": {"task": task}} if task != "auto" else {}
        if seed is not None and seed >= 0:
            generation_config["seed"] = seed

        body = {
            "model": model,
            "input": inputs,
            "response_format": response_format,
            "generation_config": generation_config,
            "stream": False,
            "background": False,
            "store": True,
        }
        if previous_interaction_id and previous_interaction_id.strip():
            body["previous_interaction_id"] = previous_interaction_id.strip()

        interaction = client.interactions.create(**body)
        raw = interaction_video_bytes(client, interaction)
        frames = video_bytes_to_frames(io.BytesIO(raw))
        audio = decode_audio(io.BytesIO(raw))
        if not frames:
            raise Exception("Failed to decode Omni video frames")
        return ([torch.cat(frames, dim=0)], audio, interaction.id)


NODE_CLASS_MAPPINGS = {"OmniFlashGen": OmniFlashGen}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniFlashGen": "OG Omni Flash"}
