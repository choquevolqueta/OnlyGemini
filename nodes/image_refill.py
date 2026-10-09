import numpy as np
import torch
from PIL import Image as PILImage

from ..utils import (
    lama_crop_around_mask,
    lama_dilate,
    lama_feather_paste,
    lama_maybe_downsample,
)

_lama_instance = None


def _get_lama():
    global _lama_instance
    if _lama_instance is None:
        from simple_lama_inpainting import SimpleLama

        _lama_instance = SimpleLama()
    return _lama_instance


class LaMaRefill:
    """Rellena zonas borradas usando IA local (sin internet).

    Cuándo usar: Tienes una imagen con un objeto que quieres eliminar y
    necesitas que el fondo quede limpio.

    Ejemplo: Foto de un paisaje con un poste de luz. Pintas el poste con
    MaskEditor, conectas a OG LaMa Refill, y el nodo rellena el área
    donde estaba el poste con textura coherente.

    Parámetros:
    - dilate_px (10): Engorda la máscara. Sube si el objeto tiene bordes
      suaves (pelo, humo). Baja para objetos definidos.
    - feather_px (8): Suaviza el borde del relleno. Mayor = transición más suave.
    - passes (1): Cuántas veces LaMa procesa. 2-3 para texturas complejas
      como madera o ladrillo. 1 para escenas limpias.

    Tips: LaMa funciona mejor con imágenes ≤1024px. El nodo hace downsample
    automático y upscale después. Para objetos grandes, sube dilate_px.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "dilate_px": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Expande la máscara. Sube para objetos con bordes suaves.",
                    },
                ),
                "feather_px": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 30,
                        "step": 1,
                        "tooltip": "Suaviza el borde del relleno. 0 = sin suavizado.",
                    },
                ),
                "passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Iteraciones de LaMa. Más pasadas = más suave pero más lento.",
                    },
                ),
                "context_padding": (
                    "INT",
                    {
                        "default": 160,
                        "min": 20,
                        "max": 400,
                        "step": 10,
                        "tooltip": "Px de contexto alrededor de la máscara. Más = mejor coherencia.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "refill"
    CATEGORY = "image/editing"

    def refill(self, image, mask, dilate_px, feather_px, passes, context_padding):
        import cv2

        bgr = cv2.cvtColor((image[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask_uint8 = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        if mask_uint8.ndim == 3:
            mask_uint8 = mask_uint8[:, :, 0]

        if not np.any(mask_uint8):
            return (image,)

        work_mask = lama_dilate(mask_uint8, dilate_px)
        crop_bgr, crop_mask, bounds = lama_crop_around_mask(bgr, work_mask, context_padding)
        proc_bgr, proc_mask, scale = lama_maybe_downsample(crop_bgr, crop_mask, 1024)

        proc_result = proc_bgr
        lama = _get_lama()
        for _ in range(passes):
            img_pil = PILImage.fromarray(cv2.cvtColor(proc_result, cv2.COLOR_BGR2RGB))
            mask_pil = PILImage.fromarray(proc_mask, mode="L")
            out_pil = lama(img_pil, mask_pil)
            proc_result = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

        if scale < 1.0:
            ch, cw = crop_bgr.shape[:2]
            proc_result = cv2.resize(proc_result, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        result_bgr = lama_feather_paste(bgr, crop_bgr, proc_result, crop_mask, bounds, feather_px)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "LaMaRefill": LaMaRefill,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LaMaRefill": "OG LaMa Refill",
}
