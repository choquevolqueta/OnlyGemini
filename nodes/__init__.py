from .image_compose import GeminiOrchestrate
from .image_gen import NanoBananaGen
from .image_refill import LaMaRefill
from .omni_gen import OmniFlashGen
from .video_gen import VeoGenerate

NODE_CLASS_MAPPINGS = {
    "NanoBananaGen": NanoBananaGen,
    "LaMaRefill": LaMaRefill,
    "GeminiOrchestrate": GeminiOrchestrate,
    "VeoGenerate": VeoGenerate,
    "OmniFlashGen": OmniFlashGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGen": "OG NanoBanana Generate",
    "LaMaRefill": "OG LaMa Refill",
    "GeminiOrchestrate": "OG Gemini Orchestrate",
    "VeoGenerate": "OG Veo Generate",
    "OmniFlashGen": "OG Omni Flash",
}
