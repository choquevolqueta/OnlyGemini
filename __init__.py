import importlib
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_node_modules = [
    "image_gen",
    "image_refill",
    "image_compose",
    "video_gen",
    "omni_gen",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod_name in _node_modules:
    _mod = importlib.import_module(f".nodes.{_mod_name}", package=__name__)
    if hasattr(_mod, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    if hasattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
