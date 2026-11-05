import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

image_refill = importlib.import_module(f"{_ROOT.name}.nodes.image_refill")


def test_lamarefill_uses_vendored_runtime():
    assert image_refill.SimpleLama.__module__ == f"{_ROOT.name}.lama.runtime"
