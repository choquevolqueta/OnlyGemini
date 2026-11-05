import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lama.runtime import (
    LAMA_MODEL_SHA256,
    ceil_modulo,
    get_image,
    prepare_img_and_mask,
    resolve_model_path,
)


def test_ceil_modulo():
    assert ceil_modulo(8, 8) == 8
    assert ceil_modulo(9, 8) == 16


def test_get_image_normalizes_pil_to_chw():
    arr = np.full((5, 7, 3), 255, dtype=np.uint8)
    out = get_image(Image.fromarray(arr))
    assert out.shape == (3, 5, 7)
    assert out.dtype == np.float32
    assert float(out.max()) == 1.0


def test_prepare_img_and_mask_pads_to_multiple_of_8():
    img = Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((5, 7), dtype=np.uint8))
    image_t, mask_t = prepare_img_and_mask(img, mask, torch.device("cpu"))
    assert image_t.shape == (1, 3, 8, 8)
    assert mask_t.shape == (1, 1, 8, 8)
    assert image_t.dtype == torch.float32
    assert mask_t.dtype == torch.int64
    assert set(mask_t.unique().tolist()) <= {0, 1}


def test_resolve_model_path_prefers_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom model.pt"
    monkeypatch.setenv("LAMA_MODEL", str(custom))
    assert resolve_model_path() == custom


def test_download_model_rejects_bad_sha(tmp_path, monkeypatch):
    import lama.runtime as runtime

    def fake_download(url, dest, progress=True):
        Path(dest).write_bytes(b"not the model")

    monkeypatch.setattr(runtime, "download_url_to_file", fake_download)
    dest = tmp_path / "dest.bin"
    with pytest.raises(RuntimeError, match="SHA256"):
        runtime.download_model(dest, url="https://example.invalid/model.pt", sha256="0" * 64)
    assert not dest.exists()


def test_sha256_pinned():
    assert len(LAMA_MODEL_SHA256) == 64
    int(LAMA_MODEL_SHA256, 16)


def test_crop_around_mask_at_border():
    from utils import lama_crop_around_mask

    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    _, _, bounds = lama_crop_around_mask(bgr, mask, padding=4)
    assert bounds == (0, 6, 0, 6)


def test_maybe_downsample_large_crop():
    from utils import lama_maybe_downsample

    bgr = np.zeros((2000, 1500, 3), dtype=np.uint8)
    mask = np.zeros((2000, 1500), dtype=np.uint8)
    small_bgr, small_mask, scale = lama_maybe_downsample(bgr, mask, max_dim=1024)
    assert scale < 1.0
    assert max(small_bgr.shape[:2]) == 1024
    assert small_mask.shape == small_bgr.shape[:2]
