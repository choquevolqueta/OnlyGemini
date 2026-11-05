# Vendored from simple-lama-inpainting 0.1.2 (Apache-2.0).
# https://github.com/enesmsahin/simple-lama-inpainting
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir

LAMA_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_MODEL_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Input image should be either PIL Image or numpy array")
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = img[np.newaxis, ...]
    if img.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got shape {img.shape}")
    return img.astype(np.float32) / 255


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)
    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    _, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)
    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)
    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)
    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)
    out_mask = (out_mask > 0) * 1
    return out_image, out_mask


def _user_model_path() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / "onlygemini" / "big-lama.pt"
    except Exception:
        return Path(__file__).resolve().parent.parent / "models" / "big-lama.pt"


def _hub_model_path() -> Path:
    return Path(get_dir()) / "checkpoints" / "big-lama.pt"


def resolve_model_path() -> Path:
    configured = os.getenv("LAMA_MODEL")
    if configured:
        return Path(configured).expanduser()
    user_path = _user_model_path()
    if user_path.exists():
        return user_path
    hub_path = _hub_model_path()
    if hub_path.exists():
        return hub_path
    return user_path


def download_model(dest: Path, url: str = LAMA_MODEL_URL, sha256: str = LAMA_MODEL_SHA256) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[OnlyGemini] Downloading LaMa model to {dest} (~196 MiB)")
    download_url_to_file(url, str(dest), progress=True)
    actual = _sha256(dest)
    if actual != sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"LaMa model SHA256 mismatch: expected {sha256}, got {actual}")
    return dest


class SimpleLama:
    def __init__(self, device=None, model_path=None):
        if device is None:
            device = os.getenv("LAMA_DEVICE", "cpu")
        self.device = torch.device(device)
        path = Path(model_path).expanduser() if model_path else resolve_model_path()
        if not path.exists():
            download_model(path)
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image, mask):
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cur_res)
