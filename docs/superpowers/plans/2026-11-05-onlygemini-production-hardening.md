# OnlyGemini Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar OnlyGemini 1.1.0 con LaMa funcionando sin dependencias rotas y con la API key leída solo de entorno/`.env`, sin persistirse en workflows.

**Architecture:** Se vendoriza el runtime Apache-2.0 de `simple-lama-inpainting` dentro del pack y se elimina la dependencia que no instala en Python 3.13. En paralelo, `get_client()` deja de resolver nombres de variables de entorno y lee únicamente `GEMINI_API_KEY` desde el entorno o desde `user/onlygemini/.env`. El widget `api_key` se conserva en su posición para no romper workflows viejos, pero se ignora y se elimina en 2.0.

**Tech Stack:** Python 3.13, ComfyUI 0.44, torch 2.12.1+xpu, numpy 2.5, Pillow 13, opencv 5, google-genai 2.34, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-05-onlygemini-production-hardening-design.md`

## Global Constraints

- Entorno de referencia: Python 3.13.12, numpy 2.5.4, Pillow 13.0.0, opencv-python 5.0.0.93, torch 2.12.1+xpu, google-genai 2.34.0.
- No agregar dependencias pip nuevas. `simple-lama-inpainting` se elimina.
- Mantener nombres de nodos, display names, tipos y orden posicional de widgets.
- El código vendorizado es Apache 2.0: incluir `LICENSE` y `NOTICE` con atribución.
- El repo fuente es `C:\Users\Personal\OneDrive\Desktop\OG\OnlyGemini`; la copia instalada en `custom_nodes/onlygemini` es solo destino de prueba.
- La versión final es `1.1.0`; la remoción del widget `api_key` se reserva para 2.0.
- No imprimir ni commitear la API key real.
- Para tests/lint, usar el Python del venv de ComfyUI (tiene torch, numpy,
  google-genai, av) y instalar `requirements-dev.txt` (`pytest`, `ruff`) en
  ese venv o en uno de desarrollo. Son dependencias dev: no van a
  `pyproject.toml` ni al runtime. Si no están en PATH, usar
  `python -m pytest` y `python -m ruff`.

## Review Focus

1. Un workflow guardado con 1.0.x y un `api_key` no vacío debe seguir cargando y ejecutando con la key del entorno, sin corrimiento de widgets ni crash.
2. Editar `user/onlygemini/.env` con ComfyUI corriendo debe tomar efecto en la siguiente ejecución, sin reiniciar.
3. `LaMaRefill` con una máscara que toca el borde de la imagen o con un crop mayor a 1024 px debe seguir recortando, bajando resolución y pegando sin errores.
4. Un `LAMA_MODEL` corrupto o con SHA256 incorrecto debe fallar con un error claro y no dejar un archivo parcial.
5. Rutas con espacios, acentos o dentro de OneDrive para `.env` y para el modelo deben funcionar.

---

### Task 1: Spike de TorchScript

**Files:** ninguno en el repo (solo archivos temporales).

**Interfaces:**
- Consumes: nada.
- Produces: `LAMA_MODEL_SHA256` (hash hex de 64 caracteres) y un go/no-go sobre TorchScript.

- [ ] **Step 1: Descargar el modelo a una carpeta temporal**

```powershell
$u = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
$d = "C:\Users\Personal\AppData\Local\Temp\opencode\big-lama.pt"
Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $d
Get-FileHash $d -Algorithm SHA256
```

Expected: archivo de 205.803.670 bytes y un SHA256 de 64 hex. Anotar el hash.

- [ ] **Step 2: Verificar que torch 2.12 lo carga y ejecuta**

```python
import torch

model = torch.jit.load(r"C:\Users\Personal\AppData\Local\Temp\opencode\big-lama.pt", map_location="cpu")
model.eval()
image = torch.rand(1, 3, 64, 64)
mask = torch.zeros(1, 1, 64, 64)
with torch.inference_mode():
    out = model(image, mask)
print(out.shape)
```

Expected: `torch.Size([1, 3, 64, 64])`.

- [ ] **Step 3: Registrar el resultado**

Si carga: anotar el SHA256 para Task 2 y continuar. Si falla: **STOP**, reportar el error y no continuar con Task 2; el plan de LaMa pasa a evaluar ONNX o re-export.

---

### Task 2: Vendorizar el runtime de LaMa

**Files:**
- Create: `lama/__init__.py`
- Create: `lama/runtime.py`
- Create: `lama/LICENSE`
- Create: `lama/NOTICE`
- Test: `tests/test_lama.py`

**Interfaces:**
- Consumes: `LAMA_MODEL_SHA256` de Task 1.
- Produces:
  - `class SimpleLama: def __init__(self, device: str | None = None, model_path: str | Path | None = None)`
  - `def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None) -> tuple[torch.Tensor, torch.Tensor]`
  - `def resolve_model_path() -> Path`
  - `def download_model(dest: Path, url: str = LAMA_MODEL_URL, sha256: str = LAMA_MODEL_SHA256) -> Path`
  - `LAMA_MODEL_URL: str`, `LAMA_MODEL_SHA256: str`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_lama.py`:

```python
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
```

- [ ] **Step 2: Ejecutar los tests y verificar que fallan**

Run: `pytest tests/test_lama.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'lama'`.

- [ ] **Step 3: Implementar `lama/runtime.py`**

Crear `lama/runtime.py` con este contenido. Reemplazar `<SHA256 de Task 1>` por el hash real antes de correr los tests; si queda el placeholder, `test_sha256_pinned` falla.

```python
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
LAMA_MODEL_SHA256 = "<SHA256 de Task 1>"


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
```

- [ ] **Step 4: Crear `lama/__init__.py`, `LICENSE` y `NOTICE`**

`lama/__init__.py`:

```python
"""Vendored LaMa runtime (Apache-2.0)."""
```

`lama/LICENSE`: copiar el texto completo de Apache 2.0 desde
`https://www.apache.org/licenses/LICENSE-2.0.txt`.

`lama/NOTICE`:

```text
Vendored from simple-lama-inpainting 0.1.2
https://github.com/enesmsahin/simple-lama-inpainting
Copyright (c) enesmsahin
Licensed under the Apache License, Version 2.0.

LaMa model and original implementation:
https://github.com/advimman/lama (Apache-2.0)
```

- [ ] **Step 5: Ejecutar los tests**

Run: `pytest tests/test_lama.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Lint**

Run: `ruff check lama tests/test_lama.py`
Expected: sin errores.

- [ ] **Step 7: Commit**

```bash
git add lama tests/test_lama.py
git commit -m "Vendor Apache-2.0 LaMa runtime"
```

---

### Task 3: Usar el runtime vendorizado y quitar la dependencia

**Files:**
- Modify: `nodes/image_refill.py:5-21`
- Modify: `pyproject.toml:15-21`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `lama.runtime.SimpleLama` de Task 2.
- Produces: `_get_lama() -> SimpleLama` sin `simple_lama_inpainting`.

- [ ] **Step 1: Cambiar el import y la carga lazy**

En `nodes/image_refill.py`, reemplazar:

```python
_lama_instance = None


def _get_lama():
    global _lama_instance
    if _lama_instance is None:
        from simple_lama_inpainting import SimpleLama

        _lama_instance = SimpleLama()
    return _lama_instance
```

por:

```python
from ..lama.runtime import SimpleLama

_lama_instance = None


def _get_lama():
    global _lama_instance
    if _lama_instance is None:
        _lama_instance = SimpleLama()
    return _lama_instance
```

- [ ] **Step 2: Quitar la dependencia**

En `pyproject.toml`, eliminar la línea `"simple-lama-inpainting>=0.1.2",`.
En `requirements.txt`, eliminar la línea `simple-lama-inpainting>=0.1.2`.

- [ ] **Step 3: Verificar que el nodo importa**

Run desde `C:\Users\Personal\OneDrive\Desktop\OG`:

```bash
python -c "import OnlyGemini.nodes.image_refill as m; print(m.LaMaRefill)"
```

Expected: imprime `<class 'OnlyGemini.nodes.image_refill.LaMaRefill'>`.

- [ ] **Step 4: Correr todos los tests y lint**

Run: `pytest -q`
Expected: PASS.

Run: `ruff check .`
Expected: sin errores.

- [ ] **Step 5: Commit**

```bash
git add nodes/image_refill.py pyproject.toml requirements.txt
git commit -m "Use vendored LaMa runtime"
```

---

### Task 4: Resolución de API key solo por entorno/`.env`

**Files:**
- Modify: `utils.py:10-22`
- Test: `tests/test_get_client.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `_env_paths(user_dir: Path | None = None) -> list[Path]`
  - `get_client(api_key: str = "") -> genai.Client`

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_get_client.py`:

```python
import os
from pathlib import Path

import pytest

import utils
from utils import get_client


def test_env_paths_prefers_user_dir(tmp_path):
    paths = utils._env_paths(user_dir=tmp_path)
    assert paths[0] == tmp_path / "onlygemini" / ".env"
    assert paths[-1] == Path(utils.__file__).parent / ".env"


def test_env_paths_handles_spaces_and_unicode(tmp_path):
    user_dir = tmp_path / "usuario con espacios" / "ñandú"
    paths = utils._env_paths(user_dir=user_dir)
    assert paths[0] == user_dir / "onlygemini" / ".env"


def test_get_client_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(utils, "_env_paths", lambda user_dir=None: [tmp_path / "missing.env"])
    assert get_client("") is not None


def test_get_client_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=dummy\n", encoding="utf-8")
    monkeypatch.setattr(utils, "_env_paths", lambda user_dir=None: [env])
    try:
        assert get_client("") is not None
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def test_get_client_ignores_widget_env_name(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_KEY", "secret")
    monkeypatch.setattr(utils, "_env_paths", lambda user_dir=None: [tmp_path / "missing.env"])
    with pytest.raises(ValueError, match="API key required"):
        get_client("OTHER_KEY")


def test_get_client_missing_key_message(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    missing = tmp_path / "missing.env"
    monkeypatch.setattr(utils, "_env_paths", lambda user_dir=None: [missing])
    with pytest.raises(ValueError, match="API key required"):
        get_client("")
```

- [ ] **Step 2: Ejecutar los tests y verificar que fallan**

Run: `pytest tests/test_get_client.py -v`
Expected: FAIL con `AttributeError: module 'utils' has no attribute '_env_paths'`.

- [ ] **Step 3: Implementar la nueva resolución**

En `utils.py`, eliminar la llamada de módulo:

```python
load_dotenv(Path(__file__).parent / ".env")
```

y reemplazar `get_client` por:

```python
def _env_paths(user_dir: Path | None = None) -> list[Path]:
    paths = []
    if user_dir is None:
        try:
            import folder_paths

            user_dir = Path(folder_paths.get_user_directory())
        except Exception:
            user_dir = None
    if user_dir is not None:
        paths.append(user_dir / "onlygemini" / ".env")
    paths.append(Path(__file__).parent / ".env")
    return paths


def get_client(api_key: str = "") -> genai.Client:
    if api_key and api_key.strip():
        print(
            "[OnlyGemini] api_key widget is deprecated and ignored. "
            "Set GEMINI_API_KEY in user/onlygemini/.env instead."
        )
    for env_path in _env_paths():
        if env_path.is_file():
            load_dotenv(env_path, override=False)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        expected = _env_paths()[0]
        raise ValueError(
            f"API key required. Set GEMINI_API_KEY in {expected} (or the environment). "
            "Do not paste it into the node: widget values are saved in the workflow."
        )
    return genai.Client(http_options={"api_version": "v1beta"}, api_key=key)
```

- [ ] **Step 4: Ejecutar los tests y verificar que pasan**

Run: `pytest tests/test_get_client.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Correr toda la suite y lint**

Run: `pytest -q`
Expected: PASS.

Run: `ruff check .`
Expected: sin errores.

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_get_client.py
git commit -m "Read Gemini API key only from env or user .env"
```

---

### Task 5: Deprecar el widget y documentar la migración

**Files:**
- Modify: `nodes/image_gen.py:35`
- Modify: `nodes/image_compose.py:47`
- Modify: `nodes/video_gen.py:49`
- Modify: `nodes/omni_gen.py:85`
- Modify: `README.md`

**Interfaces:**
- Consumes: `get_client` de Task 4.
- Produces: tooltips y documentación que apuntan a `user/onlygemini/.env`.

- [ ] **Step 1: Cambiar los cuatro tooltips**

En cada uno de los cuatro archivos, reemplazar el dict del `api_key` por:

```python
"api_key": (
    "STRING",
    {
        "default": "",
        "tooltip": (
            "Deprecated: set GEMINI_API_KEY in user/onlygemini/.env. "
            "Values typed here are saved in the workflow and are ignored."
        ),
    },
),
```

- [ ] **Step 2: Actualizar la sección de API key del README**

Reemplazar la sección `## API key setup` por:

```markdown
## API key setup

OnlyGemini reads the key from `GEMINI_API_KEY`, either from the process
environment or from a local `.env` file:

1. **Environment variable** (recommended for CI/headless):
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
2. **`.env` file** (recommended for ComfyUI Desktop):
   ```
   ComfyUI/user/onlygemini/.env
   ```
   with:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

The old per-node `api_key` widget is **deprecated and ignored** in 1.1.
It is kept only so old workflows keep loading, and it will be removed in
2.0. Do not paste keys into it: widget values are saved in the workflow
JSON and in the frontend autosave.

If you used the widget in 1.0.x, rotate the key in AI Studio, clear the
widget, and re-save the workflow.
```

- [ ] **Step 3: Actualizar LaMa en el README**

En `## Requirements`, reemplazar la línea de LaMa por:

```markdown
- `OG LaMa Refill` downloads the LaMa model (~196 MiB) to
  `ComfyUI/user/onlygemini/big-lama.pt` on first use. Set `LAMA_MODEL` to
  use a local file, or `LAMA_DEVICE` to force `cpu`, `cuda`, or `xpu`.
  No extra pip package is required.
```

En `## Installation`, mantener el `pip install -r requirements.txt` para la
instalación manual, pero aclarar que ese archivo ya no incluye LaMa.

- [ ] **Step 4: Agregar una sección de seguridad**

Agregar después de `## API key setup`:

```markdown
## Security notes

- Never commit `.env` (it is git-ignored and excluded from the registry
  archive).
- Never share a workflow that contains a key in a widget.
- Error reports include the workflow; check it before posting.
- If a key was ever saved in a workflow or in the ComfyUI frontend
  autosave, rotate it in AI Studio. Deleting the widget value does not
  erase old LevelDB/cache copies.
```

- [ ] **Step 5: Lint y tests**

Run: `ruff check .`
Expected: sin errores.

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add nodes/image_gen.py nodes/image_compose.py nodes/video_gen.py nodes/omni_gen.py README.md
git commit -m "Deprecate api_key widget and document env setup"
```

---

### Task 6: Sincronizar a ComfyUI y verificar manualmente

**Files:** ninguno en el repo; copia a `custom_nodes/onlygemini`.

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: evidencia de que ambos nodos funcionan en ComfyUI Desktop.

- [ ] **Step 1: Copiar los archivos de runtime**

```powershell
$src = "C:\Users\Personal\OneDrive\Desktop\OG\OnlyGemini"
$dst = "C:\Users\Personal\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\onlygemini"
Copy-Item "$src\utils.py" $dst -Force
Copy-Item "$src\__init__.py" $dst -Force
Copy-Item "$src\nodes\*" "$dst\nodes\" -Force
if (Test-Path "$dst\lama") { Remove-Item "$dst\lama" -Recurse -Force }
Copy-Item "$src\lama" "$dst\lama" -Recurse -Force
Copy-Item "$src\pyproject.toml" $dst -Force
```

No copiar `.env`, `.git`, `tests` ni `docs`.

- [ ] **Step 2: Mover la key al `.env` del user dir**

Crear `C:\Users\Personal\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\onlygemini\.env` con:

```
GEMINI_API_KEY=<la key real>
```

Mantener el `.env` viejo del paquete hasta verificar; luego se puede borrar.

- [ ] **Step 3: Reiniciar ComfyUI Desktop**

Esperar el log de import de `onlygemini`; no debe haber traceback.

- [ ] **Step 4: Probar NanoBanana con el widget vacío**

Abrir el workflow, confirmar que `api_key` está vacío, ejecutar. Expected: imagen generada.

- [ ] **Step 5: Probar LaMa**

Conectar una imagen y una máscara a `OG LaMa Refill`, ejecutar. Expected: en el primer run aparece `[OnlyGemini] Downloading LaMa model...`; al terminar, la zona enmascarada queda rellena.

- [ ] **Step 6: Verificar que la key no se guarda en el workflow**

Guardar el workflow y confirmar que el nodo tiene `"api_key": ""`. Buscar `AQ.` o `AIza` en el JSON:

```powershell
Select-String -Path "C:\ruta\al\workflow.json" -Pattern "AQ\.|AIza"
```

Expected: sin coincidencias.

- [ ] **Step 7: Compatibilidad con un workflow 1.0.x**

Cargar un workflow guardado antes del cambio que tenga `api_key` no vacío.
Confirmar que:

- El nodo carga sin corrimiento de widgets (prompt/model/aspect ratio siguen
  en su lugar).
- La ejecución usa `GEMINI_API_KEY` del entorno/`.env`, no el valor del widget.
- El log muestra el aviso de deprecación una vez.

- [ ] **Step 8: Rotar la key y limpiar copias locales**

La key vieja sigue en el autoguardado y la caché de ComfyUI Desktop aunque el
widget esté vacío. Lo recomendado es rotarla en AI Studio y actualizar
`user/onlygemini/.env`.

Si además se quiere borrar las copias locales, cerrar ComfyUI Desktop y borrar:

```text
C:\Users\Personal\AppData\Roaming\Comfy Desktop\Partitions\inst-1793845564532\Local Storage\leveldb
C:\Users\Personal\AppData\Roaming\Comfy Desktop\Partitions\inst-1793845564532\Cache\Cache_Data\f_00018d
C:\Users\Personal\AppData\Roaming\Comfy Desktop\Partitions\inst-1793845564532\Cache\Cache_Data\f_00018e
```

Esto puede resetear estado de la UI. Volver a escanear el perfil y el repo: la
key solo debe aparecer en `user/onlygemini/.env` (o en el `.env` legacy del
paquete, si todavía no se borró).

- [ ] **Step 9: Commit de cualquier ajuste**

Si hubo correcciones durante la prueba:

```bash
git add -A
git commit -m "Fix issues found during ComfyUI verification"
```

---

### Task 7: Versión 1.1.0, release y revisión del flag

**Files:**
- Modify: `pyproject.toml:3`
- Modify: `README.md` (notas de release)
- Modify: `docs/superpowers/specs/2026-11-05-onlygemini-production-hardening-design.md` (estado a `implementado`, opcional)

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces: rama lista, versión 1.1.0, release publicada y pedido de revisión.

- [ ] **Step 1: Subir la versión**

En `pyproject.toml`, cambiar `version = "1.0.1"` por `version = "1.1.0"`.

- [ ] **Step 2: Agregar notas de release al README**

Agregar al final:

```markdown
## 1.1.0

- LaMa now ships with a vendored Apache-2.0 runtime; no extra pip package
  and no numpy/Pillow/opencv downgrade.
- The Gemini API key is read only from `GEMINI_API_KEY` or
  `ComfyUI/user/onlygemini/.env`.
- The `api_key` widget is deprecated and ignored; it will be removed in 2.0.
- Removed the dynamic environment-variable lookup that triggered the
  registry security flag.
```

- [ ] **Step 3: Verificación final**

Run: `pytest -q`
Expected: PASS.

Run: `ruff check .`
Expected: sin errores.

Run: `git status --porcelain`
Expected: solo los archivos esperados.

- [ ] **Step 4: Commit y push**

```bash
git add pyproject.toml README.md
git commit -m "Bump version to 1.1.0"
git push -u origin fix/production-hardening
```

- [ ] **Step 5: Merge a main**

```bash
git checkout main
git pull --ff-only origin main
git merge --ff-only fix/production-hardening
git push origin main
```

- [ ] **Step 6: Publicar en el Registry (manual)**

Publicar 1.1.0 desde la web del Registry o `comfy-cli` con la cuenta
`choquevolqueta22`. Verificar que el estado de la versión no sea
`NodeVersionStatusFlagged`.

- [ ] **Step 7: Pedir revisión manual del flag (manual)**

Abrir un issue en `https://github.com/Comfy-Org/registry-backend/issues` con:

```text
Node: onlygemini
Version: 1.1.0
Previous status: NodeVersionStatusFlagged
Reason: policy-v0.6: credential-exfiltration (utils.py:19)

Fix: the dynamic environment-variable lookup was removed. The node now
reads only the fixed, documented GEMINI_API_KEY (environment or local
.env). The value is sent only to the Gemini API endpoint. There is no
dynamic env-name resolution and no credential collection.

Please re-scan / manually review 1.1.0.
```

- [ ] **Step 8: Cerrar el diseño**

Marcar el design doc como `implementado` y commitear:

```bash
git add docs/superpowers/specs/2026-11-05-onlygemini-production-hardening-design.md
git commit -m "Mark production hardening design as implemented"
git push origin main
```
