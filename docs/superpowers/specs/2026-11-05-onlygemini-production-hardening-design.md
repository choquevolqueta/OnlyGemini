# OnlyGemini — Production hardening

- **Fecha:** 2026-11-05
- **Rama:** `fix/production-hardening`
- **Estado:** propuesto
- **Alcance:** 1.1.0 (LaMa + API key + flag del Registry)

## 1. Intención

Una instalación limpia desde el Registry de ComfyUI carga el pack, pero:

1. `LaMaRefill` no funciona porque `simple-lama-inpainting` no se puede instalar en
   el entorno actual (Python 3.13, numpy 2.5, Pillow 13, opencv 5) y el Manager
   tiene `allow_pip_install = False`.
2. La API key pegada en el widget queda persistida en el workflow, en el
   autoguardado del frontend (LevelDB) y en la caché de Chromium. Eso permite
   filtrarla al compartir un workflow o un error report.
3. Las versiones 1.0.0 y 1.0.1 están `NodeVersionStatusFlagged` con
   `policy-v0.6: credential-exfiltration`, disparado por la resolución dinámica
   de variables de entorno en `utils.py`.

El objetivo es publicar una 1.1.0 que funcione en una instalación limpia, que no
persista la key y que pueda pasar la revisión del Registry.

## 2. Objetivos

- `LaMaRefill` funciona sin `pip install` manual y sin degradar numpy, Pillow ni
  opencv.
- La API key se lee solo de `GEMINI_API_KEY` o de un `.env` local; nunca del
  workflow.
- El nodo deja de resolver nombres de variables de entorno elegidos por el
  workflow.
- Se mantienen nombres de nodos, tipos de entrada/salida y el orden posicional
  de los widgets existentes.
- Tests automatizados para la lógica que no requiere el modelo y smoke test
  manual documentado para el modelo.

## 3. No objetivos

- Migrar los nodos a V3 schemas.
- Implementar un keyring del sistema operativo o un almacén de secretos cloud.
- Arreglar las versiones 1.0.0/1.0.1 ya publicadas; seguirán flagged.
- Cambiar modelos, prompts o comportamiento de la API de Gemini.
- Borrar automáticamente datos del perfil de ComfyUI Desktop.

## 4. Restricciones

- Instalación CNR; `ComfyUI-Manager` con `allow_pip_install = False`.
- Python 3.13.12, numpy 2.5.4, Pillow 13.0.0, opencv-python 5.0.0.93,
  torch 2.12.1+xpu, google-genai 2.34.0.
- El repo fuente es el clone del escritorio; la copia instalada en
  `custom_nodes/onlygemini` es solo destino de prueba.
- `big-lama.pt` pesa ~196 MiB y se descarga desde un release de GitHub.
- El repo está dentro de OneDrive; los cambios se hacen en la rama y se
  sincronizan a mano a `custom_nodes` para probar.

## 5. Enfoques considerados

### LaMa

| Enfoque | Ventajas | Costes |
|---|---|---|
| **A. Vendorizar el runtime** (elegido) | Sin pip, sin conflicto de pins, versionado, funciona con `allow_pip_install=False` | ~100 líneas a mantener; hay que validar TorchScript en torch 2.12 |
| B. `install.py` con `pip install --no-deps` | Cambio pequeño, conserva upstream | Depende de pip/Manager, red y `--no-deps`; frágil en Desktop |
| C. ONNX Runtime | Evita TorchScript | Agrega `onnxruntime` (no instalado) y un modelo ONNX que hay que alojar |

Decisión: **A**. Si el spike demuestra que `big-lama.pt` no carga en torch
2.12, se reevalúa C.

### API key

| Enfoque | Ventajas | Costes |
|---|---|---|
| **K1. Env-only + deprecar widget** (elegido) | Sin key en workflows nuevos; elimina el lookup dinámico que disparó el flag | Cambio de hábito; el widget queda visible pero ignorado en 1.1 |
| K2. Mantener widget + advertencia | Cero fricción | No resuelve la persistencia |
| K3. Extensión web para no serializar el widget | Mantiene el widget | Frágil, depende de internals del frontend |

Decisión: **K1**. En 1.1 el widget se conserva en la misma posición para no
romper `widgets_values` antiguos, pero se ignora y avisa. En 2.0 se elimina.

## 6. Diseño: LaMa

### 6.1 Estructura

```
onlygemini/
  lama/
    __init__.py
    runtime.py      # SimpleLama + prepare_img_and_mask adaptados
    LICENSE         # Apache 2.0 del paquete original
    NOTICE          # atribución a enesmsahin/simple-lama-inpainting y LaMa
```

`runtime.py` conserva la lógica del paquete (preprocesado, padding a múltiplo de
8, inferencia TorchScript y postproceso), adaptada a numpy 2 / Pillow 13.
No se copia el CLI ni `fire`.

### 6.2 Modelo

Orden de resolución:

1. `LAMA_MODEL` (ruta explícita a un `.pt` local).
2. `folder_paths.get_user_directory()/onlygemini/big-lama.pt`.
3. Cache de `torch.hub` (`<torch.hub.get_dir()>/checkpoints/big-lama.pt`) si ya
   existe.
4. Descarga desde el release oficial a la ruta 2.

La descarga usa `torch.hub.download_url_to_file` con progreso, valida el SHA256
fijado en el código y falla con un mensaje accionable si no coincide. El SHA256
se calcula en el spike y se documenta.

### 6.3 Device

Por defecto **CPU**, porque el TorchScript de LaMa no está probado en XPU y el
equipo es Intel Iris Xe. `LAMA_DEVICE` permite forzar `cpu`, `cuda` o `xpu`.
Si el device elegido falla al cargar o inferir, el error se propaga claro; no se
hace fallback silencioso.

### 6.4 Integración

- `nodes/image_refill.py` importa `SimpleLama` desde `..lama.runtime`.
- Se mantiene la carga lazy (`_get_lama()`), para que el resto del pack no
  dependa del modelo.
- `simple-lama-inpainting` se elimina de `pyproject.toml` y
  `requirements.txt`.
- El nodo conserva su nombre, categoría, entradas y salidas.

### 6.5 Tests

- Unitario: `prepare_img_and_mask` con una imagen/máscara pequeña, verificando
  shapes, dtype y padding. No requiere el modelo.
- Unitario: `lama_crop_around_mask`, `lama_dilate`, `lama_feather_paste` ya
  existentes, sin regresiones.
- Smoke test manual: descargar el modelo, ejecutar `LaMaRefill` en ComfyUI con
  una imagen y una máscara, comprobar que el resultado no tiene el objeto.
- El test del modelo no corre en CI salvo que `LAMA_MODEL` apunte a un archivo
  local.

## 7. Diseño: API key

### 7.1 Resolución

`utils.get_client()` deja de aceptar nombres de variables de entorno. Nueva
lógica:

```python
key = (api_key or "").strip()          # solo para detectar uso legacy
if key:
    print("[OnlyGemini] api_key widget is deprecated; set GEMINI_API_KEY instead.")
key = os.getenv("GEMINI_API_KEY")
if not key:
    load_dotenv(user_env_path, override=False)
    load_dotenv(package_env_path, override=False)
    key = os.getenv("GEMINI_API_KEY")
```

La lectura de `GEMINI_API_KEY` usa un nombre fijo. No se resuelve ningún nombre
elegido por el workflow.

### 7.2 Ubicación del `.env`

Orden:

1. `folder_paths.get_user_directory()/onlygemini/.env` (persiste tras actualizar
   el pack).
2. `custom_nodes/onlygemini/.env` (compatibilidad con la instalación actual).

Se recarga en cada ejecución para que editar el archivo no obligue a reiniciar
ComfyUI.

### 7.3 Widget `api_key`

- Se conserva en la misma posición de `INPUT_TYPES` para no romper workflows
  antiguos.
- Su valor se ignora; si no está vacío, se imprime una advertencia una vez por
  ejecución.
- El tooltip y el README explican que se guarda en el workflow y que no debe
  usarse.
- Se elimina en 2.0.

### 7.4 Mensaje de error

Si no hay key:

```
API key required. Set GEMINI_API_KEY in
<user>/onlygemini/.env (or the environment). Do not paste it into the node:
widget values are saved in the workflow.
```

(La ruta exacta se calcula en runtime; el texto no incluye la key.)

### 7.5 Limpieza local

La key ya está en `Local Storage/leveldb/000003.log` y en
`Cache/Cache_Data/f_00018d` y `f_00018e` del perfil de ComfyUI Desktop. El
código nuevo no puede borrar esas copias de forma segura.

- **Recomendado:** rotar la key en AI Studio y actualizar el `.env`. Las copias
  viejas quedan invalidadas.
- **Opcional:** con ComfyUI cerrado, borrar la carpeta `Local Storage/leveldb` y
  los dos archivos de `Cache_Data`, y volver a escanear para confirmar. Esto
  puede resetear estado de la UI.
- El repo no contiene la key; no hace falta reescribir historia de Git.

### 7.6 Registry

- Publicar 1.1.0 con el lookup dinámico eliminado.
- Pedir revisión manual del flag `credential-exfiltration`; el flag no se
  limpia solo.
- Las versiones 1.0.0/1.0.1 siguen flagged; la nota de release indica
  actualizar.

## 8. Compatibilidad y versionado

- Nombres de nodos, display names, tipos y orden de entradas/salidas no cambian.
- El único cambio de comportamiento es que `api_key` deja de usarse.
- `pyproject.toml` sube a `1.1.0`.
- README: sección de migración y seguridad.
- La remoción del widget se reserva para 2.0.

## 9. Verificación

1. `ruff check .` y `ruff format --check .`.
2. Tests existentes + nuevos.
3. Sync de los archivos cambiados a `custom_nodes/onlygemini`.
4. Reiniciar ComfyUI y ejecutar:
   - `NanoBananaGen` con el widget vacío y `.env` en `user/onlygemini/.env`.
   - `NanoBananaGen` con `GEMINI_API_KEY` en el entorno del proceso.
   - `LaMaRefill` con una imagen y máscara.
5. Confirmar que un workflow guardado después del cambio tiene `api_key: ""` y
   que no aparece la key en el JSON.
6. Escanear el repo y el historial de Git por la key (debe dar 0).
7. Publicar 1.1.0 y verificar el estado en el Registry.

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `big-lama.pt` no carga en torch 2.12 | Spike previo; si falla, evaluar ONNX o re-export |
| Descarga del modelo bloqueada | `LAMA_MODEL` permite usar un archivo local; documentar |
| El scanner sigue marcando la lectura fija de `GEMINI_API_KEY` | Pedir revisión manual y explicar que el nombre es fijo y documentado |
| Workflows viejos con key | Aviso al ejecutar; recomendar rotar y limpiar el widget |
| OneDrive interfiere con `.git` | Trabajar en la rama y commitear seguido; si aparece un lock, usar un worktree local |

## 11. Orden de implementación

1. Spike TorchScript (`big-lama.pt` + torch 2.12).
2. Vendorizar LaMa + tests.
3. Key env-only + `.env` en user dir + widget deprecado.
4. Sync a `custom_nodes`, reiniciar y probar.
5. README, versión 1.1.0, commit/push.
6. Publicar y pedir revisión del flag.

## 12. Decisiones tomadas por el implementador

- CPU por defecto para LaMa; `LAMA_DEVICE` como override.
- `.env` en el user dir de ComfyUI, con fallback al directorio del pack.
- El widget se conserva ignorado en 1.1 y se elimina en 2.0.
- No se borran automáticamente datos del perfil de ComfyUI Desktop.
- El SHA256 del modelo se fija tras el spike.
