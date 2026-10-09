# OnlyGemini — ComfyUI Custom Nodes

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20nodes-8a2be2.svg)

A small, focused pack of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes that wrap **Google Gemini APIs** for image generation, multi-image composition, and video generation — plus one fully **local** (offline) inpainting node.

> **5 nodes: 4 use the Gemini API (`google-genai`), 1 is 100% local (LaMa, no internet).**

---

## Features

| Node | Display name | Category | API? | What it does |
|------|--------------|----------|------|--------------|
| `NanoBananaGen` | **OG NanoBanana Generate** | image/generation | ✅ Gemini | Text-to-image and image editing (img2img), up to 5 reference images |
| `LaMaRefill` | **OG LaMa Refill** | image/editing | ❌ Local | Inpainting: remove objects with a mask (LaMa, no internet) |
| `GeminiOrchestrate` | **OG Gemini Orchestrate** | image/generation | ✅ Gemini | Compose one scene from a background + up to 5 references |
| `VeoGenerate` | **OG Veo Generate** | video/generation | ✅ Gemini | Veo video generation (text2video, first+last, ingredients, extension) |
| `OmniFlashGen` | **OG Omni Flash** | video/generation | ✅ Gemini | Edit/restyle an existing video with Gemini Omni Flash + up to 6 reference images |

---

## Requirements

- ComfyUI (recent version with the native `VIDEO` type for `OG Omni Flash`).
- Python 3.10+.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- `OG LaMa Refill` downloads the LaMa model (~200 MB) to disk on first use; afterwards it runs entirely offline.

---

## Installation

### Via ComfyUI Manager (recommended)

1. Open **ComfyUI Manager** → **Install Custom Nodes**.
2. Search for "OnlyGemini" and install.
3. Restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Choquevolqueta/OnlyGemini.git
cd OnlyGemini
pip install -r requirements.txt
```

---

## API key setup

Three options, in order of preference:

1. **Environment variable** (recommended):
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
2. **`.env` file** in the pack folder (git-ignored):
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
3. **Per-node**: type the key directly into each node's `api_key` field.

> `api_key` also accepts the **name** of an environment variable. Connect a text widget with the value `GEMINI_API_KEY` and the pack resolves it automatically.

---

## Node reference

All display names are prefixed with `OG` so they group together in the node search.

### 1. OG NanoBanana Generate (`NanoBananaGen`)

Text-to-image and image editing via `generate_content`.

- **Text-to-image**: connect only `prompt`.
- **Image editing**: connect `image` + a prompt describing the change.
- **Reference guidance**: connect `ref_image_1..5`.

**Inputs:** `prompt`, `api_key`, `model`, `aspect_ratio`, `resolution` (only `gemini-3-pro` models), `temperature`, `top_p`, `seed` (`-1` = random), optional `system_instruction`, `google_search`, `image`, `ref_image_1..5`.

**Output:** `image` — tensor `[B, H, W, C]` float32 0–1.

### 2. OG LaMa Refill (`LaMaRefill`)

Local, offline inpainting. Paint the object to remove in the `MaskEditor`, connect `image` + `mask`.

**Inputs:** `image`, `mask`, `dilate_px`, `feather_px`, `passes`, `context_padding`.

**Output:** `image` with the masked object removed.

### 3. OG Gemini Orchestrate (`GeminiOrchestrate`)

Composes a final scene from a **required background** and up to 5 reference images (subjects/objects). References are resized to the background automatically.

**Inputs:** `prompt`, `api_key`, `model`, `aspect_ratio`, `resolution`, `temperature`, `top_p`, `seed`, `system_instruction`, `background` (required), `ref_image_1..5`.

**Output:** `image`.

### 4. OG Veo Generate (`VeoGenerate`)

Veo video generation. The mode is detected automatically from what you connect:

| Mode | Connect |
|------|---------|
| Text-to-video | only `prompt` |
| First + last frame | `image` + `last_image` |
| Ingredients | `reference_images` (up to 3) |
| Extension | `video_uri` from a previous Veo run |

**Inputs:** `prompt`, `model` (`veo-3.1-generate-preview` / `-fast` / `-lite`), `aspect_ratio`, `duration_seconds` (4–8), `resolution`, `api_key`, `seed`, optional `negative_prompt`, `image`, `last_image`, `reference_images`, `video_uri`.

**Outputs:** `frames` (list), `audio`, `video_uri`.

> **Notes**
> - `seed` is exposed for workflow compatibility but is **not sent to the API** — the Gemini Developer API does not support a seed for Veo video.
> - `video_uri` only accepts a Veo-generated video (720p, referenced within 2 days); it cannot take an arbitrary file.

### 5. OG Omni Flash (`OmniFlashGen`)

The pack's video-editing node, built on **Gemini Omni Flash** (`gemini-omni-1.1-flash`) via the Interactions API. Unlike Veo, it can take **your own video** and restyle it while following its motion.

**Typical use:** render a previsualization in Blender (camera movement + character motion), then connect it together with 4–6 aesthetic reference images and a prompt. The node uploads the video, edits it, and returns the restyled frames.

| `task` | What it does |
|--------|--------------|
| `edit` | Edit/restyle an input video (requires `video`) |
| `text_to_video` | Generate from text |
| `image_to_video` | Animate an image |
| `reference_to_video` | Generate from reference images |
| `auto` | Let the model infer |

**Inputs:** `prompt`, `model`, `task`, `aspect_ratio`, `resolution`, `seed` (Omni *does* support a seed), `api_key`, optional `video` (VIDEO), `ref_image_1..6` (IMAGE), `first_frame`, `last_frame`, `previous_interaction_id`.

**Outputs:** `frames` (list), `audio`, `interaction_id` (to chain edits).

**Limits / caveats:**
- Input video for editing is **10 s max**; longer clips are trimmed to the first 10 s.
- Output is 3–10 s, **720p native** (24 fps); `1080p`/`4k` are upscaled.
- Uploaded-video editing is **not available in the EEA / Switzerland / UK**.
- Media containing recognizable people or minors is blocked by safety filters.
- The result is **not frame-perfect**: the model regenerates guided by the input motion, so camera/action are preserved approximately, not pixel-exactly.

---

## Example workflow: Blender previsualization → styled video

1. **Blender:** render a short (≤10 s) previsualization — the layout, camera path, and character motion you want to keep.
2. **ComfyUI:** `Load Video` → the `video` input of `OG Omni Flash`.
3. Generate or load 4–6 style references (e.g. `OG NanoBanana Generate` outputs) → `ref_image_1..6`.
4. Set `task = edit` and a prompt like *"Restyle this video to match the reference look; keep the camera movement and character motion."*
5. Run. `frames`/`audio` carry the result; `interaction_id` lets you iterate with `previous_interaction_id`.

---

## Troubleshooting

- **`API key required`** — no key found. Set `GEMINI_API_KEY` or create `.env`.
- **`400` / invalid argument on video** — Veo extension only accepts Veo-generated videos (720p). For arbitrary video, use `OG Omni Flash`.
- **Omni edit rejected** — your video may be >10 s, in an unsupported region, or contain people/minors.
- **`[Errno 11001] getaddrinfo failed` / timeouts** — transient network/DNS; retry.
- **LaMa slow on first run** — it is downloading the model; later runs use the cache.

## Limitations

- English prompts are fully supported; other languages may vary in quality.
- Video generation is asynchronous and can take minutes under load.
- Generated videos are SynthID-watermarked.

---

## Development

```bash
# Install dev tools
pip install -r requirements-dev.txt

# Lint / format
ruff check .
ruff format .

# Tests (standalone scripts; also runnable under pytest)
python tests/test_utils.py
python tests/test_omni_gen.py
```

The node modules live in `nodes/` and share helpers in `utils.py`. To add a node: create `nodes/<name>.py`, export `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`, and add the module name to `_node_modules` in `__init__.py`.

---

## License

[MIT](LICENSE) © 2026 Alexander Cortez.
