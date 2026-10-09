import base64
import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv(Path(__file__).parent / ".env")


def get_client(api_key):
    key = os.environ.get(api_key.strip(), api_key.strip()) or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("API key required. Provide a key directly or set GEMINI_API_KEY env var.")
    return genai.Client(http_options={"api_version": "v1beta"}, api_key=key)


def tensor_to_pil(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = image_tensor.cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def pil_to_api_image(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return types.Image(image_bytes=buf.getvalue(), mime_type="image/png")


def pil_to_part(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return types.Part.from_bytes(mime_type="image/png", data=buf.getvalue())


def tensor_to_api_image(tensor):
    return pil_to_api_image(tensor_to_pil(tensor))


def tensor_to_part(tensor):
    return pil_to_part(tensor_to_pil(tensor))


def video_bytes_to_frames(video_bytes):
    import av

    if isinstance(video_bytes, (bytes, bytearray)):
        video_bytes = io.BytesIO(video_bytes)
    container = av.open(video_bytes)
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
        frames.append(torch.from_numpy(arr).unsqueeze(0))
    container.close()
    return frames


def decode_video(client, video_obj):
    video_bytes = io.BytesIO(client.files.download(file=video_obj))
    return video_bytes_to_frames(video_bytes), video_bytes


def video_to_mp4_bytes(video_input):
    from comfy_api.latest import Types

    buf = io.BytesIO()
    video_input.save_to(buf, format=Types.VideoContainer.MP4)
    return buf.getvalue()


def upload_file(client, data, mime_type="video/mp4", display_name="og_omni_input.mp4", timeout=1800):
    suffix = os.path.splitext(display_name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        uploaded = client.files.upload(file=tmp_path, config=types.UploadFileConfig(mime_type=mime_type))
    finally:
        os.remove(tmp_path)
    start = time.time()
    while _file_state(uploaded) == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError(f"File upload processing timed out after {timeout}s")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if _file_state(uploaded) == "FAILED":
        raise Exception(f"File upload failed: {uploaded.error}")
    return uploaded


def _file_state(uploaded):
    return getattr(uploaded.state, "name", str(uploaded.state))


def interaction_video_bytes(client, interaction):
    video = interaction.output_video
    if video is None:
        raise Exception("Interaction returned no video output")
    if video.uri:
        return client.files.download(file=video.uri)
    if video.data:
        return base64.b64decode(video.data)
    raise Exception("Interaction returned no video data")


def tensor_to_omni_image(tensor):
    buf = io.BytesIO()
    tensor_to_pil(tensor).save(buf, format="PNG")
    return {
        "type": "image",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mime_type": "image/png",
    }


def decode_audio(video_bytes):
    import av

    video_bytes.seek(0)
    container = av.open(video_bytes)
    audio = None
    if container.streams.audio:
        audio_data = [f.to_ndarray() for f in container.decode(audio=0)]
        if audio_data:
            waveform = torch.from_numpy(np.concatenate(audio_data, axis=1)).float()
            if audio_data[0].dtype == np.int16:
                waveform /= 32768.0
            elif audio_data[0].dtype == np.int32:
                waveform /= 2147483648.0
            audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": container.streams.audio[0].rate,
            }
    container.close()
    return audio


def poll_operation(client, op, timeout=1800):
    print(f"Gemini operation: {op.name}")
    start = time.time()
    while not op.done:
        if time.time() - start > timeout:
            raise TimeoutError(f"Video generation timed out after {timeout}s")
        time.sleep(4)
        op = client.operations.get(op)
    if not op.response or not op.response.generated_videos:
        raise Exception("No videos generated")
    return op.response.generated_videos[0].video


def response_to_images(response):
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        reason = ""
        if candidate is not None:
            if candidate.finish_reason is not None:
                reason += f" finish_reason={candidate.finish_reason}"
            if candidate.safety_ratings:
                reason += " safety=" + ", ".join(f"{r.category}:{r.probability}" for r in candidate.safety_ratings)
        if response.prompt_feedback is not None and response.prompt_feedback.block_reason is not None:
            reason += f" block_reason={response.prompt_feedback.block_reason}"
        raise ValueError(f"Gemini returned no image content.{reason}")
    images = []
    for part in candidate.content.parts:
        if part.inline_data and part.inline_data.data:
            img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr).unsqueeze(0))
    if not images:
        raise ValueError("No images in response")
    return torch.cat(images, dim=0)


def lama_dilate(mask_np, dilate_px):
    """Expand mask edges with elliptical kernel. mask_np: uint8 HxW, returns uint8 HxW."""
    if dilate_px <= 0:
        return mask_np
    import cv2

    k = dilate_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_np, kernel)


def lama_crop_around_mask(bgr, mask, padding):
    """Crop image and mask around the masked region with context padding.
    bgr: uint8 HxWxC, mask: uint8 HxW, padding: int.
    Returns (crop_bgr, crop_mask, bounds) where bounds = (cy1, cy2, cx1, cx2)."""
    H, W = bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise RuntimeError("Empty mask")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    cy1 = max(0, y1 - padding)
    cy2 = min(H, y2 + padding)
    cx1 = max(0, x1 - padding)
    cx2 = min(W, x2 + padding)
    return bgr[cy1:cy2, cx1:cx2].copy(), mask[cy1:cy2, cx1:cx2].copy(), (cy1, cy2, cx1, cx2)


def lama_maybe_downsample(crop_bgr, crop_mask, max_dim=1024):
    """Downsample if largest dimension exceeds max_dim. Returns (small_bgr, small_mask, scale)."""
    import cv2

    ch, cw = crop_bgr.shape[:2]
    md = max(ch, cw)
    if md <= max_dim:
        return crop_bgr, crop_mask, 1.0
    scale = max_dim / md
    sw = max(1, int(round(cw * scale)))
    sh = max(1, int(round(ch * scale)))
    small_bgr = cv2.resize(crop_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(crop_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
    return small_bgr, small_mask, scale


def lama_feather_paste(bgr, crop_bgr, result_crop, crop_mask, bounds, feather_px=8):
    """Blend inpainted crop back into original. Tries Poisson seamless clone first."""
    import cv2

    cy1, cy2, cx1, cx2 = bounds
    ch, cw = crop_bgr.shape[:2]
    if result_crop.shape[:2] != (ch, cw):
        result_crop = cv2.resize(result_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

    ys, xs = np.where(crop_mask > 0)
    if ys.size > 0:
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        margin = 5
        if y_min >= margin and y_max < ch - margin and x_min >= margin and x_max < cw - margin:
            try:
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                mask8 = (crop_mask > 0).astype(np.uint8) * 255
                blended = cv2.seamlessClone(result_crop, crop_bgr, mask8, center, cv2.NORMAL_CLONE)
                out = bgr.copy()
                out[cy1:cy2, cx1:cx2] = blended
                return out
            except cv2.error:
                pass

    alpha = crop_mask.astype(np.float32) / 255.0
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = alpha[..., None]
    blended = (crop_bgr.astype(np.float32) * (1 - alpha) + result_crop.astype(np.float32) * alpha).astype(np.uint8)
    result = bgr.copy()
    result[cy1:cy2, cx1:cx2] = blended
    return result
