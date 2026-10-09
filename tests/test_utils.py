import io
import os
import sys

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import video_bytes_to_frames


def _make_mp4(n_frames=4, width=32, height=32):
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=24)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = (i * 40) % 256
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    container.mux(stream.encode())
    container.close()
    return buf.getvalue()


def test_video_bytes_to_frames_decodes_frames():
    raw = _make_mp4()
    frames = video_bytes_to_frames(io.BytesIO(raw))
    assert len(frames) == 4
    assert frames[0].shape == (1, 32, 32, 3)
    assert frames[0].dtype == torch.float32
    assert float(frames[0].max()) <= 1.0


if __name__ == "__main__":
    test_video_bytes_to_frames_decodes_frames()
    print("ALL TESTS PASSED")
