import importlib
import sys
from pathlib import Path
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT.parent))

omni_gen = importlib.import_module(f"{_ROOT.name}.nodes.omni_gen")
OmniFlashGen = omni_gen.OmniFlashGen
build_omni_prompt = omni_gen.build_omni_prompt


def test_build_prompt_with_video_and_refs():
    prompt = build_omni_prompt("turn it into anime", has_video=True, n_refs=4, has_first=False, has_last=False)
    assert "<VIDEO_0>" in prompt
    assert "<IMAGE_REF_0>" in prompt
    assert "<IMAGE_REF_3>" in prompt
    assert "turn it into anime" in prompt


def test_build_prompt_text_only():
    prompt = build_omni_prompt("a cat reading", has_video=False, n_refs=0, has_first=False, has_last=False)
    assert prompt == "a cat reading"


def test_build_prompt_frames():
    prompt = build_omni_prompt("morph", has_video=False, n_refs=0, has_first=True, has_last=True)
    assert "starting frame" in prompt
    assert "ending frame" in prompt


class _FakeVideo:
    def __init__(self, duration):
        self._duration = duration
        self.trim = None

    def get_duration(self):
        return self._duration

    def as_trimmed(self, start_time=None, duration=None):
        assert start_time is not None, "as_trimmed() called without start_time (Fix 1)"
        self.trim = (start_time, duration)
        return self


class _FakeInteraction:
    id = "interaction-123"
    output_video = object()


class _FakeInteractions:
    def __init__(self):
        self.bodies = []

    def create(self, **body):
        self.bodies.append(body)
        return _FakeInteraction()


class _FakeClient:
    def __init__(self):
        self.interactions = _FakeInteractions()


class _FakeUploaded:
    uri = "https://example.com/uploaded.mp4"


_REQUIRED = dict(
    model="gemini-omni-1.1-flash",
    task="edit",
    aspect_ratio="16:9",
    resolution="720p",
    seed=-1,
    api_key="fake-key",
)


def _generate(video=None, prompt="", **overrides):
    client = _FakeClient()
    params = {**_REQUIRED, **overrides}
    patches = [
        mock.patch.object(omni_gen, "get_client", lambda api_key: client),
        mock.patch.object(omni_gen, "video_to_mp4_bytes", lambda v: b"fake"),
        mock.patch.object(omni_gen, "upload_file", lambda c, data, mime_type=None: _FakeUploaded()),
        mock.patch.object(
            omni_gen, "tensor_to_omni_image", lambda t: {"type": "image", "data": "", "mime_type": "image/png"}
        ),
        mock.patch.object(omni_gen, "video_bytes_to_frames", lambda f: [torch.zeros(1, 2, 2, 3)]),
        mock.patch.object(omni_gen, "decode_audio", lambda f: None),
        mock.patch.object(omni_gen, "interaction_video_bytes", lambda c, i: b"fake"),
    ]
    for p in patches:
        p.start()
    try:
        OmniFlashGen().generate(prompt=prompt, video=video, **params)
    finally:
        for p in patches:
            p.stop()
    return client.interactions.bodies[-1]


def test_generate_trims_long_video():
    fake = _FakeVideo(12.0)
    body = _generate(video=fake, prompt="restyle it")
    assert fake.trim == (0, 10.0)
    assert body["model"] == "gemini-omni-1.1-flash"
    assert "input" in body
    assert "response_format" in body
    assert "generation_config" in body
    video_parts = [p for p in body["input"] if p.get("type") == "video"]
    assert len(video_parts) == 1
    assert video_parts[0]["uri"] == _FakeUploaded.uri


def test_generate_edit_without_video_raises():
    try:
        _generate(video=None, prompt="restyle it")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for edit without video")


def test_generate_empty_input_raises():
    try:
        _generate(video=None, prompt="", task="auto")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty input")


def test_generate_frames_only_does_not_raise():
    body = _generate(
        video=None,
        prompt="",
        task="auto",
        first_frame=torch.zeros(1, 2, 2, 3),
        last_frame=torch.zeros(1, 2, 2, 3),
    )
    image_parts = [p for p in body["input"] if p.get("type") == "image"]
    assert len(image_parts) == 2


if __name__ == "__main__":
    test_build_prompt_with_video_and_refs()
    test_build_prompt_text_only()
    test_build_prompt_frames()
    test_generate_trims_long_video()
    test_generate_edit_without_video_raises()
    test_generate_empty_input_raises()
    test_generate_frames_only_does_not_raise()
    print("ALL TESTS PASSED")
