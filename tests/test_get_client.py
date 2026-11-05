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
