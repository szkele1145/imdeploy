"""配置 / 下载器 / 模型库的单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.config import (  # noqa: E402
    Config,
    field_label,
    section_title,
)
from imdeploy.downloader import (  # noqa: E402
    Downloader,
    DownloadTask,
    parse_gguf_filename,
    resolve_url,
)
from imdeploy.library import ModelLibrary  # noqa: E402


# ---------------------------------------------------------------- 配置


def test_config_defaults():
    cfg = Config.default()
    assert cfg.get("app.port") == 8080
    assert cfg.get("backend.prefer") == "llama.cpp"
    assert cfg.get("recommend.prefer_small") is True


def test_config_set_get_dotted():
    cfg = Config.default()
    cfg.set("backend.context_size", 32768)
    assert cfg.get("backend.context_size") == 32768
    cfg.set("api.aliases.qwen", "qwen3-0.6b")
    assert cfg.get("api.aliases.qwen") == "qwen3-0.6b"


def test_config_roundtrip(tmp_path):
    p = tmp_path / "data" / "config.yaml"
    cfg = Config.load(p)
    cfg.set("app.port", 9099)
    cfg.save()
    cfg2 = Config.load(p)
    assert cfg2.get("app.port") == 9099
    # 未覆盖的默认值仍然在
    assert cfg2.get("backend.ollama_url") == "http://127.0.0.1:11434"


def test_config_merge_preserves_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("app:\n  port: 7777\nnew_section:\n  foo: 1\n", encoding="utf-8")
    cfg = Config.load(p)
    assert cfg.get("app.port") == 7777
    assert cfg.get("backend.prefer") == "llama.cpp"  # 默认值还在
    assert cfg.get("new_section.foo") == 1  # 新节也读进来


def test_field_labels_chinese():
    assert field_label("app.port") == "监听端口"
    assert section_title("backend") == "推理后端"
    assert field_label("not.exist.key") == "key"  # 兜底


# ---------------------------------------------------------------- GGUF 文件名解析


def test_parse_gguf_small_instruct():
    info = parse_gguf_filename("Qwen3-0.6B-Instruct-Q4_K_M.gguf")
    assert info["family"] == "qwen3"
    assert info["params_b"] == 0.6
    assert info["quant"] == "Q4_K_M"
    assert info["instruct"] is True


def test_parse_gguf_long_context():
    info = parse_gguf_filename("SmolLM2-1.7B-Instruct-128k-Q8_0.gguf")
    assert info["family"] == "smol"
    assert info["ctx_len"] >= 131072
    assert info["long_context"] is True
    assert info["quant"] == "Q8_0"


def test_parse_gguf_base_not_instruct():
    info = parse_gguf_filename("Llama-3.2-1B-base-Q4_0.gguf")
    assert info["family"] == "llama"
    assert info["instruct"] is False
    assert info["quant"] == "Q4_0"


def test_parse_gguf_params_in_millions():
    info = parse_gguf_filename("MiniCPM-500M-chat-Q5_K_M.gguf")
    assert 0.4 <= info["params_b"] <= 0.6
    assert info["family"] == "minicpm"


def test_parse_gguf_unknown():
    info = parse_gguf_filename("mystery-model-v2.gguf")
    assert info["family"] == "unknown"
    assert info["params_b"] == 0.0


# ---------------------------------------------------------------- 下载器


def test_resolve_url_modelscope():
    url = resolve_url("Qwen/Qwen3-0.6B-GGUF", "model.gguf", "modelscope")
    assert url.startswith("https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF/resolve/master/")
    assert url.endswith("model.gguf")


def test_resolve_url_huggingface_mirror():
    url = resolve_url("org/repo", "m.gguf", "huggingface", mirror=True)
    assert url.startswith("https://hf-mirror.com/org/repo/resolve/main/")


def test_download_task_percent():
    t = DownloadTask(url="x", dest=Path("x.gguf"), size=1000, downloaded=250)
    assert t.percent == 25.0


def test_downloader_local_file(tmp_path):
    """用一个本地 HTTP 不可用时的降级路径：直接测错误处理。"""
    dl = Downloader(timeout=2, resume=False)
    task = dl.download("http://127.0.0.1:1/nothing.gguf", tmp_path / "a.gguf")
    assert task.status == "error"
    assert task.error  # 有错误信息


# ---------------------------------------------------------------- 模型库


def _make_gguf(tmp_path: Path, name: str, size_mb: float = 1.0) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * int(size_mb * 1024 * 1024))
    return p


def test_library_scan_and_register(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    lib = ModelLibrary(cfg)
    f = _make_gguf(tmp_path / "models", "Qwen3-0.6B-Instruct-Q4_K_M.gguf", 2.0)
    m = lib.register(f)
    assert m is not None
    assert m.family == "qwen3"
    assert m.params_b == 0.6
    assert lib.list()[0].id == m.id


def test_library_persistence(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    lib1 = ModelLibrary(cfg)
    f = _make_gguf(tmp_path / "models", "SmolLM2-1.7B-Instruct-128k-Q8_0.gguf", 3.0)
    m = lib1.register(f)
    lib1._save()

    lib2 = ModelLibrary(cfg)
    got = lib2.get(m.id)
    assert got is not None
    assert got.family == "smol"
    assert got.ctx_len >= 131072


def test_library_best_for_alias(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    lib = ModelLibrary(cfg)
    _make_gguf(tmp_path / "models", "Qwen3-0.6B-Instruct-Q4_K_M.gguf")
    _make_gguf(tmp_path / "models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    lib.scan()

    assert lib.best_for_alias("qwen").family == "qwen3"
    assert lib.best_for_alias("llama").family == "llama"
    assert lib.best_for_alias("nonexist") is not None  # 回退到第一个


def test_library_alias_explicit(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    cfg.set("api.aliases.qwen", "llama-3.2-1b-instruct-q4_k_m")
    lib = ModelLibrary(cfg)
    _make_gguf(tmp_path / "models", "Qwen3-0.6B-Instruct-Q4_K_M.gguf")
    _make_gguf(tmp_path / "models", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    lib.scan()
    assert lib.best_for_alias("qwen").family == "llama"  # 显式映射优先


def test_library_remove(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    lib = ModelLibrary(cfg)
    f = _make_gguf(tmp_path / "models", "Gemma-2-2b-it-Q5_K_M.gguf")
    m = lib.register(f)
    assert lib.remove(m.id, delete_file=True) is True
    assert not f.exists()
    assert lib.get(m.id) is None


def test_library_scan_removes_missing(tmp_path):
    cfg = Config.default()
    cfg._base_dir = tmp_path
    lib = ModelLibrary(cfg)
    f = _make_gguf(tmp_path / "models", "Phi-3-mini-4k-instruct-Q4_K_M.gguf")
    m = lib.register(f)
    f.unlink()  # 手动删文件
    lib.scan()
    assert lib.get(m.id) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
