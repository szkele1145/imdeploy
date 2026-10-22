"""运行库自动安装的单元测试（不联网）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.runtime_install import (  # noqa: E402
    RuntimeInfo,
    detect_accel,
    find_installed,
    pick_release_asset,
)


def assets(*names: str) -> list[dict]:
    return [{"name": n, "browser_download_url": f"https://example.com/{n}",
             "size": 123} for n in names]


def test_pick_cuda_on_windows():
    a = assets(
        "llama-b4500-bin-win-cpu-x64.zip",
        "llama-b4500-bin-win-cuda-x64.zip",
        "llama-b4500-bin-win-vulkan-x64.zip",
        "llama-b4500-bin-ubuntu-x64.zip",
    )
    got = pick_release_asset(a, "cuda")
    assert got is not None
    assert "cuda" in got["name"]


def test_pick_vulkan():
    a = assets(
        "llama-b4500-bin-win-cpu-x64.zip",
        "llama-b4500-bin-win-vulkan-x64.zip",
    )
    got = pick_release_asset(a, "vulkan")
    assert got is not None
    assert "vulkan" in got["name"]


def test_pick_cpu_fallback():
    a = assets("llama-b4500-bin-win-cpu-x64.zip")
    got = pick_release_asset(a, "cuda")  # 没有 cuda 包时退回 cpu
    assert got is not None
    assert "cpu" in got["name"]


def test_pick_ignores_non_zip():
    a = assets("llama-b4500-bin-win-cpu-x64.tar.gz")
    assert pick_release_asset(a, "cpu") is None


def test_pick_empty():
    assert pick_release_asset([], "cpu") is None


def test_find_installed_direct(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    assert find_installed(tmp_path) == exe


def test_find_installed_nested(tmp_path):
    sub = tmp_path / "build" / "bin"
    sub.mkdir(parents=True)
    exe = sub / "llama-server"
    exe.write_bytes(b"x")
    assert find_installed(tmp_path) == exe


def test_find_installed_missing(tmp_path):
    assert find_installed(tmp_path) is None


def test_detect_accel_returns_known():
    assert detect_accel() in ("cuda", "vulkan", "cpu")


def test_runtime_info_defaults():
    info = RuntimeInfo(ok=False)
    assert info.exe_path == ""
    assert info.message == ""
