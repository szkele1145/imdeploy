"""llama.cpp 运行时自动安装。

懒人版体验：不用手动下载、解压、配 PATH。本模块负责

1. 查 GitHub 上 llama.cpp 的最新 release；
2. 按本机平台（Windows x64 / Linux x64 / macOS arm64）与加速后端
   （NVIDIA CUDA → Vulkan → CPU）挑选最合适的资产；
3. 下载、解压到 ``<程序目录>/bin/llama/``；
4. 把结果写进配置 ``backend.llama_server``。

所有网络请求都走系统代理（可用参数指定），失败时给中文可操作提示。
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RuntimeInfo",
    "detect_accel",
    "pick_release_asset",
    "install_llama_runtime",
    "find_installed",
]

logger = logging.getLogger(__name__)

UA = "imdeploy/0.1 (+https://github.com/szkele1145/imdeploy)"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
DEFAULT_PROXY = "http://127.0.0.1:65532"


@dataclass
class RuntimeInfo:
    """一次安装的结果。"""

    ok: bool
    exe_path: str = ""
    backend: str = ""
    version: str = ""
    message: str = ""


# ------------------------------------------------------------------ 探测

def detect_accel() -> str:
    """判断本机该用哪个加速后端。

    优先级：CUDA（有 NVIDIA 卡） > Vulkan（能跑 Vulkan 的独显/核显） > CPU。
    """
    # NVIDIA
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0 and r.stdout.strip():
            return "cuda"
    except (OSError, subprocess.SubprocessError):
        pass

    # Vulkan：Windows 上检查注册表里的显示适配器
    if sys.platform.startswith("win"):
        try:
            import winreg  # noqa: PLC0415

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Khronos\Vulkan\Drivers",
            ) as key:
                count = winreg.QueryInfoKey(key)[1]
                if count > 0:
                    return "vulkan"
        except (OSError, ImportError):
            pass
    else:
        for exe in ("vulkaninfo", "glxinfo"):
            if shutil.which(exe):
                return "vulkan"
    return "cpu"


def _platform_tag() -> str | None:
    """返回 llama.cpp 资产名里的平台标记。"""
    machine = (platform.machine() or "").lower()
    if sys.platform.startswith("win"):
        return "win" if machine in ("amd64", "x86_64") else None
    if sys.platform.startswith("linux"):
        return "ubuntu" if machine in ("x86_64", "amd64") else None
    if sys.platform == "darwin":
        return "macos"
    return None


def pick_release_asset(assets: list[dict[str, Any]], accel: str) -> dict[str, Any] | None:
    """从 release 资产里挑出最合适的一个。

    资产命名形如 ``llama-b4500-bin-win-cuda-x64.zip``、
    ``llama-b4500-bin-win-vulkan-x64.zip``、``llama-b4500-bin-win-cpu-x64.zip``。
    """
    tag = _platform_tag()
    if not tag:
        return None

    # 只看 zip
    zips = [a for a in assets if str(a.get("name", "")).lower().endswith(".zip")]
    if not zips:
        return None

    def score(name: str) -> int:
        n = name.lower()
        if tag not in n:
            return -100
        if not n.startswith("llama-"):
            return -100
        s = 0
        if "bin" not in n:
            s -= 30
        need = {"cuda": "cuda", "vulkan": "vulkan", "cpu": "cpu"}.get(accel, "cpu")
        if need in n:
            s += 50
        elif accel != "cpu" and "cpu" in n:
            s += 20  # 加速版不可用时的保底
        # 排除无关组件包
        for bad in ("cudart", "hip-", "sycl", "opencl"):
            if bad in n and accel != "cuda":
                s -= 40
        if "x64" in n or "arm64" in n:
            s += 5
        return s

    ranked = sorted(zips, key=lambda a: score(a["name"]), reverse=True)
    best = ranked[0]
    return best if score(best["name"]) > -100 else None


# ------------------------------------------------------------------ 网络

def _opener(proxy: str | None):
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _get_json(url: str, proxy: str | None, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    with _opener(proxy).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path, proxy: str | None,
              progress: Any = None, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with _opener(proxy).open(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)


# ------------------------------------------------------------------ 查找 / 安装

def find_installed(target_dir: Path) -> Path | None:
    """在目标目录里找 llama-server 可执行文件。"""
    if not target_dir.exists():
        return None
    names = ("llama-server.exe", "llama-server")
    for name in names:
        direct = target_dir / name
        if direct.is_file():
            return direct
    for name in names:
        for hit in target_dir.rglob(name):
            if hit.is_file():
                return hit
    return None


def install_llama_runtime(
    bin_dir: Path,
    proxy: str | None = DEFAULT_PROXY,
    accel: str | None = None,
    progress: Any = None,
    log: Any = None,
) -> RuntimeInfo:
    """下载并安装 llama-server 到 ``bin_dir/llama``。

    ``progress`` 签名：``(done_bytes, total_bytes_or_0)``
    ``log`` 签名：``(message: str)``
    """
    def say(msg: str) -> None:
        logger.info(msg)
        if log:
            log(msg)

    tag = _platform_tag()
    if not tag:
        return RuntimeInfo(False, message=f"暂不支持自动安装的平台：{sys.platform} / {platform.machine()}")

    accel = accel or detect_accel()
    say(f"检测到加速后端：{accel}")

    try:
        release = _get_json(GITHUB_API, proxy)
    except urllib.error.HTTPError as exc:
        return RuntimeInfo(False, message=f"查询 llama.cpp 版本失败（HTTP {exc.code}），请检查网络或代理设置")
    except Exception as exc:  # noqa: BLE001
        return RuntimeInfo(False, message=f"查询 llama.cpp 版本失败：{exc}")

    version = str(release.get("tag_name") or "")
    assets = release.get("assets") or []
    say(f"最新版本：{version}（共 {len(assets)} 个资产）")

    asset = pick_release_asset(assets, accel)
    if asset is None:
        return RuntimeInfo(False, version=version,
                           message=f"没有找到适配 {tag} / {accel} 的安装包，请到 llama.cpp releases 手动下载")

    say(f"选中：{asset['name']}")
    target = Path(bin_dir) / "llama"
    target.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="imdeploy-llama-"))
    zip_path = tmp / "runtime.zip"
    try:
        _download(asset["browser_download_url"], zip_path, proxy, progress)
        say("下载完成，正在解压…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
    except Exception as exc:  # noqa: BLE001
        return RuntimeInfo(False, version=version, message=f"下载或解压失败：{exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    exe = find_installed(target)
    if exe is None:
        return RuntimeInfo(False, version=version,
                           message="解压完成但没找到 llama-server，请检查压缩包内容")

    # Linux/macOS 补可执行权限
    if not sys.platform.startswith("win"):
        try:
            os.chmod(exe, 0o755)
        except OSError:
            pass

    say(f"安装完成：{exe}")
    return RuntimeInfo(True, exe_path=str(exe), backend=accel, version=version,
                       message=f"已安装 llama.cpp {version}（{accel}）")
