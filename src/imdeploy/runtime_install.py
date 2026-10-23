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
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
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
GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases"
RELEASES_LATEST = RELEASES_PAGE + "/latest"


def _pick_binary_release(releases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从 release 列表里挑出真正带二进制资产的那个。

    llama.cpp 的 ``/releases/latest`` 指向的往往是 ``v0.7.0`` 这种只挂
    ``nightly-tag.txt`` 的空壳发布，真正的构建产物在 ``b12345`` 形式的 tag 上。
    所以要按「有 bin 资产」来挑，不能只看 latest。
    """
    def has_bin(rel: dict[str, Any]) -> bool:
        for a in rel.get("assets") or []:
            name = str(a.get("name") or "").lower()
            if "bin" in name and name.endswith(".zip"):
                return True
        return False

    for rel in releases:
        if has_bin(rel):
            return rel
    return None
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


def _arch_tag() -> str | None:
    """返回资产名里的 CPU 架构标记：x64 / arm64。"""
    machine = (platform.machine() or "").lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return None


def pick_release_asset(assets: list[dict[str, Any]], accel: str) -> dict[str, Any] | None:
    """从 release 资产里挑出最合适的一个。

    资产命名形如 ``llama-b4500-bin-win-cuda-x64.zip``、
    ``llama-b4500-bin-win-vulkan-x64.zip``、``llama-b4500-bin-win-cpu-x64.zip``。

    架构必须严格匹配：在 x64 机器上装到 arm64 的包，运行时会报
    “该版本的 %1 与你运行的 Windows 版本不兼容”（WinError 216）。
    """
    tag = _platform_tag()
    if not tag:
        return None
    arch = _arch_tag()
    if not arch:
        return None
    wrong_arch = "arm64" if arch == "x64" else "x64"

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
        # 架构硬性要求：必须含本机架构、且不含另一种架构
        if arch not in n or wrong_arch in n:
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
        return s

    ranked = sorted(zips, key=lambda a: score(a["name"]), reverse=True)
    if not ranked:
        return None
    best = ranked[0]
    return best if score(best["name"]) > -100 else None


# ------------------------------------------------------------------ 网络

def _opener(proxy: str | None):
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _gh_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    """GitHub 请求头。有 GITHUB_TOKEN 就带上，没授权的 60 次/小时很容易打满。"""
    headers = {"User-Agent": UA, "Accept": accept}
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str, proxy: str | None, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers=_gh_headers())
    with _opener(proxy).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, proxy: str | None, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers=_gh_headers("text/html"))
    with _opener(proxy).open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _resolve_via_html(proxy: str | None) -> tuple[str, list[dict[str, str]]]:
    """API 被限流时，改从 GitHub 网页解析版本与资产列表。

    返回 ``(tag, [{"name", "browser_download_url"}, ...])``。
    纯正则解析，不引第三方 HTML 解析库。
    """
    # 1) /releases/latest 会 302 到 /releases/tag/<tag>，从 Location 里拿 tag
    req = urllib.request.Request(RELEASES_LATEST, headers=_gh_headers("text/html"),
                                 method="HEAD")
    tag = ""
    try:
        with _opener(proxy).open(req, timeout=30) as resp:
            final = resp.geturl()
            m = re.search(r"/releases/tag/([^/?#]+)", final)
            if m:
                tag = m.group(1)
    except Exception:  # noqa: BLE001
        pass

    if not tag:
        # HEAD 可能被拒，退回 GET 首页找 /releases/tag/xxx
        html = _get_text(RELEASES_PAGE, proxy)
        m = re.search(r'/releases/tag/([^"?#]+)', html)
        if not m:
            raise RuntimeError("无法从网页解析 llama.cpp 最新版本号")
        tag = m.group(1)

    # 2) expanded_assets 片段里有全部资产链接
    html = _get_text(f"https://github.com/ggml-org/llama.cpp/releases/expanded_assets/{tag}",
                     proxy)
    assets: list[dict[str, str]] = []
    for href in re.findall(r'href="(/ggml-org/llama\.cpp/releases/download/[^"]+)"', html):
        name = href.rsplit("/", 1)[-1]
        assets.append({
            "name": urllib.parse.unquote(name),
            "browser_download_url": f"https://github.com{href}",
        })
    if not assets:
        raise RuntimeError(f"网页里没有解析到 {tag} 的资产列表")
    return tag, assets



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
        releases = _get_json(GITHUB_RELEASES + "?per_page=30", proxy)
        release = _pick_binary_release(releases) if isinstance(releases, list) else None
        if release is None:
            # 兜底：退回 latest（某些版本可能确实挂在 latest 上）
            release = _get_json(GITHUB_API, proxy)
        version = str(release.get("tag_name") or "")
        assets = release.get("assets") or []
    except urllib.error.HTTPError as exc:
        # 未授权的 GitHub API 只有 60 次/小时，403/429 很常见 —— 改走网页解析
        if exc.code in (403, 429):
            say(f"GitHub API 限流（HTTP {exc.code}），改用网页解析…")
            try:
                version, assets = _resolve_via_html(proxy)
            except Exception as exc2:  # noqa: BLE001
                return RuntimeInfo(
                    False,
                    message=(f"查询 llama.cpp 版本失败（API HTTP {exc.code}，"
                             f"网页兜底也失败：{exc2}）。"
                             "可设置环境变量 GITHUB_TOKEN 提高配额，或手动下载"),
                )
        else:
            return RuntimeInfo(False, message=f"查询 llama.cpp 版本失败（HTTP {exc.code}），请检查网络或代理设置")
    except Exception as exc:  # noqa: BLE001
        return RuntimeInfo(False, message=f"查询 llama.cpp 版本失败：{exc}")

    say(f"最新版本：{version}（共 {len(assets)} 个资产）")

    asset = pick_release_asset(assets, accel)
    if asset is None:
        return RuntimeInfo(False, version=version,
                           message=f"没有找到适配 {tag} / {accel} 的安装包，请到 llama.cpp releases 手动下载")

    say(f"选中：{asset['name']}")
    target = Path(bin_dir) / "llama"
    target.mkdir(parents=True, exist_ok=True)

    # 清掉旧版本残留：混着重装不同架构的 dll 会直接导致无法启动
    for stale in target.iterdir():
        try:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
        except OSError:
            pass

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
