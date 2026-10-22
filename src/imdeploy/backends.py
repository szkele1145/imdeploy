"""推理后端抽象与实现。

提供两种本地推理后端：

- :class:`LlamaCppBackend` —— 管理 ``llama-server`` 子进程（GGUF 模型）
- :class:`OllamaBackend` —— 通过 REST 调用本机 Ollama 服务

所有 HTTP 通信统一使用标准库 ``urllib``，保持零第三方依赖。
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import Config

__all__ = [
    "Backend",
    "LlamaCppBackend",
    "OllamaBackend",
    "detect_backends",
    "find_free_port",
    "port_open",
    "http_get_json",
    "DEFAULT_OLLAMA_URL",
]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
LOG_FILE_NAME = "llama-server.log"
ENV_LLAMA_SERVER = "FEIYU_LLAMA_SERVER"

# Windows 下隐藏控制台窗口，避免启动模型时弹出黑框
if os.name == "nt":  # pragma: no cover - 平台相关
    _CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
else:  # pragma: no cover - 平台相关
    _CREATE_NO_WINDOW = 0


# ---------------------------------------------------------------- 小工具


def find_free_port(start: int, host: str = "127.0.0.1", tries: int = 64) -> int:
    """从 ``start`` 起找一个能 bind 的空闲端口。"""
    for port in range(int(start), int(start) + int(tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"从 {start} 起连续 {tries} 个端口均被占用，无法分配端口")


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """判断端口是否可以连接。"""
    if not port:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def http_get_json(url: str, timeout: float = 3.0) -> tuple[int, Any]:
    """GET 一个 JSON 接口，返回 ``(状态码, 解析结果)``；网络异常时状态码为 0。"""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return int(resp.status), json.loads(body.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return int(resp.status), body.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, None


def _no_window_kwargs() -> dict[str, Any]:
    """Windows 下附加 CREATE_NO_WINDOW 标志。"""
    if os.name == "nt" and _CREATE_NO_WINDOW:
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


# ---------------------------------------------------------------- 抽象基类


class Backend(ABC):
    """推理后端抽象基类。"""

    #: 后端标识名（llama.cpp / ollama）
    name: str = "backend"
    #: 中文显示名
    display_name: str = "推理后端"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.default()

    # ---------------- 能力探测 ----------------

    @abstractmethod
    def is_available(self) -> bool:
        """后端是否可用（可执行文件存在 / 服务可访问）。"""

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """列出后端可用的模型，返回 ``[{"id","name","family",...}, ...]``。"""

    @abstractmethod
    def start_model(self, model_id: str, **opts: Any) -> dict[str, Any]:
        """启动（或挂载）一个模型，返回 ``{"model_id","base_url","port","pid",...}``。"""

    @abstractmethod
    def stop_model(self) -> bool:
        """停止当前模型，成功返回 True。"""

    @abstractmethod
    def is_running(self, model_id: str) -> bool:
        """指定模型是否正在运行。"""

    @abstractmethod
    def base_url(self) -> str:
        """当前后端服务的基地址。"""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """健康检查，返回 ``{"ok":bool, ...}``。"""

    # ---------------- 通用辅助 ----------------

    def info(self) -> dict[str, Any]:
        """后端状态概览。"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.is_available(),
            "base_url": self.base_url(),
        }


# ---------------------------------------------------------------- llama.cpp


class LlamaCppBackend(Backend):
    """管理 ``llama-server`` 进程。

    可执行文件查找顺序：

    1. 构造参数 ``server_path``
    2. 配置项 ``backend.llama_server``
    3. 环境变量 ``FEIYU_LLAMA_SERVER``
    4. PATH 中的 ``llama-server`` / ``llama-server.exe``
    5. 与本模块同目录、``bin`` 目录下的同名可执行文件
    """

    name = "llama.cpp"
    display_name = "llama.cpp"

    def __init__(self, cfg: Config | None = None, server_path: str | os.PathLike | None = None):
        super().__init__(cfg)
        self._explicit_path = str(server_path) if server_path else ""
        self._server_path: str | None = None
        self._resolved = False
        # 运行状态
        self._proc: subprocess.Popen | None = None
        self._model_id: str = ""
        self._model_path: str = ""
        self._port: int = 0
        self._host: str = "127.0.0.1"
        self._log_handle = None
        self._started_at: float = 0.0

    # ---------------- 可执行文件 ----------------

    @property
    def server_path(self) -> str | None:
        """解析出的 llama-server 可执行文件路径。"""
        if not self._resolved:
            self._server_path = self._find_executable()
            self._resolved = True
        return self._server_path

    def _find_executable(self) -> str | None:
        candidates: list[str] = []
        if self._explicit_path:
            candidates.append(self._explicit_path)
        cfg_path = str(self.cfg.get("backend.llama_server", "") or "")
        if cfg_path:
            candidates.append(cfg_path)
        env_path = str(os.environ.get(ENV_LLAMA_SERVER, "") or "")
        if env_path:
            candidates.append(env_path)

        for cand in candidates:
            p = Path(cand).expanduser()
            if p.is_file():
                return str(p)

        # PATH
        for exe in ("llama-server", "llama-server.exe"):
            found = shutil.which(exe)
            if found:
                return found

        # 项目内常见位置：bin/ 、tools/ 、与本模块同目录
        here = Path(__file__).resolve().parent
        roots = [self.cfg.base_dir, here, here.parent, here.parent / "bin"]
        roots += [self.cfg.base_dir / "bin", self.cfg.base_dir / "tools",
                  self.cfg.base_dir / "llama.cpp" / "build" / "bin"]
        for root in roots:
            try:
                if not root or not root.exists():
                    continue
            except OSError:
                continue
            for exe in ("llama-server.exe", "llama-server"):
                cand = root / exe
                if cand.is_file():
                    return str(cand)
        return None

    # ---------------- 能力 ----------------

    def is_available(self) -> bool:
        path = self.server_path
        if not path:
            return False
        return Path(path).is_file()

    def list_models(self) -> list[dict[str, Any]]:
        """列出 models/ 目录下已下载的 GGUF（按文件名解析系列）。"""
        from .downloader import parse_gguf_filename  # 延迟导入，避免循环依赖

        models_dir = self.cfg.models_dir
        out: list[dict[str, Any]] = []
        try:
            files = sorted(models_dir.rglob("*.gguf"))
        except OSError:
            files = []
        for f in files:
            info = parse_gguf_filename(f.name)
            out.append({
                "id": f.stem.lower(),
                "name": f.stem,
                "family": info.get("family", "unknown"),
                "params_b": info.get("params_b", 0.0),
                "quant": info.get("quant", ""),
                "ctx_len": info.get("ctx_len", 0),
                "path": str(f),
            })
        return out

    # ---------------- 启动 / 停止 ----------------

    def _resolve_gguf(self, model_id: str, opts: dict[str, Any]) -> Path | None:
        """定位模型权重文件。"""
        for key in ("gguf", "model_path", "path"):
            val = opts.get(key)
            if val:
                p = Path(str(val)).expanduser()
                if p.is_file():
                    return p
        models_dir = self.cfg.models_dir
        direct = models_dir / f"{model_id}.gguf"
        if direct.is_file():
            return direct
        try:
            for f in models_dir.rglob("*.gguf"):
                if f.stem.lower() == str(model_id).lower():
                    return f
                if str(model_id).lower() in f.stem.lower():
                    return f
        except OSError:
            pass
        return None

    def _log_path(self) -> Path:
        log_dir = self.cfg.log_dir
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return log_dir / LOG_FILE_NAME

    def build_command(self, gguf: Path, port: int, opts: dict[str, Any]) -> list[str]:
        """拼出 llama-server 的命令行参数。"""
        exe = self.server_path or "llama-server"
        host = str(opts.get("host") or self._host or "127.0.0.1")
        ctx = int(opts.get("context_size") or opts.get("ctx")
                  or self.cfg.get("backend.context_size", 8192) or 8192)
        ngl_opt = opts.get("n_gpu_layers", opts.get("ngl"))
        if ngl_opt is None:
            ngl_opt = self.cfg.get("backend.gpu_layers", -1)
        ngl = int(-1 if ngl_opt is None else ngl_opt)
        cmd = [
            str(exe),
            "-m", str(gguf),
            "--host", host,
            "--port", str(port),
            "-c", str(ctx),
            "-ngl", str(ngl),
        ]
        threads = opts.get("threads", self.cfg.get("backend.threads", 0))
        if threads and int(threads) > 0:
            cmd += ["-t", str(int(threads))]
        if opts.get("jinja", True):
            cmd.append("--jinja")
        if opts.get("mmla"):
            cmd.append("--mmla")
        if opts.get("flash_attn"):
            cmd.append("--flash-attn")
        extra = opts.get("extra_args") or []
        if isinstance(extra, (list, tuple)):
            cmd += [str(x) for x in extra]
        return cmd

    def start_model(self, model_id: str, **opts: Any) -> dict[str, Any]:
        """启动 llama-server 进程加载指定 GGUF 模型。"""
        if not self.is_available():
            raise RuntimeError(
                "未找到 llama-server 可执行文件；请在设置里填写 backend.llama_server "
                f"或设置环境变量 {ENV_LLAMA_SERVER}"
            )
        gguf = self._resolve_gguf(model_id, opts)
        if gguf is None:
            raise RuntimeError(f"未找到模型权重文件：{model_id}（请先下载模型）")

        # 已有进程在跑：先停掉
        if self._proc is not None and self._proc.poll() is None:
            self.stop_model()

        base_port = int(self.cfg.get("backend.base_port", 8120) or 8120)
        port = int(opts.get("port") or 0) or find_free_port(base_port)
        host = str(opts.get("host") or "127.0.0.1")
        cmd = self.build_command(gguf, port, opts)

        log_path = self._log_path()
        try:
            self._log_handle = open(log_path, "ab")
            stdout = stderr = self._log_handle
        except OSError:
            stdout = stderr = subprocess.DEVNULL

        try:
            proc = subprocess.Popen(  # noqa: S603 - 本地可信可执行文件
                cmd,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                **_no_window_kwargs(),
            )
        except OSError as exc:
            raise RuntimeError(f"启动 llama-server 失败：{exc}") from exc

        self._proc = proc
        self._model_id = model_id
        self._model_path = str(gguf)
        self._port = port
        self._host = host
        self._started_at = time.time()

        return {
            "model_id": model_id,
            "path": str(gguf),
            "port": port,
            "pid": proc.pid,
            "base_url": self.base_url(),
            "command": cmd,
            "log_file": str(log_path),
            "started_at": self._started_at,
        }

    def stop_model(self) -> bool:
        """优雅终止 llama-server 进程（terminate → 超时后 kill）。"""
        proc = self._proc
        if proc is None:
            return False
        if proc.poll() is not None:
            self._cleanup()
            return True
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._cleanup()
        return True

    def _cleanup(self) -> None:
        self._proc = None
        self._model_id = ""
        self._model_path = ""
        self._port = 0
        self._started_at = 0.0
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    # ---------------- 状态 ----------------

    def is_running(self, model_id: str) -> bool:
        """进程存活且端口可连接才算运行中。"""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        if model_id and self._model_id and str(model_id) != self._model_id:
            return False
        return port_open(self._host, self._port)

    def base_url(self) -> str:
        if not self._port:
            return ""
        return f"http://{self._host}:{self._port}"

    def health(self) -> dict[str, Any]:
        """请求 llama-server 的 /health 接口。"""
        base = self.base_url()
        if not base:
            return {"ok": False, "backend": self.name, "error": "没有正在运行的模型"}
        status, payload = http_get_json(f"{base}/health", timeout=3.0)
        if status == 0:
            return {"ok": False, "backend": self.name, "error": "无法连接到 llama-server"}
        if status != 200:
            return {"ok": False, "backend": self.name, "status_code": status, "error": "健康检查未通过"}
        return {"ok": True, "backend": self.name, "status_code": status, "detail": payload}

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.is_available(),
            "executable": self.server_path or "",
            "running": self._proc is not None and self._proc.poll() is None,
            "model_id": self._model_id,
            "port": self._port,
            "pid": self._proc.pid if self._proc is not None and self._proc.poll() is None else None,
            "base_url": self.base_url(),
        }


# ---------------------------------------------------------------- Ollama


class OllamaBackend(Backend):
    """通过 REST 调用本机 Ollama 服务。"""

    name = "ollama"
    display_name = "Ollama"

    def __init__(self, cfg: Config | None = None, url: str | None = None):
        super().__init__(cfg)
        self._url = (url or str(self.cfg.get("backend.ollama_url", DEFAULT_OLLAMA_URL))
                     or DEFAULT_OLLAMA_URL).rstrip("/")
        self._model_id: str = ""

    # ---------------- 能力 ----------------

    def is_available(self) -> bool:
        status, _ = http_get_json(f"{self._url}/api/tags", timeout=1.5)
        return status == 200

    def list_models(self) -> list[dict[str, Any]]:
        status, payload = http_get_json(f"{self._url}/api/tags", timeout=3.0)
        if status != 200 or not isinstance(payload, dict):
            return []
        out: list[dict[str, Any]] = []
        for item in payload.get("models", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if not name:
                continue
            out.append({
                "id": name,
                "name": name,
                "family": str(item.get("family") or _guess_family(name)),
                "size": item.get("size", 0),
                "modified_at": item.get("modified_at", ""),
            })
        return out

    # ---------------- 启动 / 停止 ----------------

    def start_model(self, model_id: str, **opts: Any) -> dict[str, Any]:
        """Ollama 无需启动进程，只要模型已存在即可直接推理。"""
        models = {m["id"] for m in self.list_models()}
        target = model_id
        if target not in models:
            # Ollama 的名字常带 :latest 后缀，尝试模糊匹配
            hit = next((m for m in models if m.split(":")[0] == target.split(":")[0]), None)
            if hit:
                target = hit
            else:
                raise RuntimeError(f"Ollama 中未找到模型：{model_id}（请先 ollama pull）")
        self._model_id = target
        return {
            "model_id": target,
            "path": "",
            "port": _port_of(self._url),
            "pid": None,
            "base_url": self.base_url(),
            "command": [],
            "log_file": "",
            "started_at": time.time(),
        }

    def stop_model(self) -> bool:
        """Ollama 没有 stop 概念，仅清空当前选择。"""
        had = bool(self._model_id)
        self._model_id = ""
        return had

    def is_running(self, model_id: str) -> bool:
        """Ollama 服务可访问且模型存在即视为可用。"""
        if not self.is_available():
            return False
        target = str(model_id or self._model_id)
        if not target:
            return False
        return any(m["id"] == target or m["id"].split(":")[0] == target.split(":")[0]
                   for m in self.list_models())

    def base_url(self) -> str:
        return self._url

    def health(self) -> dict[str, Any]:
        status, payload = http_get_json(f"{self._url}/api/tags", timeout=3.0)
        if status == 0:
            return {"ok": False, "backend": self.name, "error": "无法连接到 Ollama 服务"}
        count = len((payload or {}).get("models", []) or []) if isinstance(payload, dict) else 0
        return {"ok": status == 200, "backend": self.name, "status_code": status,
                "models": count}

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.is_available(),
            "url": self._url,
            "running": self.is_available(),
            "model_id": self._model_id,
            "base_url": self.base_url(),
        }


def _guess_family(name: str) -> str:
    """从 Ollama 模型名猜系列。"""
    low = name.lower()
    for fam, pat in (
        ("qwen", "qwen"), ("llama", "llama"), ("gemma", "gemma"),
        ("phi", "phi"), ("smol", "smol"), ("internlm", "internlm"),
        ("minicpm", "minicpm"), ("mistral", "mistral"), ("deepseek", "deepseek"),
    ):
        if pat in low:
            return fam
    return "unknown"


def _port_of(url: str) -> int:
    try:
        from urllib.parse import urlsplit
        return int(urlsplit(url).port or 80)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------- 探测


def detect_backends(cfg: Config | None = None) -> list[Backend]:
    """探测当前可用的后端，按配置偏好排序。"""
    cfg = cfg or Config.default()
    prefer = str(cfg.get("backend.prefer", "auto") or "auto").strip().lower()
    llama = LlamaCppBackend(cfg)
    ollama = OllamaBackend(cfg)

    if prefer in ("llama.cpp", "llama", "llamacpp"):
        ordered = [llama, ollama]
    elif prefer == "ollama":
        ordered = [ollama, llama]
    else:
        ordered = [llama, ollama]

    return [b for b in ordered if b.is_available()]
