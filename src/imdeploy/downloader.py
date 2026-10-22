"""模型下载管理。

- 支持 ModelScope / HuggingFace 直链下载
- 断点续传（Range 请求）
- 进度回调（供 GUI 显示）
- 可选 SHA256 校验
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

UA = "imdeploy/0.1 (+https://github.com/szkele1145)"

# ModelScope 直链前缀
MS_PREFIX = "https://modelscope.cn/models/{repo}/resolve/master/"
# HuggingFace 直链前缀
HF_PREFIX = "https://huggingface.co/{repo}/resolve/main/"

# 国内镜像（可选）
HF_MIRRORS = [
    "",  # 直连
    "https://hf-mirror.com/{repo}/resolve/main/",
]


@dataclass
class DownloadTask:
    """一个下载任务。"""

    url: str
    dest: Path
    name: str = ""
    size: int = 0
    downloaded: int = 0
    status: str = "pending"   # pending / downloading / done / error / cancelled
    error: str = ""
    sha256: str = ""

    @property
    def percent(self) -> float:
        if self.size <= 0:
            return 0.0
        return min(100.0, self.downloaded * 100.0 / self.size)

    @property
    def speed_mb(self) -> float:
        return self.downloaded / 1024 / 1024


class Downloader:
    """带断点续传的下载器。"""

    def __init__(self, timeout: int = 600, resume: bool = True,
                 verify_sha256: bool = False, chunk: int = 256 * 1024):
        self.timeout = timeout
        self.resume = resume
        self.verify_sha256 = verify_sha256
        self.chunk = chunk
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def download(self, url: str, dest: Path,
                 progress: Callable[[DownloadTask], None] | None = None,
                 sha256: str = "") -> DownloadTask:
        """下载单个文件。"""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        task = DownloadTask(url=url, dest=dest, name=dest.name, sha256=sha256)

        part = dest.with_suffix(dest.suffix + ".part")

        # 已完成则跳过
        if dest.exists() and dest.stat().st_size > 0:
            if not sha256 or not self.verify_sha256:
                task.status = "done"
                task.downloaded = dest.stat().st_size
                task.size = task.downloaded
                if progress:
                    progress(task)
                return task

        headers = {"User-Agent": UA}
        mode = "wb"
        if self.resume and part.exists():
            task.downloaded = part.stat().st_size
            headers["Range"] = f"bytes={task.downloaded}-"
            mode = "ab"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 206:  # 部分内容
                    pass
                else:
                    task.downloaded = 0
                    mode = "wb"
                clen = resp.headers.get("Content-Length")
                if clen:
                    task.size = int(clen) + (task.downloaded if mode == "ab" else 0)
                task.status = "downloading"
                if progress:
                    progress(task)

                with open(part, mode) as f:
                    while True:
                        if self._cancel.is_set():
                            task.status = "cancelled"
                            if progress:
                                progress(task)
                            return task
                        data = resp.read(self.chunk)
                        if not data:
                            break
                        f.write(data)
                        task.downloaded += len(data)
                        if progress:
                            progress(task)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            task.status = "error"
            task.error = str(exc)
            if progress:
                progress(task)
            return task

        # 校验
        if sha256 and self.verify_sha256:
            actual = _sha256_file(part)
            if actual.lower() != sha256.lower():
                task.status = "error"
                task.error = f"SHA256 不匹配: 期望 {sha256[:12]}… 实际 {actual[:12]}…"
                if progress:
                    progress(task)
                return task

        shutil.move(str(part), str(dest))
        task.status = "done"
        task.size = task.downloaded
        if progress:
            progress(task)
        return task

    def download_many(self, items: list[tuple[str, Path]],
                      progress: Callable[[DownloadTask], None] | None = None
                      ) -> list[DownloadTask]:
        """串行下载多个文件。"""
        results = []
        for url, dest in items:
            results.append(self.download(url, dest, progress))
            if results[-1].status == "cancelled":
                break
        return results


def resolve_url(repo: str, filename: str, source: str = "modelscope",
                mirror: bool = False) -> str:
    """拼出模型文件的直链。"""
    repo = repo.lstrip("/")
    if source == "huggingface":
        if mirror:
            return HF_MIRRORS[1].format(repo=repo) + filename
        return HF_PREFIX.format(repo=repo) + filename
    return MS_PREFIX.format(repo=repo) + filename


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_gguf_filename(filename: str) -> dict:
    """从 GGUF 文件名里解析参数量、量化、上下文等元信息。

    例: ``Qwen3-1.7B-Instruct-UD-Q4_K_XL.gguf``
    """
    info: dict = {
        "family": "unknown",
        "params_b": 0.0,
        "quant": "",
        "ctx_len": 0,
        "instruct": False,
        "long_context": False,
    }
    name = filename.replace(".gguf", "")
    low = name.lower()

    # 参数量: 0.6B / 1.7B / 7B / 13B
    m = re.search(r"(\d+(?:\.\d+)?)\s*([BbMm])\b", name)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "b":
            info["params_b"] = num
        else:  # 百万参数换算成 B
            info["params_b"] = round(num / 1000.0, 3)

    # 量化
    q = re.search(r"\b(Q\d[_A-Za-z0-9]*|IQ\d[_A-Za-z0-9]*|F16|BF16|F32)\b", name, re.I)
    if q:
        info["quant"] = q.group(1).upper()

    # 系列
    for fam, pat in (
        ("qwen3", r"qwen\s*-?\s*3"),
        ("qwen", r"qwen"),
        ("llama", r"llama"),
        ("gemma", r"gemma"),
        ("phi", r"phi[-_]?[34]"),
        ("smol", r"smol"),
        ("internlm", r"internlm"),
        ("minicpm", r"minicpm"),
        ("mistral", r"mistral|mixtral"),
        ("deepseek", r"deepseek"),
        ("yi", r"\byi[-_]"),
    ):
        if re.search(pat, low):
            info["family"] = fam
            break

    # 指令微调
    info["instruct"] = bool(re.search(r"instruct|chat|-it\b|sft|[-_]it[-_]", low))
    if re.search(r"base|pretrained|pt\b", low) and not info["instruct"]:
        info["instruct"] = False

    # 上下文长度
    c = re.search(r"(\d+)k\b", low)
    if c:
        info["ctx_len"] = int(c.group(1)) * 1024
    if re.search(r"128k|256k|1m", low):
        info["ctx_len"] = max(info["ctx_len"], 131072)

    info["long_context"] = info["ctx_len"] >= 32768
    return info
