"""本地模型库管理。

扫描 models/ 目录下已下载的 GGUF，登记到 data/library.json；
提供「按别名选模型」「列出已装」「删除」等操作。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .downloader import parse_gguf_filename


@dataclass
class LocalModel:
    """已下载的本地模型。"""

    id: str
    name: str
    family: str
    params_b: float
    quant: str
    ctx_len: int
    size_gb: float
    path: str
    added_at: float
    instruct: bool = True
    long_context: bool = False
    description: str = ""
    download_url: str = ""
    source: str = ""

    @property
    def sort_key(self) -> tuple:
        return (0 if self.instruct else 1, -self.ctx_len, self.params_b)


class ModelLibrary:
    """本地模型库。"""

    LIB_FILE = "library.json"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models_dir = cfg.models_dir
        self.data_dir = cfg.data_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lib_path = self.data_dir / self.LIB_FILE
        self._entries: dict[str, LocalModel] = {}
        self._load()

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        if not self._lib_path.exists():
            return
        try:
            data = json.loads(self._lib_path.read_text(encoding="utf-8"))
            for item in data.get("models", []):
                m = LocalModel(**item)
                self._entries[m.id] = m
        except (json.JSONDecodeError, OSError, TypeError):
            # 库文件损坏则重建
            self._entries = {}

    def _save(self) -> None:
        payload = {"models": [asdict(m) for m in self._entries.values()],
                   "saved_at": time.time()}
        self._lib_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    # ------------------------------------------------------------ 扫描/登记

    def scan(self) -> list[LocalModel]:
        """扫描目录登记所有 GGUF（自动发现手动拷贝进来的模型）。"""
        found = 0
        for f in self.models_dir.rglob("*.gguf"):
            if self.register(f):
                found += 1
        # 已登记但文件没了 -> 移除
        for mid, m in list(self._entries.items()):
            p = Path(m.path)
            if not p.is_absolute():
                p = self.models_dir.parent / p
            if not p.exists():
                del self._entries[mid]
                found += 1
        if found:
            self._save()
        return self.list()

    def register(self, path: Path, meta: dict[str, Any] | None = None) -> LocalModel | None:
        """登记一个 GGUF 文件。"""
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".gguf":
            return None
        info = parse_gguf_filename(path.name)
        meta = meta or {}
        mid = path.stem.lower()
        m = LocalModel(
            id=mid,
            name=path.stem,
            family=meta.get("family") or info["family"],
            params_b=meta.get("params_b") or info["params_b"],
            quant=meta.get("quant") or info["quant"],
            ctx_len=meta.get("ctx_len") or info["ctx_len"],
            size_gb=round(path.stat().st_size / 1024**3, 2),
            path=str(path),
            added_at=time.time(),
            instruct=meta.get("instruct", info["instruct"]),
            long_context=meta.get("long_context", info["long_context"]),
            description=meta.get("description", ""),
            download_url=meta.get("download_url", ""),
            source=meta.get("source", ""),
        )
        self._entries[mid] = m
        self._save()
        return m

    # ------------------------------------------------------------ 查询

    def list(self) -> list[LocalModel]:
        return sorted(self._entries.values(), key=lambda m: m.sort_key)

    def get(self, mid: str) -> LocalModel | None:
        return self._entries.get(mid)

    def find_by_family(self, family: str) -> list[LocalModel]:
        return [m for m in self._entries.values() if m.family == family]

    def best_for_alias(self, alias: str) -> LocalModel | None:
        """按别名找最佳模型。

        优先级: 显式配置 > 精确 family 匹配 > 模糊匹配 > 第一个可用的。
        """
        # 1) 显式映射
        mapping = self.cfg.get("api.aliases", {}) or {}
        if alias in mapping:
            return self.get(str(mapping[alias]))
        # 2) family 精确
        exact = self.find_by_family(alias)
        if exact:
            return exact[0]
        # 3) 模糊
        pat = re.compile(re.escape(alias.lower()))
        for m in self._entries.values():
            if pat.search(m.family) or pat.search(m.id):
                return m
        # 4) default
        models = self.list()
        return models[0] if models else None

    # ------------------------------------------------------------ 删除

    def remove(self, mid: str, delete_file: bool = True) -> bool:
        m = self._entries.pop(mid, None)
        if m is None:
            return False
        if delete_file:
            p = Path(m.path)
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        self._save()
        return True
