"""配置系统。

配置以 YAML 落盘（data/config.yaml），敏感项（如访问口令）可加密存储。
所有路径相对于项目根目录解析，不写死本机绝对路径。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- 默认配置

DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "name": "本地模型部署器",
        "host": "127.0.0.1",
        "port": 8080,
        "token": "",
        "open_browser": True,
        "log_level": "INFO",
        "data_dir": "data",
        "models_dir": "models",
        "log_dir": "logs",
    },
    "backend": {
        "prefer": "llama.cpp",      # llama.cpp / ollama / auto
        "llama_server": "",          # llama-server 可执行文件路径（留空=自动查找）
        "ollama_url": "http://127.0.0.1:11434",
        "base_port": 8120,           # 后端模型服务端口起始
        "gpu_layers": -1,            # -1=自动，0=纯CPU
        "threads": 0,                # 0=自动
        "context_size": 8192,        # 启动模型的默认上下文长度
        "max_context_size": 131072,  # 允许的最大上下文（长上下文模型）
    },
    "catalog": {
        "sources": ["modelscope", "huggingface"],
        "queries": [
            "gguf qwen3", "gguf qwen2.5", "gguf llama", "gguf gemma",
            "gguf smollm", "gguf phi", "gguf internlm", "gguf minicpm",
        ],
        "limit_per_query": 30,
        "cache_hours": 12,
        "preferred_quant": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q4_K_S", "Q4_0"],
        "min_context": 8192,
    },
    "recommend": {
        "prefer_small": True,        # 0-2B 小模型优先
        "small_params_max": 2.0,
        "long_context_bonus": True,
        "min_ctx_for_bonus": 32768,
        "memory_headroom": 0.75,     # 权重占用可用内存的比例上限
    },
    "download": {
        "concurrent": 1,
        "timeout": 600,
        "verify_sha256": False,
        "resume": True,
    },
    "api": {
        "aliases": {},                # 显式别名映射: {"qwen": "model-id"}
        "expose_embedding": True,
        "stream": True,
    },
    "ui": {
        "theme": "dark",
        "auto_check_update": True,
    },
}

# ---------------------------------------------------------------- 中文化

FIELD_LABELS: dict[str, str] = {
    "app.name": "程序名称",
    "app.host": "监听地址",
    "app.port": "监听端口",
    "app.token": "访问口令（留空=不校验）",
    "app.open_browser": "启动后打开浏览器",
    "app.log_level": "日志级别",
    "app.data_dir": "数据目录",
    "app.models_dir": "模型存放目录",
    "app.log_dir": "日志目录",
    "backend.prefer": "优先后端",
    "backend.llama_server": "llama-server 路径",
    "backend.ollama_url": "Ollama 地址",
    "backend.base_port": "模型服务起始端口",
    "backend.gpu_layers": "GPU 层数（-1=自动，0=纯 CPU）",
    "backend.threads": "线程数（0=自动）",
    "backend.context_size": "默认上下文长度",
    "backend.max_context_size": "最大上下文长度",
    "catalog.sources": "模型来源",
    "catalog.queries": "抓取关键词",
    "catalog.limit_per_query": "每关键词抓取数量",
    "catalog.cache_hours": "目录缓存小时数",
    "catalog.preferred_quant": "偏好量化格式",
    "catalog.min_context": "最小上下文要求",
    "recommend.prefer_small": "优先小模型（0-2B）",
    "recommend.small_params_max": "小模型参数量上限（B）",
    "recommend.long_context_bonus": "长上下文加权",
    "recommend.min_ctx_for_bonus": "长上下文加分起点",
    "recommend.memory_headroom": "内存占用比例上限",
    "download.concurrent": "并发下载数",
    "download.timeout": "下载超时（秒）",
    "download.verify_sha256": "校验 SHA256",
    "download.resume": "断点续传",
    "api.aliases": "别名映射",
    "api.expose_embedding": "开放嵌入接口",
    "api.stream": "支持流式输出",
    "ui.theme": "界面主题",
    "ui.auto_check_update": "自动检查更新",
}

SECTION_TITLES: dict[str, str] = {
    "app": "服务设置",
    "backend": "推理后端",
    "catalog": "模型目录",
    "recommend": "推荐策略",
    "download": "下载设置",
    "api": "接口设置",
    "ui": "界面",
}

VALUE_HINTS: dict[str, dict[str, str]] = {
    "backend.prefer": {"llama.cpp": "llama.cpp（GGUF，最通用）", "ollama": "Ollama", "auto": "自动检测"},
    "app.log_level": {"DEBUG": "调试", "INFO": "普通", "WARNING": "警告", "ERROR": "错误"},
    "catalog.sources": {"modelscope": "魔搭（国内快）", "huggingface": "HuggingFace"},
}


def field_label(dotted: str) -> str:
    return FIELD_LABELS.get(dotted, dotted.split(".")[-1])


def section_title(key: str) -> str:
    return SECTION_TITLES.get(key, key)


def value_hint(dotted: str, value: Any) -> str:
    return VALUE_HINTS.get(dotted, {}).get(str(value), "")


# ---------------------------------------------------------------- Config


class Config:
    """点号路径访问的配置对象。"""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None):
        self._data = copy.deepcopy(data if data is not None else DEFAULT_CONFIG)
        self._path = Path(path) if path else None
        self._base_dir = Path(path).resolve().parent if path else Path.cwd()

    # ---------------- 构造 ----------------

    @classmethod
    def default(cls) -> "Config":
        return cls(copy.deepcopy(DEFAULT_CONFIG), None)

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else Path("data/config.yaml")
        if not p.exists():
            cfg = cls(copy.deepcopy(DEFAULT_CONFIG), p)
            return cfg
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), data)
        return cls(merged, p)

    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path else (self._path or Path("data/config.yaml"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self._data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self._path = p
        return p

    # ---------------- 访问器 ----------------

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        val = self._data.get(name, {})
        return val if isinstance(val, dict) else {}

    # ---------------- 路径 ----------------

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def path(self, dotted: str) -> Path:
        """按配置解析相对路径（相对配置文件所在目录）。"""
        raw = str(self.get(dotted, "") or "")
        if not raw:
            return self._base_dir
        p = Path(raw)
        return p if p.is_absolute() else (self._base_dir / p)

    @property
    def data_dir(self) -> Path:
        return self.path("app.data_dir")

    @property
    def models_dir(self) -> Path:
        return self.path("app.models_dir")

    @property
    def log_dir(self) -> Path:
        return self.path("app.log_dir")

    # ---------------- 其他 ----------------

    def masked(self) -> dict[str, Any]:
        """掩码敏感字段（用于面板回显）。"""
        data = copy.deepcopy(self._data)
        for dotted in ("app.token", "backend.llama_server"):
            node = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.get(p, {})
                if not isinstance(node, dict):
                    node = {}
                    break
            if isinstance(node, dict) and parts[-1] in node:
                val = node[parts[-1]]
                if isinstance(val, str) and val:
                    node[parts[-1]] = (val[:4] + "****" + val[-4:]) if len(val) > 8 else "******"
        return data


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并（override 优先）。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
