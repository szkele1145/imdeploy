"""模型目录：抓取 HuggingFace / ModelScope 上的 GGUF 模型，并做本地缓存。

设计要点：
- 只用标准库（urllib / json / dataclasses），不引入第三方 HTTP 依赖。
- 任何网络失败都必须优雅降级（返回空列表 / 抛出的异常被吞掉），不影响其它来源。
- 解析出的模型统一收敛到 :class:`ModelEntry`，供 recommend 模块做打分匹配。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "ModelEntry",
    "DEFAULT_QUERIES",
    "DEFAULT_CACHE_PATH",
    "fetch_hf_models",
    "fetch_modelscope_models",
    "parse_hf_model",
    "parse_modelscope_model",
    "load_cached_catalog",
    "load_cached_catalog_meta",
    "save_cached_catalog",
    "build_catalog",
    "detect_quant",
    "detect_family",
    "detect_params_b",
    "detect_ctx_len",
    "estimate_params_b",
]

logger = logging.getLogger(__name__)

USER_AGENT = "imdeploy/0.1 (local GGUF model recommender; python-urllib)"
HF_MODELS_API = "https://huggingface.co/api/models"
#: 国内可用的 HuggingFace 镜像，直连失败时自动回退
HF_MIRROR_API = "https://hf-mirror.com/api/models"
MODELSCOPE_MODELS_API = "https://modelscope.cn/api/v1/models"
MODELSCOPE_FILES_API = "https://modelscope.cn/api/v1/models/{path}/repo/files"
#: 镜像的下载前缀（resolve 路径）
HF_MIRROR_DL = "https://hf-mirror.com"

DEFAULT_TIMEOUT = 20.0
DEFAULT_LIMIT = 20
DEFAULT_DETAIL_WORKERS = 6

#: 默认查询词（覆盖主流小模型家族 + 长上下文优化版）
DEFAULT_QUERIES: tuple[str, ...] = (
    "Qwen2.5 GGUF",
    "Qwen3 GGUF",
    "Llama-3.2 GGUF",
    "SmolLM2 GGUF",
    "gemma-2 GGUF",
    "Phi-3.5 GGUF",
    "MiniCPM GGUF",
    "TinyLlama GGUF",
    "InternLM2 GGUF",
)

#: 默认缓存位置：<项目根>/data/catalog.json（相对包路径推导，不写死绝对路径）
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "catalog.json"

_GB = 1024.0**3


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ModelEntry:
    """一个 GGUF 模型的统一描述。"""

    id: str = ""
    name: str = ""
    family: str = ""
    params_b: float = 0.0
    quant: str = ""
    ctx_len: int = 0
    size_gb: float = 0.0
    download_url: str = ""
    sha256: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    description: str = ""
    repo_url: str = ""
    license: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ModelEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        data = {k: v for k, v in (obj or {}).items() if k in known}
        data["tags"] = [str(t) for t in (data.get("tags") or [])]
        data["params_b"] = float(data.get("params_b") or 0.0)
        data["ctx_len"] = int(data.get("ctx_len") or 0)
        data["size_gb"] = float(data.get("size_gb") or 0.0)
        return cls(**data)

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"{self.id} ({self.params_b}B, {self.quant}, {self.ctx_len // 1024}K, {self.size_gb:.1f}GB)"


@dataclass
class _File:
    """仓库内的一个文件。"""

    name: str
    size: int = 0
    sha256: str = ""


# --------------------------------------------------------------------------- #
# 字段识别工具
# --------------------------------------------------------------------------- #
_QUANT_RE = re.compile(r"\b(iq\d+(?:_[a-z0-9]+)*|q\d+(?:_[a-z0-9]+)*|f16|f32|bf16)\b", re.IGNORECASE)
_PARAM_B_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-z0-9])")
_PARAM_M_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*[mM](?![a-z0-9])")
_CTX_K_RE = re.compile(r"(?<![a-z0-9.])(\d{1,4})\s*[kK](?![a-z0-9])")
_CTX_NUM_RE = re.compile(r"(?<![a-z0-9.])(\d{4,7})(?![a-z0-9])")

_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("smollm", r"smollm|smol-?lm"),
    ("qwen", r"qwen"),
    ("gemma", r"gemma"),
    ("llama", r"llama|tinyllama"),
    ("phi", r"phi[-_ ]?\d"),
    ("internlm", r"internlm|intern-?lm"),
    ("minicpm", r"minicpm"),
    ("mistral", r"mistral|mixtral|ministral"),
    ("deepseek", r"deepseek"),
    ("gptoss", r"gpt-?oss"),
    ("olmo", r"olmo"),
    ("granite", r"granite"),
    ("baichuan", r"baichuan"),
    ("yi", r"(^|[/_.-])yi[-_/]"),
    ("cohere", r"aya"),
)

#: 量化格式的体积系数（GB / B 参数），用于在缺少体积信息时估算
_BYTES_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.35,
    "Q3_K_M": 0.45,
    "Q4_0": 0.55,
    "Q4_K_S": 0.58,
    "Q4_K_M": 0.60,
    "IQ4_XS": 0.58,
    "Q5_0": 0.68,
    "Q5_K_M": 0.70,
    "Q6_K": 0.80,
    "Q8_0": 1.00,
    "F16": 2.00,
    "BF16": 2.00,
    "F32": 4.00,
}


def detect_quant(text: str) -> str:
    """从文件名/仓库名中识别量化格式，例如 ``Q4_K_M``。"""
    if not text:
        return ""
    match = _QUANT_RE.search(text)
    if not match:
        return ""
    quant = match.group(1).upper()
    if quant.startswith("IQ"):
        return "IQ" + quant[2:]
    if quant in ("F16", "F32", "BF16"):
        return quant
    return quant


def detect_family(text: str) -> str:
    """从名称中识别模型家族（qwen / llama / gemma ...）。"""
    if not text:
        return ""
    low = text.lower()
    for family, pattern in _FAMILY_RULES:
        if re.search(pattern, low):
            return family
    return ""


def detect_params_b(text: str) -> float:
    """从名称中识别参数量（单位 B）。识别不到返回 0.0。"""
    if not text:
        return 0.0
    match = _PARAM_B_RE.search(text)
    if match:
        try:
            value = float(match.group(1))
        except ValueError:
            return 0.0
        return value if 0 < value < 1000 else 0.0
    match = _PARAM_M_RE.search(text)
    if match:
        try:
            value = float(match.group(1))
        except ValueError:
            return 0.0
        return round(value / 1000.0, 3) if 0 < value < 100000 else 0.0
    return 0.0


def detect_ctx_len(text: str) -> int:
    """从名称中识别上下文长度（支持 ``32k`` / ``128K`` / ``32768`` 写法）。"""
    if not text:
        return 0
    match = _CTX_K_RE.search(text)
    if match:
        try:
            value = int(match.group(1)) * 1024
        except ValueError:
            value = 0
        if 1024 <= value <= 4_194_304:
            return value
    match = _CTX_NUM_RE.search(text)
    if match:
        try:
            value = int(match.group(1))
        except ValueError:
            value = 0
        if 1024 <= value <= 4_194_304:
            return value
    return 0


def estimate_params_b(size_gb: float, quant: str) -> float:
    """在缺少参数量信息时，用体积与量化格式反推参数量。"""
    if size_gb <= 0:
        return 0.0
    factor = _BYTES_PER_PARAM.get((quant or "").upper(), 0.65)
    return round(size_gb / factor, 3)


# --------------------------------------------------------------------------- #
# HTTP 基础
# --------------------------------------------------------------------------- #
def _http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT, headers: dict[str, str] | None = None) -> Any:
    """用 urllib 发 GET 请求并解析 JSON。"""
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return json.loads(raw)


def _quote(value: str) -> str:
    return urllib.parse.quote(str(value), safe="/")


# --------------------------------------------------------------------------- #
# HuggingFace
# --------------------------------------------------------------------------- #
def _hf_files(siblings: Any) -> list[_File]:
    """把 HF 的 siblings 列表转成 _File 列表（尽量取到体积和 sha256）。"""
    files: list[_File] = []
    for sibling in siblings or []:
        if isinstance(sibling, str):
            files.append(_File(name=sibling))
            continue
        if not isinstance(sibling, dict):
            continue
        name = str(sibling.get("rfilename") or sibling.get("name") or "")
        if not name:
            continue
        size = sibling.get("size")
        if not size:
            lfs = sibling.get("lfs") or {}
            if isinstance(lfs, dict):
                size = lfs.get("size") or lfs.get("pointerSize") or 0
        sha = str(sibling.get("sha256") or "")
        lfs = sibling.get("lfs") or {}
        if not sha and isinstance(lfs, dict):
            oid = str(lfs.get("oid") or "")
            if oid:
                sha = oid.split(":", 1)[1] if ":" in oid else oid
        try:
            size_int = int(size or 0)
        except (TypeError, ValueError):
            size_int = 0
        files.append(_File(name=name, size=size_int, sha256=sha))
    return files


_QUANT_PREFERENCE = (
    "Q4_K_M",
    "Q5_K_M",
    "Q6_K",
    "Q4_K_S",
    "Q8_0",
    "IQ4_XS",
    "Q5_0",
    "Q4_0",
    "Q3_K_M",
    "Q2_K",
)


def _choose_gguf(files: Sequence[_File]) -> _File | None:
    """从多个 gguf 文件里挑一个代表文件（优先 Q4_K_M 这类通用量化）。"""
    ggufs = [f for f in files if f.name.lower().endswith(".gguf")]
    if not ggufs:
        return None
    for preferred in _QUANT_PREFERENCE:
        for candidate in ggufs:
            if detect_quant(candidate.name) == preferred:
                return candidate
    return ggufs[0]


def _hf_ctx_len(obj: dict[str, Any], files: Sequence[_File]) -> int:
    """尽力从多个位置提取上下文长度。"""
    gguf_meta = obj.get("gguf")
    if isinstance(gguf_meta, dict):
        value = gguf_meta.get("context_length")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    for holder in (obj.get("config"), obj.get("cardData"), obj.get("gguf"), obj):
        if not isinstance(holder, dict):
            continue
        for key in ("context_length", "max_position_embeddings", "n_ctx", "max_sequence_length"):
            value = holder.get(key)
            if isinstance(value, (int, float)) and 0 < int(value) <= 4_194_304:
                return int(value)
        rope = holder.get("rope_scaling")
        if isinstance(rope, dict):
            for key in ("original_max_position_embeddings", "max_position_embeddings"):
                value = rope.get(key)
                if isinstance(value, (int, float)) and 0 < int(value) <= 4_194_304:
                    return int(value)

    for tag in obj.get("tags") or []:
        if not isinstance(tag, str):
            continue
        if tag.startswith("context_length:"):
            try:
                value = int(tag.split(":", 1)[1])
            except ValueError:
                value = 0
            if 0 < value <= 4_194_304:
                return value
        if tag.lower() == "gguf":
            continue
        # HF 有时把 gguf 元数据以 JSON 形式塞进 tag
        if tag.startswith("{"):
            try:
                meta = json.loads(tag)
            except (ValueError, TypeError):
                meta = None
            if isinstance(meta, dict):
                value = meta.get("context_length")
                if isinstance(value, (int, float)) and 0 < int(value) <= 4_194_304:
                    return int(value)

    for f in files:
        value = detect_ctx_len(f.name)
        if value:
            return value
    return 0


def _hf_params_b(obj: dict[str, Any], files: Sequence[_File], quant: str, size_gb: float) -> float:
    gguf_meta = obj.get("gguf")
    if isinstance(gguf_meta, dict):
        total = gguf_meta.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return round(float(total) / 1e9, 3)
    for text in (obj.get("modelId") or "", obj.get("id") or ""):
        value = detect_params_b(str(text))
        if value:
            return value
    for f in files:
        value = detect_params_b(f.name)
        if value:
            return value
    return estimate_params_b(size_gb, quant)


def parse_hf_model(obj: dict[str, Any]) -> ModelEntry | None:
    """把一条 HuggingFace API 数据解析成 :class:`ModelEntry`；不是 GGUF 则返回 ``None``。"""
    if not isinstance(obj, dict):
        return None
    repo_id = str(obj.get("modelId") or obj.get("id") or "").strip()
    if not repo_id:
        return None
    files = _hf_files(obj.get("siblings"))
    chosen = _choose_gguf(files)
    if chosen is None:
        return None

    quant = detect_quant(chosen.name) or detect_quant(repo_id)
    size_gb = round(chosen.size / _GB, 3) if chosen.size else 0.0
    tags = [str(t) for t in (obj.get("tags") or []) if isinstance(t, (str, int, float))]
    card = obj.get("cardData") if isinstance(obj.get("cardData"), dict) else {}
    license_name = str(card.get("license") or "")
    if not license_name:
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_name = tag.split(":", 1)[1]
                break
    description = str(obj.get("description") or card.get("pretty_name") or "")
    display_name = str(card.get("pretty_name") or repo_id.split("/")[-1])
    params_b = _hf_params_b(obj, files, quant, size_gb)
    ctx_len = _hf_ctx_len(obj, files)
    # 下载地址走镜像（国内 huggingface.co 直连基本不通）
    download_url = f"{HF_MIRROR_DL}/{_quote(repo_id)}/resolve/main/{_quote(chosen.name)}"
    return ModelEntry(
        id=repo_id,
        name=display_name,
        family=detect_family(repo_id),
        params_b=params_b,
        quant=quant,
        ctx_len=ctx_len,
        size_gb=size_gb,
        download_url=download_url,
        sha256=chosen.sha256,
        tags=tags,
        source="hf",
        description=description,
        repo_url=f"https://huggingface.co/{_quote(repo_id)}",
        license=license_name,
    )


def _hf_detail(model_id: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """拉取单个模型的完整信息（带文件体积）。失败返回空 dict。"""
    for base in (HF_MIRROR_API, HF_MODELS_API):
        url = f"{base}/{_quote(model_id)}?blobs=true"
        try:
            payload = _http_get_json(url, timeout)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:  # noqa: BLE001 - 网络异常统一降级
            logger.debug("HF 详情抓取失败 %s: %s", model_id, exc)
    return {}


def fetch_hf_models(
    query: str,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    with_details: bool = True,
) -> list[ModelEntry]:
    """搜索 HuggingFace 上的 GGUF 模型。失败时自动回退 hf-mirror；都失败返回空列表。"""
    params = urllib.parse.urlencode({"search": query, "limit": max(1, int(limit)), "full": "true"})
    payload: Any = None
    used_mirror = False
    # 镜像优先：国内直连 huggingface.co 经常超时（实测 60s+）
    for base in (HF_MIRROR_API, HF_MODELS_API):
        url = f"{base}?{params}"
        try:
            payload = _http_get_json(url, timeout)
            used_mirror = base == HF_MIRROR_API
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("模型目录抓取失败（%s / query=%s）: %s", base, query, exc)
    if payload is None:
        return []
    if used_mirror:
        logger.info("已通过 hf-mirror 镜像获取目录（query=%s）", query)
    if not isinstance(payload, list):
        logger.warning("HuggingFace 返回格式异常（query=%s）", query)
        return []

    entries: list[ModelEntry] = []
    for obj in payload:
        entry = parse_hf_model(obj)
        if entry is not None:
            entries.append(entry)

    if with_details and entries:
        entries = _enrich_hf_entries(entries, timeout)
    return entries


def _enrich_hf_entries(entries: list[ModelEntry], timeout: float) -> list[ModelEntry]:
    """给缺少体积/上下文信息的条目补一次详情请求。"""
    todo = [e for e in entries if not e.size_gb or not e.ctx_len]
    if not todo:
        return entries

    def _work(entry: ModelEntry) -> tuple[ModelEntry, dict[str, Any]]:
        return entry, _hf_detail(entry.id, timeout)

    workers = min(DEFAULT_DETAIL_WORKERS, max(1, len(todo)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_work, todo))

    by_id = {e.id: e for e in entries}
    for entry, detail in results:
        if not detail:
            continue
        files = _hf_files(detail.get("siblings"))
        chosen = _choose_gguf(files)
        if chosen is not None:
            if chosen.size:
                entry.size_gb = round(chosen.size / _GB, 3)
            if chosen.sha256:
                entry.sha256 = chosen.sha256
            entry.download_url = (
                f"https://huggingface.co/{_quote(entry.id)}/resolve/main/{_quote(chosen.name)}"
            )
        if not entry.ctx_len:
            entry.ctx_len = _hf_ctx_len(detail, files)
        if not entry.params_b:
            entry.params_b = _hf_params_b(detail, files, entry.quant, entry.size_gb)
        if not entry.license:
            card = detail.get("cardData") if isinstance(detail.get("cardData"), dict) else {}
            entry.license = str(card.get("license") or "")
    _ = by_id
    return entries


# --------------------------------------------------------------------------- #
# ModelScope
# --------------------------------------------------------------------------- #
def _modelscope_model_list(payload: Any) -> list[dict[str, Any]]:
    """从 ModelScope 的嵌套返回结构里挖出模型列表。"""
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("Data")
    if isinstance(data, dict):
        model = data.get("Model")
        if isinstance(model, dict):
            models = model.get("Models")
            if isinstance(models, list):
                return [m for m in models if isinstance(m, dict)]
        models = data.get("Models")
        if isinstance(models, list):
            return [m for m in models if isinstance(m, dict)]
    models = payload.get("Models")
    if isinstance(models, list):
        return [m for m in models if isinstance(m, dict)]
    return []


def _modelscope_files(payload: Any) -> list[_File]:
    """从 ModelScope 文件列表返回结构里挖出文件。"""
    candidates: list[Any] = []
    if isinstance(payload, dict):
        data = payload.get("Data")
        if isinstance(data, dict):
            files = data.get("Files")
            if isinstance(files, list):
                candidates = files
            elif isinstance(data, list):
                candidates = data
    elif isinstance(payload, list):
        candidates = payload

    out: list[_File] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or item.get("Path") or item.get("name") or "")
        if not name:
            continue
        size = item.get("Size") or item.get("size") or 0
        try:
            size_int = int(size or 0)
        except (TypeError, ValueError):
            size_int = 0
        sha = str(item.get("Sha256") or item.get("sha256") or item.get("Sha256Hash") or "")
        out.append(_File(name=name, size=size_int, sha256=sha))
    return out


def _modelscope_repo_files(path: str, timeout: float = DEFAULT_TIMEOUT) -> list[_File]:
    """拉取 ModelScope 仓库的文件列表。失败返回空列表。"""
    params = urllib.parse.urlencode({"Revision": "master", "Root": ""})
    url = f"{MODELSCOPE_FILES_API.format(path=_quote(path))}?{params}"
    try:
        payload = _http_get_json(url, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ModelScope 文件列表抓取失败 %s: %s", path, exc)
        return []
    return _modelscope_files(payload)


def parse_modelscope_model(obj: dict[str, Any], require_gguf: bool = True) -> ModelEntry | None:
    """把一条 ModelScope 模型数据解析成 :class:`ModelEntry`。

    ModelScope 的列表接口不返回文件列表，因此需要先把 ``Files`` 塞进 obj，
    只有确认仓库里有 gguf 文件时才会返回条目（``require_gguf=False`` 时宽松返回）。
    """
    if not isinstance(obj, dict):
        return None
    repo_id = str(obj.get("Path") or obj.get("ModelId") or obj.get("Name") or "").strip()
    if not repo_id:
        return None

    raw_files = obj.get("Files") if obj.get("Files") is not None else obj.get("files")
    files = _modelscope_files({"Data": {"Files": raw_files}}) if raw_files else []
    if not files:
        # 某些返回把文件列表直接放在 obj["Data"] 或 obj 顶层
        files = _modelscope_files({"Data": obj}) + _modelscope_files(obj)
    if not files:
        # 已经是 _File 对象列表（内部流程 enrich 后）
        files = [f for f in (raw_files or []) if isinstance(f, _File)]
    ggufs = [f for f in files if f.name.lower().endswith(".gguf")]
    if require_gguf and not ggufs:
        return None

    chosen = _choose_gguf(files) if ggufs else None
    quant = detect_quant(chosen.name) if chosen else detect_quant(repo_id)
    size_gb = round(chosen.size / _GB, 3) if (chosen and chosen.size) else 0.0
    tags = [str(t) for t in (obj.get("Tags") or obj.get("tags") or []) if isinstance(t, (str, int, float))]
    tasks = obj.get("Tasks")
    if isinstance(tasks, list):
        tags.extend(str(t) for t in tasks if isinstance(t, str))
    frameworks = obj.get("Frameworks")
    if isinstance(frameworks, list):
        tags.extend(str(t) for t in frameworks if isinstance(t, str))
    description = str(obj.get("Description") or obj.get("ChineseName") or "")
    display_name = str(obj.get("Name") or repo_id.split("/")[-1])
    params_b = detect_params_b(repo_id) or detect_params_b(display_name) or estimate_params_b(size_gb, quant)
    ctx_len = detect_ctx_len(repo_id) or detect_ctx_len(display_name)
    if not ctx_len and chosen is not None:
        ctx_len = detect_ctx_len(chosen.name)
    license_name = str(obj.get("License") or obj.get("LicenseName") or "")
    download_url = (
        f"https://modelscope.cn/models/{_quote(repo_id)}/resolve/master/{_quote(chosen.name)}"
        if chosen
        else ""
    )
    return ModelEntry(
        id=repo_id,
        name=display_name,
        family=detect_family(repo_id),
        params_b=params_b,
        quant=quant,
        ctx_len=ctx_len,
        size_gb=size_gb,
        download_url=download_url,
        sha256=chosen.sha256 if chosen else "",
        tags=tags,
        source="modelscope",
        description=description,
        repo_url=f"https://modelscope.cn/models/{_quote(repo_id)}",
        license=license_name,
    )


def fetch_modelscope_models(
    query: str,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    with_files: bool = True,
) -> list[ModelEntry]:
    """搜索 ModelScope 上的 GGUF 模型。失败返回空列表（不抛异常）。"""
    params = urllib.parse.urlencode(
        {"Query": query, "PageNumber": 1, "PageSize": max(1, int(limit))}
    )
    url = f"{MODELSCOPE_MODELS_API}?{params}"
    try:
        payload = _http_get_json(url, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ModelScope 抓取失败（query=%s）: %s", query, exc)
        return []

    models = _modelscope_model_list(payload)
    if not models:
        logger.warning("ModelScope 无结果（query=%s）", query)
        return []
    if not with_files:
        return []

    def _work(obj: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(obj)
        path = str(obj.get("Path") or obj.get("ModelId") or obj.get("Name") or "")
        if path:
            enriched["Files"] = _modelscope_repo_files(path, timeout)
        return enriched

    workers = min(DEFAULT_DETAIL_WORKERS, max(1, len(models)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        enriched_list = list(pool.map(_work, models))

    entries: list[ModelEntry] = []
    for obj in enriched_list:
        entry = parse_modelscope_model(obj)
        if entry is not None:
            entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_cached_catalog(path: str | Path, entries: Iterable[ModelEntry], fetched_at: str | None = None) -> Path:
    """把目录写入 JSON 缓存（带 fetched_at）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    items = [e.to_dict() for e in entries]
    payload = {
        "fetched_at": fetched_at or _utc_now_iso(),
        "count": len(items),
        "models": items,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_cached_catalog_meta(path: str | Path) -> dict[str, Any]:
    """读取缓存原始内容（含 fetched_at）；文件缺失或损坏返回空 dict。"""
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, list):
        return {"fetched_at": "", "count": len(data), "models": data}
    if isinstance(data, dict):
        return data
    return {}


def load_cached_catalog(path: str | Path) -> list[ModelEntry]:
    """读取缓存里的模型列表；文件缺失或损坏返回空列表。"""
    meta = load_cached_catalog_meta(path)
    models = meta.get("models")
    if not isinstance(models, list):
        return []
    entries: list[ModelEntry] = []
    for obj in models:
        if not isinstance(obj, dict):
            continue
        try:
            entries.append(ModelEntry.from_dict(obj))
        except Exception as exc:  # noqa: BLE001
            logger.debug("缓存条目解析失败: %s", exc)
    return entries


def _cache_is_fresh(fetched_at: str, ttl_hours: float) -> bool:
    if not fetched_at:
        return False
    if ttl_hours <= 0:
        return False
    try:
        stamp = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - stamp).total_seconds() / 3600.0
    # 未来时间戳（时钟偏差）也视为新鲜
    if age_hours < 0:
        return True
    return age_hours <= ttl_hours


def _dedupe(entries: Iterable[ModelEntry]) -> list[ModelEntry]:
    """按 id 去重合并（保留信息更全的一条）。"""
    out: list[ModelEntry] = []
    index: dict[str, ModelEntry] = {}
    for entry in entries:
        key = (entry.id or entry.download_url or "").strip().lower()
        if not key:
            continue
        prev = index.get(key)
        if prev is None:
            index[key] = entry
            out.append(entry)
            continue
        if prev.size_gb <= 0 < entry.size_gb:
            prev.size_gb = entry.size_gb
            prev.download_url = entry.download_url or prev.download_url
            prev.sha256 = entry.sha256 or prev.sha256
        if prev.ctx_len <= 0 < entry.ctx_len:
            prev.ctx_len = entry.ctx_len
        if prev.params_b <= 0 < entry.params_b:
            prev.params_b = entry.params_b
        if not prev.quant and entry.quant:
            prev.quant = entry.quant
        if not prev.license and entry.license:
            prev.license = entry.license
        for tag in entry.tags:
            if tag not in prev.tags:
                prev.tags.append(tag)
    return out


def build_catalog(
    queries: Sequence[str] | None = None,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
    limit: int = DEFAULT_LIMIT,
    sources: Sequence[str] = ("hf",),
    cache_ttl_hours: float = 24.0,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[ModelEntry]:
    """抓取多个来源的模型目录并按 id 去重合并。

    - ``queries`` 默认为 :data:`DEFAULT_QUERIES`。
    - ``use_cache=True`` 且缓存未过期时直接返回缓存。
    """
    query_list = tuple(queries) if queries else DEFAULT_QUERIES
    target = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    if use_cache:
        meta = load_cached_catalog_meta(target)
        if _cache_is_fresh(str(meta.get("fetched_at") or ""), cache_ttl_hours):
            cached = load_cached_catalog(target)
            if cached:
                logger.info("命中模型目录缓存：%s（%d 条）", target, len(cached))
                return cached

    fetchers = []
    for source in sources:
        key = str(source).lower()
        if key in ("hf", "huggingface"):
            fetchers.append(fetch_hf_models)
        elif key in ("ms", "modelscope"):
            fetchers.append(fetch_modelscope_models)

    # 各查询词并发抓取（串行 8 个查询极易被用户当成"卡死"）
    def _fetch_one(query: str) -> list[ModelEntry]:
        out: list[ModelEntry] = []
        for fetcher in fetchers:
            try:
                out.extend(fetcher(query, limit) or [])
            except TypeError:
                try:
                    out.extend(fetcher(query) or [])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("抓取失败（%s / %s）: %s",
                                   getattr(fetcher, "__name__", fetcher), query, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("抓取失败（%s / %s）: %s",
                               getattr(fetcher, "__name__", fetcher), query, exc)
        return out

    collected: list[ModelEntry] = []
    if fetchers:
        workers = min(6, max(1, len(query_list)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for got in pool.map(_fetch_one, query_list):
                collected.extend(got)

    merged = _dedupe(collected)
    try:
        save_cached_catalog(target, merged)
    except OSError as exc:
        logger.warning("写入缓存失败 %s: %s", target, exc)
    logger.info("模型目录构建完成：%d 条（原始 %d 条）", len(merged), len(collected))
    return merged
