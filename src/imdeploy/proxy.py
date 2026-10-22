"""转发工具。

把 API 层的请求转发给当前激活的推理后端，统一用 ``urllib``，
并提供「按别名挑模型」的映射逻辑。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator

__all__ = [
    "forward_json",
    "forward_stream",
    "pick_model_for_alias",
    "DEFAULT_ALIAS",
    "KNOWN_FAMILIES",
    "infer_family_map",
]

#: 默认兜底别名
DEFAULT_ALIAS = "default"

#: 已知模型系列（用于自动推断 alias → family）
KNOWN_FAMILIES: tuple[str, ...] = (
    "qwen", "llama", "gemma", "phi", "smol", "minicpm", "internlm",
    "mistral", "deepseek", "yi",
)


def forward_json(backend_base_url: str, path: str, payload: dict[str, Any],
                 timeout: float = 120.0,
                 extra_headers: dict[str, str] | None = None
                 ) -> tuple[int, dict[str, Any]]:
    """把一个 JSON 请求 POST 给后端，返回 ``(状态码, 响应体)``。

    - 网络异常 → ``(0, {"error": {"message": ..., "type": "upstream_error"}})``
    - HTTP 错误 → 透传后端返回的状态码与错误体
    - 响应不是 JSON → 包成 ``{"raw": "..."}``
    """
    url = _join_url(backend_base_url, path)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            pass
        return status, _parse_or_error(raw, status, "上游返回错误状态")
    except urllib.error.URLError as exc:
        return 0, _err_payload(f"无法连接推理后端：{exc.reason}", "upstream_error")
    except (OSError, TimeoutError) as exc:
        return 0, _err_payload(f"请求推理后端失败：{exc}", "upstream_error")

    return status, _parse_body(raw)


def forward_stream(backend_base_url: str, path: str, payload: dict[str, Any],
                   timeout: float = 120.0,
                   extra_headers: dict[str, str] | None = None
                   ) -> Iterator[str]:
    """转发一个 SSE 流式请求，逐行产出后端返回的原始行。

    产出的是「不含换行符」的一行一行文本；调用方负责按 SSE 协议写回客户端。
    网络异常时抛出 :class:`RuntimeError`。
    """
    url = _join_url(backend_base_url, path)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"上游流式请求失败：HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接推理后端：{exc.reason}") from exc
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f"请求推理后端失败：{exc}") from exc

    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            yield line
    finally:
        try:
            resp.close()
        except (OSError, AttributeError):
            pass


def pick_model_for_alias(alias: str, loaded_models: Iterable[Any],
                         family_map: dict[str, str] | None = None) -> str:
    """按别名挑一个已加载的模型。

    参数：

    - ``alias``：请求路径里的别名，如 ``qwen`` / ``default``
    - ``loaded_models``：已加载模型列表，元素可以是字符串 id 或带 ``id``/``family``
      字段的字典
    - ``family_map``：``family → 别名`` 的映射（由 catalog 的 family 字段推断，
      或配置里显式指定）

    匹配顺序：

    1. 别名对应的 family 精确匹配
    2. 别名出现在模型 id / family 里（子串匹配）
    3. ``default`` 别名对应的 family
    4. 列表第一个模型

    找不到时返回空字符串。
    """
    models = [_normalize_model(m) for m in (loaded_models or [])]
    if not models:
        return ""

    alias_key = str(alias or DEFAULT_ALIAS).strip().lower()
    fmap = {str(k).lower(): str(v).lower() for k, v in (family_map or {}).items()}

    # 1) family_map 反查：alias -> family
    target_family = ""
    for fam, mapped_alias in fmap.items():
        if mapped_alias == alias_key:
            target_family = fam
            break

    if target_family:
        for m in models:
            if m["family"].lower() == target_family.lower():
                return m["id"]

    # 2) 直接把别名当 family 用
    for m in models:
        if m["family"].lower() == alias_key:
            return m["id"]

    # 3) 子串匹配（模型 id 或 family 里含别名）
    for m in models:
        if alias_key and (alias_key in m["id"].lower() or alias_key in m["family"].lower()):
            return m["id"]

    # 4) default 兜底
    default_family = ""
    for fam, mapped_alias in fmap.items():
        if mapped_alias == DEFAULT_ALIAS:
            default_family = fam
            break
    if default_family:
        for m in models:
            if m["family"].lower() == default_family.lower():
                return m["id"]

    # 5) 第一个
    return models[0]["id"]


def infer_family_map(models: Iterable[Any]) -> dict[str, str]:
    """从 catalog / 模型列表推断 ``family → alias`` 映射。

    已知系列（qwen / llama / gemma / phi / smol / minicpm / internlm ...）
    会被映射到同名别名；其余系列统一归到 ``default``。
    """
    fmap: dict[str, str] = {}
    for m in (models or []):
        fam = str(_normalize_model(m)["family"]).lower()
        if not fam or fam == "unknown":
            continue
        alias = fam if fam in KNOWN_FAMILIES else DEFAULT_ALIAS
        fmap.setdefault(fam, alias)
    return fmap


# ---------------------------------------------------------------- 内部工具


def _normalize_model(m: Any) -> dict[str, str]:
    """把模型条目统一成 ``{"id","family"}``。"""
    if isinstance(m, dict):
        mid = str(m.get("id") or m.get("model") or m.get("name") or "")
        fam = str(m.get("family") or "")
    else:
        mid = str(m or "")
        fam = ""
    if not fam:
        fam = _family_from_id(mid)
    return {"id": mid, "family": fam}


def _family_from_id(model_id: str) -> str:
    """从模型 id 里猜系列。"""
    low = model_id.lower()
    for fam in KNOWN_FAMILIES:
        if fam in low:
            return fam
    return "unknown"


def _join_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    if not path:
        return base
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base}/{path.lstrip('/')}"


def _parse_body(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(data, dict):
        return data
    return {"data": data}


def _parse_or_error(raw: str, status: int, fallback: str) -> dict[str, Any]:
    parsed = _parse_body(raw) if raw.strip() else {}
    if isinstance(parsed, dict) and parsed.get("error"):
        return parsed
    return _err_payload(raw or fallback, "upstream_error", status)


def _err_payload(message: str, err_type: str = "invalid_request_error",
                 code: int | str | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"message": message, "type": err_type}
    if code is not None:
        err["code"] = code
    return {"error": err}
