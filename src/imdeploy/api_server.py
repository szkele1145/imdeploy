"""OpenAI 兼容 API 层。

用标准库 ``http.server.ThreadingHTTPServer`` 实现零依赖 API：

- ``GET  /v1/models``            列出模型
- ``POST /v1/chat/completions``  对话补全
- ``POST /v1/completions``       文本补全
- ``POST /api/embeddings``       嵌入
- ``POST /api/{alias}``          按模型系列别名转发（qwen / llama / ...）
- ``GET  /api/status``           服务状态
- ``GET  /healthz``              健康检查

支持 CORS、Bearer Token 校验、流式（SSE）转发与请求统计。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import proxy
from .backends import Backend, detect_backends
from .config import Config
from .proxy import DEFAULT_ALIAS, forward_json, pick_model_for_alias

__all__ = ["ApiServer", "create_handler", "build_server", "main"]

# ---------------------------------------------------------------- 内置网页面板

_WEBUI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ImDeploy 本地模型部署器</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2129;--border:#30363d;
--text:#e6edf3;--dim:#8b949e;--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;--err:#f85149}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.6 "Microsoft YaHei UI",system-ui,sans-serif;padding:24px}
.wrap{max-width:1080px;margin:0 auto}
header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--border)}
h1{font-size:19px;font-weight:600}
.badge{font-size:12px;padding:3px 10px;border-radius:20px;background:rgba(63,185,80,.15);color:var(--ok);border:1px solid rgba(63,185,80,.3)}
.badge.off{background:rgba(139,148,158,.15);color:var(--dim);border-color:rgba(139,148,158,.3)}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:14px;animation:fadeUp .35s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
h2{font-size:14px;color:var(--accent);margin-bottom:12px;font-weight:600}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.kv{display:grid;grid-template-columns:110px 1fr;gap:6px 12px;font-size:13px}
.kv dt{color:var(--dim)}
.kv dd{color:var(--text);word-break:break-all}
button{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:7px;
padding:8px 16px;cursor:pointer;font-size:13px;font-family:inherit;transition:.18s}
button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
button.primary{background:var(--accent);color:#06101f;border-color:var(--accent);font-weight:600}
button.primary:hover{color:#06101f;filter:brightness(1.12)}
input{background:var(--panel2);border:1px solid var(--border);border-radius:7px;padding:8px 12px;
color:var(--text);font-family:inherit;font-size:13px;flex:1;min-width:180px}
input:focus{outline:none;border-color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 10px;color:var(--dim);font-weight:500;border-bottom:1px solid var(--border);font-size:12px}
td{padding:8px 10px;border-bottom:1px solid rgba(48,54,61,.5)}
tr:hover td{background:var(--panel2)}
.mono{font-family:Consolas,monospace;font-size:12px}
#log{background:#0a0d12;border:1px solid var(--border);border-radius:8px;padding:12px;
height:180px;overflow-y:auto;font-family:Consolas,monospace;font-size:12px;color:var(--dim);white-space:pre-wrap}
.tip{color:var(--dim);font-size:12px;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>ImDeploy 本地模型部署器</h1>
  <span class="badge" id="status">检测中</span>
</header>

<div class="card">
  <h2>服务状态</h2>
  <dl class="kv" id="statusKv"><dt>加载中…</dt><dd></dd></dl>
</div>

<div class="card">
  <h2>推荐模型（按本机硬件匹配）</h2>
  <div class="row">
    <button class="primary" onclick="loadRecs(true)">重新推荐</button>
    <span class="tip" id="recTip">点「重新推荐」从模型站抓取并匹配</span>
  </div>
  <table>
    <thead><tr><th>模型</th><th>档位</th><th>参数量</th><th>上下文</th><th>体积</th><th>操作</th></tr></thead>
    <tbody id="recBody"><tr><td colspan="6" style="color:var(--dim)">尚未加载推荐列表</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>下载进度</h2>
  <div id="dlBox"><span class="tip">暂无下载任务</span></div>
</div>

<div class="card">
  <h2>模型库</h2>
  <div class="row">
    <button onclick="loadModels()">刷新列表</button>
    <span class="tip" id="modelTip"></span>
  </div>
  <table>
    <thead><tr><th>模型</th><th>系列</th><th>参数量</th><th>上下文</th><th>量化</th></tr></thead>
    <tbody id="modelBody"><tr><td colspan="5" style="color:var(--dim)">暂无已下载模型</td></tr></tbody>
  </table>
</div>

<div class="card">
  <h2>请求测试</h2>
  <div class="row">
    <input id="q" placeholder="输入一句话测试…" value="你好，介绍一下你自己">
    <button class="primary" onclick="ask()">发送</button>
  </div>
  <div id="log">就绪。先在设置里加载模型，然后在这里测试对话。</div>
</div>

<div class="card">
  <h2>接口地址</h2>
  <dl class="kv">
    <dt>OpenAI 兼容</dt><dd class="mono" id="u1"></dd>
    <dt>模型列表</dt><dd class="mono" id="u2"></dd>
    <dt>多模型别名</dt><dd class="mono" id="u3"></dd>
  </dl>
</div>
</div>

<script>
const base = location.origin;
const $ = id => document.getElementById(id);

function log(msg){
  const el = $('log');
  el.textContent = msg;
  el.scrollTop = el.scrollHeight;
}

async function loadStatus(){
  try{
    const r = await fetch(base + '/api/status');
    const d = await r.json();
    $('status').textContent = '服务运行中';
    $('status').className = 'badge';
    const backends = (d.backends||[]).map(b=>b.name||b).join('、') || '未检测到';
    const loaded = Object.keys(d.loaded_models||{});
    $('statusKv').innerHTML =
      '<dt>监听地址</dt><dd class="mono">' + d.host + ':' + d.port + '</dd>' +
      '<dt>推理后端</dt><dd>' + backends + '</dd>' +
      '<dt>已加载模型</dt><dd>' + (loaded.length ? loaded.join('、') : '无') + '</dd>';
    $('u1').textContent = base + '/v1/chat/completions';
    $('u2').textContent = base + '/v1/models';
    $('u3').textContent = base + '/api/qwen  ' + base + '/api/llama';
  }catch(e){
    $('status').textContent = '服务异常';
    $('status').className = 'badge off';
    $('statusKv').innerHTML = '<dt>错误</dt><dd>' + e + '</dd>';
  }
}

async function loadModels(){
  try{
    const r = await fetch(base + '/v1/models');
    const d = await r.json();
    const items = d.data || [];
    if(!items.length){
      $('modelBody').innerHTML = '<tr><td colspan="5" style="color:var(--dim)">暂无已下载模型，请把 GGUF 放进 models 目录</td></tr>';
      $('modelTip').textContent = '';
      return;
    }
    $('modelBody').innerHTML = items.map(m =>
      '<tr><td>' + m.id + '</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>').join('');
    $('modelTip').textContent = '共 ' + items.length + ' 个';
  }catch(e){
    $('modelBody').innerHTML = '<tr><td colspan="5" style="color:var(--err)">加载失败：' + e + '</td></tr>';
  }
}

async function ask(){
  const q = $('q').value.trim();
  if(!q) return;
  log('请求中…');
  try{
    const r = await fetch(base + '/v1/chat/completions', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model:'qwen', messages:[{role:'user',content:q}]})
    });
    const d = await r.json();
    if(d.error){ log('错误：' + (d.error.message || JSON.stringify(d.error))); return; }
    const txt = d.choices?.[0]?.message?.content || JSON.stringify(d);
    log(txt);
  }catch(e){ log('请求失败：' + e); }
}

async function loadRecs(refresh){
  $('recTip').textContent = refresh ? '正在抓取模型目录，可能需要十几秒…' : '加载中…';
  try{
    const r = await fetch(base + '/api/recommend?limit=20&refresh=' + (refresh ? 1 : 0));
    const d = await r.json();
    if(!d.ok){ $('recTip').textContent = '失败：' + (d.error?.message || '未知错误'); return; }
    const ms = d.models || [];
    if(!ms.length){
      $('recBody').innerHTML = '<tr><td colspan="6" style="color:var(--dim)">没有匹配到模型</td></tr>';
      $('recTip').textContent = '没有匹配到模型';
      return;
    }
    $('recBody').innerHTML = ms.map(m => {
      const tier = m.tier || '-';
      const color = tier === '首选' ? 'var(--ok)' : tier === '备选' ? 'var(--accent)'
                  : tier === '不适合' ? 'var(--err)' : 'var(--warn)';
      const size = m.size_gb ? m.size_gb.toFixed(2) + ' GB' : '-';
      const ctx = m.ctx_len ? Math.round(m.ctx_len/1024) + 'K' : '-';
      const btn = '<button onclick="startDl(\'' + m.id.replace(/'/g,"\\'") + '\')">下载</button>';
      return '<tr><td>' + m.name + '</td>'
        + '<td style="color:' + color + '">' + tier + '</td>'
        + '<td>' + (m.params_b ? m.params_b + 'B' : '-') + '</td>'
        + '<td>' + ctx + '</td><td>' + size + '</td><td>' + btn + '</td></tr>';
    }).join('');
    $('recTip').textContent = '共 ' + ms.length + ' 个候选';
  }catch(e){ $('recTip').textContent = '请求失败：' + e; }
}

async function startDl(mid){
  try{
    const r = await fetch(base + '/api/download', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model_id: mid})
    });
    const d = await r.json();
    if(!d.ok){ $('recTip').textContent = '下载失败：' + (d.error || ''); return; }
    $('recTip').textContent = '已开始下载：' + mid;
    pollDownloads();
  }catch(e){ $('recTip').textContent = '请求失败：' + e; }
}

async function pollDownloads(){
  try{
    const r = await fetch(base + '/api/downloads');
    const d = await r.json();
    const tasks = Object.values(d.tasks || {});
    if(!tasks.length){ $('dlBox').innerHTML = '<span class="tip">暂无下载任务</span>'; return; }
    $('dlBox').innerHTML = tasks.map(t => {
      const pct = t.percent || 0;
      const color = t.status === 'done' ? 'var(--ok)'
                  : t.status === 'error' ? 'var(--err)' : 'var(--accent)';
      const bar = '<div style="background:var(--panel2);border-radius:4px;height:6px;margin-top:6px;overflow:hidden">'
        + '<div style="width:' + pct + '%;height:100%;background:' + color + ';transition:width .3s"></div></div>';
      const label = t.status === 'done' ? '已完成'
                  : t.status === 'error' ? ('失败：' + (t.error||''))
                  : (pct.toFixed(1) + '%');
      var extra = '';
      if(t.status === 'done') extra = ' <button onclick="loadModels()">刷新模型库</button>';
      return '<div style="margin-bottom:10px"><b>' + t.model_id + '</b> '
        + '<span style="color:' + color + '">' + label + '</span>' + extra + bar + '</div>';
    }).join('');
  }catch(e){}
}

loadStatus(); loadModels(); pollDownloads();
setInterval(loadStatus, 5000);
setInterval(pollDownloads, 2000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 统计


class Stats:
    """线程安全的请求/Token 统计。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_failed = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.started_at = time.time()

    def count_request(self, failed: bool = False) -> None:
        with self._lock:
            self.requests_total += 1
            if failed:
                self.requests_failed += 1

    def count_tokens(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        with self._lock:
            self.tokens_in += int(tokens_in)
            self.tokens_out += int(tokens_out)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }


# ---------------------------------------------------------------- API 服务


class ApiServer:
    """聚合后端、模型目录、别名映射与统计信息。"""

    def __init__(self, cfg: Config | None = None, backends: list[Backend] | None = None,
                 host: str = "127.0.0.1", port: int = 8080, token: str = ""):
        self.cfg = cfg or Config.default()
        self.backends = backends if backends is not None else detect_backends(self.cfg)
        self.host = host
        self.port = int(port)
        self.token = str(token or self.cfg.get("app.token", "") or "")
        self.stats = Stats()

        # 已加载模型登记: id -> {"backend": Backend, "base_url": str, "family": str}
        self._loaded: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

        # family -> alias 映射（catalog family 字段推断）
        self._family_map: dict[str, str] = {}

        # 显式别名: alias -> model_id
        self._explicit_aliases: dict[str, str] = {
            str(k): str(v)
            for k, v in (self.cfg.get("api.aliases", {}) or {}).items()
        }

        # 模型目录 / 推荐结果 / 下载任务 缓存
        self._catalog_cache: list[Any] | None = None
        self._rec_cache: list[Any] | None = None
        self._hw_cache: Any = None
        self._downloads: dict[str, dict[str, Any]] = {}

    # ---------------- 后端 ----------------

    @property
    def active_backend(self) -> Backend | None:
        """当前激活后端（第一个可用的）。"""
        for b in self.backends:
            if b.is_available():
                return b
        return self.backends[0] if self.backends else None

    def register_loaded(self, model_id: str, backend: Backend, base_url: str,
                        family: str = "") -> None:
        """登记一个已加载的模型。"""
        with self._lock:
            self._loaded[str(model_id)] = {
                "backend": backend,
                "base_url": base_url,
                "family": family,
                "loaded_at": time.time(),
            }

    def unregister_loaded(self, model_id: str) -> None:
        with self._lock:
            self._loaded.pop(str(model_id), None)

    def set_family_map(self, family_map: dict[str, str]) -> None:
        """设置 ``family → alias`` 映射（自动从 catalog 推断后传入）。"""
        with self._lock:
            self._family_map = dict(family_map or {})

    def refresh_family_map(self, catalog_models: list[dict[str, Any]] | None = None) -> dict[str, str]:
        """从 catalog 模型列表推断 family → alias 映射。"""
        source = catalog_models
        if source is None:
            source = []
            for b in self.backends:
                try:
                    source += b.list_models()
                except Exception:  # noqa: BLE001 - 列模型失败不应影响启动
                    pass
        fmap = proxy.infer_family_map(source)
        self.set_family_map(fmap)
        return fmap

    # ---------------- 别名 ----------------

    def aliases(self) -> dict[str, Any]:
        """返回「别名 → 模型 id」视图。"""
        with self._lock:
            out: dict[str, Any] = {}
            for model_id, info in self._loaded.items():
                fam = str(info.get("family") or "").lower()
                alias = self._family_map.get(fam) or (fam if fam else DEFAULT_ALIAS)
                out.setdefault(alias, model_id)
            for alias, mid in self._explicit_aliases.items():
                out[alias] = mid
            if DEFAULT_ALIAS not in out and self._loaded:
                first = next(iter(self._loaded))
                out[DEFAULT_ALIAS] = first
            return out

    def resolve_alias(self, alias: str) -> str | None:
        """解析别名为模型 id（失败返回 None）。"""
        alias_key = str(alias or "").strip().lower()
        with self._lock:
            # 1) 显式配置
            if alias_key in self._explicit_aliases:
                mid = self._explicit_aliases[alias_key]
                if not self._loaded or mid in self._loaded:
                    return mid
            # 2) 已登记别名
            amap = self.aliases()
            if alias_key in amap:
                return amap[alias_key]
            # 3) 从已加载列表挑
            if self._loaded:
                return pick_model_for_alias(
                    alias_key,
                    [{"id": mid, "family": info.get("family", "")}
                     for mid, info in self._loaded.items()],
                    self._family_map,
                )
        return None

    # ---------------- 状态 ----------------

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            loaded = {
                mid: {
                    "backend": info["backend"].name,
                    "family": info.get("family", ""),
                    "base_url": info.get("base_url", ""),
                    "loaded_at": info.get("loaded_at", 0),
                }
                for mid, info in self._loaded.items()
            }
            family_map = dict(self._family_map)
            explicit = dict(self._explicit_aliases)
        return {
            "service": "imdeploy",
            "host": self.host,
            "port": self.port,
            "loaded_models": loaded,
            "backends": [b.status() if hasattr(b, "status") else b.info() for b in self.backends],
            "active_backend": self.active_backend.name if self.active_backend else None,
            "family_map": family_map,
            "explicit_aliases": explicit,
            "aliases": self.aliases(),
            "stats": self.stats.snapshot(),
            "token_required": bool(self.token),
        }

    # ---------------- 转发 ----------------

    def backend_url_for(self, model_id: str) -> str:
        """取指定模型对应后端的 base_url。"""
        with self._lock:
            info = self._loaded.get(str(model_id))
            if info:
                return str(info.get("base_url") or "")
        backend = self.active_backend
        if backend is None:
            return ""
        try:
            return backend.base_url() or ""
        except Exception:  # noqa: BLE001
            return ""

    def forward_for_model(self, model_id: str, path: str, payload: dict[str, Any],
                          timeout: float = 120.0) -> tuple[int, dict[str, Any]]:
        """把请求转发到模型对应的后端。"""
        base = self.backend_url_for(model_id)
        if not base:
            return 0, {"error": {
                "message": f"模型未加载或后端不可用：{model_id}",
                "type": "api_error",
            }}
        return forward_json(base, path, payload, timeout=timeout)

    # ---------------- 模型目录 / 推荐 / 下载 ----------------

    def get_recommendations(self, limit: int = 20, min_ctx: int = 4096,
                            refresh: bool = False) -> list[dict[str, Any]]:
        """返回按本机硬件排序的推荐模型。结果缓存，避免每次重复抓取。"""
        from .catalog import build_catalog
        from .hardware import detect
        from .recommend import recommend

        with self._lock:
            cat = self._catalog_cache
            recs = self._rec_cache

        if refresh or cat is None:
            try:
                cat = build_catalog(use_cache=not refresh)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"抓取模型目录失败：{exc}") from exc
            with self._lock:
                self._catalog_cache = cat

        hw = self._hw_cache
        if hw is None:
            hw = detect()
            with self._lock:
                self._hw_cache = hw

        recs = recommend(cat, hw, limit=limit, min_ctx=min_ctx)
        with self._lock:
            self._rec_cache = recs

        out: list[dict[str, Any]] = []
        for r in recs:
            e = r.entry
            out.append({
                "id": e.id,
                "name": e.name,
                "family": e.family,
                "params_b": e.params_b,
                "quant": e.quant,
                "ctx_len": e.ctx_len,
                "size_gb": r.estimated_size_gb or e.size_gb,
                "tier": r.tier,
                "score": r.score,
                "fit": r.fit,
                "reasons": r.reasons,
                "download_url": e.download_url,
                "repo_url": e.repo_url,
            })
        return out

    def start_download(self, model_id: str) -> dict[str, Any]:
        """后台下载一个模型（按 model_id 从缓存目录里找下载地址）。"""
        from .downloader import Downloader

        with self._lock:
            if model_id in self._downloads and self._downloads[model_id].get("status") == "downloading":
                return {"ok": False, "error": "该模型正在下载中", "model_id": model_id}

        url = ""
        size_hint = 0.0
        with self._lock:
            for r in (self._rec_cache or []):
                if r.entry.id == model_id:
                    url = r.entry.download_url
                    size_hint = r.estimated_size_gb
                    break
        if not url:
            return {"ok": False, "error": f"找不到模型的下载地址：{model_id}"}

        dest = self.cfg.models_dir / f"{model_id.replace('/', '_')}.gguf"
        state: dict[str, Any] = {
            "model_id": model_id, "status": "downloading", "percent": 0.0,
            "downloaded": 0, "total": 0, "error": "",
            "dest": str(dest), "size_hint_gb": size_hint,
        }
        with self._lock:
            self._downloads[model_id] = state

        def worker() -> None:
            dl = Downloader()

            def progress(task) -> None:
                with self._lock:
                    state["percent"] = round(task.percent, 1)
                    state["downloaded"] = task.downloaded
                    state["total"] = task.size

            try:
                task = dl.download(url, dest, progress=progress)
                with self._lock:
                    state["status"] = "done" if task.status == "done" else "error"
                    state["error"] = task.error or ""
                    state["percent"] = 100.0 if task.status == "done" else state["percent"]
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    state["status"] = "error"
                    state["error"] = str(exc)

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "model_id": model_id, "dest": str(dest)}

    def download_status(self) -> dict[str, Any]:
        with self._lock:
            return {k: dict(v) for k, v in self._downloads.items()}


# ---------------------------------------------------------------- Handler


def create_handler(server: ApiServer) -> type[BaseHTTPRequestHandler]:
    """闭包绑定一个 ApiServer，生成请求处理器类。"""

    class Handler(BaseHTTPRequestHandler):
        # 关闭默认 stderr 日志刷屏
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            pass

        # ---------------- 通用 ----------------

        def _send_json(self, data: Any, status: int = 200, extra_headers: dict | None = None) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

        def _send_error_json(self, message: str, err_type: str = "invalid_request_error",
                             status: int = 400, code: Any = None) -> None:
            err: dict[str, Any] = {"message": message, "type": err_type}
            if code is not None:
                err["code"] = code
            self._send_json({"error": err}, status=status)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8", "replace"))

        # ---------------- 鉴权 ----------------

        def _check_token(self) -> bool:
            """token 非空时要求 Authorization: Bearer 或 ?token=。"""
            tok = server.token
            if not tok:
                return True
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer ") and auth[len("Bearer "):].strip() == tok:
                return True
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            got = (qs.get("token") or [""])[0]
            return got == tok

        # ---------------- OPTIONS ----------------

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors_headers()
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()

        # ---------------- GET ----------------

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if not self._check_token():
                server.stats.count_request(failed=True)
                self._send_error_json("访问令牌校验失败", "invalid_request_error", 401, "invalid_token")
                return
            if path in ("/healthz", "/api/status/health"):
                self._send_json({"ok": True, "service": "imdeploy"})
                server.stats.count_request()
                return
            if path == "/api/status":
                self._send_json(server.status_payload())
                server.stats.count_request()
                return
            if path in ("/v1/models", "/api/models"):
                self._handle_models()
                return
            if path in ("/api/recommend", "/api/catalog"):
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                try:
                    limit = int((q.get("limit") or ["20"])[0])
                    min_ctx = int((q.get("min_ctx") or ["4096"])[0])
                    refresh = (q.get("refresh") or ["0"])[0] in ("1", "true", "yes")
                    data = server.get_recommendations(limit=limit, min_ctx=min_ctx,
                                                      refresh=refresh)
                except Exception as exc:  # noqa: BLE001
                    self._send_error_json(f"获取推荐失败：{exc}", "api_error", 500, "recommend_failed")
                    return
                self._send_json({"ok": True, "count": len(data), "models": data})
                server.stats.count_request()
                return
            if path == "/api/downloads":
                self._send_json({"ok": True, "tasks": server.download_status()})
                server.stats.count_request()
                return
            if path in ("/", "/index.html", "/ui", "/docs"):
                self._send_html(_WEBUI_HTML)
                server.stats.count_request()
                return
            self._send_error_json(f"未找到路径：{path}", "invalid_request_error", 404, "not_found")
            server.stats.count_request(failed=True)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_models(self) -> None:
            data: list[dict[str, Any]] = []
            with server._lock:
                model_ids = list(server._loaded.keys())
            for b in server.backends:
                try:
                    models = b.list_models()
                except Exception:  # noqa: BLE001
                    models = []
                for m in models:
                    if m.get("id") not in model_ids:
                        data.append({
                            "id": m.get("id", ""),
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        })
            for mid in model_ids:
                data.append({
                    "id": mid,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                })
            # 去重
            seen: set[str] = set()
            unique: list[dict[str, Any]] = []
            for item in data:
                if item["id"] and item["id"] not in seen:
                    seen.add(item["id"])
                    unique.append(item)
            self._send_json({"object": "list", "data": unique})
            server.stats.count_request()

        # ---------------- POST ----------------

        def do_POST(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            if not self._check_token():
                server.stats.count_request(failed=True)
                self._send_error_json("访问令牌校验失败", "invalid_request_error", 401, "invalid_token")
                return
            try:
                payload = self._read_body()
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是 JSON 对象")
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"请求体 JSON 解析失败：{exc}", "invalid_request_error", 400, "invalid_json")
                return

            if path == "/v1/chat/completions":
                self._handle_chat(payload)
            elif path == "/v1/completions":
                self._handle_completion(payload)
            elif path == "/api/embeddings" or path == "/v1/embeddings":
                self._handle_embeddings(payload)
            elif path == "/api/download":
                mid = str(payload.get("model_id") or payload.get("id") or "").strip()
                if not mid:
                    server.stats.count_request(failed=True)
                    self._send_error_json("缺少 model_id", "invalid_request_error", 400, "missing_model_id")
                    return
                res = server.start_download(mid)
                self._send_json(res, 200 if res.get("ok") else 400)
                server.stats.count_request(failed=not res.get("ok"))
            elif path.startswith("/api/") and len(path) > 5:
                self._handle_alias(path[len("/api/"):], payload)
            else:
                server.stats.count_request(failed=True)
                self._send_error_json(f"未找到路径：{path}", "invalid_request_error", 404, "not_found")

        # ---------------- 各端点实现 ----------------

        def _pick_model(self, payload: dict[str, Any], alias: str = "") -> str | None:
            """从请求体或别名解析目标模型。"""
            explicit = str(payload.get("model") or "").strip()
            if explicit:
                return explicit
            if alias:
                resolved = server.resolve_alias(alias)
                if resolved:
                    return resolved
            return None

        def _forward_or_error(self, model_id: str, upstream_path: str,
                             payload: dict[str, Any]
                             ) -> tuple[int, dict[str, Any]] | None:
            """转发并处理通用错误；返回 None 表示已写回响应。"""
            status, data = server.forward_for_model(model_id, upstream_path, payload)
            if status == 0:
                err = (data or {}).get("error") or {}
                server.stats.count_request(failed=True)
                self._send_error_json(
                    str(err.get("message") or "推理后端不可用"),
                    "api_error", 502, "backend_unavailable",
                )
                return None
            if status >= 400:
                err = (data or {}).get("error") or {}
                server.stats.count_request(failed=True)
                self._send_error_json(
                    str(err.get("message") or f"上游返回错误状态 {status}"),
                    str(err.get("type") or "api_error"), status, err.get("code"),
                )
                return None
            return status, data

        def _handle_chat(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            # 流式
            if payload.get("stream"):
                self._handle_chat_stream(payload, model_id)
                return
            status_data = self._forward_or_error(model_id, "/v1/chat/completions", payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_chat(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

        def _handle_chat_stream(self, payload: dict[str, Any], model_id: str) -> None:
            base = server.backend_url_for(model_id)
            if not base:
                server.stats.count_request(failed=True)
                self._send_error_json(f"模型未加载：{model_id}", "api_error", 502, "backend_unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors_headers()
            self.end_headers()
            try:
                for line in proxy.forward_stream(base, "/v1/chat/completions", payload):
                    if line:
                        self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.write(b"\n")
                    self.wfile.flush()
            except (RuntimeError, OSError) as exc:
                try:
                    self.wfile.write(f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except OSError:
                    pass
                server.stats.count_request(failed=True)
                return
            server.stats.count_request()

        def _handle_completion(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            status_data = self._forward_or_error(model_id, "/v1/completions", payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_completion(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

        def _handle_embeddings(self, payload: dict[str, Any]) -> None:
            model_id = self._pick_model(payload)
            if not model_id:
                server.stats.count_request(failed=True)
                self._send_error_json("缺少 model 参数，且没有可用的已加载模型", "invalid_request_error", 400, "model_not_found")
                return
            if not server.cfg.get("api.expose_embedding", True):
                server.stats.count_request(failed=True)
                self._send_error_json("嵌入接口未开放（api.expose_embedding=False）", "invalid_request_error", 404, "embedding_disabled")
                return
            status_data = self._forward_or_error(model_id, "/v1/embeddings", payload)
            if status_data is None:
                return
            _, data = status_data
            out = _to_openai_embedding(data, model_id, payload)
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), 0)
            self._send_json(out)
            server.stats.count_request()

        def _handle_alias(self, alias: str, payload: dict[str, Any]) -> None:
            """POST /api/{alias} —— 按模型系列别名转发。"""
            model_id = self._pick_model(payload, alias=alias)
            if not model_id:
                server.stats.count_request(failed=True)
                amap = server.aliases()
                self._send_error_json(
                    f"别名「{alias}」没有对应的已加载模型（已注册别名：{', '.join(sorted(amap)) or '无'}）",
                    "invalid_request_error", 404, "alias_not_found",
                )
                return
            upstream_path = "/v1/chat/completions"
            status_data = self._forward_or_error(model_id, upstream_path, payload)
            if status_data is None:
                return
            _, data = status_data
            try:
                out = _to_openai_chat(data, model_id)
            except (KeyError, TypeError, ValueError) as exc:
                server.stats.count_request(failed=True)
                self._send_error_json(f"解析上游响应失败：{exc}", "api_error", 502, "bad_upstream")
                return
            usage = out.get("usage") or {}
            server.stats.count_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
            self._send_json(out)
            server.stats.count_request()

    return Handler


# ---------------------------------------------------------------- 响应归一化


def _to_openai_chat(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """把后端响应转成 OpenAI chat.completion 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    # 已经是 OpenAI 格式：直接补齐字段
    choices = data.get("choices")
    message = None
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
    if message is not None:
        return _fill_chat_fields(data, model_id)

    # Ollama /api/generate 风格: {"response": "..."}
    if "response" in data and "choices" not in data:
        return {
            "id": f"chatcmpl-{_now_ms()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": str(data.get("response") or "")},
                "finish_reason": str(data.get("done_reason") or "stop"),
            }],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }

    # Ollama /api/chat 风格: {"message": {"role","content"}}
    msg = data.get("message")
    if isinstance(msg, dict):
        return {
            "id": f"chatcmpl-{_now_ms()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{
                "index": 0,
                "message": {"role": msg.get("role", "assistant"),
                            "content": str(msg.get("content") or "")},
                "finish_reason": "stop" if data.get("done", True) else "length",
            }],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }

    raise KeyError("上游响应中没有 choices / response / message 字段")


def _fill_chat_fields(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """补齐 OpenAI 格式缺的字段。"""
    out = dict(data)
    out.setdefault("id", f"chatcmpl-{_now_ms()}")
    out["object"] = "chat.completion"
    out.setdefault("created", int(time.time()))
    out.setdefault("model", model_id)
    choices = out.get("choices") or [{}]
    for ch in choices:
        if isinstance(ch, dict):
            ch.setdefault("index", 0)
            ch.setdefault("finish_reason", "stop")
            msg = ch.get("message")
            if not isinstance(msg, dict):
                ch["message"] = {"role": "assistant", "content": ch.get("text") or ""}
            else:
                msg.setdefault("role", "assistant")
                msg.setdefault("content", "")
    out["choices"] = choices
    out.setdefault("usage", _usage(data))
    return out


def _to_openai_completion(data: dict[str, Any], model_id: str) -> dict[str, Any]:
    """把后端响应转成 OpenAI text_completion 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    if data.get("choices") and isinstance(data["choices"][0], dict):
        out = dict(data)
        out.setdefault("id", f"cmpl-{_now_ms()}")
        out["object"] = "text_completion"
        out.setdefault("created", int(time.time()))
        out.setdefault("model", model_id)
        for ch in out["choices"]:
            if isinstance(ch, dict):
                ch.setdefault("index", 0)
                ch.setdefault("finish_reason", "stop")
                ch.setdefault("text", ch.get("text") or "")
        out.setdefault("usage", _usage(data))
        return out
    # Ollama generate 风格
    if "response" in data:
        return {
            "id": f"cmpl-{_now_ms()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": str(data.get("model") or model_id),
            "choices": [{"index": 0, "text": str(data.get("response") or ""),
                         "finish_reason": "stop"}],
            "usage": _usage(data, prompt_eval=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count")),
        }
    raise KeyError("上游响应中没有 choices / response 字段")


def _to_openai_embedding(data: dict[str, Any], model_id: str,
                         payload: dict[str, Any]) -> dict[str, Any]:
    """把后端响应转成 OpenAI embedding 格式。"""
    if not isinstance(data, dict):
        raise TypeError("上游响应不是 JSON 对象")
    if data.get("data"):
        out = dict(data)
        out.setdefault("object", "list")
        out.setdefault("model", model_id)
        out.setdefault("usage", _usage(data))
        return out
    # Ollama /api/embeddings 风格: {"embedding": [...]}
    if "embedding" in data:
        emb = data.get("embedding") or []
        return {
            "object": "list",
            "model": str(data.get("model") or model_id),
            "data": [{"object": "embedding", "index": 0, "embedding": emb}],
            "usage": _usage(data),
        }
    raise KeyError("上游响应中没有 embedding / data 字段")


def _usage(data: dict[str, Any], prompt_eval: Any = None, eval_count: Any = None) -> dict[str, int]:
    """从各种字段里凑出 usage。"""
    u = data.get("usage") or {}
    if isinstance(u, dict):
        pt = u.get("prompt_tokens")
        ct = u.get("completion_tokens")
        if pt is not None or ct is not None:
            return {
                "prompt_tokens": int(pt or 0),
                "completion_tokens": int(ct or 0),
                "total_tokens": int(u.get("total_tokens") or (int(pt or 0) + int(ct or 0))),
            }
    pe = prompt_eval if prompt_eval is not None else data.get("prompt_eval_count")
    ec = eval_count if eval_count is not None else data.get("eval_count")
    if pe is not None or ec is not None:
        p, c = int(pe or 0), int(ec or 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------- 组装


def build_server(cfg: Config | None = None, host: str | None = None,
                 port: int | None = None, token: str | None = None,
                 backends: list[Backend] | None = None) -> tuple[ThreadingHTTPServer, ApiServer]:
    """创建 HTTP 服务器与对应的 ApiServer。"""
    cfg = cfg or Config.load()
    api = ApiServer(
        cfg,
        backends=backends,
        host=host or str(cfg.get("app.host", "127.0.0.1")),
        port=int(port if port is not None else cfg.get("app.port", 8080)),
        token=token if token is not None else "",
    )
    handler = create_handler(api)
    httpd = ThreadingHTTPServer((api.host, api.port), handler)
    httpd.daemon_threads = True
    api.port = httpd.server_address[1]
    return httpd, api


def main(argv: list[str] | None = None) -> int:
    """命令行入口：``python -m imdeploy.api_server --host --port --token``。"""
    parser = argparse.ArgumentParser(
        prog="feiyu-api-server",
        description="imdeploy：本地大模型部署器 OpenAI 兼容 API 服务",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    parser.add_argument("--token", default="", help="访问口令（留空=不校验）")
    parser.add_argument("--config", default="data/config.yaml", help="配置文件路径")
    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except Exception:  # noqa: BLE001 - 配置加载失败时用默认配置
        cfg = Config.default()

    httpd, api = build_server(cfg, host=args.host, port=args.port, token=args.token)
    print(f"imdeploy API 服务已启动：http://{api.host}:{api.port}")
    print(f"  后端：{', '.join(b.name for b in api.backends) or '未检测到'}")
    print(f"  口令：{'已启用' if api.token else '未启用（免鉴权）'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
