"""实机验证 HTTP 层：/api/load_model -> /v1/chat/completions 全链路。

需要先装好 llama.cpp 运行库并有一个 gguf 模型（默认用 e2e_real_model 下好的那个）。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PORT = 8921
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, payload=None, timeout=300):
    url = BASE + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {}


def main() -> int:
    from imdeploy.api_server import build_server
    from imdeploy.config import Config

    cfg_path = ROOT / "data" / "config.yaml"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config.default()

    print("[0/4] 准备服务")
    httpd, api = build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    print("[1/4] 扫描本地模型库")
    st, d = call("GET", "/api/library_scan")
    check("GET /api/library_scan", st == 200 and d.get("ok"), f"-> added={d.get('added')}")

    st, d = call("GET", "/api/library")
    models = d.get("models") or []
    check("库里有模型", bool(models), f"-> {[m['name'] for m in models][:2]}")
    if not models:
        print("  没有本地模型，先跑 tests/e2e_real_model.py")
        httpd.shutdown()
        return 1

    target = models[0]
    print(f"[2/4] 加载模型 {target['name']}")
    st, d = call("POST", "/api/load_model", {"model_id": target["id"]})
    check("POST /api/load_model", st == 200 and d.get("ok"),
          f"-> {d.get('error', '') or d.get('base_url', '')}")

    st, d = call("GET", "/api/loaded")
    check("已加载列表非空", bool(d.get("models")))

    print("[3/4] 等模型就绪")
    ok_ready = False
    for i in range(120):
        time.sleep(1)
        st, d = call("GET", "/v1/models", timeout=5)
        if st == 200:
            ok_ready = True
            break
    check("GET /v1/models 就绪", ok_ready)

    print("[4/4] 对话")
    st, d = call("POST", "/v1/chat/completions", {
        "model": target["id"],
        "messages": [{"role": "user", "content": "只回复两个字：你好"}],
        "max_tokens": 32,
    })
    text = ""
    if st == 200:
        try:
            text = d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
    check("POST /v1/chat/completions", st == 200 and bool(text.strip()),
          f"-> {text.strip()[:80]}" if text else f"HTTP {st} {json.dumps(d, ensure_ascii=False)[:300]}")

    call("POST", "/api/unload_model", {"model_id": target["id"]})
    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
