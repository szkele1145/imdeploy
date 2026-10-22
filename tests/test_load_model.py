"""验证 imdeploy 的模型加载/卸载接口（用假后端，不依赖真实 llama-server）。"""
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8911
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, payload=None, timeout=20):
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


class FakeBackend:
    """假后端，模拟 llama-server。"""

    name = "llama.cpp"

    def __init__(self):
        self.running = None
        self.stopped = []

    def is_available(self):
        return True

    def start_model(self, model_id, **opts):
        if self.running:
            raise RuntimeError("已有模型在运行")
        self.running = model_id
        return {"model_id": model_id, "base_url": f"http://127.0.0.1:8120",
                "path": f"/models/{model_id}.gguf", "pid": 1234}

    def stop_model(self):
        if self.running:
            self.stopped.append(self.running)
            self.running = None
            return True
        return False

    def is_running(self, model_id):
        return self.running == model_id

    def base_url(self):
        return "http://127.0.0.1:8120" if self.running else ""

    def list_models(self):
        return []

    def health(self):
        return {"ok": bool(self.running)}


def main() -> int:
    from imdeploy import api_server as api_mod
    from imdeploy import backends as backends_mod
    from imdeploy.api_server import build_server
    from imdeploy.config import Config

    fake = FakeBackend()
    # start_model 里是 `from .backends import detect_backends`（局部导入），
    # 所以要打在 backends 模块上才生效
    backends_mod.detect_backends = lambda cfg=None: [fake]

    tmp = Path(tempfile.mkdtemp(prefix="imd-load-"))
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(tmp / "data" / "config.yaml")
    cfg = Config.load(tmp / "data" / "config.yaml")

    httpd, api = build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    print("[模型加载]")
    st, d = call("POST", "/api/load_model", {"model_id": "qwen3-0.6b-q4"})
    check("POST /api/load_model", st == 200 and d.get("ok"), f"-> {d.get('error', '')}")
    check("返回 base_url", bool(d.get("base_url")))

    st, d = call("GET", "/api/loaded")
    loaded = (d.get("models") or {})
    check("GET /api/loaded 记录已加载", "qwen3-0.6b-q4" in loaded, f"-> {list(loaded)}")

    # 重复加载应提示已加载
    st, d = call("POST", "/api/load_model", {"model_id": "qwen3-0.6b-q4"})
    check("重复加载幂等", st == 200 and d.get("ok"))

    # 缺参数
    st, d = call("POST", "/api/load_model", {})
    check("缺 model_id 报错", st == 400 and not d.get("ok"))

    # 卸载
    st, d = call("POST", "/api/unload_model", {"model_id": "qwen3-0.6b-q4"})
    check("POST /api/unload_model", st == 200 and d.get("ok"))
    st, d = call("GET", "/api/loaded")
    check("卸载后清空记录", not (d.get("models") or {}))

    # 后端异常时优雅报错
    def boom(mid, **opts):
        raise RuntimeError("llama-server 不存在")
    fake.start_model = boom
    st, d = call("POST", "/api/load_model", {"model_id": "x"})
    check("后端异常优雅报错", st == 400 and "llama-server" in (d.get("error") or ""),
          f"-> {d.get('error', '')[:50]}")

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
