"""验证本地模型库的删除接口（不碰真实模型）。"""
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8931
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, payload=None, timeout=20):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
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

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    tmp = Path(tempfile.mkdtemp(prefix="imd-lib-"))
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(tmp / "data" / "config.yaml")
    cfg = Config.load(tmp / "data" / "config.yaml")

    models = cfg.models_dir
    models.mkdir(parents=True, exist_ok=True)
    fake = models / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    fake.write_bytes(b"GGUF" + b"\x00" * 2048)

    httpd, api = build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    print("[模型库]")
    st, d = call("GET", "/api/library_scan")
    check("扫描登记", st == 200 and d.get("ok"), f"-> added={d.get('added')}")

    st, d = call("GET", "/api/library")
    models_list = d.get("models") or []
    check("库里有条目", len(models_list) == 1, f"-> {[m['name'] for m in models_list]}")
    mid = models_list[0]["id"] if models_list else ""

    st, d = call("POST", "/api/library_remove", {})
    check("缺 model_id 报错", st == 400 and not d.get("ok"))

    st, d = call("POST", "/api/library_remove", {"model_id": mid, "keep_file": True})
    check("仅移除登记（保留文件）", st == 200 and d.get("ok"))
    check("文件仍在", fake.is_file())

    # 重新登记再真删
    call("GET", "/api/library_scan")
    st, d = call("POST", "/api/library_remove", {"model_id": mid})
    check("删除模型+文件", st == 200 and d.get("ok"))
    check("文件已删", not fake.is_file())

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
