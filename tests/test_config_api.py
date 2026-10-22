"""验证 imdeploy 的配置表单 API。"""
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8907
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


def main() -> int:
    from imdeploy.api_server import build_server
    from imdeploy.config import Config

    tmp = Path(tempfile.mkdtemp(prefix="imd-cfg-"))
    cfg_path = tmp / "data" / "config.yaml"
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(cfg_path)
    cfg = Config.load(cfg_path)

    httpd, api = build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    print("[配置 API]")
    st, d = call("GET", "/api/config")
    check("GET /api/config", st == 200 and d.get("ok"))
    fields = d.get("fields") or {}
    print(f"    字段数: {len(fields)}")
    check("返回字段", len(fields) > 10, f"-> {len(fields)} 项")
    check("返回分组", bool(d.get("sections")))

    # 类型检查
    types = {k: v.get("type") for k, v in fields.items()}
    check("布尔字段识别为 bool", types.get("app.open_browser") == "bool",
          f"-> {types.get('app.open_browser')}")
    check("数字字段识别为 int", types.get("app.port") == "int",
          f"-> {types.get('app.port')}")
    check("文本字段识别为 text", types.get("backend.llama_server") == "text",
          f"-> {types.get('backend.llama_server')}")
    print(f"    app.port 类型 = {types.get('app.port')}, 标签 = {fields.get('app.port', {}).get('label')}")

    # 保存
    st, d = call("POST", "/api/config", {"config": {"app.port": 9099, "app.name": "我的部署器"}})
    check("POST /api/config", st == 200 and d.get("ok"), f"-> {d.get('error', '')}")
    check("报告保存项", len(d.get("saved") or []) == 2, f"-> {d.get('saved')}")

    # 校验真的写进去了
    st, d = call("GET", "/api/config")
    f2 = d.get("fields") or {}
    check("端口已更新", f2.get("app.port", {}).get("value") == 9099,
          f"-> {f2.get('app.port', {}).get('value')}")
    check("名称已更新", f2.get("app.name", {}).get("value") == "我的部署器")

    # 落盘校验
    txt = cfg_path.read_text(encoding="utf-8")
    check("已落盘", "9099" in txt)

    # 非法键应被忽略
    st, d = call("POST", "/api/config", {"config": {"不存在的节.字段": 1}})
    check("忽略非法键", st == 200 and not (d.get("saved") or []),
          f"-> {d.get('saved')}")

    # 布尔
    st, d = call("POST", "/api/config", {"config": {"app.open_browser": False}})
    check("布尔可保存", st == 200 and d.get("ok"))
    st, d = call("GET", "/api/config")
    check("布尔已更新",
          (d.get("fields", {}).get("app.open_browser", {}).get("value")) is False)

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
