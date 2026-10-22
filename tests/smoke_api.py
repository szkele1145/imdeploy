"""API 服务器冒烟测试：真起一个服务，用 urllib 请求各端点。"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.api_server import build_server  # noqa: E402
from imdeploy.config import Config  # noqa: E402


def get(url: str, timeout: float = 10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def post(url: str, payload: dict, timeout: float = 10):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main() -> int:
    cfg = Config.default()
    cfg.set("app.port", 8931)
    cfg.set("app.token", "")

    httpd, api = build_server(cfg, host="127.0.0.1", port=8931)
    import threading

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{api.port}"

    checks = []
    checks.append(("GET /v1/models", *get(f"{base}/v1/models")))
    checks.append(("GET /api/status", *get(f"{base}/api/status")))
    checks.append(("GET /healthz", *get(f"{base}/healthz")))
    checks.append(("POST /v1/chat/completions",
                   *post(f"{base}/v1/chat/completions",
                         {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]})))
    checks.append(("POST /api/qwen (别名)",
                   *post(f"{base}/api/qwen",
                         {"messages": [{"role": "user", "content": "hi"}]})))
    checks.append(("POST /api/llama (别名)",
                   *post(f"{base}/api/llama",
                         {"messages": [{"role": "user", "content": "hi"}]})))

    bad = []
    for name, st, body in checks:
        print(f"{name:32s} -> {st}  {body[:130]}")
        if st >= 500:
            bad.append(name)

    httpd.shutdown()
    httpd.server_close()
    print("\nSMOKE " + ("OK" if not bad else "FAILED: " + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
