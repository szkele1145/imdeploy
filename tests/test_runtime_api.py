"""验证推理运行库相关接口（不真下载）。"""
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8913
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, payload=None, timeout=25):
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
    from imdeploy import backends as backends_mod
    from imdeploy import runtime_install as ri
    from imdeploy.api_server import build_server
    from imdeploy.config import Config

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    tmp = Path(tempfile.mkdtemp(prefix="imd-rt-"))
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(tmp / "data" / "config.yaml")
    cfg = Config.load(tmp / "data" / "config.yaml")

    httpd, api = build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    print("[运行库接口]")

    # 1) 状态查询（未安装）
    st, d = call("GET", "/api/runtime")
    check("GET /api/runtime", st == 200 and d.get("ok"), f"-> installed={d.get('installed')}")
    check("返回可用后端列表", "backends" in d)

    # 2) 假安装器：模拟安装成功
    def fake_install(bin_dir, proxy=None, accel=None, progress=None, log=None):
        if log:
            log("检测到加速后端：vulkan")
            log("下载中 50%")
        exe = Path(bin_dir) / "llama" / "llama-server.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"x")
        return ri.RuntimeInfo(True, exe_path=str(exe), backend="vulkan",
                              version="b9999", message="已安装 llama.cpp b9999（vulkan）")

    ri.install_llama_runtime = fake_install
    st, d = call("POST", "/api/install_runtime")
    check("POST /api/install_runtime", st == 200 and d.get("ok"), f"-> {d.get('error','')}")

    # 等后台线程完成
    for _ in range(20):
        time.sleep(0.2)
        st, d = call("GET", "/api/runtime_progress")
        if not d.get("installing"):
            break
    res = d.get("result") or {}
    check("后台安装完成", bool(res.get("ok")), f"-> {res.get('message','')}")
    check("返回安装路径", bool(res.get("path")))
    check("记录安装日志", len(d.get("log") or []) > 0, f"-> {d.get('log')}")

    # 3) 装完后状态应为已就绪
    st, d = call("GET", "/api/runtime")
    check("安装后状态已就绪", bool(d.get("installed")),
          f"-> installed={d.get('installed')} path={d.get('path','')}")

    # 4) 配置里应写入了路径
    cfg2 = Config.load(tmp / "data" / "config.yaml")
    check("配置已写入 llama_server 路径",
          bool(cfg2.get("backend.llama_server")),
          f"-> {cfg2.get('backend.llama_server','')}")

    # 5) 装完后再次安装应被接受（说明并发锁已释放）
    st, d = call("POST", "/api/install_runtime")
    check("安装完成后可再次安装", st == 200 and d.get("ok"), f"-> {d.get('error','')}")
    time.sleep(0.6)

    # 6) 安装进行中再次点，应被拒绝
    slow = {"n": 0}

    def slow_install(bin_dir, proxy=None, accel=None, progress=None, log=None):
        slow["n"] += 1
        if log:
            log("开始")
        time.sleep(2.0)
        return ri.RuntimeInfo(True, exe_path="x", backend="cpu", version="v")

    ri.install_llama_runtime = slow_install
    st, d = call("POST", "/api/install_runtime")
    check("启动安装", st == 200 and d.get("ok"))
    st, d = call("POST", "/api/install_runtime")
    check("安装进行中重复点被拒", st == 400 and not d.get("ok"),
          f"-> {d.get('error','')}")

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
