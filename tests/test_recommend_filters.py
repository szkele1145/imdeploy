"""验证推荐接口的筛选参数（体积上限 / 系列）。"""
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8941
BASE = f"http://127.0.0.1:{PORT}"


def get(path, timeout=60):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {}


def main() -> int:
    from imdeploy import api_server as api_mod
    from imdeploy.api_server import ApiServer
    from imdeploy.catalog import ModelEntry
    from imdeploy.config import Config
    from imdeploy.hardware import HardwareInfo
    from imdeploy.recommend import Recommendation

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    def entry(mid, family, size_gb, ctx=32768):
        return ModelEntry(id=mid, name=mid, family=family, params_b=1.0, quant="Q4_K_M",
                          ctx_len=ctx, size_gb=size_gb, download_url=f"https://x/{mid}")

    entries = [
        entry("qwen-small", "qwen3", 0.6),
        entry("qwen-big", "qwen3", 8.0),
        entry("llama-mid", "llama", 3.5),
        entry("huge-one", "other", 30.0),
    ]

    # 打桩，避免真抓网络
    api_mod.build_catalog = lambda use_cache=True: entries  # type: ignore[attr-defined]
    import imdeploy.catalog as cat_mod
    cat_mod.build_catalog = lambda use_cache=True: entries  # type: ignore[attr-defined]
    import imdeploy.hardware as hw_mod
    hw_mod.detect = lambda: HardwareInfo(
        cpu_arch="x86_64", cpu_cores=8, cpu_model_name="test-cpu",
        total_ram_gb=32.0, available_ram_gb=24.0,
        gpu_name="", gpu_vram_gb=0.0, gpu_backend="cpu",
    )

    tmp = Path(tempfile.mkdtemp(prefix="imd-rec-"))
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(tmp / "data" / "config.yaml")
    cfg = Config.load(tmp / "data" / "config.yaml")

    api = ApiServer(cfg, backends=[])

    print("[推荐筛选]")
    base_recs = api.get_recommendations(limit=40, min_ctx=4096)
    check("无筛选返回全部", len(base_recs) >= 3, f"-> {len(base_recs)}")

    capped = api.get_recommendations(limit=40, min_ctx=4096, max_size_gb=4.0)
    sizes = [r["size_gb"] for r in capped]
    check("体积上限生效", all(s <= 4.0 for s in sizes), f"-> {sizes}")
    check("过滤掉了大模型", "huge-one" not in [r["id"] for r in capped])

    qwen = api.get_recommendations(limit=40, min_ctx=4096, family="qwen")
    check("系列筛选生效", all("qwen" in r["family"] for r in qwen),
          f"-> {[r['id'] for r in qwen]}")
    check("系列筛选非空", len(qwen) >= 1)

    both = api.get_recommendations(limit=40, min_ctx=4096, max_size_gb=1.0, family="qwen")
    check("组合筛选", len(both) == 1 and both[0]["id"] == "qwen-small",
          f"-> {[r['id'] for r in both]}")

    # HTTP 层参数解析
    httpd, srv = api_mod.build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    st, d = get("/api/recommend?limit=40&max_size=4")
    check("HTTP max_size 解析", st == 200 and d.get("ok"),
          f"-> count={d.get('count')}")
    if st == 200:
        sizes = [m["size_gb"] for m in d.get("models", [])]
        check("HTTP 结果受体积限制", all(s <= 4.0 for s in sizes), f"-> {sizes}")

    st, d = get("/api/recommend?limit=40&family=llama")
    check("HTTP family 解析", st == 200 and d.get("ok") and d.get("count", 0) >= 1,
          f"-> {[m['id'] for m in d.get('models', [])]}")

    st, d = get("/api/recommend?limit=40&max_size=notanumber")
    check("非法 max_size 不崩", st == 200 and d.get("ok"), f"-> {st}")

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
