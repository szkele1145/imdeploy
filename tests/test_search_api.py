"""验证 /api/search 多源搜索（HF + ModelScope）。"""
import json
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 8951
BASE = f"http://127.0.0.1:{PORT}"


def main() -> int:
    from imdeploy import api_server as api_mod
    from imdeploy import catalog
    from imdeploy.api_server import ApiServer
    from imdeploy.catalog import ModelEntry
    from imdeploy.config import Config

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    def entry(mid, family="qwen", size=1.0, params=1.0, source="hf"):
        return ModelEntry(id=mid, name=mid, family=family, params_b=params,
                          quant="Q4_K_M", ctx_len=32768, size_gb=size,
                          download_url=f"https://x/{mid}", source=source)

    # 打桩：search_models 走桩，避免真联网
    calls = []

    def fake_search(q, limit=30):
        calls.append(q)
        if "qwen" in q.lower():
            return [entry("Qwen/Qwen2.5-7B-GGUF", source="hf"),
                    entry("ms/qwen2.5-7b", source="modelscope")]
        if "llama" in q.lower():
            return [entry("Meta/Llama-3-8B-GGUF", family="llama", source="hf")]
        return []

    catalog.search_models = fake_search  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="imd-srch-"))
    cfg = Config.default()
    cfg.set("app.base_dir", str(tmp))
    cfg.save(tmp / "data" / "config.yaml")
    cfg = Config.load(tmp / "data" / "config.yaml")

    httpd, srv = api_mod.build_server(cfg, host="127.0.0.1", port=PORT)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(1.0)

    def get(path, timeout=30):
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return json.loads(r.read())

    print("[/api/search]")
    d = get("/api/search?" + urllib.parse.urlencode({"q": "qwen"}))
    check("搜索 qwen 返回 200", d.get("ok") and d.get("count", 0) >= 2,
          f"-> {d.get('count')}")
    check("多源去重后返回", {m["source"] for m in d.get("models", [])} >= {"hf", "modelscope"},
          f"-> {[m['source'] for m in d.get('models', [])]}")

    d2 = get("/api/search?" + urllib.parse.urlencode({"q": "llama"}))
    check("搜索 llama 命中", d2.get("count", 0) >= 1
          and d2["models"][0]["id"] == "Meta/Llama-3-8B-GGUF")

    d3 = get("/api/search?" + urllib.parse.urlencode({"q": "zzz-nothing"}))
    check("无结果返回空列表", d3.get("ok") and d3.get("count") == 0)

    d4 = get("/api/search")
    check("缺 q 参数不崩", d4.get("ok") and d4.get("count") == 0,
          f"-> {d4.get('count')}")

    httpd.shutdown()
    httpd.server_close()

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
