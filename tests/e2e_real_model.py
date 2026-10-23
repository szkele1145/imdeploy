"""端到端实机验证：下载小模型 -> 启动 llama-server -> 发对话请求。

这个脚本会真的下几百 MB，只在需要验证链路时手工跑。
用法：python tests/e2e_real_model.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 选一个足够小、镜像上确实存在的模型（Qwen2.5-0.5B-Instruct Q4_K_M ≈ 400MB）
MODEL_URL = ("https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
             "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def main() -> int:
    from imdeploy.backends import detect_backends
    from imdeploy.config import Config
    from imdeploy.downloader import Downloader

    cfg = Config.load(ROOT / "data" / "config.yaml") if (ROOT / "data" / "config.yaml").exists() else Config.default()
    models_dir = cfg.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / MODEL_NAME

    print(f"[1/4] 模型文件: {dest}")
    if dest.exists() and dest.stat().st_size > 50 * 1024 * 1024:
        print(f"      已存在（{dest.stat().st_size / 1048576:.0f} MB），跳过下载")
    else:
        print("      开始下载（首次约 400MB，请耐心）…")
        dl = Downloader()

        last = [0.0]

        def progress(task) -> None:
            now = time.time()
            if now - last[0] >= 3:
                last[0] = now
                print(f"      {task.percent:5.1f}%  "
                      f"{task.downloaded / 1048576:.0f}/{task.size / 1048576:.0f} MB")

        task = dl.download(MODEL_URL, dest, progress=progress)
        if task.status != "done":
            print(f"      下载失败: {task.error}")
            return 1
        print(f"      下载完成（{dest.stat().st_size / 1048576:.0f} MB）")

    print("[2/4] 检测后端")
    backends = detect_backends(cfg)
    for b in backends:
        print(f"      {b.name}: available={b.is_available()}")
    llama = next((b for b in backends if b.name == "llama.cpp"), None)
    if llama is None or not llama.is_available():
        print("      llama.cpp 后端不可用，先跑 --install-runtime")
        return 1

    print("[3/4] 启动 llama-server")
    info = llama.start_model(str(dest), ctx_size=2048, n_gpu_layers=0, threads=4)
    print(f"      base_url={info.get('base_url')}")
    base_url = info.get("base_url") or ""

    # 等它 ready（/health 或 /v1/models 返回 200）
    ready = False
    for i in range(120):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:  # noqa: BLE001
            if i % 10 == 9:
                print(f"      等待就绪… {i + 1}s")
    if not ready:
        print("      服务未能就绪")
        llama.stop_model()
        return 1
    print("      服务已就绪")

    print("[4/4] 发对话请求")
    payload = json.dumps({
        "model": dest.stem,
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "max_tokens": 64,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"]
        print(f"      模型回复：{text.strip()[:200]}")
        print(f"      usage: {data.get('usage')}")
        ok = bool(text.strip())
    except Exception as exc:  # noqa: BLE001
        print(f"      请求失败：{exc}")
        ok = False
    finally:
        llama.stop_model()

    print("\n结果:", "端到端链路打通 ✅" if ok else "链路未打通 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
