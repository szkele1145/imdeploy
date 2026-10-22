"""测试模型目录抓取速度与可用性。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imdeploy.catalog import DEFAULT_QUERIES, build_catalog, fetch_hf_models  # noqa: E402
from imdeploy.hardware import detect  # noqa: E402
from imdeploy.recommend import recommend  # noqa: E402


def main():
    t0 = time.time()
    r = fetch_hf_models("gguf qwen3", 10)
    print(f"[单查询] {len(r)} 条, {time.time() - t0:.1f}s")
    for e in r[:3]:
        print(f"   {e.name} | {e.params_b}B | {e.ctx_len // 1024}K | {e.quant} | {e.size_gb}GB")

    t0 = time.time()
    cat = build_catalog(use_cache=False)
    dt = time.time() - t0
    print(f"\n[完整目录] {len(cat)} 条, {dt:.1f}s")

    hw = detect()
    t0 = time.time()
    recs = recommend(cat, hw, limit=12)
    print(f"[推荐] {len(recs)} 条, {time.time() - t0:.2f}s")
    for rec in recs[:6]:
        e = rec.entry
        print(f"   [{rec.tier}] {e.name[:40]:40s} {e.params_b}B {e.ctx_len // 1024}K "
              f"{e.quant:8s} 分数{rec.score:.0f}")
    return 0 if cat else 1


if __name__ == "__main__":
    sys.exit(main())
