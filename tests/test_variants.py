"""验证量化版本聚合与按量化下载。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.catalog import ModelEntry, group_by_model  # noqa: E402


def entry(name, quant, size):
    return ModelEntry(
        id="Qwen/Qwen3-0.6B-GGUF",
        name=name,
        family="qwen3",
        params_b=0.6,
        quant=quant,
        ctx_len=32768,
        size_gb=size,
        download_url=f"https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/{name}",
    )


def test_group_by_model():
    items = [
        entry("model-Q8_0.gguf", "Q8_0", 0.98),
        entry("model-Q4_K_M.gguf", "Q4_K_M", 0.50),
        entry("model-Q5_K_M.gguf", "Q5_K_M", 0.60),
    ]
    groups = group_by_model(items)
    assert len(groups) == 1, f"应聚合成 1 组，实际 {len(groups)}"
    g = groups[0]
    print(f"聚合组: {g['name']}  推荐={g['recommended']}")
    quants = [v["quant"] for v in g["variants"]]
    print("  变体:", quants)
    print("  体积:", [v["size_gb"] for v in g["variants"]])
    assert quants == ["Q4_K_M", "Q5_K_M", "Q8_0"], f"排序错: {quants}"
    assert g["recommended"] == "Q4_K_M"
    assert g["variants"][0]["filename"] == "model-Q4_K_M.gguf"


def test_group_multiple_repos():
    a = entry("a-Q4_K_M.gguf", "Q4_K_M", 0.5)
    b = ModelEntry(id="Other/Model-GGUF", name="Other", family="other",
                   params_b=1.0, quant="Q4_0", ctx_len=8192, size_gb=0.9)
    groups = group_by_model([a, b])
    assert len(groups) == 2
    print(f"两个仓库 -> {len(groups)} 组")
    for g in groups:
        print(f"  {g['id']} 变体数={len(g['variants'])}")


def test_group_empty():
    assert group_by_model([]) == []
    print("空目录 OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}\n")
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}\n")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {e}\n")
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)
