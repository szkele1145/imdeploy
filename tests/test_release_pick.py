"""验证运行库安装的 release 选择逻辑（不掉网、不下载）。

真实踩过的坑：llama.cpp 的 /releases/latest 指向 v0.7.0 这种只挂
nightly-tag.txt 的空壳发布，真正的二进制在 b12345 上。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.runtime_install import _pick_binary_release, pick_release_asset  # noqa: E402


def rel(tag, assets):
    return {"tag_name": tag, "assets": [{"name": n, "browser_download_url": f"u/{n}"} for n in assets]}


def test_skips_empty_shell_release():
    releases = [
        rel("v0.7.0", ["nightly-tag.txt"]),
        rel("b11829", ["llama-b11829-bin-win-cpu-x64.zip",
                       "llama-b11829-bin-win-vulkan-x64.zip"]),
        rel("b11828", ["llama-b11828-bin-win-cpu-x64.zip"]),
    ]
    got = _pick_binary_release(releases)
    assert got is not None
    assert got["tag_name"] == "b11829", got["tag_name"]
    print(f"跳过空壳 v0.7.0，选中 {got['tag_name']}")


def test_returns_none_when_no_binaries():
    releases = [rel("v0.7.0", ["nightly-tag.txt"]), rel("v0.6.0", [])]
    assert _pick_binary_release(releases) is None
    print("没有二进制时返回 None")


def test_empty_list():
    assert _pick_binary_release([]) is None
    print("空列表返回 None")


def test_pick_asset_prefers_vulkan_cpu_fallback():
    assets = [
        {"name": "llama-b1-bin-win-cuda-cu12.4-x64.zip", "browser_download_url": "cuda"},
        {"name": "llama-b1-bin-win-vulkan-x64.zip", "browser_download_url": "vulkan"},
        {"name": "llama-b1-bin-win-cpu-x64.zip", "browser_download_url": "cpu"},
    ]
    picked = pick_release_asset(assets, "vulkan")
    assert picked is not None
    assert "vulkan" in picked["name"], picked["name"]
    print(f"vulkan 优先: {picked['name']}")

    picked_cpu = pick_release_asset(assets, "cpu")
    assert picked_cpu is not None
    assert "cpu" in picked_cpu["name"], picked_cpu["name"]
    print(f"cpu 优先: {picked_cpu['name']}")


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
