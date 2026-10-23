"""验证下载多镜像回退。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.api_server import _mirror_candidates  # noqa: E402


def test_mirror_to_huggingface():
    url = "https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/m.gguf"
    got = _mirror_candidates(url)
    assert got[0] == url
    assert any("huggingface.co" in u for u in got), got
    assert not any("hf-mirror.com" in u for u in got[1:]), got
    print("hf-mirror -> huggingface 回退")


def test_mirror_to_hfmirror():
    url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/m.gguf"
    got = _mirror_candidates(url)
    assert got[0] == url
    assert any("hf-mirror.com" in u for u in got), got
    print("huggingface -> hf-mirror 回退")


def test_modelscope_no_mirror():
    url = "https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF/resolve/master/m.gguf"
    got = _mirror_candidates(url)
    assert got == [url], got
    print("modelscope 无镜像，只试原地址")


def test_empty_and_unknown():
    assert _mirror_candidates("") == []
    assert _mirror_candidates("https://example.com/a.gguf") == ["https://example.com/a.gguf"]
    print("空地址/未知域名安全返回")


def test_no_duplicates():
    url = "https://hf-mirror.com/a/m.gguf"
    got = _mirror_candidates(url)
    assert len(got) == len(set(got)), got
    print("候选地址无重复")


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
