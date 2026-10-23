"""验证「模型加载中」的识别逻辑。

llama-server 加载权重时 /v1/models 已经 200，但 /v1/chat/completions 回
503 {"error":{"message":"Loading model"}}，需要自动重试而不是直接报错。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.api_server import _is_loading_model  # noqa: E402


def test_detects_loading_message():
    assert _is_loading_model({"error": {"message": "Loading model"}})
    assert _is_loading_model({"error": {"message": "loading model, please wait"}})
    print("识别 Loading model 消息")


def test_detects_loading_code():
    assert _is_loading_model({"error": {"code": "loading"}})
    print("识别 code=loading")


def test_detects_string_error():
    assert _is_loading_model({"error": "Loading model"})
    print("识别字符串形式 error")


def test_detects_top_level_message():
    assert _is_loading_model({"message": "Loading model"})
    print("识别顶层 message")


def test_rejects_real_errors():
    assert not _is_loading_model({"error": {"message": "model not found"}})
    assert not _is_loading_model({"error": {"message": "out of memory"}})
    assert not _is_loading_model({})
    assert not _is_loading_model(None)
    assert not _is_loading_model("Loading model")
    print("真实错误不会被误判为加载中")


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
