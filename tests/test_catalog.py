"""离线单元测试：模型目录解析 / 硬件数据结构 / 推荐引擎 / 缓存。

全部测试不联网：HF / ModelScope 的响应都用构造数据 + monkeypatch。
运行：``.venv\Scripts\python.exe -m pytest tests/ -q``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imdeploy.catalog import (  # noqa: E402
    DEFAULT_QUERIES,
    ModelEntry,
    build_catalog,
    detect_ctx_len,
    detect_family,
    detect_params_b,
    detect_quant,
    fetch_hf_models,
    fetch_modelscope_models,
    load_cached_catalog,
    parse_hf_model,
    parse_modelscope_model,
    save_cached_catalog,
)
from imdeploy.hardware import HardwareInfo, available_ram_gb, detect, summarize  # noqa: E402
from imdeploy.recommend import (  # noqa: E402
    TIERS,
    Recommendation,
    recommend,
    score_model,
    summarize_hw,
)

_GB = 1024.0**3


# --------------------------------------------------------------------------- #
# 测试数据（模拟 API 响应）
# --------------------------------------------------------------------------- #
def make_hf_model(
    model_id: str = "Qwen/Qwen3-0.6B-GGUF",
    gguf_name: str = "qwen3-0.6b-q4_k_m.gguf",
    gguf_size: int = 470 * 1024 * 1024,
    tags: list[str] | None = None,
    description: str = "",
    sha256: str = "deadbeef" * 8,
    extra: dict | None = None,
) -> dict:
    """构造一个 HuggingFace /api/models 列表项。"""
    obj: dict = {
        "modelId": model_id,
        "id": model_id,
        "tags": tags if tags is not None else ["gguf", "qwen3", "instruct", "text-generation"],
        "siblings": [
            {"rfilename": "README.md"},
            {
                "rfilename": gguf_name,
                "size": gguf_size,
                "sha256": sha256,
            },
        ],
        "downloads": 12345,
        "description": description,
        "cardData": {"license": "apache-2.0"},
    }
    if extra:
        obj.update(extra)
    return obj


def make_modelscope_model(
    path: str = "Qwen/Qwen3-0.6B-GGUF",
    name: str = "Qwen3-0.6B-GGUF",
    gguf_name: str = "qwen3-0.6b-q4_k_m.gguf",
    gguf_size: int = 470 * 1024 * 1024,
    description: str = "",
    tags: list[str] | None = None,
) -> dict:
    """构造一个 ModelScope api/v1/models 列表项（带 Files）。"""
    return {
        "Path": path,
        "Name": name,
        "Description": description,
        "Tags": tags if tags is not None else ["qwen3", "instruct", "gguf"],
        "License": "Apache-2.0",
        "Files": [
            {"Name": "README.md", "Size": 1000},
            {"Name": gguf_name, "Size": gguf_size, "Sha256": "cafebabe" * 8},
        ],
    }


@pytest.fixture()
def hw_low_end() -> HardwareInfo:
    """低配机器：8GB 内存 / 无 GPU。"""
    return HardwareInfo(
        cpu_arch="x86_64",
        cpu_cores=8,
        cpu_model_name="Intel i5-12400",
        total_ram_gb=8.0,
        available_ram_gb=6.0,
        gpu_backend="cpu",
        is_arm=False,
    )


@pytest.fixture()
def hw_with_gpu() -> HardwareInfo:
    """带 NVIDIA GPU 的中配机器：16GB 内存 / 8GB 显存。"""
    return HardwareInfo(
        quant=None,  # type: ignore[arg-type]
        cpu_arch="x86_64",
        cpu_cores=12,
        cpu_model_name="AMD Ryzen 5 7600",
        total_ram_gb=16.0,
        available_ram_gb=12.0,
        gpu_name="NVIDIA RTX 4070",
        gpu_vram_gb=8.0,
        gpu_backend="nvidia",
        is_arm=False,
    )


def make_entry(**kwargs) -> ModelEntry:
    """快速构造 ModelEntry。"""
    defaults = dict(
        id="test/model-a",
        name="Model A",
        family="qwen",
        params_b=0.6,
        quant="Q4_K_M",
        ctx_len=32768,
        size_gb=0.45,
        download_url="https://example.com/model-a.gguf",
        sha256="ab" * 32,
        tags=["gguf", "instruct"],
        source="hf",
        description="小模型，长上下文",
        repo_url="https://example.com/repo",
        license="apache-2.0",
    )
    defaults.update(kwargs)
    return ModelEntry(**defaults)


# --------------------------------------------------------------------------- #
# 1. parse_hf_model
# --------------------------------------------------------------------------- #
class TestParseHfModel:
    def test_basic_parse(self):
        obj = make_hf_model()
        entry = parse_hf_model(obj)
        assert entry is not None
        assert entry.id == "Qwen/Qwen3-0.6B-GGUF"
        assert entry.source == "hf"
        assert entry.family == "qwen"
        assert entry.params_b == 0.6
        assert entry.quant == "Q4_K_M"
        assert entry.size_gb == pytest.approx(0.45, abs=0.05)
        assert entry.download_url.endswith(".gguf")
        assert entry.sha256 == "deadbeef" * 8
        assert entry.license == "apache-2.0"
        assert "instruct" in entry.tags

    def test_rejects_non_gguf(self):
        obj = make_hf_model()
        obj["siblings"] = [{"rfilename": "model.safetensors", "size": 123}]
        assert parse_hf_model(obj) is None

    def test_rejects_missing_id(self):
        obj = make_hf_model()
        obj["modelId"] = ""
        obj["id"] = ""
        assert parse_hf_model(obj) is None

    def test_lfs_size_and_ctx_from_gguf_meta(self):
        obj = make_hf_model(
            extra={
                "gguf": {"context_length": 131072, "total": 600_000_000},
                "siblings": [
                    {
                        "rfilename": "model-q5_k_m.gguf",
                        "lfs": {"size": 1_000_000_000, "oid": "sha256:" + "aa" * 32},
                    }
                ],
            }
        )
        entry = parse_hf_gguf = parse_hf_model(obj)
        assert entry is not None
        assert entry.ctx_len == 131072
        assert entry.params_b == pytest.approx(0.6, abs=0.01)
        assert entry.size_gb == pytest.approx(0.931, abs=0.02)
        assert entry.sha256 == "aa" * 32

    def test_tag_ctx_len(self):
        obj = make_hf_model(extra={"tags": ["gguf", "context_length:65536", "qwen3"]})
        entry = parse_hf_model(obj)
        assert entry is not None
        assert entry.ctx_len == 65536


# --------------------------------------------------------------------------- #
# 2. parse_modelscope_model
# --------------------------------------------------------------------------- #
class TestParseModelscopeModel:
    def test_basic_parse(self):
        obj = make_modelscope_model()
        entry = parse_modelscope_model(obj)
        assert entry is not None
        strict = None
        assert entry.id == "Qwen/Qwen3-0.6B-GGUF"
        assert entry.source == "modelscope"
        assert entry.params_b == 0.6
        assert entry.quant == "Q4_K_M"
        assert entry.size_gb == pytest.approx(0.45, abs=0.03)
        assert "modelscope.cn/models/" in entry.repo_url
        assert entry.download_url.endswith(".gguf")

    def test_require_gguf_default_rejects_empty_files(self):
        obj = make_modelscope_model()
        obj["Files"] = []
        assert parse_modelscope_model(obj) is None

    def test_no_gguf_files_returns_none(self):
        obj = make_modelscope_model()
        obj["Files"] = [{"Name": "model.safetensors", "Size": 100}]
        assert parse_modelscope_model(obj) is None

    def test_loose_mode_returns_entry(self):
        obj = make_modelscope_model()
        obj["Files"] = []
        entry = parse_modelscope_model(obj, require_gguf=False)
        assert entry is not None
        assert entry.source == "modelscope"
        assert entry.download_url == ""


# --------------------------------------------------------------------------- #
# 3. build_catalog 去重
# # -- ...
# --------------------------------------------------------------------------- #
class TestBuildCatalog:
    def test_merge_and_dedupe(self, monkeypatch, tmp_path):
        a = make_entry(id="Qwen/Qwen3-1.7B-GGUF", ctx_len=32768, size_gb=1.1)
        b = make_entry(id="qwen/qwen3-1.7b-gguf", ctx_len=0, size_gb=0.0)  # 同 id 小写
        c = make_entry(id="Others/Distinct-7B-GGUF", params_b=7.0, quant="Q4_K_M", size_gb=4.0)

        def fake_hf(query, limit, **kwargs):
            return [a]

        def fake_ms(query, limit, **kwargs):
            return [b, c]

        monkeypatch.setattr("imdeploy.catalog.fetch_hf_models", fake_hf)
        monkeypatch.setattr("imdeploy.catalog.fetch_modelscope_models", fake_ms)

        catalog = build_catalog(
            queries=["gguf qwen3"],
            use_cache=False,
            cache_path=tmp_path / "cat.json",
            limit=5,
            sources=("hf", "modelscope"),
        )
        ids = {e.id.lower() for e in catalog}
        assert len(ids) == 2
        assert ids == {"qwen/qwen3-1.7b-gguf", "others/distinct-7b-gguf"}
        # 去重合并：b 的空字段不应拉低 a 的信息（b 与 a 同 id）
        merged = [e for e in catalog if e.id.lower() == "qwen/qwen3-1.7b-gguf"][0]
        assert merged.size_gb == 1.1  # b 的 0 不该覆盖 a 的 1.1
        assert merged.ctx_len == 32768

    def test_default_queries_cover_required_families(self):
        required = ["qwen3", "qwen2.5", "llama", "gemma", "smollm", "phi", "internlm", "minicpm"]
        blob = " ".join(DEFAULT_QUERIES).lower()
        for keyword in required:
            assert keyword in blob, f"默认查询词缺少 {keyword}"

    def test_fetch_failure_degrades_to_empty(self, monkeypatch, tmp_path):
        def boom(query, limit=None, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("imdeploy.catalog.fetch_hf_models", boom)
        monkeypatch.setattr("imdeploy.catalog.fetch_modelscope_models", boom)

        catalog = build_catalog(
            queries=["gguf qwen3"],
            use_cache=False,
            cache_path=tmp_path / "cat.json",
        )
        assert catalog == []

    def test_cache_hit_returns_cached_entries(self, monkeypatch, tmp_path):
        a = make_entry(id="X/cached-model", size_gb=0.5, ctx_len=32768)
        cache = tmp_path / "cache.json"
        save_cached_catalog(cache, [a], fetched_at="2999-01-01T00:00:00+00:00")

        calls = []

        def fake_fetch(query, limit, **kwargs):
            calls.append(query)
            return [make_entry(id="Y/fresh")]

        monkeypatch.setattr("feiyu_local_llfings := None", None) if False else None
        monkeypatch.setattr("imdeploy.catalog.fetch_hf_models", fake_fetch)

        catalog = build_catalog(
            queries=["gguf qwen3"],
            use_cache=True,
            cache_path=cache,
            cache_ttl_hours=100000.0,
        )
        assert calls == []  # 不应触发抓取
        assert [e.id for e in catalog] == ["X/cached-model"]


# --------------------------------------------------------------------------- #
# 4. HardwareInfo
# --------------------------------------------------------------------------- #
class TestHardwareInfo:
    def test_defaults(self):
        hw = HardwareInfo()
        assert hw.cpu_arch == ""
        assert hw.cpu_cores == 0
        total = hw.total_ram_gb
        assert total == 0.0
        assert hw.gpu_backend == "cpu"
        assert hw.is_arm is False
        assert hw.is_android is False
        assert hw.has_gpu is False

    def test_is_arm_detection(self):
        hw = HardwareInfo(cpu_arch="aarch64")
        assert hw.is_arm is False  # is_arm 默认 False，需要 detect() 或手动设置
        hw_arm = HardwareInfo(cpu_arch="aarch64", is_arm=True)
        assert hw_arm.is_arm is True

    def test_detect_returns_full_object(self):
        hw = detect()
        assert isinstance(hw, HardwareInfo)
        assert hw.cpu_arch  # 架构总能探测到
        assert hw.cpu_cores > 0
        assert hw.cpu_model_name  # 本机（Windows）能拿到型号
        assert hw.total_ram_gb > 0
        assert hw.available_ram_gb > 0
        assert hw.gpu_backend in ("nvidia", "vulkan", "rocm", "cpu")

    def test_available_ram_gb_positive(self):
        assert available_ram_gb() > 0

    def test_summarize_is_chinese_sentence(self):
        hw = HardwareInfo(
            cpu_arch="x86_64",
            cpu_cores=8,
            cpu_model_name="Intel i5",
            total_ram_gb=16.0,
            available_ram_gb=10.0,
            gpu_name="RTX 4060",
            gpu_vram_gb=8.0,
            gpu_backend="nvidia",
        )
        text = summarize(hw)
        assert text.endswith("。")
        assert "Intel i5" in text
        assert "16.0GB" in text
        assert "RTX 4060" in text
        assert "无独立 GPU" not in text

    def test_summarize_no_gpu(self):
        hw = HardwareInfo(cpu_arch="x86_64", cpu_cores=4, total_ram_gb=8.0, available_ram_gb=4.0)
        text = summarize(hw)
        assert "无独立 GPU" in text


# --------------------------------------------------------------------------- #
# 5. score_model
# --------------------------------------------------------------------------- #
class TestScoreModel:
    def test_memory_shortage_fit_false(self, hw_low_end):
        big = make_entry(size_gb=9.0, params_b=7.0, ctx_len=8192)
        rec = score_model(big, hw_low_end)
        assert rec.fit is False
        assert rec.tier == "不适合"
        assert rec.score < 50
        assert any("超出可用内存预算" in r for r in rec.reasons)

    def test_gpu_vram_shortage_fit_false(self, hw_with_gpu):
        big = make_entry(size_gb=7.5, params_b=7.0, ctx_len=8192)
        rec = score_model(big, hw_with_gpu)
        assert rec.fit is True  # 内存装得下
        assert any("大于显存" in r for r in rec.reasons)
        # 7.5GB > 8GB*0.9=7.2 → 混合推理提示，但不影响 fit（内存维度）

    def test_longer_ctx_scores_higher(self, hw_low_end):
        e8k = make_entry(ctx_len=8192, id="m/8k")
        e32k = make_entry(ctx_len=32768, id="m/32k")
        e64k = make_entry(ctx_len=65536, id="m/64k")
        e128k = make_entry(ctx_len=131072, id="m/128k")
        s8, s32, s64, s128 = (
            score_model(e8k, hw_low_end).score,
            score_model(e32k, hw_low_end).score,
            score_model(e64k, hw_low_end).score,
            score_model(e128k, hw_low_end).score,
        )
        assert s8 < s32 < s64 < s128
        assert s128 >= 80

    def test_base_scores_lower_than_instruct(self, hw_low_end):
        instruct = make_entry(tags=["gguf", "instruct"], id="m/instruct")
        base = make_entry(tags=["gguf", "base"], id="m/base")
        s_inst = score_model(instruct, hw_low_end).score
        s_base = score_model(base, hw_low_end).score
        assert s_inst > s_base
        # 差距应至少来自 tag 加减分（6 + 6 = 12 分）
        assert s_inst - s_base >= 12.0

    def test_small_model_bonus(self, hw_low_end):
        small = make_entry(params_b=0.6, id="m/small")
        medium = make_entry(params_b=7.0, size_gb=4.0, id="m/medium")
        s_small = score_model(small, hw_low_end).score
        s_med = score_model(medium, hw_low_end).score
        assert s_small > s_med

    def test_quant_preference(self, hw_low_end):
        q4km = make_entry(quant="Q4_K_M", id="m/q4km")
        q4_0 = make_entry(quant="Q4_0", id="m/q4_0")
        assert score_model(q4km, hw_low_end).score > score_model(q4_0, hw_low_end).score

    def test_ctx_preference_long_boosts_long_ctx(self, hw_low_end):
        e = make_entry(ctx_len=65536)
        balanced = score_model(e, hw_low_end, "balanced").score
        long = score_model(e, hw_low_end, "long").score
        short = score_model(e, hw_low_end, "short").score
        assert long > balanced > short

    def test_long_context_in_text_gets_bonus(self, hw_low_end):
        with_text = make_entry(name="Qwen3 128K", description="long context", id="m/128k-text")
        plain = make_entry(name="Qwen3", description="", id="m/plain")
        assert score_model(with_text, hw_low_end).score > score_model(plain, hw_low_end).score

    def test_fit_false_low_reasons_present(self, hw_low_end):
        big = make_entry(size_gb=100.0, params_b=70.0, ctx_len=8192)
        rec = score_model(big, hw_low_end)
        assert rec.fit is False
        assert rec.score == 0.0  # 超预算+大参数+超内存 → 钳到 0
        assert rec.tier == "不适合"


# --------------------------------------------------------------------------- #
# 6. recommend
# --------------------------------------------------------------------------- #
class TestRecommend:
    def test_sorted_desc_and_tiers(self, hw_low_end):
        catalog = [
            make_entry(id="m/weak", params_b=7.0, size_gb=4.5, ctx_len=8192, tags=["gguf", "base"]),
            make_entry(id="m/fit-fit", params_b=0.6, size_gb=0.45, ctx_len=131072, tags=["gguf", "instruct"]),
            make_entry(id="m/tiny-weak", params_b=0.4, size_gb=0.3, ctx_len=4096, tags=["gguf", "instruct"]),
            make_entry(id="m/oversized", params_b=13.0, size_gb=9.0, ctx_len=8192),
        ]
        recs = recommend(catalog, hw_low_end, limit=10, min_ctx=0)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)
        tiers = {r.entry.id: r.tier for r in recs}
        assert tiers["m/oversized"] == "不适合"
        assert tiers["m/fit-fit"] == "首选"
        assert tiers["m/weak"] in ("备选", "偏大")

    def test_min_ctx_filters_short_ctx(self, hw_low_end):
        catalog = [
            make_entry(id="m/short", ctx_len=4096),
            make_entry(id="m/long", ctx_len=65536),
            make_entry(id="m/unknown-ctx", ctx_len=0),
        ]
        recs = recommend(catalog, hw_low_end, limit=10, min_ctx=8192)
        ids = {r.entry.id for r in recs}
        assert "m/short" not in ids
        assert {"m/long", "m/unknown-ctx"} <= ids

    def test_limit(self, hw_low_end):
        catalog = [make_entry(id=f"m/model-{i}", params_b=0.6, size_gb=0.4, ctx_len=32768) for i in range(20)]
        recs = recommend(catalog, hw_low_end, limit=5)
        assert len(recs) == 5

    def test_unfit_sorted_after_fit(self, hw_low_end):
        catalog = [
            make_entry(id="m/unfit-high", params_b=1.0, size_gb=9.0, ctx_len=131072),
            make_entry(id="m/fit-low", params_b=0.6, size_gb=0.45, ctx_len=8192),
        ]
        recs = recommend(catalog, hw_low_end, limit=10, min_ctx=0)
        assert recs[0].entry.id == "m/fit-low"
        assert recs[-1].entry.id == "m/unfit-high"

    def test_summarize_hw_delegates(self, hw_with_gpu):
        assert "GPU" in summarize_hw(hw_with_gpu) or "无独立 GPU" in summarize_hw(hw_with_gpu)


# --------------------------------------------------------------------------- #
# 7. 缓存读写往返
# --------------------------------------------------------------------------- #
class TestCache:
    def test_roundtrip(self, tmp_path):
        entries = [
            make_entry(id="A/x", size_gb=0.45),
            make_entry(id="B/y", size_gb=1.2, ctx_len=131072, tags=["gguf", "instruct", "extra"]),
        ]
        cache = tmp_path / "catalog.json"
        save_cached_catalog(cache, entries, fetched_at="2025-01-01T00:00:00+00:00")
        assert cache.exists()
        raw = json.loads(cache.read_text(encoding="utf-8"))
        assert raw["fetched_at"] == "2025-01-01T00:00:00+00:00"
        assert raw["count"] == 2
        loaded = load_cached_catalog(cache)
        assert len(loaded) == 2
        by_id = {e.id: e for e in loaded}
        assert by_id["B/y"].ctx_len == 131072
        assert by_id["A/x"].size_gb == pytest.approx(0.45)
        assert by_id["B/y"].tags == ["gguf", "instruct", "extra"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_cached_catalog(tmp_path / "nope.json") == []

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json", encoding="utf-8")
        assert load_cached_catalog(p) == []


# --------------------------------------------------------------------------- #
# 8. 字段识别工具
# --------------------------------------------------------------------------- #
class TestDetectors:
    def test_detect_quant(self):
        assert detect_quant("qwen3-0.6b-q4_k_m.gguf") == "Q4_K_M"
        assert detect_quant("model-Q6_K.gguf") == "Q6_K"
        assert detect_quant("model.Q8_0.gguf") == "Q8_0"
        assert detect_quant("plain.gguf") == ""

    def test_detect_params(self):
        assert detect_params_b("qwen3-0.6b") == pytest.approx(0.6)
        assert detect_params_b("llama-1350m") == pytest.approx(1.35)
        assert detect_params_b("no-number") == 0.0

    def test_detect_ctx_len(self):
        assert detect_ctx_len("model-128k") == 131072
        assert detect_ctx_len("model-32768") == 32768
        assert detect_ctx_len("nothing") == 0

    def test_detect_family(self):
        assert detect_family("Qwen/Qwen3-0.6B") == "qwen"
        assert detect_family("google/gemma-2-2b") == "gemma"
        assert detect_family("meta-llama/Llama-3.2-1B") == "llama"
        assert detect_family("unknown/whatever") == ""


# --------------------------------------------------------------------------- #
# 9. fetch 函数的优雅降级（不联网）
# --------------------------------------------------------------------------- #
class TestFetchDegrade:
    def test_hf_fetch_network_error_returns_empty(self, monkeypatch):
        def boom(url, timeout=None, headers=None):
            raise OSError("network unreachable")

        monkeypatch.setattr("imdeploy.catalog._http_get_json", boom)
        assert fetch_hf_models("gguf qwen3", 5, with_details=False) == []

    def test_modelscope_fetch_network_error_returns_empty(self, monkeypatch):
        def boom(url, timeout=None, headers=None):
            raise OSError("network unreachable")

        monkeypatch.setattr("imdeploy.catalog._http_get_json", boom)
        assert fetch_modelscope_models("gguf qwen3", 5) == []

    def test_hf_fetch_parses_fake_response(self, monkeypatch):
        payload = [make_hf_model(), make_hf_model(model_id="B/no-gguf", gguf_name="model.safetensors")]

        def fake_get(url, timeout=None, headers=None):
            return payload

        monkeypatch.setattr("imdeploy.catalog._http_get_json", fake_get)
        entries = fetch_hf_models("gguf qwen3", 5, with_details=False)
        assert len(entries) == 1
        assert entries[0].id == "Qwen/Qwen3-0.6B-GGUF"

    def test_modelscope_fetch_parses_fake_response(self, monkeypatch):
        raw = {"Data": {"Model": {"Models": [make_modelscope_model()]}}}

        def fake_get(url, timeout=None, headers=None):
            if "repo/files" in url:
                return {"Data": {"Files": make_modelscope_model()["Files"]}}
            return raw

        monkeypatch.setattr("imdeploy.catalog._http_get_json", fake_get)
        entries = fetch_modelscope_models("gguf qwen3", 5)
        assert len(entries) == 1
        assert entries[0].source == "modelscope"
        assert entries[0].quant == "Q4_K_M"


# --------------------------------------------------------------------------- #
# 10. TIERS 常量
# --------------------------------------------------------------------------- #
def test_tiers_constants():
    assert TIERS == ("首选", "备选", "偏大", "偏小", "不适合")
