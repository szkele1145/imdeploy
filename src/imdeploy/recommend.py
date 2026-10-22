"""推荐引擎：按本机硬件给模型打分、分档，并给出建议上下文长度。

核心思路（用户明确要求优化 0-2B 小模型的上下文与记忆能力）：
- 0-2B 小模型加权最高，参数量越大得分越低；
- 上下文长度是核心指标：32K 起加分，64K 更高，128K 最高；
- 内存必须装得下（size_gb ≤ 可用内存 × 75%），装不下直接 fit=False；
- 量化偏好 Q4_K_M / Q5_K_M / Q6_K（长上下文下质量不塌），Q4_0 降分；
- instruct/chat 类加分（对话记忆好），base/pretrained 降分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .catalog import ModelEntry
from .hardware import HardwareInfo, summarize as _hw_summarize

__all__ = [
    "Recommendation",
    "TIERS",
    "score_model",
    "recommend",
    "summarize_hw",
    "estimate_download_gb",
    "recommended_context_for",
]

#: 分档名称（中文）
TIERS: tuple[str, ...] = ("首选", "备选", "偏大", "偏小", "不适合")

#: 内存余量系数：模型体积必须 ≤ 可用内存 × 该系数
RAM_BUDGET_RATIO = 0.75
#: 小模型优化的上限（B）
SMALL_MODEL_MAX_B = 2.0
#: KV 缓存粗略预算：每 1B 参数、每 32K 上下文约占的内存（GB）
KV_GB_PER_B_PER_32K = 0.20


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    """一条推荐结果。"""

    entry: ModelEntry
    score: float = 0.0
    tier: str = ""
    reasons: list[str] = field(default_factory=list)
    estimated_size_gb: float = 0.0
    recommended_ctx_len: int = 0
    fit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": round(self.score, 2),
            "tier": self.tier,
            "reasons": list(self.reasons),
            "estimated_size_gb": round(self.estimated_size_gb, 2),
            "recommended_ctx_len": self.recommended_ctx_len,
            "fit": self.fit,
        }


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _mem_budget(hw: HardwareInfo) -> float:
    """模型体积上限 = 可用内存 × 0.75（可用内存未知时退回总内存）。"""
    avail = hw.available_ram_gb or hw.total_ram_gb or 0.0
    return avail * RAM_BUDGET_RATIO


def _tags_of(entry: ModelEntry) -> list[str]:
    return [_norm(t) for t in (entry.tags or [])]


def _quant_normalized(quant: str) -> str:
    return _norm(quant).replace(" ", "_").upper()


#: 上下文加分阶梯：逐级叠加
_CTX_LADDER: tuple[tuple[int, float, str], ...] = (
    (8192, 1.0, "支持 8K 上下文"),
    (16384, 2.0, "支持 16K 上下文"),
    (32768, 4.0, "支持 32K 长上下文"),
    (65536, 7.0, "支持 64K 超长上下文"),
    (131072, 11.0, "支持 128K 超长上下文（顶级）"),
)

#: 量化偏好评分
_QUANT_SCORES: dict[str, float] = {
    "Q4_K_M": 10.0,
    "Q5_K_M": 9.5,
    "Q6_K": 9.0,
    "Q5_K_S": 8.0,
    "Q4_K_S": 7.5,
    "IQ4_XS": 7.0,
    "Q8_0": 5.0,
    "Q5_0": 4.0,
    "Q4_0": 2.0,
    "Q3_K_M": 1.0,
    "Q2_K": -1.0,
    "F16": -2.0,
    "BF16": -2.0,
    "F32": -4.0,
}

#: 按量化格式估算的每 B 参数体积（GB），用于 size_gb 缺失时
_GB_PER_PARAM: dict[str, float] = {
    "Q2_K": 0.35,
    "Q3_K_M": 0.45,
    "Q4_0": 0.55,
    "Q4_K_S": 0.58,
    "Q4_K_M": 0.60,
    "IQ4_XS": 0.58,
    "Q5_0": 0.68,
    "Q5_K_M": 0.70,
    "Q6_K": 0.80,
    "Q8_0": 1.00,
    "F16": 2.00,
    "BF16": 2.00,
    "F32": 4.00,
}
_DEFAULT_GB_PER_PARAM = 0.65


def _ctx_text_bonus(entry: ModelEntry) -> tuple[float, str]:
    """名称/描述里含 “128k / 32k / long context” 等字样时补一点分。"""
    haystack = f"{entry.name} {entry.description}".lower()
    matched: list[str] = []
    bonus = 0.0
    if re.search(r"(?<![a-z0-9])128\s*k\b", haystack):
        bonus += 2.0
        matched.append("宣传 128K 上下文")
    if re.search(r"(?<![a-z0-9])256\s*k\b", haystack):
        bonus += 1.0
        matched.append("宣传 256K 上下文")
    if re.search(r"(?<![a-z0-9])(?:32|64)\s*k\b", haystack):
        bonus += 1.0
        matched.append("宣传长上下文")
    if re.search(r"long[ _-]?context|长上下文|超长上下文", haystack):
        bonus += 1.5
        matched.append("主打长上下文")
    return bonus, "、".join(matched)


def _params_score(params_b: float) -> tuple[float, str]:
    """参数量得分：0-2B 加权最高（用户核心诉求），越大递减。"""
    if params_b <= 0:
        return 3.0, ""
    if params_b <= 1.0:
        return 18.0, "0-1B 超小模型，本地推理极快"
    if params_b <= SMALL_MODEL_MAX_B:
        return 16.0, "0-2B 小模型，贴合低配硬件（重点优化档）"
    if params_b <= 4.0:
        return 11.0, "2-4B 模型，多数设备可跑"
    if params_b <= 8.0:
        return 7.0, "4-8B 模型，需要中配硬件"
    if params_b <= 13.0:
        return 4.0, "8-13B 模型，偏大"
    if params_b <= 32.0:
        return 1.0, "13-32B 模型，需要高端硬件"
    return -6.0, "32B+ 模型，本地部署成本很高"


def _ctx_score(ctx_len: int) -> tuple[float, list[str]]:
    """上下文得分（核心指标）：32K 起显著加分，64K 更高，128K 最高。"""
    score = 0.0
    reasons: list[str] = []
    for threshold, points, label in _CTX_LADDER:
        if ctx_len >= threshold:
            score += points
            if points >= 4.0:
                reasons.append(label)
    if 0 < ctx_len < 8192:
        score -= 1.5
        reasons.append(f"上下文仅约 {ctx_len // 1024}K，偏短")
    return score, reasons


def _gpu_bonus(entry: ModelEntry, hw: HardwareInfo, estimated_size_gb: float) -> tuple[float, list[str]]:
    """GPU 相关加分：显存装得下模型时按后端与量化奖励。"""
    bonuses: list[str] = []
    total = 0.0
    backend = _norm(hw.gpu_backend)
    vram = hw.gpu_vram_gb or 0.0
    if backend in ("nvidia", "vulkan", "rocm") and vram > 0 and estimated_size_gb > 0:
        if estimated_size_gb <= vram * 0.9:
            total += 8.0
            bonuses.append("模型可完整装进 GPU 显存，推理速度快")
            if backend == "nvidia":
                total += 2.0
                bonuses.append("NVIDIA CUDA 加速")
            elif backend == "rocm":
                total += 1.0
                bonuses.append("ROCm 加速")
            quant = _quant_normalized(entry.quant)
            if quant in ("Q4_K_M", "Q5_K_M", "Q6_K"):
                total += 1.0
                bonuses.append("量化格式适合常驻 GPU 显存")
            elif quant == "Q8_0":
                total -= 1.0
                bonuses.append("Q8_0 量化体积偏大，需更多显存")
        elif estimated_size_gb <= vram:
            total += 3.0
            bonuses.append("模型体积大于显存可直接承载的部分，需要 CPU+GPU 混合推理")
        else:
            total -= 3.0
            bonuses.append("模型大于显存，需要 CPU+GPU 混合推理，速度会下降")
    return total, bonuses


def _fit_check(estimated_size_gb: float, hw: HardwareInfo) -> tuple[bool, list[str]]:
    """内存匹配检查：体积超出预算（可用内存×75%）→ fit=False。"""
    reasons: list[str] = []
    budget = _mem_budget(hw)
    if budget <= 0:
        return True, ["未探测到内存信息，跳过内存匹配"]
    if estimated_size_gb <= 0:
        return True, ["模型体积未知，跳过内存匹配"]
    if estimated_size_gb > budget:
        reasons.append(
            f"需要约 {estimated_size_gb:.1f}GB，超出可用内存预算 {budget:.1f}GB（75%）"
        )
        return False, reasons
    if estimated_size_gb > budget * 0.85:
        reasons.append("体积接近内存预算上限，建议控制并发与上下文长度")
    return True, reasons


# --------------------------------------------------------------------------- #
# 公开估算函数
# --------------------------------------------------------------------------- #
def estimate_download_gb(entry: ModelEntry, hw: HardwareInfo | None = None) -> float:
    """估算模型实际占用体积（GB）。

    有 size_gb 用 size_gb；没有则按参数量与量化位数粗估（含 20% 冗余）。
    """
    if entry.size_gb > 0:
        return round(entry.size_gb, 3)
    if entry.params_b > 0:
        per_b = _GB_PER_PARAM.get(_quant_normalized(entry.quant), _DEFAULT_GB_PER_PARAM)
        return round(entry.params_b * per_b * 1.2, 2)
    return 0.0


def _estimate_params_b_fallback(entry: ModelEntry) -> float:
    """参数量未知时，用体积与量化格式反推。"""
    if entry.params_b > 0:
        return entry.params_b
    per_b = _GB_PER_PARAM.get(_quant_normalized(entry.quant), _DEFAULT_GB_PER_PARAM)
    if entry.size_gb > 0 and per_b > 0:
        return round(entry.size_gb / per_b, 3)
    return 0.0


def recommended_context_for(entry: ModelEntry, hw: HardwareInfo) -> int:
    """在内存预算内为模型推荐一个上下文长度。

    返回值不会超过模型自身声明的 ctx_len；预算不足时保底 2048（或更小）。
    """
    declared = entry.ctx_len or 0
    if declared <= 0:
        return 0
    budget = _mem_budget(hw)
    weights_gb = estimate_download_gb(entry, hw)
    params_b = entry.params_b if entry.params_b > 0 else _estimate_params_b_fallback(entry)
    if budget <= 0 or params_b <= 0 or weights_gb <= 0:
        return min(declared, 8192)
    kv_gb_per_32k = params_b * KV_GB_PER_B_PER_32K
    headroom = max(budget - weights_gb, 0.0)
    if kv_gb_per_32k <= 0:
        return min(declared, 8192)
    max_ctx_budget = int(headroom / kv_gb_per_32k * 32768)
    cap = min(declared, 131072 if params_b <= SMALL_MODEL_MAX_B else 32768)
    result = min(declared, cap, max_ctx_budget)
    floor = min(declared, 2048)
    return max(floor, result)


# --------------------------------------------------------------------------- #
# 打分与分档
# --------------------------------------------------------------------------- #
def _tier(
    score: float,
    fit: bool,
    params_b: float,
    estimated_size_gb: float,
    hw: HardwareInfo,
    ctx_len: int,
) -> str:
    """把分数翻译成中文分档：首选 / 备选 / 偏大 / 偏小 / 不适合。"""
    if not fit:
        return "不适合"
    budget = _mem_budget(hw)
    size_ratio = (estimated_size_gb / budget) if budget > 0 else 0.0
    if params_b > 8.0 or size_ratio > 0.85:
        return "偏大"
    if 0 < params_b < 0.5 and ctx_len < 16384:
        return "偏小"
    if score >= 70.0:
        return "首选"
    return "备选"


def score_model(entry: ModelEntry, hw: HardwareInfo, ctx_preference: str = "balanced") -> Recommendation:
    """给单个模型打分（0-100），生成推荐理由、建议上下文与分档。

    ``ctx_preference`` 可选：
    - ``balanced``：默认权重；
    - ``long``：放大长上下文权重（1.5 倍）；
    - ``short``：弱化上下文权重（0.5 倍），偏向轻量部署。
    """
    reasons: list[str] = []
    estimated_size_gb = estimate_download_gb(entry, hw)
    fit, fit_reasons = _fit_check(estimated_size_gb, hw)
    reasons.extend(fit_reasons)

    # --- 参数量（0-2B 加权，核心诉求）---
    params_b = entry.params_b if entry.params_b > 0 else _estimate_params_b_fallback(entry)
    p_score, p_reason = _params_score(params_b)
    if p_reason:
        reasons.append(p_reason)

    # --- 上下文（核心指标）---
    ctx_len = entry.ctx_len or 0
    c_score, c_reasons = _ctx_score(ctx_len)
    reasons.extend(c_reasons)
    ctx_pref = _norm(ctx_preference)
    if ctx_pref == "long":
        c_score = c_score * 1.5
        if ctx_len >= 65536:
            reasons.append("按“长上下文优先”加权")
    elif ctx_pref == "short":
        c_score = c_score * 0.5

    # --- 量化 ---
    quant = _quant_normalized(entry.quant)
    q_score = _QUANT_SCORES.get(quant, 0.0)
    if quant == "Q4_0" and ctx_len >= 32768:
        q_score -= 2.0
        reasons.append("Q4_0 在长上下文下质量易塌，降分")
    elif quant in ("Q4_K_M", "Q5_K_M", "Q6_K"):
        reasons.append(f"{quant} 量化，质量与体积均衡（长上下文友好）")
    elif quant == "Q8_0":
        reasons.append("Q8_0 接近无损，但体积约为 Q4 的两倍")

    # --- GPU ---
    g_score, g_reasons = _gpu_bonus(entry, hw, estimated_size_gb)
    reasons.extend(g_reasons)

    # --- tags 与文本信号：instruct/chat 加分，base 降分 ---
    tags = _tags_of(entry)
    t_score = 0.0
    if any(t in ("instruct", "chat", "it", "sft") for t in tags):
        t_score += 6.0
        reasons.append("带 instruct/chat 标签，对话与记忆能力好")
    if any(t in ("base", "pretrained") for t in tags):
        t_score -= 6.0
        reasons.append("base/pretrained 模型，指令遵循弱，长对话记忆差")
    t_bonus, t_match = _ctx_text_bonus(entry)
    if t_bonus:
        t_score += t_bonus
        if t_match:
            reasons.append(t_match)

    # --- 汇总（基准 50 分）---
    raw = 50.0 + p_score + c_score + q_score + g_score + t_score
    if not fit:
        # 装不下就是装不下：分数归零，避免排在"备选"里误导用户
        raw = 0.0
    score = max(0.0, min(100.0, raw))

    tier = _tier(score, fit, params_b, estimated_size_gb, hw, ctx_len)

    rec_ctx = recommended_context_for(entry, hw)
    if rec_ctx > 0 and fit:
        reasons.append(f"建议上下文长度约 {rec_ctx // 1024}K")

    return Recommendation(
        entry=entry,
        score=round(score, 2),
        tier=tier,
        reasons=reasons,
        estimated_size_gb=estimated_size_gb,
        recommended_ctx_len=rec_ctx,
        fit=fit,
    )


# --------------------------------------------------------------------------- #
# 推荐入口
# --------------------------------------------------------------------------- #
def recommend(
    catalog: Iterable[ModelEntry],
    hw: HardwareInfo,
    limit: int = 12,
    min_ctx: int = 8192,
    ctx_preference: str = "balanced",
) -> list[Recommendation]:
    """对整个目录打分，返回按分数降序的推荐列表。

    - ``min_ctx``：过滤掉已知上下文且小于该值的模型（未知 ctx_len 的保留）；
    - ``fit=False``（内存装不下）的排到列表末尾；
    - ``limit``：最多返回多少条。
    """
    scored: list[Recommendation] = []
    for entry in catalog:
        if not isinstance(entry, ModelEntry):
            continue
        if 0 < (entry.ctx_len or 0) < min_ctx:
            continue
        scored.append(score_model(entry, hw, ctx_preference))

    scored.sort(key=lambda r: (not r.fit, -r.score))
    if limit and limit > 0:
        scored = scored[:limit]
    return scored


def summarize_hw(hw: HardwareInfo) -> str:
    """人类可读的中文硬件摘要（一句话）。"""
    return _hw_summarize(hw)
