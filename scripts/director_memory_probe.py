"""Print the director's sequence-side memory estimate for representative prompts.

The director is the only stage that feeds a very long sequence to a language
model, and Spark-X2.5 computes attention with plain matmuls (no SDPA/flash-attn),
so every layer materialises a full prompt x prompt score matrix. That cost grows
with the square of the prompt, so this probe shows why the prompt cap exists and
how much of a 12GB card is left for weights at different prompt lengths.

Run: uv run python scripts/director_memory_probe.py

This probe only does the arithmetic; `scripts/director_model_probe.py` is the one
that actually runs the remote model code (on CPU, with a few million parameters)
to check the masks. Neither needs the 7.7GB checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import AIConfig  # noqa: E402
from beatforge.models.ai_director import (  # noqa: E402
    DIRECTOR_OVERHEAD_GB,
    MIN_DIRECTOR_RESERVE_GB,
    REPAIR_TOKENS,
    RETRY_BUDGET_SCALE,
    _sequence_reserve_gb,
)

# XHToken/Spark-X2.5-4B config.json, verified against the published repository.
SPARK = SimpleNamespace(
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=4,
    hidden_size=2560,
    head_dim=256,
    intermediate_size=10240,
    vocab_size=131072,
    sliding_window=512,
    layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 9,
    torch_dtype="bfloat16",
)

CARD_GB = 12.0
# A 12GB card under a Windows desktop typically reports ~10GiB free, which is
# what ``torch.cuda.mem_get_info`` sees and what the budget is actually based on.
FREE_GB = 10.0
# (label, prompt tokens) - max_new_tokens comes from AIConfig.
CASES = (
    ("修复前 · 默认质量组合（实测 23218 字符）", 13000),
    ("修复前 · 长歌曲", 15500),
    ("修复后 · 默认上限 2600 命中阶梯末级", 2581),
    ("修复后 · CPU 兼容配置上限 2000", 2000),
    ("修复后 · 极简提示词", 1031),
)


def main() -> int:
    config = AIConfig()
    full_layers = sum(1 for kind in SPARK.layer_types if kind == "full_attention")
    print(f"显卡 {CARD_GB:.0f}GB · 实测空闲 {FREE_GB:.1f}GB · "
          f"director_gpu_memory_gb = {config.director_gpu_memory_gb:.1f} · "
          f"max_new_tokens = {config.director_max_new_tokens}")
    print(f"Spark-X2.5-4B：{SPARK.num_hidden_layers} 层（{full_layers} 全注意力 + "
          f"{SPARK.num_hidden_layers - full_layers} 滑动窗口 {SPARK.sliding_window}）· "
          f"{SPARK.num_attention_heads} 注意力头 / {SPARK.num_key_value_heads} KV 头 · "
          f"head_dim {SPARK.head_dim} · bf16")
    print()
    header = f"{'场景':<36}{'提示词':>8}{'序列预留':>10}{'权重预算':>10}  结论"
    print(header)
    print("-" * len(header))

    failures = 0
    for label, prompt_tokens in CASES:
        reserve = _sequence_reserve_gb(config, SPARK, prompt_tokens)
        # Mirrors _gpu_budgets: free memory minus the sequence reserve, capped by
        # the configured ceiling.
        budget = max(1.0, min(config.director_gpu_memory_gb, FREE_GB - reserve))
        retry = max(1.0, budget * RETRY_BUDGET_SCALE)
        if budget < 4.0:
            verdict = f"权重装不下 → OOM（重试仅 {retry:.1f}GiB）"
            failures += 1
        elif budget < config.director_gpu_memory_gb:
            verdict = "需卸载部分权重"
        else:
            verdict = "可全量驻留"
        print(f"{label:<36}{prompt_tokens:>8}{reserve:>9.1f}G{budget:>9.1f}G  {verdict}")

    print()
    print(f"序列预留 = 注意力分数矩阵 + 掩码 + 保守的 prefill logits + KV 缓存 + 激活，"
          f"再加固定开销 {DIRECTOR_OVERHEAD_GB:.1f}GiB，下限 {MIN_DIRECTOR_RESERVE_GB:.1f}GiB；"
          f"JSON 修复轮次另按 +{REPAIR_TOKENS} tokens 计入。")
    print(f"权重预算按空闲显存而非显卡总容量计算，因此先关掉其他占显存的程序更有效。")
    print(f"以上 {len(CASES)} 个场景中，有 {failures} 个在修复前的提示词长度下必然 OOM。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
