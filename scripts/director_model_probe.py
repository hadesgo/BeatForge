"""Verify the director's remote-code model on CPU with a few million parameters.

Spark-X2.5 ships ``modeling_spark.py`` written against transformers 4.57 while the
project pins transformers >= 5.16, so the remote file has to be rewritten before
it can run at all (see ``_repair_remote_model_code``). That rewrite is a regex
over the model source, and everything it gets wrong only shows up at the first
forward pass - which normally needs the 7.7GB checkpoint and a free GPU.

This probe shrinks the published config to a few million parameters and runs the
*real* remote code on CPU, so the mask plumbing is checked without weights:

  * eager masks   - the remote attention adds a 4D float mask by hand, so the
                    model must not silently resolve to sdpa (whose mask is None)
  * causality     - a prefix forward must reproduce the matching logits slice
  * sliding window- a token older than ``sliding_window`` must be invisible
  * cache agreement - cached and cache-free greedy decoding must agree

Run: uv run python scripts/director_model_probe.py [model_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beatforge.config import AIConfig  # noqa: E402
from beatforge.models.ai_director import _repair_remote_model_code  # noqa: E402

# Small enough to build and run on CPU in seconds, but it keeps every structural
# feature that matters: a hybrid sliding/full layer stack, GQA, and the partial
# rotary factor that differs between the two layer types.
TINY = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "sliding_window": 4,
}
TINY_ROPE = {
    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 5000000},
    "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
}
LAYER_TYPES = ("sliding_attention", "full_attention", "sliding_attention", "full_attention")
SEQUENCE = 12
PREFIX = 7


def _find_model_dir(repo_id: str) -> Path | None:
    """Locate a downloaded snapshot of ``repo_id`` without hitting the network."""
    candidates = [
        *Path.cwd().glob(f"**/.beatforge/models/modelscope/{repo_id}"),
        *Path.home().glob(f".cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots/*"),
    ]
    for candidate in candidates:
        if (candidate / "modeling_spark.py").exists():
            return candidate
    return None


def _shrink(config, layer_types: tuple[str, ...]) -> None:
    """Apply TINY to a loaded Spark config in place."""
    for name, value in TINY.items():
        setattr(config, name, value)
    config.num_hidden_layers = len(layer_types)
    config.layer_types = list(layer_types)
    config.rope_parameters = dict(TINY_ROPE)
    config.tie_word_embeddings = True


def _greedy(model, ids, steps: int, use_cache: bool) -> list[int]:
    import torch

    current, generated = ids, []
    for _ in range(steps):
        with torch.inference_mode():
            out = model(
                input_ids=current,
                attention_mask=torch.ones_like(current),
                use_cache=use_cache,
            )
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(nxt))
        current = torch.cat([current, nxt], dim=1)
    return generated


def main() -> int:
    import torch
    import transformers

    config_default = AIConfig()
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_model_dir(config_default.director_model)
    if model_dir is None or not model_dir.exists():
        print(f"未找到 {config_default.director_model} 的本地快照，跳过（可显式传入路径）")
        return 0

    print(f"模型目录 {model_dir}")
    print(f"transformers {transformers.__version__} · torch {torch.__version__}")
    _repair_remote_model_code(str(model_dir))

    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )
    _shrink(config, LAYER_TYPES)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()

    ids = torch.randint(0, TINY["vocab_size"], (1, SEQUENCE))
    attention_mask = torch.ones_like(ids)
    failures: list[str] = []

    # 1. The remote attention adds a materialised 4D mask by hand. On sdpa the
    #    mask helper returns None and attention would quietly become unmasked.
    print(f"注意力实现 {config._attn_implementation}")
    if config._attn_implementation != "eager":
        failures.append("注意力实现不是 eager，远程注意力会拿到 None 掩码")

    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
        prefix = model(
            input_ids=ids[:, :PREFIX], attention_mask=attention_mask[:, :PREFIX], use_cache=False,
        ).logits
    delta = (full[:, :PREFIX] - prefix).abs().max().item()
    print(f"因果性      前 {PREFIX} 个位置的最大偏差 {delta:.3e}")
    if delta != 0:
        failures.append("掩码没有挡住未来位置（前缀不一致）")

    # 2. With a single sliding layer, a token outside the window cannot influence
    #    the last position, while the last token itself must. Stacking sliding
    #    layers widens the receptive field by one window per layer, so the layer
    #    count has to stay at 1 for the window to be the only thing under test.
    sliding_config = transformers.AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
    )
    _shrink(sliding_config, ("sliding_attention",))
    sliding_model = transformers.AutoModelForCausalLM.from_config(sliding_config, trust_remote_code=True)
    sliding_model.eval()
    with torch.inference_mode():
        base = sliding_model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits[:, -1]
        far_ids = ids.clone()
        far_ids[0, 0] = (far_ids[0, 0] + 1) % TINY["vocab_size"]
        far = sliding_model(input_ids=far_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1]
        near_ids = ids.clone()
        near_ids[0, -1] = (near_ids[0, -1] + 1) % TINY["vocab_size"]
        near = sliding_model(input_ids=near_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1]
    window = TINY["sliding_window"]
    far_delta = (base - far).abs().max().item()
    near_delta = (base - near).abs().max().item()
    print(f"滑动窗口    窗口外 token 影响 {far_delta:.3e} · 窗口内 token 影响 {near_delta:.3e}")
    if far_delta != 0:
        failures.append(f"第 0 个 token 越过了 {window} 的滑动窗口")
    if near_delta == 0:
        failures.append("滑动窗口把窗口内的 token 也挡住了")

    # 3. Decoding rebuilds its mask from the cache instead of `cache_position`,
    #    which is the argument the 5.x repair has to drop from `mask_kwargs`.
    cached = _greedy(model, ids, 5, use_cache=True)
    uncached = _greedy(model, ids, 5, use_cache=False)
    print(f"缓存一致性  带缓存 {cached} · 无缓存 {uncached}")
    if cached != uncached:
        failures.append("带缓存与无缓存的贪心解码结果不一致")

    print()
    if failures:
        for failure in failures:
            print(f"失败：{failure}")
        return 1
    print("全部通过：远程代码在已安装的 transformers 上可运行，且掩码语义正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
