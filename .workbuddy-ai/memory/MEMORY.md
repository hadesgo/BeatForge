# BeatForge 项目长期记忆

## 设计约定
- 选镜（`beatforge/planner.py`）的素材复用策略：按"已出现次数"分层，优先最少使用层；
  素材够用（最少使用层数量 ≥ 剩余镜头数）时保证零复用。多图合成属于"奢侈品"，
  只在 `富余素材 = 最少使用层数量 - 剩余镜头数 >= 0` 时消耗富余素材。
  开关为 `render.avoid_asset_repeats`（默认 true）。细节见 README「素材复用规则」。
- 规划器保持确定性：不使用随机数，平票靠素材发现顺序（列表顺序）决出，便于测试与复现。
- 所有剪辑决策都要写进 `.beatforge/plan.json`，方便人工复核；新增 render 配置会自动出现在其中。

## AI 导演（`beatforge/models/ai_director.py`）的显存设计
Spark-X2.5-4B 的 remote code 有两个必须靠外部兜住的特性，任何"把长序列喂给它"的改动都要重新核对：
1. 注意力是手写 `torch.matmul` + softmax，**没有 SDPA / flash-attn 后端**；滑动窗口层
   （27/36 层，window=512）是给完整的 `[heads, n, n]` 分数矩阵加掩码，**不切 KV**。
   分数矩阵峰值按 `heads * n² * 6 字节`（bf16 矩阵 + fp32 softmax 副本）估算。
2. 未定义 `prepare_inputs_for_generation`，`logits_to_keep` 默认 0，prefill 会对
   **每个 prompt 位置**跑 lm_head：`n * vocab * dtype_bytes`（vocab=131072）。
以上两项都随提示词长度**平方**增长，所以提示词必须限长。

- `AIConfig.director_prompt_tokens`（默认 2600）限制提示词长度；`PROMPT_LADDER` 逐级降级
  （先裁候选表 → 再减素材数 → 最后才对歌词抽样），每级用 tokenizer 实测。
  裁掉候选表时 `_trim_context` 会同步改写 `instruction`，不能指向已删除的表。
- `_sequence_reserve_gb` 按上面两个公式 + KV + 激活估算序列侧开销；
  `_gpu_budgets` 用 **`torch.cuda.mem_get_info` 的空闲显存**减预留，再与
  `director_gpu_memory_gb` 取小。Accelerate 的 `max_memory` 只预算权重，不能替代这一步。
- OOM 时自动用 `RETRY_BUDGET_SCALE`（0.65）更小的权重预算重试一次。
- `scripts/director_memory_probe.py` 可在无显卡、不加载模型的情况下打印各提示词长度的预留量。

## 本机环境注意
- `uv run pytest` 默认临时目录无权限，需要 `--basetemp=<可写目录> -p no:cacheprovider`。
- 无 NVIDIA 显卡：AI 链路用 `--no-ai` 验证，模型相关测试用 mock。显存相关改动靠单元测试 + 数值估算验证，不做 GPU 实机验证。
- `AIConfig.device` 只允许 `auto|cuda|cpu`，无法选多卡；但 `beatforge/models/` 下多处仍用
  `device == "cuda"` 判断（`transcriber.py`、`vision_index.py`、`audio_semantics.py`），
  一旦放开 `cuda:N` 会静默退回 fp32/CPU。导演模块已改为 `_is_cuda()` 前缀匹配，其余待跟进。
