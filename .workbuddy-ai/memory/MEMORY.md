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

## 图片运镜（`beatforge/renderer.py::_image_filter_graph`）
- **禁止再用 `zoompan` 做图片运镜**。它把 `x`/`y` 截断到输入帧的整数像素，而本项目运镜只有
  0.03–0.1 px/帧，结果是"连续多帧静止 + 突然跳 1 像素"的顿挫。改用
  `perspective=...:interpolation=cubic:sense=source:eval=frame`，逐帧求值裁剪四边形，亚像素重采样。
- `perspective` 的表达式**只暴露 `on`（帧序号）**，不暴露 `t`；用 `t` 直接报错。
- 几何量别写反：**裁剪框宽 = `W/zoom`**，**可平移量程 = `W - W/zoom`**。写反会变成极端硬推镜。
- 回归测试在 `tests/test_renderer.py`：断言无 zoompan、裁剪框是矩形且不越界、保持宽高比、
  相邻帧步长不超过 `max(平均步长*6, 0.5px)`。改运镜必须让这些测试继续通过。

## 运镜抖动怎么测（别拿成片直接测）
- 相位相关测帧间位移的前提是**相邻帧互为刚体变换**。成片里的暗角、颗粒、调色固定在画面坐标上，
  会破坏这一前提，使单帧步长正负相消、读数全是噪声——曾因此在真实片段上报出 58/101 静止帧、
  1.75px 抖动的假结论（一致性只有 0.033）。
- 判据是**一致性** `|Σstep| / Σ|step|`：≈1 才有效，< 0.5 时脚本直接判"测量不可信"。
  `scripts/jitter_probe.py` 与技能的 `subpixel_motion.py` 都已内置该判据。
- 要测真实运镜是否平滑，用 `scripts/camera_move_probe.py --controlled`（关掉颗粒/暗角/调色、
  单层铺满）或 `scripts/zoompan_probe.py`。可信结论：旧 zoompan 抖动 1.67px / 比值 18.6
  → 新 perspective 0.117px / 比值 1.5。
- 自己写测量时的三个硬坑：float32 会让 FFT 退化成 complex64 且 DC 基底淹没真实峰（必须去均值
  + float64）；三点抛物线拟合峰值在小平移下低偏差约 40%（用局部上采样 DFT）；测试卡要用类照片的
  1/f 噪声且中灰，不能用平滑噪声或条纹。

## 本机环境注意
- `uv run pytest` 默认临时目录无权限，需要 `--basetemp=<可写目录> -p no:cacheprovider`。
- 无 NVIDIA 显卡：AI 链路用 `--no-ai` 验证，模型相关测试用 mock。显存相关改动靠单元测试 + 数值估算验证，不做 GPU 实机验证。
- `AIConfig.device` 只允许 `auto|cuda|cpu`，无法选多卡；但 `beatforge/models/` 下多处仍用
  `device == "cuda"` 判断（`transcriber.py`、`vision_index.py`、`audio_semantics.py`），
  一旦放开 `cuda:N` 会静默退回 fp32/CPU。导演模块已改为 `_is_cuda()` 前缀匹配，其余待跟进。
