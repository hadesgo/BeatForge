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
2. `logits_to_keep` **不是**瓶颈：remote `forward` 的默认值确实是 0，但 transformers 5.16 的
   `generate` 会在 `_supports_logits_to_keep()` 为真时显式传 `logits_to_keep=1`（已实测 spy 到
   `[1, 1]`），prefill 只对最后一个位置跑 lm_head。`_sequence_reserve_gb` 里的
   `prefill * vocab * dtype_bytes` 现在是**刻意的保守余量**（≈0.66GiB @2600 tokens），
   不是真实开销；不要因为"省显存"就删掉它，删除只会在 transformers 改行为时换来一次 OOM 重载。
   旧结论"prefill 对每个 prompt 位置跑 lm_head"已作废（README 与注释同步改过）。

真正随提示词长度**平方**增长的是 1，所以提示词必须限长。

- `AIConfig.director_prompt_tokens`（默认 2600）限制提示词长度；`PROMPT_LADDER` 逐级降级
  （先裁候选表 → 再减素材数 → 最后才对歌词抽样），每级用 tokenizer 实测。
  裁掉候选表时 `_trim_context` 会同步改写 `instruction`，不能指向已删除的表。
- `_sequence_reserve_gb` 按上面两个公式 + KV + 激活估算序列侧开销；
  `_gpu_budgets` 用 **`torch.cuda.mem_get_info` 的空闲显存**减预留，再与
  `director_gpu_memory_gb` 取小。Accelerate 的 `max_memory` 只预算权重，不能替代这一步。
- OOM 时自动用 `RETRY_BUDGET_SCALE`（0.65）更小的权重预算重试一次。
- `scripts/director_memory_probe.py` 可在无显卡、不加载模型的情况下打印各提示词长度的预留量。

## 远程代码兼容层（`_repair_remote_model_code`）
Spark-X2.5 自带的 `modeling_spark.py` 是按 `transformers==4.57` 写的（`config.json` 里
`transformers_version: 4.57.1`），而项目锁 `transformers>=5.16.1`。加载前必须就地改写本地模型目录里
的这份文件，目前两处：
1. `_tied_weights_keys` 旧列表写法 → `{target: source}` 映射（否则权重加载就炸）。
2. `mask_kwargs` 里 `create_causal_mask(input_embeds=..., cache_position=...)` → 5.16 已改名为
   `inputs_embeds` 并删掉 `cache_position`，否则首次前向报
   `TypeError: create_causal_mask() got an unexpected keyword argument 'input_embeds'`。

- 改写依据是**已安装**的 `create_causal_mask` 签名（`inspect.signature` 逐项过滤 + `MASK_KWARG_ALIASES`
  别名映射），所以新旧 transformers 都对，且幂等；签名里有 `**kwargs` 时一律不动（无法验证就别乱删参数）。
- 只打补丁到 HF 的 modules 缓存目录是没用的：缓存目录名是源码哈希，改了本地文件就会换目录重建。
- **`config._attn_implementation` 必须是 `eager`**。remote 注意力手工加 4D float 掩码，若落到 sdpa，
  `create_causal_mask` 会返回 `None`，注意力**静默失去掩码**。`Spark2_5PreTrainedModel` 没声明
  `_supports_sdpa`，所以现在默认就是 eager；改模型/改 transformers 后要用探针复核这一条。
- `scripts/director_model_probe.py`：把 config 缩到几百万参数后在 **CPU** 上跑同一份远程代码，
  不需要 7.7GB 权重就能验四件事：注意力实现、因果性（前缀一致）、滑动窗口、带缓存/无缓存贪心一致。
  滑动窗口那一项**必须用单层**——感受野 = 层数 × window，多层会掩盖窗口外的 token。

## 提示词长度必须用 `_rendered_token_count` 量
`processor.apply_chat_template(..., tokenize=True)` 返回的是含 `input_ids`/`attention_mask` 的
`BatchEncoding`，`len()` 是**键的个数（2）**，不是 token 数。曾因此把 `_prompt_tokens` 写成
`len(rendered)`，导致：提示词上限全程失效（`_fit_prompt` 每级都判"装得下"，从没裁过）、
序列预留按 2 tokens 算成 1.5GiB 下限，日志打印「导演提示词约 2 tokens · 权重预算 9.0GiB」。
修复后同一场景是「2848 tokens · 预留 3.5GiB · 预算 7.3GiB」。改任何长度测量都要覆盖
BatchEncoding / 张量 / 列表 / 嵌套列表四种形态（测试在 `tests/test_ai_director.py`）。

## 图片运镜（`beatforge/renderer.py::_image_filter_graph`）
- **禁止再用 `zoompan` 做图片运镜**。它把 `x`/`y` 截断到输入帧的整数像素，而本项目运镜只有
  0.03–0.1 px/帧，结果是"连续多帧静止 + 突然跳 1 像素"的顿挫。改用
  `perspective=...:interpolation=cubic:sense=source:eval=frame`，逐帧求值裁剪四边形，亚像素重采样。
- 几何量别写反：**裁剪框宽 = `W/zoom`**，**可平移量程 = `W - W/zoom`**。写反会变成极端硬推镜。
- 运镜参数化在 `_CameraMove`（冻结 dataclass）+ `_CAMERA_MOVES`（12 种）。`zoom_from`/`zoom_to`
  是**增量倍率**（保证 zoom ≥ 1），`x_from`..`y_to` 是**平移量程的比例**（±0.15 以内，避免撞 `clip()` 边界而中途停住）。
- 回归测试在 `tests/test_renderer.py`：断言无 zoompan、裁剪框是矩形且不越界、保持宽高比、
  相邻帧步长不超过 `max(平均步长*6, 0.5px)`、以及**超出镜头长度后几何保持不变**。改运镜必须让这些测试继续通过。

## `perspective` 的三个硬坑（改任何运镜前必读）
1. **`on` 是 1-based，而且会越过镜头长度继续增长。**
   `vf_perspective.c` 里 `VAR_ON = outl->frame_count_in + 1`；静态图来自 `-loop 1`（无限长），
   下游的 `fps`/`trim` 为了确定自己的时间戳会多拉几帧，所以 `on` 会一直涨。
   进度必须写成 **`clip((on-1)/(frames-1),0,1)`**：
   - 少写 `-1` → 末帧过冲一帧；
   - 少写 `clip` → 递减型运镜在末尾**反转**，zoom 掉到 1 以下 → `W-W/zoom` 变负 → 该帧被 ffmpeg 拒绝。
   症状极具误导性：只有 `dolly_out`/`pull_back`/`breathe` 这类**递减**运镜崩，`cinematic_depth` 全正常。
2. **`clip()` 不交换上下界**，它算的是 `min(max(x,min),max)`；上界为负就原样返回负数。
   实测 `perspective=x0='clip(5,0,-1)'` 直接 `Invalid argument`。
   因此每个运镜额外抬高 `_MIN_ZOOM = 0.004` 的放大率，消除"裁剪框正好等于整帧"的退化映射。
3. **`geq` 是另一套方言**：没有 `on`，用 0 基的 `N`。而且逐像素求值，1080p 实测 0.18s/帧（≈5.5fps），
   比其它效果加起来还贵——iris 遮罩改在 1/8 画布上生成再 `scale` 上采样。

## 合成平面与帧数（`_plane_duration`）
- 用 `color` 生成的平面（分屏分隔条、iris 遮罩/底、渐变）经 `overlay=shortest=1` 与画面合成时，
  若长度正好等于镜头时长，它会是**最短流**从而把整镜截短一帧。统一按 `duration + 1/fps` 生成，
  多出的一帧由收尾的 `trim` 切掉。曾因此发现 `split_screen` 一直在丢最后一帧（既有 bug）。
- `tests/test_renderer.py::test_all_still_image_effects_render` 断言帧数**精确相等**，用来守住这条。

## 转场（`beatforge/renderer.py` + `planner.py`）
- 规划器只下发**转场族**（14 族，`_transition_family`），`_transition_spec` 再结合进入镜头的
  `transition_tone` 与离开镜头的漂移方向解析成 39 个具体 `xfade` 名（`_TRANSITION_LIBRARY`）。
- `_TRANSITION_PACE` 逐族定时长；方向性族在 `_TRANSITION_DIRECTION`。未知族名回退成溶解，不是硬切。
- `render.transition_density`（默认 0.35）只管**段落内部**的可见转场比例；段落切换/乐段边界/
  导演冲击点不受约束。`_break_transition_repeats` 防止同一族连续出现（含蓄转场除外）。
- 测试会拿 `ffmpeg -h filter=xfade` 校验名字，并**真的跑一遍**每个转场（`xfade` 的 duration 必须是
  `0.3` 这种写法，`.3` 会报 `Unable to parse "duration" option value`）。

## 运镜抖动怎么测（别拿成片直接测）
- 相位相关测帧间位移的前提是**相邻帧互为刚体变换**。成片里的暗角、颗粒、调色固定在画面坐标上，
  会破坏这一前提，使单帧步长正负相消、读数全是噪声——曾因此在真实片段上报出 58/101 静止帧、
  1.75px 抖动的假结论（一致性只有 0.033）。
- 判据是**一致性** `|Σstep| / Σ|step|`：≈1 才有效，< 0.5 时脚本直接判"测量不可信"。
  `scripts/jitter_probe.py` 已内置该判据。（本文件曾提到"技能的 `subpixel_motion.py`"，
  但本机与项目里都没有任何 skills 目录，该文件不存在——需要时按本文自行重建。）
- 要测真实运镜是否平滑，用 `scripts/camera_move_probe.py --controlled`（关掉颗粒/暗角/调色、
  单层铺满）或 `scripts/zoompan_probe.py`。可信结论：旧 zoompan 抖动 1.67px / 比值 18.6
  → 新 perspective 0.117px / 比值 1.5。
- 自己写测量时的三个硬坑：float32 会让 FFT 退化成 complex64 且 DC 基底淹没真实峰（必须去均值
  + float64）；三点抛物线拟合峰值在小平移下低偏差约 40%（用局部上采样 DFT）；测试卡要用类照片的
  1/f 噪声且中灰，不能用平滑噪声或条纹。

## 本机环境注意
- `uv run pytest` 默认临时目录无权限，需要 `--basetemp=<可写目录> -p no:cacheprovider`。
- **本机有可用 CUDA 显卡**：RTX 5070 12GB（`torch.cuda.is_available()` 为 True，空闲约 10.8GiB），
  但 `nvidia-smi` 会报 `Failed to initialize NVML: Unknown Error`，所以别用 nvidia-smi 判断显存，
  用 `torch.cuda.mem_get_info()`。AI 链路可以真机验证：`--no-ai` 仍是最快的回归方式。
- `AIConfig.device` 只允许 `auto|cuda|cpu`，无法选多卡；但 `beatforge/models/` 下多处仍用
  `device == "cuda"` 判断（`transcriber.py`、`vision_index.py`、`audio_semantics.py`），
  一旦放开 `cuda:N` 会静默退回 fp32/CPU。导演模块已改为 `_is_cuda()` 前缀匹配，其余待跟进。
