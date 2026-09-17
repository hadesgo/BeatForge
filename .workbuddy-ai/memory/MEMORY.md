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
- 运镜参数化在 `_CameraMove`（冻结 dataclass）+ `_CAMERA_MOVES`（21 种）。`zoom_from`/`zoom_to`
  是**增量倍率**（保证 zoom ≥ 1），`x_from`..`y_to` 是**平移量程的比例**（±0.15 以内，避免撞 `clip()` 边界而中途停住）。
  另外几个字段负责"不只是推一下"的运镜：`curve`(ramp/breathe/pulse)、`roll`(滚转角度)、
  `keystone`/`keystone_v`(梯形变形)、`shake`(手持抖动)、`drag`(甩镜拖影帧数)。
- **可平移量程正比于 zoom**：pan 的位移 = `(W - W/zoom) × 比例`，所以浅裁的运镜只能平移几个像素。
  写"甩镜"这类大位移运镜必须把 zoom 调深（`whip_pan` 用 0.42），否则无论名义上多快都读起来像慢漂移
  —— 曾因此写出位移只有 8px 的"甩镜"（修复后 46px @640）。
- 四边形由 `_camera_quad` 统一构造：**同一个中心点 + 同一个半尺寸**，再整体旋转或做梯形变形。
  这样滚转和 3D 转向都复用**同一次** `perspective`，不叠加第二次重采样。
- 回归测试在 `tests/test_renderer.py`：断言无 zoompan、四边形是凸的且四角都在画面内、保持宽高比、
  相邻帧步长不超过 `max(平均步长*6, 0.5px)`、以及**超出镜头长度后几何保持不变**。改运镜必须让这些测试继续通过。
  注意四边形的角点顺序：过滤器里的 `x0..x3` 是 TL/TR/**BL/BR**，直接按索引连起来是蝴蝶结，
  面积和转向判定必须按 TL→TR→BR→BL 的**环序**（`_corners` 已经换过序）。

## 滚转/梯形必须按包围盒算量程（踩过的坑）
- 转过的角点会甩到裁剪框外，梯形较宽的那条边会顶到画面边缘。所以能平移的量程必须按四边形的
  **包围盒**算，而不是 `W - W/zoom`。用 `W - W/zoom` 会在渲染中途被 ffmpeg 拒绝某一帧
  （实测 `tilt3d_back` 越界 24.5px、`tilt3d_left` 越界 12.1px、`roll_drift` 越界 4.1px）。
- `_required_zoom(move, amount, low, high, aspect)` 采样 zoom 曲线，算出包围盒需要的**最小放大率**，
  然后把**整条曲线**抬高到那个高度。抬高整条而不是夹住某几帧，是为了保住运镜原本的推进幅度。
  轴对齐的运镜包围盒 == 裁剪框，所以这个改动对它们完全无影响（现有运镜的四边形逐帧不变）。
- 采样 9 个点就够：所有曲线形状在采样点之间对 `unit` 单调，而 breathe/pulse 这类非单调曲线
  本来就会扫过整个 0..1 区间。
- `scripts/quad_probe.py` 逐帧求值全部运镜的四边形，报越界像素与形变量。**调滚转角度或梯形强度先跑它**，
  比等 ffmpeg 在渲染中途拒绝某一帧快得多。

## `perspective` 的三个硬坑（改任何运镜前必读）
1. **`on` 是 1-based，而且会越过镜头长度继续增长。**
   `vf_perspective.c` 里 `VAR_ON = outl->frame_count_in + 1`；静态图来自 `-loop 1`（无限长），
   下游的 `fps`/`trim` 为了确定自己的时间戳会多拉几帧，所以 `on` 会一直涨。
   进度必须写成 **`clip((on-1)/(frames-1),0,1)`**：
   - 少写 `-1` → 末帧过冲一帧；
   - 少写 `clip` → 递减型运镜在末尾**反转**，zoom 掉到 1 以下 → `W-W/zoom` 变负 → 该帧被 ffmpeg 拒绝。
   症状极具误导性：只有 `dolly_out`/`pull_back`/`breathe` 这类**递减**运镜崩，`cinematic_depth` 全正常。
   - 同理，手持抖动用 `on` 计时（`(max(1,min(on,frames))-1)/fps`，两端都要钳）：
     用 progress 计时会让长镜头上的抖动频率变慢成"摇摆"，不钳上界则镜头结束后还在抖。
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
- 规划器只下发**转场族**（`_transition_family`），`_transition_spec` 再结合进入镜头的
  `transition_tone` 与离开镜头的漂移方向解析成具体动作。共 **21 个 xfade 族 / 58 个具体名**
  （`_TRANSITION_LIBRARY`）+ **3 个效果型转场**（`_EFFECT_TRANSITIONS`）。
- `_transition_spec` 返回 `_Transition(kind, name, seconds)`，`kind ∈ {cut, xfade, effect}`，
  带 `.handle` 属性：**只有 `xfade` 需要额外 footage 补重叠**，`cut` 和 `effect` 都是 0。
  这个区分很关键——效果型转场是在镜头自己的帧里烧出来的，多渲染 footage 会直接改变总时长。
- `_TRANSITION_PACE` 逐族定时长；方向性族在 `_TRANSITION_DIRECTION`。未知族名回退成溶解，不是硬切。
- **效果型转场**（`glitch`/`light_leak`/`film_burn`）不用 xfade：滤镜直接烧进两个镜头各自的帧里，
  时间线上是硬切。冲击发生在切点**上**而不是横跨切点，这是它和溶解的本质区别。
  滤镜都带 `enable=` 时间门，只在切点附近生效；`rgbashift`/`noise`/`fade`/`tmix` 都支持 timeline。
  - **闪光时序坑**：`fade` 在 `st + d` 才到达目标色，所以出点一侧要从 `duration - flash - 1/fps` 起算。
    停在片段末尾的话最后一帧只走到三成，闪光退化成一记轻微提亮（实测峰值只比常态高 1.1）。
  - 验证必须同时看**均值亮度 + 空间标准差 + 暖度**：噪点和通道分离几乎不改变均值，
    只看平均亮度会得出"什么都没做"的结论（第一版探针就踩了这个）。
- `render.transition_density`（默认 0.35）只管**段落内部**的可见转场比例；段落切换/乐段边界/
  导演冲击点不受约束。`_break_transition_repeats` 防止同一族连续出现（含蓄转场除外），
  `_TRANSITION_ALTERNATIVES` 里效果型转场的替代项也保持同等强度——把故障换成溶解会把这一刀泄掉。
- 测试会拿 `ffmpeg -h filter=xfade` 校验名字，并**真的跑一遍**每个转场（`xfade` 的 duration 必须是
  `0.3` 这种写法，`.3` 会报 `Unable to parse "duration" option value`）。

## 字幕特效（`beatforge/lyrics.py`）
- 16 种，按剪映式"入场/持续/异类"三类：`cinematic`/`bounce`/`typewriter`/`punch`/`slide`/`flip_in`、
  `karaoke`/`float`/`glow`/`neon`/`neon_flicker`/`shake`/`wave`/`rainbow`/`spotlight`、`glitch`。
- `SUBTITLE_EFFECTS` 是**唯一真源**：`config.SubtitleEffectChoice` 与
  `ai_director.SubtitleEffect` 都用 `Literal[*SUBTITLE_EFFECTS]` 生成（Python 3.11+ 支持解包，
  pydantic 也认）。别再在两处手抄名单。
- `_subtitle_effect(name, line, *, width, height, margin) -> (prefix, text)` 是唯一的分派点；
  逐字特效（`wave`/`flip_in`）走 `_per_character(text, tag_for)` 做 stagger。
- ASS 颜色是 `&HAABBGGRR&`，不是 RGB，写反了会得到完全不同的颜色。
- **libass 会静默忽略不认识的标签**——渲染完美无缺却什么都没做。所以
  `scripts/subtitle_effect_probe.py` 要同时查"墨量"（真的画出来了吗）和"相邻帧差"（真的在动吗）。
  `--sheet` 可输出接触表。
- `\jitter` 是 libass 扩展；`shake` 同时叠了一层 `\frz` 摇摆作为保底，这样即使某个构建忽略
  `\jitter`，这一句仍然在动。

## 视觉检索的输入预算（`beatforge/models/vision_index.py`）
- Qwen-VL 系 processor 在视觉塔之前会自己把图缩到 `max_pixels = 1280*28*28` ≈ 1.0MP。
  所以喂原图**换不来任何模型能看到的细节**，只换来全分辨率解码 + 缩放 + 两者同时驻留。
  任何"把素材喂给视觉模型"的改动都要先过 `_fit_within(w, h, input_pixels)`。
- 按**像素预算**缩放，不要按长边：长边规则的失效模式正是竖屏（768x2048 仍是 1.5MP）。
  只缩不放，取偶（yuv420p 与 patch 网格都要偶数）。
- 图片素材走 `_model_image()` 拿预算内缓存副本（`cache/model-input/`），编码与重排共用；
  在预算内的素材原样透传，不建缓存。存副本用 `source.draft("RGB", target)` 走 JPEG 的
  1/2、1/4、1/8 免解码降采样。视频帧的缓存 key **必须包含目标尺寸**，
  否则改预算会静默复用旧尺寸的帧。
- 关键帧用完就释放：`similarities` 聚合后把 `spans[i]` 的 frames 置 None，
  重排用 `_video_frame_at()` 只回读需要的那一帧。重排对同一素材的重复候选是常态，
  "每个候选重新解码一次"是乘以候选数的浪费。
- `VisionIndex.input_pixels` 是**类属性**，这样用 `__new__` 绕过构造器的测试也有合理默认值。
- 验证用 `scripts/vision_input_probe.py`（两条路径各起一个子进程，峰值读数才各归各的）。
  实测 8 张 6000x4000：耗时 1.29s→0.80s，峰值内存 224→61 MiB，
  送进模型的像素 192→8.02 MP。**最后一项才是显存那本账**——视觉塔的 patch 数与激活显存
  直接按像素数走，重排阶段还要乘以候选数。
- Windows 上 `ctypes` 调 `GetProcessMemoryInfo` **必须显式声明 `argtypes`/`restype`**，
  否则不报错、直接返回 0。`resource` 模块是 Unix-only。

## 规划器的选择器必须用独立计数器（踩过的坑）
- 运镜轮转（`_single_image_effect` 里的 `cursor % N`）按**已选出的单图镜头数**推进，
  而不是按镜头总序号。多图合成门控会吃掉一部分镜头，若两者共用全局 `index`，
  被吃掉的序号对应的轮转槽位就永远轮不到——实测同一首歌只用到 13/21 种运镜，
  改成独立计数后用满 21 种。`create_plan` 维护 `single_image_cursor`，
  `_choose_image_effect` 返回 `layer_count == 0` 时递增。
- 转场轮转同理，但成因不同：`_transition_family` 接收**两个**计数器——
  `index`（镜头序号，给密度门控用）和 `visible`（已分配的可见转场数，给轮转用）。
  转场分支很窄（`open` 只在 dreamy 歌曲的段落切换处出现，一首歌两三次），
  按镜头序号轮转会让这几次触发全落在同一余数上，`close`/`open`/`radial` 因此出现 0 次。
  两者回答不同问题，所以都保留；两个 int 挨着容易传错，签名用关键字参数。
- 这类"饿死"用单个样本歌曲测不出来：覆盖率取决于该曲的段落与能量分布。
  已加两个跨歌曲扫描的测试，断言**每个**运镜 / 每个转场族都至少被选中一次：
  `test_every_camera_move_is_reachable_from_some_song`（28 首）、
  `test_every_transition_family_is_reachable_from_some_song`（8 组，是脚本穷举出的最小覆盖面）。
  **写新的选择分支时要一起扩充这两个扫描**，否则新加的名字可能根本轮不到。
- 注意"某族扫不到"未必是代码 bug：固定段落布局下副歌边界能量恒 >.78 会被 `flash` 分支抢占，
  导致 `radial` 永远轮不到。先换几组段落布局和能量曲线再下结论。

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
