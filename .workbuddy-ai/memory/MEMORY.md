# BeatForge 项目长期记忆

> 只放高频规则与坑。**保持当前量级（约 15k 字符）**——再长会在注入时被截断，后半部分等于不存在。
> 长尾细节（人声分离、视觉检索预算、剪辑风格、字幕版式/镂空、抖动测量、素材复用推导、
> 效果型转场与合成平面、字幕特效明细、提示词实测值）在 **`MEMORY-details.md`**，
> 改对应模块前先读那一节。

## 本机环境
- `uv run pytest` 需 `--basetemp=<可写目录> -p no:cacheprovider`（默认临时目录无权限）。
- **有可用 CUDA 显卡**：RTX 5070 12GB，空闲约 10.8GiB。`nvidia-smi` 报 NVML 初始化失败，
  判断显存要用 `torch.cuda.mem_get_info()`。AI 链路可真机验证。
- **ffmpeg 9.0.1 删了老选项**：`-vsync`（改 `-fps_mode passthrough`）和 `-filter_complex_script`
  都报 `Unrecognized option`，退出码 `2880417800`（这个怪数字 = 选项不存在）。
  从文件读滤镜图用通用的 **`-/filter_complex <文件>`**。
- **不要用 `git stash` 做"临时还原代码验证 bug"**：命令被信号打断后 `.git/refs` 整个目录消失，
  git 报 `not a git repository`（`packed-refs`/`HEAD` 还在）。恢复：按 `.git/logs/HEAD` 与
  `.git/logs/refs/**` 的 reflog 末尾哈希用 `printf` 重建 `refs/heads/*`、`refs/remotes/origin/*`。
  临时验证改用**运行时替换函数**，别碰 git 索引。
- `AIConfig.device` 只允许 `auto|cuda|cpu`；但 `transcriber.py`、`vision_index.py`、
  `audio_semantics.py` 仍用 `device == "cuda"` 判断，放开 `cuda:N` 会静默退回 fp32/CPU
  （导演模块已改 `_is_cuda()` 前缀匹配）。
- All-In-One-Infer 在 **CWD** 建 `demix/`、`spec/`（正常结束自清，中断则留下），已 gitignore。
  它批量生成/删临时文件，**在 agent 会话里会被安全钩子按 turn 统计删除数后中止**
  （`SAFE_DELETE_BULK_CONFIRM_REQUIRED`）——用户自己终端不受影响。所以本会话验证流水线时，
  改为从现有 `plan.json` 直接驱动渲染阶段。

## 规划器（`beatforge/planner.py`）
- **确定性**：不用随机数，平票靠素材发现顺序（列表顺序）决出。
- **素材复用**按"已出现次数"分层，优先最少使用层；最少使用层数量 ≥ 剩余镜头数时零复用。
  开关 `render.avoid_asset_repeats`（默认 true）。推导与示例见 `MEMORY-details.md`。
- 决策全写进 `.beatforge/plan.json`；**`shots[i].start` 是权威时间轴**，渲染必须对齐它。
- **选择器必须用独立计数器，否则槽位被"饿死"**：
  - 运镜轮转按**已选出的单图镜头数**（`single_image_cursor`，`_choose_image_effect` 返回
    `layer_count == 0` 时递增），不是镜头总序号（多图合成会吃掉序号）。共用 `index` 时实测
    只用到 13/21 种运镜。
  - 转场轮转同理但成因不同：`_transition_family` 收**两个**关键字计数器——`index`（密度门控用）
    与 `visible`（已分配的可见转场数，轮转用）。分支很窄（`open` 只在 dreamy 的段落切换处出现，
    一首歌两三次），按 `index` 轮转会全落同一余数上，`close`/`open`/`radial` 出现 0 次。
- 这类"饿死"单曲测不出来。守门测试 `test_every_camera_move_is_reachable_from_some_song`（28 首）、
  `test_every_transition_family_is_reachable_from_some_song`（8 组，脚本穷举的最小覆盖面）——
  **加新选择分支要一起扩充**。
- "某族扫不到"未必是 bug：固定段落布局下副歌边界能量恒 >.78 会被 `flash` 抢占。先换布局再下结论。

## AI 导演：显存（`beatforge/models/ai_director.py`）
remote code 有两个必须外部兜住的特性：
1. 注意力是手写 `torch.matmul` + softmax，**无 SDPA / flash-attn**；滑动窗口层（27/36 层，
   window=512）给完整 `[heads, n, n]` 矩阵加掩码，**不切 KV**。峰值按 `heads * n² * 6 字节` 估。
2. `logits_to_keep` **不是**瓶颈：remote `forward` 默认 0，但 transformers 5.16 的 `generate` 在
   `_supports_logits_to_keep()` 为真时显式传 `1`（已 spy 到 `[1,1]`）。`_sequence_reserve_gb` 里的
   `prefill * vocab * dtype_bytes` 是**刻意的保守余量**（≈0.66GiB @2600 tokens），别为省显存删掉。

真正平方增长的是 1，所以提示词必须限长。
- `director_prompt_tokens`（2600）+ `PROMPT_LADDER` 逐级降级（先裁候选表 → 再减素材 → 最后才抽样），
  每级用 tokenizer 实测。裁掉候选表时 `_trim_context` 必须同步改写 `instruction`。
  实测值见 `MEMORY-details.md`。
- `_gpu_budgets` 用 **`torch.cuda.mem_get_info` 的空闲显存**减 `_sequence_reserve_gb`，再与
  `director_gpu_memory_gb` 取小。Accelerate 的 `max_memory` 只预算权重，不能替代这一步。
- OOM 时用 `RETRY_BUDGET_SCALE`（0.65）重试一次。`scripts/director_memory_probe.py` 是纯算术探针。
- `_dtype_bytes` 要**先读 `dtype`**：`config.torch_dtype` 在 5.x 是弃用属性，一读就告警
  （而 `from_pretrained(torch_dtype=...)` 不告警）。

## AI 导演：远程代码兼容层（`_repair_remote_model_code`）
`modeling_spark.py` 按 `transformers==4.57` 写（config 里 `transformers_version: 4.57.1`），
项目锁 `>=5.16.1`。加载前必须就地改写**本地模型目录**里的这份文件，两处：
1. `_tied_weights_keys` 列表 → `{target: source}` 映射（否则权重加载就炸）。
2. `create_causal_mask(input_embeds=..., cache_position=...)` → 5.16 改名 `inputs_embeds` 且删了
   `cache_position`，否则首次前向报 `TypeError: ... unexpected keyword argument 'input_embeds'`。
- 改写依据**已安装**签名（`inspect.signature` 过滤 + `MASK_KWARG_ALIASES` 别名），新旧都对且幂等；
  签名带 `**kwargs` 时不动。`mask_kwargs` 同时喂给 `create_causal_mask` 与
  `create_sliding_window_causal_mask`，所以取两者签名的**交集**。
- 只补到 HF modules 缓存目录没用：缓存目录名是源码哈希，改本地文件就换目录重建。
- **`config._attn_implementation` 必须是 `eager`**：remote 手工加 4D float 掩码，落到 sdpa 时
  `create_causal_mask` 返回 `None`，注意力**静默失去掩码**。`Spark2_5PreTrainedModel` 没声明
  `_supports_sdpa`，所以现在默认就是 eager；改模型/改 transformers 后用探针复核。
- `scripts/director_model_probe.py`：config 缩到几百万参数在 **CPU** 跑同一份远程代码，无需 7.7GB
  权重即可验：注意力实现、因果性（前缀一致）、滑动窗口、带缓存/无缓存贪心一致、**分层 RoPE**。
  滑动窗口那项**必须用单层**（感受野 = 层数 × window，多层会掩盖窗口外 token）。
- 日志里的 `Unrecognized keys in rope_parameters ... {'full_attention','sliding_attention'}`
  **是噪音但值得理解**（分层 RoPE 的嵌套 schema 与 transformers 的扁平 schema 不匹配）——
  含义、为什么无害、以及它掩盖的静默失效风险见 `MEMORY-details.md`。

## AI 导演：提示词长度测量
`processor.apply_chat_template(..., tokenize=True)` 返回含 `input_ids`/`attention_mask` 的
`BatchEncoding`，`len()` 是**键的个数（2）**不是 token 数。曾把 `_prompt_tokens` 写成
`len(rendered)`：提示词上限全程失效（`_fit_prompt` 每级都判"装得下"）、预留按 2 tokens 算成
1.5GiB 下限（日志「约 2 tokens · 预算 9.0GiB」）。修复后同场景「2848 tokens · 预留 3.5GiB ·
预算 7.3GiB」。`_rendered_token_count` 要覆盖 BatchEncoding / 张量 / 列表 / 嵌套列表四种形态。

## 图片运镜：四边形与 `perspective`
- **禁止 `zoompan`**：它把 `x`/`y` 截断到整数像素，而本项目运镜只有 0.03–0.1 px/帧 → "多帧静止 +
  突然跳 1 像素"。改用 `perspective=...:interpolation=cubic:sense=source:eval=frame` 逐帧求值四边形。
- 几何量别写反：**裁剪框宽 = `W/zoom`**，**可平移量程 = `W - W/zoom`**。
- **可平移量程正比于 zoom**：位移 = `(W - W/zoom) × 比例`。`whip_pan` 因此把 zoom_to 设到 0.42
  （46px @640）；曾写 0.045 只得 8px，读起来像慢漂移。
- 参数在 `_CameraMove`（冻结 dataclass）+ `_CAMERA_MOVES`（21 种）：`zoom_from/to` 是**增量倍率**，
  `x_from..y_to` 是**平移量程比例**（±0.15 内，避免撞 `clip()` 边界而中途停住）；
  `curve`(ramp/breathe/pulse)、`roll`、`keystone`/`keystone_v`、`shake`、`drag`。
- 四边形由 `_camera_quad` 统一构造（**同一中心点 + 同一半尺寸**再旋转/梯形），所以滚转与 3D 转向
  复用**同一次** `perspective`，不叠加第二次重采样。
- 角点顺序：过滤器里 `x0..x3` 是 TL/TR/**BL/BR**，按索引连起来是蝴蝶结；面积与转向判定必须按
  TL→TR→BR→BL 的**环序**（`_corners` 已换序）。
- 三个硬坑：
  1. **`on` 是 1-based 且会越过镜头长度继续涨**（`VAR_ON = frame_count_in + 1`；静态图来自
     `-loop 1`，下游 `fps`/`trim` 会多拉几帧）。进度必须写 `clip((on-1)/(frames-1),0,1)`：
     少 `-1` 末帧过冲；少 `clip` 则递减型运镜末尾**反转**，zoom < 1 → `W-W/zoom` 变负 → 该帧被拒。
     症状极具误导性：只有 `dolly_out`/`pull_back`/`breathe` 这类**递减**运镜崩。
     手持抖动用 `on` 计时（`(max(1,min(on,frames))-1)/fps`，两端都钳）。
  2. **`clip()` 不交换上下界**（`min(max(x,min),max)`），上界为负就原样返回
     （`x0='clip(5,0,-1)'` 直接 `Invalid argument`）。所以每个运镜额外抬高 `_MIN_ZOOM = 0.004`。
  3. **`geq` 是另一套方言**：没有 `on`，用 0 基 `N`；逐像素求值，1080p 实测 0.18s/帧（≈5.5fps）——
     iris 遮罩改在 1/8 画布生成再 `scale` 上采样。

## 图片运镜：包围盒量程与 `amount`
- 转过的角点会甩出裁剪框、梯形较宽的边会顶到边缘，所以量程必须按四边形**包围盒**算，不是
  `W - W/zoom`（实测 `tilt3d_back` 越界 24.5px、`tilt3d_left` 12.1px、`roll_drift` 4.1px）。
- `_required_zoom(move, amount, low, high, aspect)` 采样 9 点求包围盒需要的最小放大率，然后把
  **整条曲线**抬高（不是夹住某几帧，以保住推进幅度）。轴对齐运镜的包围盒 == 裁剪框，故不受影响。
- **`amount` 必须同时进估算和生成**。`amount` = 每镜运镜强度
  （`max(.35, art.camera_intensity) * (1 + melody*.18) * {"dynamic":1.35,"gentle":.7}`，
  实测 0.245~3.2），缩放 zoom/roll/shake/keystone。曾 `_quad_spread` 乘了 `amount` 而
  `_camera_quad` **没乘**：估算小最多 14%，`amount < 1` 的镜头四角出画
  （`tilt3d_front` @0.8714 越界 23px @1920，第 218/253 镜被 EINVAL 拒）。
  **只要估算与生成各写一份公式就一定会漂移**，而 `amount == 1` 时完全等价 ——
  所以测试与探针**必须扫 amount**。
- 守门：`scripts/quad_probe.py` 逐帧求值全部运镜、**扫 10 个 amount**，报最紧余量（负值=越界像素）；
  `test_every_camera_move_stays_in_frame_at_every_camera_intensity` 是测试版。
  `tests/test_renderer.py` 另断言：无 zoompan、四边形凸且四角在画面内、保持宽高比、
  相邻帧步长 ≤ `max(平均步长*6, 0.5px)`、**超出镜头长度后几何不变**。

## 时间线：转场与拼接
- 规划器只下发**转场族**（`_transition_family`），`_transition_spec` 结合进入镜头的
  `transition_tone` 与离开镜头的漂移方向解析成动作。**21 个 xfade 族 / 58 个具体名**
  （`_TRANSITION_LIBRARY`）+ **3 个效果型转场**（`_EFFECT_TRANSITIONS`）。
  `_TRANSITION_PACE` 逐族定时长，方向性族在 `_TRANSITION_DIRECTION`；未知族名回退溶解而非硬切。
- `_Transition(kind, name, seconds)`，`kind ∈ {cut, xfade, effect}`，带 `.handle`：
  **只有 `xfade` 需要额外 footage**（`cut`/`effect` 都是 0）。
- **补的必须是「入场」转场的重叠量**。整条时间线是**一次** ffmpeg 调用，滤镜链
  `x_k = xfade(x_{k-1}, clip_k)`：第一输入是"已拼好的全部"，第二是当前镜头，混合段展示第二输入的
  **第 0 帧起**。所以 **`A_k = D_k + s_{k-1}`**（入场）。曾写成出场侧 `D_k + s_k`：slack 放错端
  （出场侧没人消费，xfade 直接丢弃），自己的入场混合又缺帧。253 镜实测 198.133s vs plan 197.555s
  （**+0.578s**），`-shortest` 把它从片尾悄悄切掉（末镜 1.725s 只播 1.147s，**丢 33%**）。
- **时间轴必须锚定 `shots[i].start`，不能跟着实测剪辑时长累加**。剪辑是整数帧而
  `trim=duration=` **向上取整**（`ceil(D*fps)`），每镜最多长一帧，253 镜累加就是那 0.578s。
  xfade 的 `offset` 取 `min(plan 位置, 实测流上限)`——xfade 丢弃第一输入超出 `offset+seconds` 的
  部分，回锚不需要 trim；`concat`（cut/效果型）无法平移流，先 `trim=end=<plan 位置>` 再拼。
  修后 picture 197.5667s（+0.0117s），末镜播满。残留 −0.06s（≈2 帧）来自 `trim=end=` 帧粒度。
- **拼接命令行会超 Windows 32767 上限**：253 镜 = 滤镜图约 27k + `-i` 约 9k →
  `FileNotFoundError: [WinError 206]`。**253 镜全部渲染成功后才在拼接时炸**，症状是 `output.mp4`
  根本不存在、`picture.mp4` 还是上一次的。滤镜图已写进 `<picture>.filter` 用 `-/filter_complex`
  传入（守门 `test_a_long_edit_keeps_the_filter_graph_off_the_command_line`）。残留上限：`-i` 仍内联
  （约 36 字符/镜），约 900 镜后会再次超限。
- `render.transition_density`（0.35）只管**段落内部**；段落切换/乐段边界/导演冲击点不受约束。
  `_break_transition_repeats` 防同族连续（含蓄转场除外）；`_TRANSITION_ALTERNATIVES` 里效果型
  转场的替代项保持同等强度——把故障换成溶解会把这一刀泄掉。
- 测试拿 `ffmpeg -h filter=xfade` 校验名字并**真的跑一遍**每个转场（duration 必须写 `0.3`，
  `.3` 报 `Unable to parse "duration" option value`）。

## 其余模块的要点（细节见 `MEMORY-details.md`）
- **效果型转场**（`glitch`/`light_leak`/`film_burn`）：不用 xfade，滤镜烧进两个镜头各自的帧，
  时间线是硬切——冲击发生在切点**上**而非横跨切点。`fade` 在 `st+d` 才到目标色，
  出点一侧要从 `duration - flash - 1/fps` 起算（否则闪光退化成轻微提亮）。
  验证必须同时看**均值亮度 + 空间标准差 + 暖度**，只看平均亮度会得出"什么都没做"。
- **合成平面**（`_plane_duration`）：`color` 平面经 `overlay=shortest=1` 合成时若长度正好等于镜头
  时长，它会成为最短流把整镜截短一帧；统一按 `duration + 1/fps` 生成再由收尾 `trim` 切掉。
  `test_all_still_image_effects_render` 断言帧数精确相等。
- **字幕特效**：16 种，`SUBTITLE_EFFECTS` 是**唯一真源**（`config` 与 `ai_director` 都用
  `Literal[*SUBTITLE_EFFECTS]` 生成，别手抄名单）；`_subtitle_effect()` 是唯一分派点；
  ASS 颜色是 `&HAABBGGRR&`；**libass 会静默忽略不认识的标签**，所以
  `scripts/subtitle_effect_probe.py` 要同时查"墨量"与"相邻帧差"。
- **字幕版式/镂空**（`free` 分片淡入、断句取最长静音、`knockout` 必须同时压暗画面）、
  **剪辑风格**（`editing.py` 只管时间决策；密度门控必须在意图分支之前）、
  **人声分离**（选轨按分句标记而非子串）、**视觉检索输入预算**（按像素而非长边）、
  **运镜抖动测量**（判据是一致性 `|Σstep|/Σ|step|`）——都在 `MEMORY-details.md`。
