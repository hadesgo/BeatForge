# BeatForge 项目长期记忆

> 只放高频规则与坑。**保持当前量级（约 15k 字符）**——再长会在注入时被截断，后半部分等于不存在。
> 长尾细节（人声分离、视觉检索预算、剪辑风格、字幕版式/镂空、抖动测量、素材复用推导、
> 效果型转场与合成平面、字幕特效明细、提示词阶梯）在 **`MEMORY-details.md`**，
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
  `audio_semantics.py` 仍用 `device == "cuda"` 判断，放开 `cuda:N` 会静默退回 fp32/CPU。
  **导演模块不再接收 device**——它走 `llama-server` 子进程，显存由 `-ngl` 决定。
- **llama.cpp 的 CUDA 构建不一定自带 CUDA 运行时**：实测 `ggml-cuda.dll` 静态导入
  `cublas64_13.dll`，缺它就 `STATUS_DLL_NOT_FOUND (0xC0000135)`，进程在加载期死掉且**零输出**。
  机器上没有独立 CUDA toolkit，但 `.venv\Lib\site-packages\torch\lib` 里有 cublas64_13 /
  cublasLt64_13 / cudart64_13（torch 2.14+cu130 自带）。`llama_server._launch_environment`
  会把它加到子进程 PATH（**bin 目录在前**，保证自带 DLL 优先），所以不用手工复制那 ~500MB。
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

## AI 导演：只剩一条路径（`beatforge/models/ai_director.py`）
- **只有 `llama-server` 子进程跑 GGUF**。曾经的 in-process `transformers` 引擎（Spark-X2.5 +
  `_repair_remote_model_code` 远程代码兼容层 + Accelerate 显存预算/`_gpu_budgets`/卸载 +
  多模态联系表 + `director_gpu_memory_gb`/`director_offload`/`director_backend` 等配置项）
  **已整体删除**。看到旧文档讲"两个引擎"、`XHToken/Spark-X2.5-4B`、`_sequence_reserve_gb`
  的一律当成过时内容。
- **提示词上限的约束是 KV 缓存**（不是平方注意力）：Bonsai 约 75% 层是线性注意力，瓶颈由 `-c`
  决定——llama.cpp 按整个窗口一次性分配。**`-c` 同时装提示词和回复，提示词拿不到全部**：
  实际预算 = `min(director_prompt_tokens, -c − director_max_new_tokens − PROMPT_SAFETY_TOKENS(64))`。
  默认 `-c 16384` + `max_new_tokens 4096` → 约 **12224**。把 `-c` 当提示词预算就会得到
  `exceed_context_size_error`（真机就是这么炸的：prompt 16495 > n_ctx 16384）。想让导演读更多
  要抬 `-c`，不是抬 `director_prompt_tokens`（那只是上界）。
- **提示词长度必须服务端实测，不能字符估算**：中文约 1 字符/token，模板自带英文样板接近
  5 字符/token，所以估算两个方向都会错。**低估是致命的**——真机估 14240 / 实测 16495（低 16%），
  "装得下"判定通过、llama.cpp 直接拒收整个请求。现在阶梯用 `/apply-template`（按真实模板渲染，
  同样带 chat_template_kwargs 关闭思考）+ `/tokenize` 计数。**这两个端点在服务器起来之后才可能调用**，
  所以 `_treat_with_llamacpp` 的顺序是「先起服务 → 算预算 → 拟合提示词 → 生成」，不是先拟合。
  老构建没有这两个端点时退回 `_estimated_tokens`（`server.tokenizer_available is False`）。
- **上下文按最宽规格构建再由阶梯裁剪**——阶梯只能做减法，没放进去的细节用不到；裁剪顺序是
  先清逐句候选表、再减素材、最后才抽样歌词。**候选列表必须按检索得分排序**：裁剪是普通切片，
  按发现顺序排会先丢最强的。裁掉候选表时 `_trim_context` 必须同步改写 `instruction`。
- 三值 GGUF 必须用 **PrismML 的 llama.cpp 分支**，否则要么拒绝该量化类型、要么**加载成功但输出乱码**。
- **每次请求都必须关掉思考**：`chat_template_kwargs = {"enable_thinking": false}`（`llama_server.chat` 已固定带上）。
  Bonsai 是 R1 风格推理模型，开着思考它会把整个 token 预算烧在 `reasoning_content` 上、`content`
  返回**空**（表现为 `finish_reason=length`，不是"变慢"）。实测同一句"返回一行 JSON"：
  开思考 170 tokens 且可能拿不到答案，关掉只要 14 tokens 直接给 JSON，模板还会顺手去掉推理指令
  让提示词更短。**`reasoning_budget: 0` 在 llama.cpp b10709 上无效**，只有模型自带 Jinja 模板的
  `enable_thinking` 开关管用。

## 配置里的 `Path | None`：空白值必须按"未配置"处理
- pydantic 的 `Path | None` 会把 `""` 折叠成 `Path(".")`——**不是 None**，于是所有"留空即自动查找"
  的选项静默失效。真机踩到：`director_llama_server = ""` 走了"显式路径"分支，在 PATH 与
  `<cache>/bin` 查找之前就报「指向的文件不存在：.」，而权重和客户端其实都装好了。
- 修法：`config._blank_path_is_none()` + 各模型的 `field_validator(mode="before")`，已覆盖
  `director_gguf`、`director_llama_server`、`render.subtitle_fonts_dir`、`project.lyrics`。
  **新增任何 `Path | None` 配置项都要挂上这个校验器**，否则同一个坑会再来一次。
- 顺带：pydantic 默认**忽略**未知键，所以删掉的配置项留在旧 `project.toml` 里不报错也不生效。
  守门测试 `test_the_shipped_project_files_only_name_real_config_fields` 就是防这个。

## 模型下载：导演 GGUF 默认走魔搭
- 导演 GGUF 与快照走**同一套 provider 顺序**（`auto` = modelscope → huggingface 回退），
  但选项不同：`_gguf_options` 给它 `allow_patterns` + `local_dir`，只下一个量化文件
  （整仓快照要 13GB 而只需要其中一个）。
- **`allow_patterns` 对 Hugging Face 必须是 list**：它按 pattern 迭代，传裸字符串会逐字符匹配、
  一个文件都下不到（modelscope 两种都吃）。
- 落盘目录 `cache_dir/director` 必须与 `llama_server.find_gguf` 的查找目录一致，
  否则下载成功但运行时找不到、首次使用又被重拉一遍。

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
