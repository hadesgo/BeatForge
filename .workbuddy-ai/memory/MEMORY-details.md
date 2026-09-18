# BeatForge 细节参考

`MEMORY.md` 只放高频规则；这里是按需查阅的长尾细节。**改动相关模块前先读对应小节。**

## 人声分离（`beatforge/models/separator.py`）
- **作用域严格限定在 ASR**：`pipeline.speech_source()` 是唯一入口，只给人声轨做转录**和强制对齐**；
  拍点/能量/段落分析仍跑整轨混音（那些要的就是鼓）。
- **模型名必须进缓存 key**（`cache/<stem>-<digest>-vocals.wav`）：两个检查点给出两条不同人声轨，
  混用在下游完全看不出来。
- **选轨按文件名里的分句标记 `_\(([^)]+)\)_`，绝不能按子串匹配**。输出命名
  `<输入>_(<分句>)_<模型>.wav`，而默认检查点就叫 `vocals_mel_band_roformer`——**每个输出文件名都含
  "vocals"，包括伴奏**。子串匹配会把伴奏喂给识别器，症状是转录出一片空白，看起来像模型坏了
  （实跑中招：缓存下来的"人声轨"低频占 52.5%，比原始混音还高）。测试必须用**真实文件名**，
  `song_(Vocals)_model.wav` 这种模型名不含 vocals 的假名字会一路绿灯。
- **别按架构推测依赖，读源码或直接 import**：`audio_separator/separator/separator.py` 顶层无条件
  `import onnxruntime`；`uvr_lib_v5/spec_utils.py` 顶层 `import audioread`，而它没写进依赖表。
- CPU 推理约 **204 秒 / 分钟音频**（45 秒实测 153 秒）。`Separator.separate()` 返回**完整路径**
  而非裸文件名，解析要判断 `is_absolute()`。
- 装不上时**退回整轨并明确说明**（`plan.json` 的 `models.separation` 记录）。降级本身没问题，
  悄悄降级才是问题——整轨转录是另一份更差的歌词，下游分辨不出来。
- 选模型依据包自带的 `audio-separator/models-scores.json`（115 个模型的实测 SDR），不是猜：
  默认 `vocals_mel_band_roformer.ckpt`（均 SDR 11.49，与最高 11.53 持平）。卡拉 OK 模型主轨是
  伴奏，不要当人声分离用。
- **下载步骤与运行时必须用同一个模型目录**。分离检查点不是 HF/ModelScope 快照，而是
  `audio-separator` 自己目录里的单个 `.ckpt`，走它**只下载不加载**的
  `Separator.download_model_files(filename)`。两边目录都从 `separator.SEPARATOR_SUBDIR` 派生并有
  测试断言——不一致的后果是静默的：下载看起来成功，运行时找不到、首次使用重拉一遍。
- **可选 extra 缺失时下载要"跳过"而不是"失败"**：`separate_vocals` 默认 true，让下载失败会让默认
  配置在所有没装该 extra 的机器上跑不通。区分"没装包"（跳过 + 说明缺哪个）与"真的下载失败"
  （进 failures 抛 `ModelDownloadError`）。
- `release_gpu()` 只 `except ImportError` 不够：torch 存在但不可用（半装完、没有 `cuda` 属性、驱动
  拒绝初始化）会抛 `AttributeError` 把调用它的阶段带崩。它是**尽力而为的清理**，已放宽到
  `(ImportError, AttributeError, RuntimeError, OSError)`。

## 视觉检索输入预算（`beatforge/models/vision_index.py`）
- Qwen-VL 系 processor 在视觉塔前会自己把图缩到 `max_pixels = 1280*28*28` ≈ 1.0MP，所以喂原图
  **换不来任何模型能看到的细节**，只换来全分辨率解码 + 缩放 + 两者同时驻留。任何"把素材喂给视觉
  模型"的改动都要先过 `_fit_within(w, h, input_pixels)`。
- 按**像素预算**缩放，不要按长边（长边规则的失效模式正是竖屏：768x2048 仍是 1.5MP）。只缩不放，
  取偶（yuv420p 与 patch 网格都要偶数）。
- 图片素材走 `_model_image()` 拿预算内缓存副本（`cache/model-input/`），编码与重排共用；预算内的
  原样透传、不建缓存。存副本用 `source.draft("RGB", target)` 走 JPEG 1/2、1/4、1/8 免解码降采样。
  视频帧的缓存 key **必须含目标尺寸**，否则改预算会静默复用旧尺寸的帧。
- 关键帧用完就释放：`similarities` 聚合后把 `spans[i]` 的 frames 置 None，重排用 `_video_frame_at()`
  只回读需要的那一帧。重排对同一素材的重复候选是常态，"每个候选重新解码一次"是乘以候选数的浪费。
- `VisionIndex.input_pixels` 是**类属性**，这样用 `__new__` 绕过构造器的测试也有合理默认值。
- 验证用 `scripts/vision_input_probe.py`（两条路径各起子进程，峰值读数才各归各的）。实测 8 张
  6000x4000：1.29s→0.80s，峰值内存 224→61 MiB，送进模型的像素 192→8.02 MP。
  **最后一项才是显存那本账**——视觉塔 patch 数与激活显存按像素数走，重排还要乘以候选数。
- Windows 上 `ctypes` 调 `GetProcessMemoryInfo` **必须显式声明 `argtypes`/`restype`**，否则不报错、
  直接返回 0。`resource` 模块是 Unix-only。

## 剪辑风格（`beatforge/editing.py`）
- **剪辑不是 look**：调色/字体/颗粒属于艺术指导（`director.py`）；这层只管**关于时间的决定**——
  镜长、切点网格、转场音量、景别对比、运镜强度。新的剪辑决策放这里，别塞进 config 散键。
- `EDIT_STYLES` 八种风格是**自洽的习惯**不是独立旋钮——挑一半"卡点"再挑一半"长镜"，剪出来就是
  没有观点的片子。`edit_style="auto"` 按情绪选，`manual` 表示"别接管"。
  **风格接管它表过态的项**（镜长、`transition_density`、`composite_ratio`），其余走参数——
  风格和手工值是对同一问题的两个回答。
- **`cut_alignment` 的乐句网格必须按风格自己的镜长窗口自适应**（`_phrase_grid`）：固定四小节在
  120bpm 下是 8 秒，镜长上限 6 秒的风格永远找不到候选点，会静默退化成按小节切。
  加新网格类型先问：这个网格的间距会不会大于风格自己的 `shot_max`？
- **密度门控必须在"意图"分支之前**：`impact`/`breathe` 曾排在前面，导致
  `transition_density=0.10` 的纪实风格实测仍有 47% 可见转场。
- 转场降级（`_quieten`）要**在所有出口统一过滤**，否则含蓄风格的结构性转场仍会闪白/故障；
  降级用**轮转替代名**，避免全塌成同一个。
- 验证用 `scripts/edit_style_probe.py`（同一首歌喂给全部风格，对比镜长/转场/景别）。
  **探针素材的景别必须成块分布**：按索引交替的话所有风格都会 100% 换景别，指标失去意义。
  实测镜头中位 0.50s（冲击碎剪）~ 8.00s（极简留白），可见转场 10%（纪实）~ 70%（冲击）。

## 字幕：版式与镂空（`beatforge/lyrics.py` + `renderer.py`）
- `subtitle_layout` 默认 `free`：不做底部字幕条，而是**按演唱位置把一句切成分片**摆到主体不占处，
  **每片在自己被唱到的时刻淡入**（参考片：利比《跳楼机》官方歌词 MV 的核心）。
- **断句不能按字数平分**：中文没有词间空格，"黎明照亮天空"对半分会把"照亮"劈开。断点取**逐字
  时间轴里最长的静音**（`_MIN_BREAK_SECONDS=0.22`）→ 退到标点 → 都没有则整句不拆。
- **版式表 10 套**（上下夹持/升降对角/同基线拉开/左右堆叠/居中紧堆/对角两角/两侧贴边/下沉居中），
  轮转用**与表长互质的步长**（`_FREE_STEP=3`）而非手写循环：十套走完才重复，表按"给中间留多少
  空间"排序，相邻两句构图必然不同。每套混用 4/5/6 对齐——只换位置不换对齐仍是同一套构图。
- **避让主体是"整体平移 + 跳版式"，不是"替换成固定位置"**。早先主体一偏就把所有行换成两组固定
  锚点，一支主体偏侧的歌整首都只有 2 种落点——这才是"太死板"的真正原因。现在先平移出主体方框，
  平移不了就换轮转里的下一套（最多 `_FREE_TRIES` 套）。
- **避让判定必须是二维方框**（`_CLEAR_X=0.11` / `_CLEAR_Y=0.10`）。参考片把字摆在主体**旁边**的
  次数和上下一样多，只按垂直避让会把这类版式全否掉。
- 取整到像素会让"刚好达标"的间隙掉回框内（实测 .0994 < .10），`_fit` 要多留 `_CLEAR_MARGIN=0.005`。
- 断言要写**真正的不变式**（没有任何一片落在主体方框内），别写实现细节的排序——
  "偏上的版式整体高于偏下的版式"只在旧实现下成立。
- **分片延迟出场**：`\alpha&HFF&` + `\t(delay, delay+260, \alpha&H00&)`，**必须追加在特效标签之后**
  （libass 按序应用 `\t` 链，后写的对同一属性胜出）；同时把特效自带 `\fad(in,out)` 的入场半边改成
  `\fad(0,out)`，否则两个 alpha 动画互相打架。
- 自由版式下 `karaoke` **不再逐字扫光**（分片已承载时间信息，再扫一遍是说两遍）；`band` 版式保留。
- `subtitle_fill="knockout"`：没有字幕层，只有画面里的字形空洞，字中透出提亮虚化的同一帧。
  遮罩 = 同一份脚本渲染成白字黑底，`format=gray` + `alphamerge` 取亮度当 alpha，所以脚本里任何
  动画都会让空洞同步跟随。**只提亮填充不够**：自由版式把字放在平滑而暗的空处，实测对比度 −6.2
  （比周围还暗）；必须**同时压暗画面**（`_KNOCKOUT_LIFT=0.16` / `_KNOCKOUT_DIM=-0.10`），
  对比度才成为滤镜图的固有属性（实测 +47/+10）而不是碰巧。
- 验证用 `scripts/subtitle_layout_probe.py`：**不依赖 LRC**（"歌手在哪换气"LRC 带不了），直接构造
  带逐字时间轴的歌词走真实 `render()`，输出接触表。`--layout free|band --fill solid|knockout`。

## 运镜抖动怎么测（别拿成片直接测）
- 相位相关测帧间位移的前提是**相邻帧互为刚体变换**。成片里的暗角、颗粒、调色固定在画面坐标上，
  会破坏这一前提，使单帧步长正负相消、读数全是噪声——曾在真实片段上报出 58/101 静止帧、1.75px
  抖动的假结论（一致性只有 0.033）。
- 判据是**一致性** `|Σstep| / Σ|step|`：≈1 才有效，< 0.5 时脚本判"测量不可信"。
  `scripts/jitter_probe.py` 已内置。（本文件曾提到"技能的 `subpixel_motion.py`"，但本机与项目里
  没有任何 skills 目录，该文件不存在——需要时按本文重建。）
- 要测真实运镜是否平滑，用 `scripts/camera_move_probe.py --controlled`（关掉颗粒/暗角/调色、
  单层铺满）或 `scripts/zoompan_probe.py`。可信结论：旧 zoompan 抖动 1.67px / 比值 18.6 →
  新 perspective 0.117px / 比值 1.5。
- 自己写测量的三个硬坑：float32 会让 FFT 退化成 complex64 且 DC 基底淹没真实峰（必须去均值 +
  float64）；三点抛物线拟合峰值在小平移下低偏差约 40%（用局部上采样 DFT）；测试卡要用类照片的
  1/f 噪声且中灰，不能用平滑噪声或条纹。

## 素材复用与多图合成的推导（`planner.py`）
- 分层的定义：按素材"已出现次数"分组，优先取次数最少的层；层内按发现顺序（列表顺序）。
- `富余素材 = 最少使用层数量 - 剩余镜头数`：≥ 0 时才允许把富余的素材用于多图合成，
  这样"零复用"的承诺不会被合成打破。
- 多图辅助素材来自同一句歌词的视觉语义排序，同时受质量、色彩连续性、重复使用和分辨率惩罚约束；
  默认约 24% 图片镜头使用多图，副歌与导演冲击段提高、呼吸段降低。
- 细节与示例见 README「素材复用规则」。

## AI 导演：提示词阶梯与实测值
- `PROMPT_LADDER` 逐级降级顺序：先裁逐句候选表 → 再减送入的素材条数 → 最后才对歌词抽样。
  理由：候选表是最冗余的部分；歌词承载叙事，最后才动。
- 实测：一首 68 句歌词、24 个素材的歌曲，未裁剪提示词 **6390 tokens**；阶梯裁剪后 **2285 tokens**
  （34 句歌词、候选表清空、8 个素材），`instruction` 同步改写。
- 显存后果（空闲 10.78GiB）：未裁剪时预留 8.3GiB → 权重预算只剩 2.5GiB（8GB 权重约 5.5GB 被压到
  CPU，慢到不可用）；裁剪后预留 3.0GiB → 预算 7.8GiB。**别用"预算 < 权重体积"判 OOM**：
  `director_offload = true` 本来就靠卸载兜底，7.3GiB 预算的实机运行正常跑完。

## 效果型转场（`glitch` / `light_leak` / `film_burn`）
不用 xfade：滤镜直接烧进两个镜头各自的帧，时间线上是硬切——冲击发生在切点**上**而非横跨切点，
这是它与溶解的本质区别。滤镜都带 `enable=` 时间门，只在切点附近生效
（`rgbashift`/`noise`/`fade`/`tmix` 都支持 timeline）。
- **闪光时序坑**：`fade` 在 `st + d` 才到达目标色，出点一侧要从 `duration - flash - 1/fps` 起算；
  停在末尾的话最后一帧只走到三成，闪光退化成轻微提亮（实测峰值只高 1.1）。
- 验证必须同时看**均值亮度 + 空间标准差 + 暖度**：噪点与通道分离几乎不改均值，只看平均亮度会得出
  "什么都没做"（第一版探针就踩了）。`scripts/cut_effect_probe.py` 在真实切点上测这三项。

## 多图合成：留边处理只给单图和 beat_montage（`beatforge/renderer.py`）
- `_adapt_image` 是给**一张填不满画面的图片**用的：92% 缩放居中，周围用**同一张图的虚化副本**
  （`gblur` + 压暗）补满。**"同时显示多张图"的合成里不能用它**——每张图各带一片虚化底，
  一帧就有两个竞争背景；同一张图还会以两个尺度出现两遍。
- `split_screen`/`photo_stack`/`double_exposure` 用 `_fill_frame`（`increase` + `crop`，全出血）。
  堆叠的底图额外压暗去饱和让卡片读得出来。
- **`beat_montage` 有意保留留边**（`_adapt_image`），这是用户拍板的决定：它一次只显示一张图，
  没有两片虚化底竞争、也没有同图两个尺度并存——那两点才是"同时显示"类效果出问题的原因；
  每帧内缩还让蒙太奇像一串照片而不是整帧硬切。**别去"统一"它**，
  `test_beat_montage_keeps_its_letterbox_on_purpose` 正面断言了 `gblur` + `decrease` 必须在。
- 症状对照：**堆叠乱** = 底图清晰自己叠自己虚化副本再压卡片，三个尺度同框；
  **并排割裂** = 一格留边、一格全出血，两边取景曝光不一致，像两张恰好挨着的图而不是一张切成两半。
- 调色是在**合成之后**统一套的（`_image_filter_graph` 返回的标签才进 finishing），
  所以割裂与调色无关。改合成时别把 finishing 挪到前面去。
- 测试要**双向**守：合成图里不得有 `gblur`/`decrease`，**但单图仍必须有**——
  只写前一半会让修复越界，把留边处理从它该在的地方也拿掉。
- 探针 `.probe/composite_probe.py` 要用 `_render_shot()` 而不是 `render()`
  （单镜头走 `render()` 会在 concat 步骤失败）。

## 合成平面与帧数（`_plane_duration`）
`color` 生成的平面（分屏条、iris 遮罩/底、渐变）经 `overlay=shortest=1` 合成时，若长度正好等于
镜头时长，它会成为**最短流**把整镜截短一帧。统一按 `duration + 1/fps` 生成，多出的一帧由收尾
`trim` 切掉（曾因此发现 `split_screen` 一直在丢最后一帧）。
`tests/test_renderer.py::test_all_still_image_effects_render` 断言帧数**精确相等**守这条。

## 字幕特效明细（`beatforge/lyrics.py`）
16 种：`cinematic`/`bounce`/`typewriter`/`punch`/`slide`/`flip_in`、`karaoke`/`float`/`glow`/
`neon`/`neon_flicker`/`shake`/`wave`/`rainbow`/`spotlight`、`glitch`。
- `SUBTITLE_EFFECTS` 是**唯一真源**：`config.SubtitleEffectChoice` 与 `ai_director.SubtitleEffect`
  都用 `Literal[*SUBTITLE_EFFECTS]` 生成（3.11+ 解包，pydantic 也认）。别在两处手抄名单。
- `_subtitle_effect(name, line, *, width, height, margin) -> (prefix, text)` 是唯一分派点；
  逐字特效（`wave`/`flip_in`）走 `_per_character(text, tag_for)` 做 stagger。
- ASS 颜色是 `&HAABBGGRR&` 不是 RGB。`\jitter` 是 libass 扩展；`shake` 另叠一层 `\frz` 摇摆保底。
- **libass 会静默忽略不认识的标签**——渲染完美无缺却什么都没做。`scripts/subtitle_effect_probe.py`
  要同时查"墨量"（真画出来了吗）与"相邻帧差"（真在动吗），`--sheet` 出接触表。

## 那条 `rope_parameters` 告警是什么意思（无害，但有静默失效风险）
日志里每次加载导演模型会出现 1~3 条：
```
[transformers] Unrecognized keys in `rope_parameters` for 'rope_type'='default': {'full_attention', 'sliding_attention'}
```
**含义**：Spark-X2.5 用的是**分层 RoPE**，配置写成嵌套字典
```json
"rope_parameters": {"full_attention":    {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
                    "sliding_attention": {"partial_rotary_factor": 1.0,  "rope_theta": 10000}},
"layer_types": ["sliding_attention", ... 36 项]，  // 27 个滑动 + 9 个全注意力
```
而 transformers 5.x 的 `rope_parameters` 期望的是**扁平的 RoPE 参数名**（`rope_type`/`rope_theta`/
`partial_rotary_factor`）。`modeling_rope_utils._check_received_keys` 会往字典里注入扁平的
`rope_type='default'`/`rope_theta`，然后把 `full_attention`/`sliding_attention` 判为"不认识的键"
并告警。**每次 `AutoConfig` 实例化触发 1 条**，导演阶段加载 3 次配置（`_load_model_config`、
tokenizer、模型）→ 3 条。

**为什么无害**：模型根本不走 transformers 的 RoPE 机制。远程配置类自己提供了
`Spark2_5Config.get_rope_theta(layer_type)` / `get_partial_rotary_factor(layer_type)`，
`modeling_spark.py` 按 `set(config.layer_types)` 各建一套 cos/sin 缓存
（`compute_rope_cos_sin`），逐层取用。实测两种层确实不同：
`full_attention` cos.shape=(8, 64)（256 维里只旋转 25%）vs `sliding_attention` (8, 256)。

**风险**：这属于"警告是噪音"的少数情况——如果哪天 transformers 不再把嵌套子字典透传下来，
远程代码会**静默**回退到 θ=10000 + 全旋转，模型照样跑，只是变差。所以
`scripts/director_model_probe.py` 加了第 5 项断言（`分层 RoPE`），把"警告是安全的"变成**被验证的事实**，
而不是靠人记住。改 transformers 版本后跑一次探针即可。

