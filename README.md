# BeatForge

BeatForge 是一个 Python + uv 的本地 AI 音乐视频剪辑器。输入音乐、图片/视频以及可选的 LRC 歌词，输出自动编排的 MV 和一份可审阅的镜头决策文件。

## AI 流程

```text
音乐 ── Qwen3-ASR + ForcedAligner ── 逐字时间轴 ──┐
  └── All-In-One + CLAP ── 旋律/节拍/章节/意境 ──┼── Qwen3.5 AI 导演 ── 确定性规划器 ── FFmpeg
图片/视频 ── WeMM-Embedding + Qwen3-VL Reranker ─────┘        │
                                                      导演方案 JSON
```

- `Qwen/Qwen3-ASR-1.7B-hf`：Transformers 原生歌曲识别模型；
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`：Transformers 原生字符/单词级演唱时间对齐；
- `laion/clap-htsat-fused`：音乐情绪、质感和强度的零样本分类；
- `tencent/WeMM-Embedding-9B`：基于 Qwen3.5 的高质量文本、图片和视频统一向量检索；
- `Qwen/Qwen3-VL-Reranker-8B`：对初选画面进行歌词意境、构图和叙事适配精排；
- `All-In-One-Infer`：识别 intro、verse、chorus、bridge、solo、outro 和强拍；
- `Beat This!`：可选的高精度 beat/downbeat 后备；
- `Qwen/Qwen3.5-9B`：本地多模态 AI 导演，负责全片概念、叙事弧、视觉母题和分段剪辑策略；
- `librosa`：旋律变化、节拍密度、能量、音色亮度和章节边界；
- `FFmpeg`：裁切、图片运镜、调色、字幕和最终编码。

模型分阶段加载并释放，不会同时占用显存。完整 AI 流程以 **12GB 显存的 NVIDIA 显卡**作为最低目标规格，不限定具体型号；驱动必须能够运行项目使用的 PyTorch 2.14.0 + CUDA 13.2 构建。无 NVIDIA 显卡的电脑仍可完成开发、单元测试和 `--no-ai` 渲染验证，但完整模型推理速度不作为支持目标。

默认质量优先组合面向12GB显存设计：Qwen3-ASR 1.7B保持BF16，视觉召回使用 WeMM-Embedding-9B，精排使用 Qwen3-VL-Reranker-8B，导演使用Qwen3.5-9B；后三者按阶段加载，其中视觉召回、精排和导演使用 bitsandbytes NF4 双重量化、BF16计算。WeMM 使用完整4096维归一化向量，不为节省少量内存而截断检索维度。CUDA运行时还会启用TF32、高精度矩阵乘策略和cuDNN形状调优。

> 当前开发电脑没有 NVIDIA 显卡，也没有下载真实模型权重，因此12GB方案是项目的目标下限，并非已经在所有12GB显卡上实测通过的保证。代码、单元测试和无模型渲染链路可以在当前电脑验证；首次部署到GPU电脑时，请先执行 `doctor` 和 `--plan-only` 烟雾测试。若模型加载阶段就显存不足，应改用示例CPU配置中的 WeMM-Embedding-2B 和2B精排模型；只降低批量大小无法减少模型权重本身的占用。

## 安装

基础开发环境，不包含任何模型框架：

```powershell
uv sync
```

当前 CPU 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cpu --extra qwen --extra music-ai
```

12GB 显存及以上 NVIDIA 显卡的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
```

CPU 和 CUDA profile 互斥，uv 会阻止二者同时安装。模型权重不会在 `uv sync` 时下载。建议先用下面的统一下载命令准备权重；完成后，运行时会直接使用下载清单中的本地目录。

### 模型兼容性与安全

- Qwen3-ASR 从 Transformers 5.13.0 起提供原生支持；BeatForge 当前统一要求 `transformers>=5.16.1`，以同时满足 ASR 和视觉模型集成，不建议只为复现某个模型示例而单独降级。
- WeMM 官方示例曾推荐 `transformers==5.2.0`，BeatForge 使用更新版本与 `sentence-transformers[image]>=5.7`。这个组合已通过接口级测试，但尚未在本机用真实 WeMM 权重完成GPU烟雾测试；部署时以 `uv.lock` 创建环境后先运行本文的烟雾测试。
- WeMM 通过 `trust_remote_code=True` 加载腾讯官方模型仓库中的自定义 Python 代码。只应下载可信的官方仓库；把 `vision_model` 改成其他仓库时，也等于信任并执行该仓库的模型代码。权重准备完成后建议设置 `offline = true`，避免运行过程中访问网络或获取变化后的代码。
- `doctor` 检查运行库、FFmpeg、CUDA和显存是否就绪，不会真正加载数十GB模型；完整兼容性以一次真实的 `--plan-only` 运行为准。

### 统一下载模型

统一下载项目配置中启用的 ASR、强制对齐、音乐情绪、视觉检索、视觉精排和 AI 导演模型：

```powershell
uv run beatforge download-models my-mv/project.toml
```

默认 `auto` 模式优先从 ModelScope（魔搭社区）下载，适合中国大陆网络；某个仓库在魔搭不存在时才回退到 Hugging Face。Qwen3-ASR、ForcedAligner、Qwen3-VL Reranker 和 Qwen3.5 可使用魔搭的同名官方仓库。

WeMM-Embedding 若尚未被 ModelScope 收录，会由 `auto` 模式自动转到 Hugging Face；已经下载后，BeatForge 始终从清单记录的本地目录加载。使用 `--source modelscope --no-fallback` 时，这类未收录模型会按预期报告失败，而不会静默换源。

指定独立缓存目录和单模型下载并发数：

```powershell
uv run beatforge download-models my-mv/project.toml --cache-dir D:\ai-models --workers 4
```

只允许魔搭下载、完全禁止回退：

```powershell
uv run beatforge download-models my-mv/project.toml --source modelscope --no-fallback
```

需要恢复 Hugging Face 下载时使用 `--source huggingface`。完成后会在项目 `.beatforge/models.json` 写入包含来源和本地路径的统一模型清单，随后可以在配置中设置 `offline = true`。如果清单中的目录被移动或删除，运行时会自动退回配置里的仓库 ID。All-In-One 和 Beat This! 的结构分析权重由各自安装包管理，不属于统一模型清单。

运行时量化不会缩小下载到磁盘的官方BF16模型文件。WeMM-Embedding-9B 权重约18.8GB，默认完整模型缓存建议预留约70GB磁盘空间。若显存更大，可把 `vision_quantization` 或 `director_quantization` 改为 `int8`；24GB以上显存可尝试 `none` 获得最高保真度。12GB配置应保持 `nf4`。`vision_batch_size` 默认是4，发生CUDA显存不足时会自动降到2或1重试；16GB以上显存可尝试手动提高到8。

## 使用

### 推荐的完整流程

在目标 NVIDIA GPU 电脑上，建议按下面的顺序完成首次部署：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
uv run beatforge init my-mv
```

将音乐放到 `my-mv/music.mp3`，图片和视频放到 `my-mv/media/`。如果已有 LRC，保存为 `my-mv/lyrics.lrc`；如果要使用 Qwen3-ASR，删除该文件并删除或注释 `project.toml` 的 `lyrics` 配置。

先统一下载配置中用到的权重，再检查环境：

```powershell
uv run beatforge download-models my-mv/project.toml
uv run beatforge doctor
```

下载成功后可把 `project.toml` 中的 `offline = false` 改为 `offline = true`。第一次不要直接渲染全片，先用真实模型生成剪辑决策，确认 ASR、WeMM、精排和导演都能加载：

```powershell
uv run beatforge run my-mv/project.toml --plan-only
uv run beatforge run my-mv/project.toml
```

仓库中的 `demo/project.toml` 是12GB显存质量目标的示例配置。当前没有 NVIDIA 显卡的开发电脑使用下面的流程即可验证基础分析、字幕、动效、调色和 FFmpeg 渲染，不会下载或加载模型：

```powershell
uv sync
uv run python scripts/create_demo.py
uv run beatforge run demo/project.toml --no-ai
```

只生成剪辑决策，不渲染：

```powershell
uv run beatforge run my-mv/project.toml --plan-only
```

禁用模型，验证基础分析和渲染：

```powershell
uv run beatforge run my-mv/project.toml --no-ai
```

输出包括 `output.mp4`、`.beatforge/plan.json`、`.beatforge/lyrics.ass` 和视频分析所用的缓存关键帧。`plan.json` 包含音乐结构、模型配置、素材信息、逐镜头语义得分和剪辑参数。

### 关键配置速查

| 配置项 | 默认值 | 对成片或资源的影响 | 12GB建议 |
| --- | --- | --- | --- |
| `device` | `auto` | 自动选择CUDA或CPU；正式GPU运行可设为 `cuda` 以尽早暴露环境问题 | `cuda` |
| `offline` | `false` | 为 `true` 时仅使用模型清单和本地缓存 | 下载完成后设为 `true` |
| `vision_model` | `tencent/WeMM-Embedding-9B` | 决定歌词与画面的语义召回质量，也是视觉阶段主要显存占用 | 首次加载OOM时换2B |
| `vision_quantization` | `nf4` | `nf4` 最省显存，`int8`/`none` 需要更多显存 | 保持 `nf4` |
| `vision_batch_size` | `4` | 影响视觉编码吞吐和激活显存；OOM会自动按4→2→1重试 | `4`，仍OOM时设 `1` |
| `vision_rerank_top_k` | `8` | 每句歌词进入精排的候选数；更高可能改善选镜，但更慢 | `8` |
| `frame_samples` | `8` | 长视频关键帧覆盖率；更高更容易找到对应画面，但分析更慢 | `8`，长素材可到 `12` |
| `director_model` | `Qwen/Qwen3.5-9B` | 统一叙事、色彩弧、母题与章节策略 | 保持9B + NF4 |
| `director_gpu_memory_gb` | `9.0` | 导演阶段允许使用的显存上限，其余可卸载到内存/磁盘 | 不要直接填满12GB |
| `director_contact_sheet_assets` | `32` | 给导演观看的高价值素材数量；越多上下文越完整，处理越慢 | `24`–`32` |
| `crf` / `intermediate_crf` | `19` / `14` | 数值越低画质越高、文件越大；中间文件应比最终文件更高质量 | 保持默认 |
| `look_strength` | `0.72` | AI导演色彩弧的应用强度 | 写实人像可降至 `0.55`–`0.7` |
| `shot_match_strength` | `0.3` | 不同设备和来源素材的曝光/饱和度匹配强度 | `0.25`–`0.4` |
| `film_grain` | `1.6` | 用轻微统一颗粒掩盖素材来源差异 | 干净数字风格可降至 `0.5`–`1.0` |
| `subtitle_effect` / `subtitle_font` | `auto` / `auto` | AI按旋律、情绪和段落选择字幕动效与字体 | 保持 `auto` |

`vision_batch_size` 只影响编码时的激活显存，不能解决模型权重加载就OOM的问题。`frame_samples` 和 `director_contact_sheet_assets` 主要交换分析时间与选择信息量，并不会让最终视频分辨率变高。

## 本地 AI 导演

导演模型由 BeatForge 直接通过 Transformers 加载，不需要 llama.cpp、Ollama、LM Studio 或额外服务。视觉检索结束并释放显存后才加载导演；导演方案完成后立即删除模型、执行垃圾回收并清空 CUDA allocator，再进入 FFmpeg 渲染。加载或输出校验失败时自动使用规则导演，流程不会中断。

```toml
[ai]
director_enabled = true
director_model = "Qwen/Qwen3.5-9B"
director_quantization = "nf4"
director_temperature = 0.18
director_max_new_tokens = 3072
director_gpu_memory_gb = 9.0
director_cpu_memory_gb = 20.0
director_offload = true
director_contact_sheet_assets = 32
```

`director_gpu_memory_gb` 是 Accelerate 的显存上限；12GB 显卡默认只允许导演使用 9GB。9B导演和8B视觉模型采用运行时NF4双重量化，计算类型保持BF16；各模型严格分阶段加载，不会同时驻留显存。超出部分在 `director_offload = true` 时卸载到内存和 `.beatforge/director-offload/`。BeatForge 会从检索结果中选出最多32个高价值素材，为图片和视频相关帧生成带素材ID的联系表。设 `director_contact_sheet_assets = 0` 可以关闭这项功能。

导演同时接收歌曲统计、逐句歌词和乐段信息，输出经 Pydantic 校验的结构化方案；第一次 JSON 不合法会在同一次模型生命周期内自动修正一次。它不会生成时间码或直接执行 FFmpeg，具体剪辑点仍由节拍模型和确定性规划器控制。

## 字幕和画面动效

字幕使用 ASS 渲染。`auto` 会根据 CLAP 情绪、局部能量、节奏密度和旋律变化为每句歌词单独选择效果：

```toml
subtitle_font = "auto"
subtitle_fonts_dir = "fonts"
subtitle_effect = "auto"
subtitle_margin = 72
subtitle_highlight_color = "&H0000D7FF"
visual_effects = true
vignette = true
film_grain = 1.6
look_strength = 0.72
shot_match_strength = 0.3
```

- `karaoke`：逐字高亮，并带轻微缩放入场；
- `cinematic`：模糊消散和长淡入淡出；
- `bounce`：随句子出现的弹跳缩放。
- `float`：伴随舒缓旋律缓慢上浮；
- `glow`：适合浪漫和梦幻段落的柔光入场；
- `typewriter`：适合暗黑、叙事感段落的逐字出现。

BeatForge 内置七组跨平台字体候选：`modern`、`cinematic`、`lyrical`、`energetic`、`dreamy`、`minimal` 和 `dark`。每组会依次尝试思源/Noto、霞鹜文楷、MiSans 等开源字体以及当前系统常见中文字体；未找到时使用系统中文字体回退，避免直接回退 Arial 导致方框字。可以把 `.ttf`、`.otf` 或 `.ttc` 放入项目的 `fonts/`，程序会读取字体内部的真实家族名，而不是猜测文件名。

默认情绪映射如下，也可以把右侧替换成任意字体家族名：

```toml
[render.subtitle_fonts]
energetic = "preset:energetic"
uplifting = "preset:modern"
melancholic = "preset:cinematic"
dreamy = "preset:dreamy"
romantic = "preset:lyrical"
dark = "preset:dark"
cinematic = "preset:cinematic"
```

若希望整首歌固定使用某个预设或自定义字体，可分别设置 `subtitle_font = "preset:minimal"` 或 `subtitle_font = "My MV Font"`。自动字幕还会对异常长的歌词单独缩小字号，普通短句保持原字号；这不会破坏逐字高亮和打字机时序。

图片和视频的推拉幅度同时参考局部能量、旋律变化率和歌曲意境。高能/高节奏密度段落使用锐化与亮色闪切，低能段落使用柔化与长淡入，梦幻和抒情歌曲降低镜头运动，所有镜头可选暗角和动态胶片颗粒。

为提高成片统一性，渲染器会把 AI 导演的 `color_arc` 真正转换成按章节推进的轻量色彩风格，而不是只写入 JSON；同时依据素材代表色做受限的曝光和饱和度匹配，最大修正量受到 `shot_match_strength` 控制，不会把夜景强行拉成白天。`look_strength` 控制导演色彩弧强度，设为 `0` 可完全关闭。缩放统一使用 Lanczos，并校正像素宽高比。

中间片段默认使用 `intermediate_crf = 14`，最终交付使用 `crf = 19`，减少转场合成和字幕压制中的多代有损损失；程序会自动保证中间 CRF 不高于最终 CRF。`encoder_tune = "film"` 用于保留肤质、自然纹理和细颗粒，也可根据动画素材改为 `animation`，明显颗粒化素材可用 `grain`。最终选择与每个镜头的代表色都会写入 `plan.json`。

## CPU 与 NVIDIA GPU 配置

CPU兼容配置（用于功能验证，完整推理会很慢）：

```toml
[ai]
device = "auto"
asr_backend = "qwen3"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "wemm-embedding"
vision_model = "tencent/WeMM-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
vision_quantization = "none"
vision_batch_size = 2
music_structure_backend = "allin1"
frame_samples = 3
director_enabled = true
director_model = "Qwen/Qwen3.5-4B"
director_quantization = "none"
director_gpu_memory_gb = 9.0
```

12GB 显存及以上 NVIDIA 显卡推荐配置：

```toml
[ai]
device = "cuda"
asr_backend = "qwen3"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "wemm-embedding"
vision_model = "tencent/WeMM-Embedding-9B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-8B"
vision_quantization = "nf4"
vision_batch_size = 4
music_structure_backend = "allin1"
frame_samples = 8
director_enabled = true
director_model = "Qwen/Qwen3.5-9B"
director_quantization = "nf4"
director_gpu_memory_gb = 9.0
```

运行 `uv run beatforge doctor` 检查实际使用 CPU 还是 CUDA。Faster Whisper、原 Qwen3-VL-Embedding 和 SigLIP2 仍可通过 `asr_backend`/`vision_backend` 作为兼容后备。

## 素材语义

WeMM-Embedding 会分别通过 `encode_query` 和 `encode_document` 比较歌词、图片和视频关键帧。文件名和 sidecar 标签作为模型关闭时的后备。例如 `海边_日落_回忆.jpg`，或创建 `portrait.jpg.json`：

```json
{
  "description": "女孩在海边回头，夕阳逆光",
  "tags": ["女孩", "海边", "日落", "回忆", "离别"],
  "mood": "melancholic",
  "shot_size": "medium_close_up",
  "camera_motion": "slow_push_in",
  "quality_score": 0.9,
  "dominant_color": [184, 112, 74],
  "focus_point": [0.62, 0.43]
}
```

`focus_point` 是归一化的主体中心坐标，左上角为 `[0, 0]`、右下角为 `[1, 1]`。不填写时，图片会通过局部细节、对比度、色彩边缘和保守的中心先验自动估计；视频使用多个采样帧的中位焦点，减少单帧误判。

视频默认均匀抽取8个关键帧并分别匹配歌词，以最相关的两帧计算稳健召回分数，再从语义最相关的时刻附近取材。视觉精排会继续使用这个歌词对应帧，不再退回视频中间帧。提高 `frame_samples` 会提升长视频覆盖率，也会增加分析时间。

## 专业剪辑策略

- 优先在 downbeat 和乐段边界切镜，而不是每句歌词机械切换；
- 歌词决定镜头内容，但不再强制每句换画面，避免歌词幻灯片感；
- 对视频的采样帧分别计算歌词相似度，从最相关画面附近开始取材；连续使用同一视频时顺接时间轴，避免随机跳段；AI 导演同时收到每句歌词的前四名候选及视频时间点；
- 主歌保持色彩和主体连续，副歌提高视频镜头比例和切换强度；
- 副歌复现少量视觉母题，让成片有记忆点；
- 同景别连续出现会扣分，画质、曝光、清晰度和分辨率参与选镜；
- 相邻镜头主色差异过大时降低分数，冲击型剪辑除外；
- 常规节拍点以硬切为主，只在乐段变化和呼吸段使用带 handle 的溶解、闪白或淡黑；
- 原视频保留自身摄影运动，不再叠加周期性摇摆；静态图片根据 `impact/continuity/breathe` 使用不同的推近、漂移或缓慢拉远，不再按镜头序号机械横移；
- 图片和视频按主体焦点进行保守的智能裁切；对短于目标镜头的视频、需要严重放大的低分辨率素材和极端画幅素材降权，减少可见循环、糊画面和主体裁掉；
- intro/outro 留呼吸，chorus 紧凑，bridge/solo 给旋律性镜头更长时间。
- AI 导演统一概念、叙事弧、调色倾向和视觉母题，并对各乐段给出剪辑强度、景别、素材偏好、字幕与转场意见；

这些决策会写入 `plan.json` 的 `section`、`edit_intent`、`melody`、`quality_score` 和 `art_direction`，方便人工复核。

## 常见问题

### `doctor` 显示 CUDA 未启用

先确认安装的是 `ai-cuda` 而不是 `ai-cpu`，再检查 NVIDIA 驱动是否能够运行 PyTorch 2.14.0 + CUDA 13.2 构建。`nvidia-smi` 能显示显卡不代表当前 Python 环境中的 PyTorch 一定启用了CUDA；以 `uv run beatforge doctor` 的结果为准。修改依赖组合后重新执行对应的 `uv sync`，不要在同一环境混装 CPU 和 CUDA profile。

### ModelScope 找不到 WeMM

这是预期的下载源差异。使用默认 `--source auto`，BeatForge 会只为缺失仓库回退到 Hugging Face；中国大陆网络需要能够访问该站点或使用已经下载好的本地缓存。若使用 `--source modelscope --no-fallback`，WeMM 未收录时会直接失败。成功下载后检查 `.beatforge/models.json`，并启用 `offline = true`。

### WeMM 报自定义代码、Processor 或配置加载错误

确认使用 `tencent/WeMM-Embedding-*` 官方仓库，并已安装 `qwen` extra：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
```

不要绕过锁文件随意降级 Transformers。先记录完整异常、当前 `transformers` 和 `sentence-transformers` 版本，再确认本地模型目录是否下载完整。模型仓库更新后，如需重新获取自定义代码，应在联网模式下明确重新下载并复测，确认无误后再恢复离线模式。

### CUDA 显存不足

- 如果在视觉编码过程中OOM，先把 `vision_batch_size` 调为 `1`；程序也会自动减半重试。
- 如果在模型刚加载时OOM，改用示例CPU配置中的 WeMM-Embedding-2B 和2B精排模型；降低批量大小对此无效。
- 如果导演阶段OOM，保持 `director_quantization = "nf4"`，降低 `director_gpu_memory_gb` 和 `director_contact_sheet_assets`，或改用 Qwen3.5-4B。
- 每次只改一个参数并重新运行 `--plan-only`，从日志确认失败发生在哪个模型阶段。

### 离线模式提示找不到模型

先暂时设为 `offline = false` 并重新运行 `download-models`。模型目录被移动、磁盘盘符变化或 `.beatforge/models.json` 仍指向旧路径时，需要重新生成清单；仅复制清单而不复制其指向的模型目录无效。

### 字幕乱码、方框字或自定义字体没有生效

确认 FFmpeg 构建包含 libass，把 `.ttf`、`.otf` 或 `.ttc` 放入项目 `fonts/`，并让 `subtitle_fonts_dir = "fonts"`。固定字体时填写字体内部的家族名，而不是文件名；不确定时优先使用 `preset:cinematic`、`preset:modern` 等内置预设。字体授权由素材提供者自行确认。

### 成片能生成，但看起来像素材幻灯片

先补充素材 sidecar 的人物、地点、情绪、景别与运镜信息，并确保视频有足够长度和镜头变化。随后查看 `plan.json` 中的语义得分和候选画面：长视频可把 `frame_samples` 提高到 `10`–`12`，召回相近但精排不稳定时可适度提高 `vision_rerank_top_k`。不要单纯堆叠转场；统一的主体、色彩、运动方向和重复母题通常比更多特效更接近真人精剪。

## 测试

纯单元测试，不加载或下载模型：

```powershell
uv run pytest -q
```

生成合成素材并执行无模型端到端测试：

```powershell
uv run python scripts/create_demo.py
uv run beatforge run demo/project.toml --no-ai
```

在你自己的 AI 环境中执行模型烟雾测试：

```powershell
uv run beatforge doctor
uv run beatforge run demo/project.toml --plan-only
```

## 代码结构

```text
beatforge/audio.py                    节拍、章节与能量分析
beatforge/director.py                 导演方案到字幕与视觉艺术指导
beatforge/models/ai_director.py       本地 Qwen3.5 导演协议与校验
beatforge/models/transcriber.py       Qwen3-ASR/Whisper 时间轴
beatforge/models/audio_semantics.py   CLAP 音乐语义
beatforge/models/music_structure.py   All-In-One/Beat This 结构分析
beatforge/models/vision_index.py      WeMM/Qwen3-VL-Embedding/SigLIP2 检索
beatforge/planner.py                  多目标镜头编排
beatforge/renderer.py                 FFmpeg 成片渲染
beatforge/pipeline.py                 分阶段模型生命周期
tests/                                不下载模型的测试
```
