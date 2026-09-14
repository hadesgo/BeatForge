# BeatForge

BeatForge 是一个 Python + uv 的本地 AI 音乐视频剪辑器。输入音乐、图片/视频以及可选的 LRC 歌词，输出自动编排的 MV 和一份可审阅的镜头决策文件。

## AI 流程

```text
音乐 ── Qwen3-ASR + ForcedAligner ── 逐字时间轴 ──┐
  └── All-In-One + CLAP ── 旋律/节拍/章节/意境 ──┼── Spark-X2.5 AI 导演 ── 确定性规划器 ── FFmpeg
图片/视频 ── WeMM-Embedding + Qwen3-VL Reranker ─────┘        │
                                                      导演方案 JSON
```

- `Qwen/Qwen3-ASR-1.7B-hf`：Transformers 原生歌曲识别模型；
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`：Transformers 原生字符/单词级演唱时间对齐；
- `laion/clap-htsat-fused`：音乐情绪、质感和强度的零样本分类；
- `tencent/WeMM-Embedding-2B`：文本、图片和视频统一向量检索；
- `Qwen/Qwen3-VL-Reranker-2B`：对初选画面进行歌词意境、构图和叙事适配精排；
- `All-In-One-Infer`：识别 intro、verse、chorus、bridge、solo、outro 和强拍；
- `Beat This!`：可选的高精度 beat/downbeat 后备；
- `XHToken/Spark-X2.5-4B`：本地文本 AI 导演，负责全片概念、叙事弧、视觉母题和分段剪辑策略；
- `librosa`：旋律变化、节拍密度、能量、音色亮度和章节边界；
- `FFmpeg`：裁切、图片运镜、调色、字幕和最终编码。

模型分阶段加载并释放，不会同时占用显存。完整 AI 流程以 **12GB 显存的 NVIDIA 显卡**作为最低目标规格，不限定具体型号；基准环境为 Python 3.13、PyTorch 2.14.0 + CUDA 13.0。无 NVIDIA 显卡的电脑仍可完成开发、单元测试和 `--no-ai` 渲染验证，但完整模型推理速度不作为支持目标。

默认质量优先组合面向12GB显存设计：Qwen3-ASR 1.7B、WeMM-Embedding-2B、Qwen3-VL-Reranker-2B 和 Spark-X2.5-4B 均使用原生 BF16/模型原始精度，不依赖 bitsandbytes 等运行时量化库。视觉召回完成后会先删除 WeMM 并释放 CUDA 缓存，再加载精排模型；导演又在整个视觉索引释放后加载，因此三个大模型不会同时驻留显存。CUDA运行时仍启用TF32、高精度矩阵乘策略和cuDNN形状调优。导演阶段额外限制提示词长度并为注意力矩阵预留显存，详见[提示词上限与显存预留](#提示词上限与显存预留)。

> 当前开发电脑没有 NVIDIA 显卡，也没有下载真实模型权重，因此12GB方案是项目的目标下限，并非已经在所有12GB显卡上实测通过的保证。代码、单元测试和无模型渲染链路可以在当前电脑验证；首次部署到GPU电脑时，请先执行 `doctor` 和 `--plan-only` 烟雾测试。若视觉编码出现瞬时显存不足，先把 `vision_batch_size` 降到1。

## 安装

基础开发环境，不包含任何模型框架：

```powershell
uv python install 3.13
uv sync
```

当前 CPU 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cpu --extra qwen --extra music-ai
```

12GB 显存及以上、驱动支持 CUDA 13.x 的 NVIDIA 显卡（默认高性能 profile）：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
```

`ai-cpu` 和 `ai-cuda` 两个 profile 互斥，uv 会阻止混装。`ai-cuda` 从 CUDA 13.0 源安装 `torch 2.14.0+cu130`、`torchvision 0.29.0+cu130`、`torchaudio 2.11.0+cu130` 和 `torchcodec 0.16.0+cu130`。以上包均提供 Python 3.13 轮子。不要再单独覆盖其中任何一个包。模型权重不会在 `uv sync` 时下载。

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

默认 `auto` 模式优先从 ModelScope（魔搭社区）下载，适合中国大陆网络；某个仓库在魔搭不存在时才回退到 Hugging Face。Spark-X2.5 若魔搭没有同名仓库，会自动回退 Hugging Face。

下载器会把 Hugging Face 配置 ID `tencent/WeMM-Embedding-2B` 映射为魔搭命名空间 `tencent-community/WeMM-Embedding-2B`。若该镜像暂时不可用，默认 `auto` 会回退到 Hugging Face；不要对整套默认模型使用 `--source modelscope --no-fallback`，除非已确认每个仓库都存在。

指定独立缓存目录和单模型下载并发数：

```powershell
uv run beatforge download-models my-mv/project.toml --cache-dir D:\ai-models --workers 4
```

只允许魔搭下载、完全禁止回退：

```powershell
uv run beatforge download-models my-mv/project.toml --source modelscope --no-fallback
```

需要恢复 Hugging Face 下载时使用 `--source huggingface`。完成后会在项目 `.beatforge/models.json` 写入包含来源和本地路径的统一模型清单，随后可以在配置中设置 `offline = true`。如果清单中的目录被移动或删除，运行时会自动退回配置里的仓库 ID。All-In-One 和 Beat This! 的结构分析权重由各自安装包管理，不属于统一模型清单。

项目不再安装或调用量化库，模型使用官方原始权重。`vision_batch_size` 默认是4，发生CUDA显存不足时会自动降到2或1重试；批量大小只影响激活显存，不改变模型权重精度。

旧项目中的 `vision_quantization` 和 `director_quantization` 配置应删除；当前版本不会读取这两个选项。

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
| `vision_model` | `tencent/WeMM-Embedding-2B` | 决定歌词与画面的语义召回质量，也是视觉阶段主要显存占用 | 保持4B原始精度 |
| `vision_batch_size` | `4` | 影响视觉编码吞吐和激活显存；OOM会自动按4→2→1重试 | `4`，仍OOM时设 `1` |
| `vision_rerank_top_k` | `8` | 每句歌词进入精排的候选数；更高可能改善选镜，但更慢 | `8` |
| `frame_samples` | `8` | 长视频关键帧覆盖率；更高更容易找到对应画面，但分析更慢 | `8`，长素材可到 `12` |
| `director_model` | `XHToken/Spark-X2.5-4B` | 统一叙事、色彩弧、母题与章节策略 | 保持4B原始精度 |
| `director_prompt_tokens` | `2600` | 导演提示词的 token 上限；超长歌曲会自动抽样歌词并裁剪候选表 | 显存紧张时降到 `1600`–`2000` |
| `director_gpu_memory_gb` | `9.0` | 导演阶段允许使用的显存上限，其余可卸载到内存/磁盘 | 不要直接填满12GB |
| `director_contact_sheet_assets` | `0` | 仅供可选多模态导演观看联系表；Spark文本导演不会使用 | 保持 `0` |
| `crf` / `intermediate_crf` | `19` / `14` | 数值越低画质越高、文件越大；中间文件应比最终文件更高质量 | 保持默认 |
| `look_strength` | `0.72` | AI导演色彩弧的应用强度 | 写实人像可降至 `0.55`–`0.7` |
| `shot_match_strength` | `0.3` | 不同设备和来源素材的曝光/饱和度匹配强度 | `0.25`–`0.4` |
| `film_grain` | `1.6` | 用轻微统一颗粒掩盖素材来源差异 | 干净数字风格可降至 `0.5`–`1.0` |
| `image_composites` | `true` | 多图镜头的分屏、堆叠、双重曝光和节拍蒙太奇 | 保持 `true` |
| `image_composite_ratio` | `0.24` | 多图镜头的基础占比；副歌和导演标记的冲击段会自动提高 | `0.18`–`0.30` |
| `max_composite_images` | `3` | 分屏、照片堆叠和节拍蒙太奇的同镜头素材上限 | 建议保持 `3` |
| `avoid_asset_repeats` | `true` | 素材够用时保证每个镜头都用没出现过的素材，只在素材不足时才复用 | 保持 `true` |
| `image_background_blur` | `26.0` | 原比例图片周围的满屏模糊背景强度 | 人像可用 `22`–`32` |
| `subtitle_effect` / `subtitle_font` | `auto` / `auto` | AI按旋律、情绪和段落选择字幕动效与字体 | 保持 `auto` |

`vision_batch_size` 只影响编码时的激活显存，不能解决模型权重加载就OOM的问题。`frame_samples` 和 `director_contact_sheet_assets` 主要交换分析时间与选择信息量，并不会让最终视频分辨率变高。`director_gpu_memory_gb` 只是权重上限，导演的注意力矩阵和 KV 缓存会另外占用显存，所以它必须明显低于显卡总容量。

## 本地 AI 导演

导演模型由 BeatForge 直接通过 Transformers 加载，不需要 llama.cpp、Ollama、LM Studio 或额外服务。视觉检索结束并释放显存后才加载导演；导演方案完成后立即删除模型、执行垃圾回收并清空 CUDA allocator，再进入 FFmpeg 渲染。加载或输出校验失败时自动使用规则导演，流程不会中断。

```toml
[ai]
director_enabled = true
director_model = "XHToken/Spark-X2.5-4B"
director_backend = "text"
director_temperature = 0.18
director_max_new_tokens = 3072
director_prompt_tokens = 2600
director_gpu_memory_gb = 9.0
director_cpu_memory_gb = 20.0
director_offload = true
director_contact_sheet_assets = 0
```

`director_gpu_memory_gb` 是 Accelerate 的权重上限；12GB 显卡默认只允许导演使用9GB。Spark-X2.5-4B 使用 `AutoModelForCausalLM` 和原始权重直接加载，超出部分在 `director_offload = true` 时卸载到内存和 `.beatforge/director-offload/`。Spark 是文本模型，因此它读取 WeMM/Qwen 精排后的素材描述、候选得分和视频时间点，不直接读取联系表图片；`director_contact_sheet_assets` 对默认导演保持为0。若以后切回多模态导演，需同时设置 `director_backend = "multimodal"`，才会生成并传入联系表。

Spark-X2.5 发布的 `modeling_spark.py` 是按 `transformers==4.57` 写的，与本项目要求的 `transformers>=5.16.1` 有两处不兼容：`_tied_weights_keys` 的旧列表写法会让权重加载直接失败，`create_causal_mask` 的参数名也从 `input_embeds` 改成了 `inputs_embeds` 并去掉了 `cache_position`。BeatForge 会在加载前就地改写本地模型目录里的这份远程代码，并打印「已修复 … 的远程代码兼容性：…」；改写依据是**已安装**的 `create_causal_mask` 签名，所以新旧参数名都能正确处理，且重复运行不会重复改写。`scripts/director_model_probe.py` 用一份只有几百万参数的迷你模型在 CPU 上跑通同一份远程代码，不需要 7.7GB 权重和显卡就能验证注意力实现、因果性、滑动窗口和 KV 缓存一致性：

```bash
uv run python scripts/director_model_probe.py
```

### 提示词上限与显存预留

导演是整条链路里唯一会把超长序列喂给语言模型的一步，而 Spark-X2.5 的注意力是手写的 `torch.matmul` + softmax：没有 SDPA 或 flash-attn 后端，滑动窗口层也只是给完整的 `[头数, 提示词, 提示词]` 分数矩阵加掩码，并不切掉 KV。这个开销随提示词长度平方增长，因此提示词必须限长，且显存预算必须为它单独留出空间。

BeatForge 用两道闸门处理：

- `director_prompt_tokens`（默认 `2600`）限制提示词长度。超长歌曲会按阶梯逐级降级——先裁剪逐句候选表，再减少送入的素材条数，最后才对歌词抽样——每一级都用 tokenizer 实测 token 数，直到装进预算为止。被裁掉候选表时，提示词里的字段说明会同步改写，不会指向已经不存在的表。
- 加载前先按公式估算序列侧开销（注意力分数矩阵、掩码、保守估的 logits、KV 缓存、激活），再从**当前空闲显存**而不是显卡总容量里扣除，剩下的才是权重预算，并且权重预算不会超过 `director_gpu_memory_gb`。加载和生成期间若仍抛显存不足，会自动用更小的权重预算重试一次。

启动时会打印一行诊断，例如：

```text
导演提示词约 2581 tokens · 空闲显存 10.0GiB · 权重预算 6.7GiB（其余 3.3GiB 留给注意力矩阵、日志张量和 KV 缓存）
```

如果这行显示权重预算明显偏低（比如低于 4GiB），说明这首歌的提示词偏长或显卡上还有其他进程占用显存，可以降低 `director_prompt_tokens`，或先关闭占用显存的程序再渲染。`scripts/director_memory_probe.py` 可以在不加载模型的情况下打印不同提示词长度对应的显存预留：

```bash
uv run python scripts/director_memory_probe.py
```

按 12GB 显卡、实测空闲 10GiB 计算，提示词从修复前的约 13000 tokens 降到 2581 tokens 后，序列预留从 23.4GiB（远超显卡容量，必然 OOM）降到 3.3GiB，权重预算从「负数被夹到 1.0GiB」变成 6.7GiB。

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
image_composites = true
image_composite_ratio = 0.24
max_composite_images = 3
avoid_asset_repeats = true
blurred_image_background = true
image_background_blur = 26.0
image_foreground_scale = 0.92
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

图片素材始终保持原始宽高比，不会为填满横屏或竖屏而拉伸。默认把同一图片等比放大、裁切并模糊为满屏背景，再把清晰原图等比缩放到前景；`image_foreground_scale` 控制前景占画面比例。关闭 `blurred_image_background` 后仍保持原比例，但不再施加背景模糊。

图片镜头不再只有随机推拉。规划器会参考歌曲段落、局部能量、旋律变化以及 AI 导演给出的 `edit_intent`，自动选择以下效果，并把结果和辅助图层写入 `plan.json`：

- `cinematic_depth`：克制的景深推拉，适合主歌、前奏和尾奏；
- `focus_pull`：清晰前景配合柔化背景，随旋律缓慢推进；
- `pan_reveal`：具有明确方向的画面揭示；
- `split_screen`：两张语义相关图片分屏并置；
- `photo_stack`：图片按真实节拍依次滑入并轻微旋转叠放；
- `double_exposure`：双图银幕混合，适合桥段、梦幻或抽象意境；
- `beat_montage`：最多四张图片在镜头内部按音乐节拍依次切换。

图片运镜使用 `perspective` 滤镜逐帧求值裁剪四边形，而不是 `zoompan`。`zoompan` 会把裁剪原点截断到输入帧的整数像素上：本项目的运镜速度通常只有每帧 0.03–0.1 像素，于是画面会连续多帧完全静止、再突然跳一个整像素，在任何直线边缘上都能看出规律性的顿挫。`perspective` 对每帧做真正的亚像素仿射重采样，单次重采样即可完成，既不会像超采样那样二次损失清晰度，也不会出现整像素跳变。裁剪框宽度是 `W/zoom`，可平移量程是 `W - W/zoom`，两者不可互换。

多图辅助素材来自同一句歌词的视觉语义排序，同时受质量、色彩连续性、重复使用和分辨率惩罚约束。默认仅约 24% 图片镜头使用多图组合，副歌和 AI 导演标记的冲击段会提高概率，呼吸段会降低概率，避免整支 MV 变成模板化电子相册。`image_composites = false` 可只保留单图景深和方向性运镜。

### 素材复用规则

`avoid_asset_repeats = true`（默认）时，规划器把素材分成"已出现次数"分层，每个镜头都从出现次数最少的素材里选，因此只要素材数量够，同一个素材就不会出现两次：素材数量不少于镜头数时，全片零复用；素材不足时，复用次数在所有素材之间尽量均分，而不是反复使用语义最匹配的那几张。相邻镜头之间的复用会被额外扣分，同一句重复歌词也不会用同一张画面。

多图合成按"富余素材"计费：只有当素材在满足剩余镜头所需之后还有剩余时，才允许把它用作分屏、照片堆叠或双重曝光的辅助图层，否则该镜头退回单图运镜。素材非常紧张时（例如 3 张图配 40 个镜头）多图合成仍然保留，因为此时复用已经无法避免。想恢复旧的"按得分自由复用"行为（包括副歌视觉母题复现），把 `avoid_asset_repeats` 设为 `false`。

图片和视频的运动强度同时参考局部能量、旋律变化率和歌曲意境。高能/高节奏密度段落使用锐化与亮色闪切，低能段落使用柔化与长淡入，梦幻和抒情歌曲降低镜头运动，所有镜头可选暗角和动态胶片颗粒。

为提高成片统一性，渲染器会把 AI 导演的 `color_arc` 真正转换成按章节推进的轻量色彩风格，而不是只写入 JSON；同时依据素材代表色做受限的曝光和饱和度匹配，最大修正量受到 `shot_match_strength` 控制，不会把夜景强行拉成白天。`look_strength` 控制导演色彩弧强度，设为 `0` 可完全关闭。缩放统一使用 Lanczos，并校正像素宽高比。

中间片段默认使用 `intermediate_crf = 14`，最终交付使用 `crf = 19`，减少转场合成和字幕压制中的多代有损损失；程序会自动保证中间 CRF 不高于最终 CRF。`encoder_tune = "film"` 用于保留肤质、自然纹理和细颗粒，也可根据动画素材改为 `animation`，明显颗粒化素材可用 `grain`。最终选择与每个镜头的代表色都会写入 `plan.json`。

## CPU 与 NVIDIA GPU 配置

CPU兼容配置（用于功能验证，完整推理会很慢）：

```toml
[ai]
device = "auto"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "wemm-embedding"
vision_model = "tencent/WeMM-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
vision_batch_size = 2
music_structure_backend = "allin1"
frame_samples = 3
director_enabled = true
director_model = "XHToken/Spark-X2.5-4B"
director_backend = "text"
director_prompt_tokens = 2000
director_gpu_memory_gb = 9.0
```

12GB 显存及以上 NVIDIA 显卡推荐配置：

```toml
[ai]
device = "cuda"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "wemm-embedding"
vision_model = "tencent/WeMM-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
vision_batch_size = 4
music_structure_backend = "allin1"
frame_samples = 8
director_enabled = true
director_model = "XHToken/Spark-X2.5-4B"
director_backend = "text"
director_prompt_tokens = 2600
director_gpu_memory_gb = 9.0
```

运行 `uv run beatforge doctor` 检查实际使用 CPU 还是 CUDA。歌词识别统一使用 Qwen3-ASR；原 Qwen3-VL-Embedding 和 SigLIP2 仍可通过 `vision_backend` 作为兼容后备。

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
- 素材够用时每个镜头都换新素材，同一素材只在素材数量不足时才再次出现，且复用会在所有素材之间均摊；
- 主歌保持色彩和主体连续，副歌提高视频镜头比例和切换强度；
- 副歌复现少量视觉母题，让成片有记忆点（素材充足时以不复用同一素材优先，母题主要在素材不足时体现）；
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

先确认安装的是 `ai-cuda`，而不是 `ai-cpu`。`ai-cuda` 需要能够运行 CUDA 13.0 构建的驱动；CUDA 13.x 至少需要 R580 系列驱动。`nvidia-smi` 能显示显卡不代表当前 Python 环境中的 PyTorch 一定启用了 CUDA；以 `uv run beatforge doctor` 的结果为准。

如果警告中显示 `found version 12080`，说明当前驱动无法运行 CUDA 13.0 PyTorch。请升级到支持 CUDA 13.x 的 R580 或更新驱动并重启，然后重新同步 `ai-cuda` 环境。

### PyTorch 与 TorchAudio 的 CUDA 版本不同

例如旧环境中的 `PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0`，表示 PyTorch 包来自不同 CUDA 软件源。更新代码后用 `ai-cuda` profile 强制刷新；不要单独运行 `pip install torchaudio`。修复后，下面四个导入必须同时成功，版本后缀应全部为 `+cu130`：

```bash
uv run python -c "import torch, torchvision, torchaudio, torchcodec; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torchcodec.__version__, torch.version.cuda)"
```

### ModelScope 下载 WeMM 失败

BeatForge 会把 `tencent/WeMM-Embedding-2B` 自动映射为魔搭命名空间的 `tencent-community/WeMM-Embedding-2B`。先检查网络、磁盘空间和 ModelScope 登录或访问限制。默认 `--source auto` 会在魔搭下载失败后回退到 Hugging Face；成功下载后检查 `.beatforge/models.json`，并启用 `offline = true`。

### Qwen3-VL-Reranker 提示缺少 `true_token_id`

部分 ModelScope 本地快照即使已经包含 `1_LogitScore/config.json`，Sentence Transformers 的自动模块加载仍可能没有把 token ID 正确传给 `LogitScore`。BeatForge 对 Qwen3-VL-Reranker 显式构造官方的 `Transformer(any-to-any) + LogitScore` 模块链，并从模型自身 tokenizer 获取 yes/no token ID，因此不依赖自动读取这份模块配置。程序还会覆盖不兼容的旧聊天模板，使 `query`、`document` 以及图片占位符都能进入实际提示词；无需重新下载整套权重。

### WeMM 报自定义代码、Processor 或配置加载错误

确认使用 `tencent/WeMM-Embedding-*` 官方仓库，并已安装 `qwen` extra：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
```

不要绕过锁文件随意降级 Transformers。先记录完整异常、当前 `transformers` 和 `sentence-transformers` 版本，再确认本地模型目录是否下载完整。模型仓库更新后，如需重新获取自定义代码，应在联网模式下明确重新下载并复测，确认无误后再恢复离线模式。

### CUDA 显存不足

- 如果在视觉编码过程中OOM，先把 `vision_batch_size` 调为 `1`；程序也会自动减半重试。
- 如果 WeMM 模型刚加载时OOM，改用 WeMM-Embedding-2B；降低批量大小对此无效。
- 如果导演阶段OOM，先降低 `director_prompt_tokens`（提示词长度是显存占用的平方项），再降低 `director_gpu_memory_gb` 和 `director_max_new_tokens`，由 Accelerate 把更多权重卸载到内存。程序本身也会自动改用更小的权重预算重试一次，日志里会出现「导演阶段显存不足，改用 X GiB 权重预算重试」。
- 渲染前先关闭其他占用显存的程序：导演的权重预算按**空闲显存**计算，被浏览器或游戏占掉几个 GB 会直接压低预算。
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

查看不同素材数量下的复用分布（验证"素材够用时不复用"是否生效）：

```powershell
uv run python scripts/reuse_probe.py
```

查看不同提示词长度对应的导演显存预留（不需要显卡，也不加载模型）：

```powershell
uv run python scripts/director_memory_probe.py
```

排查图片运镜的帧间抖动（需要 FFmpeg，不需要显卡）。`jitter_probe.py` 用相位相关测量相邻帧之间的亚像素位移，并列出每一帧的步长；`zoompan_probe.py` 用受控光斑对比 `zoompan`、`perspective` 与超采样三种实现；`camera_move_probe.py` 走真实渲染路径，`--controlled` 会关闭颗粒、暗角和调色，让位移测量不被噪声干扰：

```powershell
uv run python scripts/jitter_probe.py demo/.beatforge/clips/00002.mp4
uv run python scripts/zoompan_probe.py
uv run python scripts/camera_move_probe.py --controlled
```

这三个工具都会先打印一个**一致性**（`|累计位移| / Σ|单帧位移|`）：测量成立的前提是相邻帧互为刚体变换，而暗角、颗粒、调色都固定在画面坐标上，会破坏这个前提。一致性低于 0.5 时脚本会直接判定"测量不可信"并拒绝给出结论——此时看到的"抖动"是噪声，不是成片的问题。要判断真实运镜是否平滑，请用 `camera_move_probe.py --controlled` 或 `zoompan_probe.py` 的受控片段。

在你自己的 AI 环境中执行模型烟雾测试：

```powershell
uv run beatforge doctor
uv run beatforge run demo/project.toml --plan-only
```

## 代码结构

```text
beatforge/audio.py                    节拍、章节与能量分析
beatforge/director.py                 导演方案到字幕与视觉艺术指导
beatforge/models/ai_director.py       本地 Spark-X2.5 导演协议与校验
beatforge/models/transcriber.py       Qwen3-ASR 与强制对齐时间轴
beatforge/models/audio_semantics.py   CLAP 音乐语义
beatforge/models/music_structure.py   All-In-One/Beat This 结构分析
beatforge/models/vision_index.py      WeMM/Qwen3-VL-Embedding/SigLIP2 检索
beatforge/planner.py                  多目标镜头编排
beatforge/renderer.py                 FFmpeg 成片渲染
beatforge/pipeline.py                 分阶段模型生命周期
scripts/                              演示素材、复用/显存/抖动探针
tests/                                不下载模型的测试
```
