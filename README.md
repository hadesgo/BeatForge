# BeatForge

BeatForge 是一个 Python + uv 的本地 AI 音乐视频剪辑器。输入音乐、图片/视频以及可选的 LRC 歌词，输出自动编排的 MV 和一份可审阅的镜头决策文件。

## AI 流程

```text
音乐 ── MelBand-RoFormer 人声分离 ── Qwen3-ASR + ForcedAligner ── 逐字时间轴 ──┐
  └── All-In-One + CLAP ── 旋律/节拍/章节/意境 ──┼── Spark-X2.5 AI 导演 ── 确定性规划器 ── FFmpeg
图片/视频 ── WeMM-Embedding + Qwen3-VL Reranker ─────┘        │
                                                      导演方案 JSON
```

- `vocals_mel_band_roformer.ckpt`：MelBand-RoFormer 人声分离，只作用于歌词识别，见下文；
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

人声分离是独立的可选 extra：`uv sync --extra ai --extra ai-cpu --extra separation`（CUDA 机器把 `ai-cpu` 换成 `ai-cuda`）。它需要 PyTorch，并且**绕不开两个未声明的依赖**：`audio-separator` 在模块顶层就 import `onnxruntime`（即使 RoFormer 检查点本身走 torch 推理），`uvr_lib_v5/spec_utils` 又在顶层 import `audioread`——后者根本没写进它的依赖表。两个都已在 extra 里显式声明，CPU 版就够。

`ai-cpu` 和 `ai-cuda` 两个 profile 互斥，uv 会阻止混装。`ai-cuda` 从 CUDA 13.0 源安装 `torch 2.14.0+cu130`、`torchvision 0.29.0+cu130`、`torchaudio 2.11.0+cu130` 和 `torchcodec 0.16.0+cu130`。以上包均提供 Python 3.13 轮子。不要再单独覆盖其中任何一个包。模型权重不会在 `uv sync` 时下载。

### 转录前的人声分离

ASR 模型被要求从**整轨混音**里听人声，等于要求它隔着鼓组听人说话。词确实在那里，但它要和所有东西争抢模型的注意力，而失败方式不是听不出来——是**听出一份自信的错误歌词**，恰好在编曲最密的地方。所以先分离再转录：

```
音乐 ── MelBand-RoFormer ── 人声轨 ── Qwen3-ASR ── 歌词
                    └─────── 伴奏轨（丢弃）
```

- **只给人声轨做转录，但强制对齐也用同一条人声轨**。对齐问的是"这个词是**什么时候**唱的"，而这个问题在没有军鼓压着的时候好回答得多；
- **拍点、能量、段落分析仍然跑整轨混音**。那些要的就是鼓，人声轨里没有鼓；
- 结果按「源文件 + 模型」缓存，重渲染不会重复付分离的代价。**换分离模型会让缓存失效**——两个检查点给出两条不同的人声轨，而混用它们在下游完全看不出来，只会表现成歌词略有不同；
- 分离是可选依赖，**装不上时会退回整轨并明确说明**。降级本身没问题，悄悄降级才是问题：整轨转录出来的是另一份、更差的歌词，而下游没有任何环节能分辨。

**选轨必须按文件名里的分句标记，不能按子串匹配。** 该包的输出命名是 `<输入>_(<分句>)_<模型>.wav`，而默认检查点就叫 `vocals_mel_band_roformer`——于是**每个输出文件名都含 "vocals"，包括伴奏**。按子串匹配会选中伴奏轨喂给识别器，症状是转录出一片空白，看起来像模型坏了而不是选轨错了。这个坑是在第一次端到端实跑时抓到的（缓存下来的"人声轨"低频占 52.5%，比原始混音还高）。

CPU 推理速度约 **204 秒 / 分钟音频**（45 秒音频实测 153 秒），一首 3.5 分钟的歌约 12 分钟。结果有缓存，重渲染不会重复付这个代价。

分离用的检查点由 `separation_model` 指定，默认 `vocals_mel_band_roformer.ckpt`。`beatforge download-models` 会把它一起下载——它不是 Hugging Face / ModelScope 的仓库快照，而是 `audio-separator` 自己目录里的单个检查点，所以走一条单独的下载路径（该包的 `download_model_files()`，只下载不加载，不需要显存）。下载目录与运行时查找目录由同一个常量派生，否则下载步骤会静默失效、模型在首次使用时被重新拉一遍。

没装 `separation` extra 时，下载步骤会**跳过并说明缺哪个 extra**，而不是让整条命令失败——`separate_vocals` 默认为 `true`，如果这里直接报错，默认配置就会在没装 extra 的机器上整体跑不通。

`audio-separator` 自带一份各模型的实测基准（`models-scores.json`），按人声 SDR 排下来：

| 平均 SDR | 最高 SDR | 检查点 |
| --- | --- | --- |
| 11.53 | 15.57 | `mel_band_roformer_kim_ft_unwa.ckpt` |
| **11.49** | **15.63** | **`vocals_mel_band_roformer.ckpt`（默认）** |
| 10.02 | 16.13 | `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` |

默认值与前两名在噪声范围内持平，取的是最通用、被引用最多的那一个。想换更强的（例如针对特定语种或编曲微调过的 RoFormer）只改这一行，缓存会自动按新模型重建。用卡拉 OK 模型（`mel_band_roformer_karaoke_*.ckpt`）也能分离人声，但它们是为去人声设计的，主轨是伴奏，平均 SDR 明显更低。

### 模型兼容性与安全

- Qwen3-ASR 从 Transformers 5.13.0 起提供原生支持；BeatForge 当前统一要求 `transformers>=5.16.1`，以同时满足 ASR 和视觉模型集成，不建议只为复现某个模型示例而单独降级。
- WeMM 官方示例曾推荐 `transformers==5.2.0`，BeatForge 使用更新版本与 `sentence-transformers[image]>=5.7`。这个组合已通过接口级测试，但尚未在本机用真实 WeMM 权重完成GPU烟雾测试；部署时以 `uv.lock` 创建环境后先运行本文的烟雾测试。
- WeMM 通过 `trust_remote_code=True` 加载腾讯官方模型仓库中的自定义 Python 代码。只应下载可信的官方仓库；把 `vision_model` 改成其他仓库时，也等于信任并执行该仓库的模型代码。权重准备完成后建议设置 `offline = true`，避免运行过程中访问网络或获取变化后的代码。
- `doctor` 检查运行库、FFmpeg、CUDA和显存是否就绪，不会真正加载数十GB模型；完整兼容性以一次真实的 `--plan-only` 运行为准。

### 统一下载模型

统一下载项目配置中启用的 ASR、强制对齐、音乐情绪、视觉检索、视觉精排、AI 导演和人声分离模型：

```powershell
uv run beatforge download-models my-mv/project.toml
```

默认 `auto` 模式优先从 ModelScope（魔搭社区）下载，适合中国大陆网络；某个仓库在魔搭不存在时才回退到 Hugging Face。Spark-X2.5 若魔搭没有同名仓库，会自动回退 Hugging Face。人声分离检查点来自 `audio-separator` 自己的目录，走单独路径，不受 `--source` 影响；未安装 `separation` extra 时会显示"跳过"并说明缺哪个 extra，其余模型照常下载。

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
| `edit_style` | `auto` | 剪辑风格：镜长、切点网格、转场密度与取向、景别对比、运镜强度的整套预设 | 想换一种节奏语言时固定成某个风格名 |
| `device` | `auto` | 自动选择CUDA或CPU；正式GPU运行可设为 `cuda` 以尽早暴露环境问题 | `cuda` |
| `offline` | `false` | 为 `true` 时仅使用模型清单和本地缓存 | 下载完成后设为 `true` |
| `vision_model` | `tencent/WeMM-Embedding-2B` | 决定歌词与画面的语义召回质量，也是视觉阶段主要显存占用 | 保持4B原始精度 |
| `vision_batch_size` | `4` | 影响视觉编码吞吐和激活显存；OOM会自动按4→2→1重试 | `4`，仍OOM时设 `1` |
| `vision_rerank_top_k` | `8` | 每句歌词进入精排的候选数；更高可能改善选镜，但更慢 | `8` |
| `vision_input_pixels` | `1003520` | 每张图送进视觉编码器前的像素上限（= 1280×28×28） | 显存紧张时降到 `501760`（约 0.5MP） |
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
| `transition_density` | `0.35` | 段落内部使用可见转场的比例；`0` 只在段落切换时转场，`1` 每个切点都转场 | `0.25`–`0.45` |
| `image_background_blur` | `26.0` | 原比例图片周围的满屏模糊背景强度 | 人像可用 `22`–`32` |
| `subtitle_effect` / `subtitle_font` | `auto` / `auto` | AI按旋律、情绪和段落从 16 种字幕动效里选，并选择字体 | 保持 `auto`，也可固定成某个特效名 |
| `subtitle_layout` | `free` | `free` 分句自由排版并避开主体；`band` 为传统底部居中一行 | 想做成官方歌词 MV 那样就用 `free` |
| `subtitle_fill` | `solid` | `knockout` 让文字从画面里镂空，字中透出提亮虚化的同一帧 | 想要"融入画面"就用 `knockout` |
| `subtitle_outline` | `1.1` | 描边宽度；越小越融入画面，`0` 为无描边 | 画面偏暗时用 `0`，杂时用 `1.5`–`2.2` |

`edit_style` 不为 `manual` 时会**接管**镜长、转场密度和多图比例——风格和手工值是对同一个问题的两个回答，同时给会互相打架；风格没有表态的项（分辨率、调色、字幕特效）仍走各自的配置。`vision_batch_size` 只影响编码时的激活显存，不能解决模型权重加载就OOM的问题。`vision_input_pixels` 决定每张图进编码器前的像素上限，直接决定视觉塔的 patch 数量和激活显存；它只压缩超出预算的素材，小图不受影响。`frame_samples` 和 `director_contact_sheet_assets` 主要交换分析时间与选择信息量，并不会让最终视频分辨率变高。`director_gpu_memory_gb` 只是权重上限，导演的注意力矩阵和 KV 缓存会另外占用显存，所以它必须明显低于显卡总容量。

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
subtitle_layout = "free"
subtitle_fill = "solid"
subtitle_outline = 1.1
subtitle_margin = 72
subtitle_highlight_color = "&H0000D7FF"
visual_effects = true
image_composites = true
image_composite_ratio = 0.24
max_composite_images = 3
avoid_asset_repeats = true
transition_density = 0.35
blurred_image_background = true
image_background_blur = 26.0
image_foreground_scale = 0.92
vignette = true
film_grain = 1.6
look_strength = 0.72
shot_match_strength = 0.3
```

### 版式：让字幕成为画面的一部分

默认的 `subtitle_layout = "free"` 不做底部字幕条。它参考官方歌词 MV 的做法，把一句歌词**按歌手换气的位置切开**，分片摆到画面里主体不占的地方，每片在自己被唱到的时刻淡入：

- **断句在停顿处**，不在字数中点。中文没有词间空格，按字数平分会把词切开——"黎明照亮天空"对半分成"黎明照 / 亮天空"，把"照亮"劈成两半。所以断点取**逐字时间轴里最长的那段静音**（`_MIN_BREAK_SECONDS = 0.22`）；没有逐字时间轴时退到标点；两者都没有就**整句不拆**，只做自由摆放；
- **十套版式轮转**，每句走不同的构图：上下夹持、升/降对角、同基线拉开间距、左对齐堆叠、右对齐堆叠、居中紧堆、对角两角、两侧贴边、下沉居中。版式表按"给中间留多少空间"排序，轮转步长与表长互质，所以**十套走完才重复**，且相邻两句的构图必然不同。每套版式都混用左/中/右对齐——只换位置不换对齐，读起来仍然像同一套构图换了地方；
- **摆位避开主体**，用 `focus_point`（素材阶段已经估好的主体中心）。**避让是二维的**：参考片把字摆在主体**旁边**的次数和摆在上下一样多，只按垂直方向避让会把这类版式全部否掉——早先的版本就是这样，结果主体一偏，整首歌塌回两组固定位置。现在的做法是把候选版式**整体平移**出主体所在的方框，平移到不了就**换下一套版式**（最多试 4 套），所以主体偏侧时版式的多样性也保得住；
- **每句有一点点确定性微偏移**（由行号算出，不用随机数），免得同一套版式在一首歌里第二次出现时落在完全相同的像素上；
- **每片按演唱时刻出现**。整句不再一次性出现，而是随着演唱在画面各处依次亮起——这是参考片最核心的一条，也是它读起来"在画面里"而不是"浮在画面上"的原因。所以自由版式下 `karaoke` 不再逐字扫光：分片本身已经承载了时间信息，再扫一遍是把同一件事说两遍。

`subtitle_layout = "band"` 恢复传统的底部居中一行，逐字扫光的 `karaoke` 也在那个版式下保留。`subtitle_outline` 默认 1.1（发丝描边）；参考片是**完全无描边**，设成 `0` 可以得到同样的效果，但需要画面本身够暗——白字落在亮部会读不出来。

`subtitle_fill = "knockout"` 是"融入画面"的字面做法：**没有字幕层，只有画面里一个字形空洞**，字中透出同一帧提亮虚化后的样子。

- 遮罩就是同一份歌词脚本渲染成白字黑底，`alphamerge` 取它的亮度当 alpha，所以脚本里任何动画（淡入、缩放、分片延迟）都会让空洞同步跟随；
- 字被提亮的同时**画面整体被压暗**。只提亮是不够的——自由版式故意把字放在画面空的地方，那里往往平滑而暗，只提亮会让字直接消失。压暗之后对比度成为滤镜图的固有属性，而不是碰巧取决于字背后是什么。

字幕动效共 16 种，按剪映式文字动画的三类组织——入场、持续、以及故障那一种异类：

**入场**（只在句子开头演一次）

- `cinematic`：模糊消散和长淡入淡出，最中性的一种；
- `bounce`：随句子出现的弹跳缩放；
- `typewriter`：逐字出现，适合暗黑、叙事感段落；
- `punch`：从 185% 快速砸回原位并带走运动模糊，用于冲击句；
- `slide`：从侧面滑入；
- `flip_in`：逐字从下方翻转立起，整句像被"组装"出来。

**持续**（句子在屏期间一直在演）

- `karaoke`：逐字高亮，并带轻微缩放入场；
- `float`：伴随舒缓旋律缓慢上浮；
- `glow`：柔光入场，适合浪漫和梦幻段落；
- `neon`：白色字芯外套青色光晕，光晕缓慢呼吸；
- `neon_flicker`：霓虹还没热起来——大部分时间是亮的，中间抖两次；
- `shake`：`\jitter` 抖动叠一层轻微摇摆（`\jitter` 是 libass 扩展，所以摇摆是保底动作）；
- `wave`：逐字绕 Y 轴翻转，一道波浪从句子左端走到右端；
- `rainbow`：六种色相在句内循环一圈，首尾同色所以接得上；
- `spotlight`：从暗色低亮开始，像灯光刚找到这句话。

**异类**

- `glitch`：品红描边配青色阴影做出色差，再叠两次透明度抖动，用于信号中断般的瞬间。

效果名可以直接写进配置固定使用，也可以交给 AI 导演逐句挑选：`subtitle_effect = "auto"` 时，句子按段落、能量、旋律和情绪从上面这张表里选——副歌高能量句走冲击组（`punch`/`shake`/`neon_flicker`），旋律句走流动组（`wave`/`neon`），暗黑情绪走 `glitch`/`spotlight`/`typewriter`。

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

图片镜头不再只有随机推拉。规划器会参考歌曲段落、局部能量、旋律变化以及 AI 导演给出的 `edit_intent`，自动选择效果，并把结果和辅助图层写入 `plan.json`。单图运镜按**性格**区分，而不是按幅度区分——漂移保持构图只做横移，推轨则明确承诺一次推进，冲击快到位后定住，呼吸是膨胀后回到原位：

- 铺垫型，用于主歌、前奏和尾奏：`cinematic_depth` 克制的景深推拉、`focus_pull` 清晰前景配合柔化背景的缓推；
- 揭示型，画面要"露出"什么：`pan_reveal` 横向平移揭示、`tilt_up` / `tilt_down` 垂直摇移、`arc` 弧线绕行、`drift` 几乎不缩放的纯横移；
- 承诺型，切点要求一个明确动作：`dolly_in` 推进、`dolly_out` 拉远、`punch_in` 快速推入后定住、`pull_back` 从近景退回；
- 呼吸型：`breathe` 在中段膨胀再回到原位，用于长镜头和低能量段落。

在此之上还有一组剪映式运镜，它们改变的是**画面的空间感**而不只是取景范围：

- `whip_pan`：甩镜。能甩多远取决于可平移量程 `W - W/zoom`，所以这个运镜裁得很深——浅裁的话整段位移只有几个像素，无论名义上多快都读起来像慢漂移；配 `tmix` 沿路径混邻近帧，没有这层拖影快摇会像跳切；
- `spiral_in`：推进的同时缓慢滚转，读起来像绕着主体走而不是直冲过去；
- `roll_drift`：几乎不推的荷兰角漂移，由滚转承担全部动感；
- `handheld`：手持晃动。抖动由多个不成倍数的正弦叠加而成（`perspective` 不提供随机源），并按 `on/fps` 计时，所以长镜头上的抖动频率不会变慢成"摇摆"；
- `tilt3d_back` / `tilt3d_front` / `tilt3d_left` / `tilt3d_right`：3D 转向。上下边宽度（或左右边高度）按不同速率变化，平面因此读起来有正面和背面——这是唯一让静态图产生"在空间里转动"错觉的运镜；
- `pulse_in`：一次镜头里推两下，两次都落在拍子上。

三种复合动效按约十二分之一的概率落在单图镜头上：

- `film_bars`：叠加 2.35:1 上下遮幅，画面本身不裁切；
- `iris`：圆形遮罩从中心一点开到全画幅，像锁孔逐渐打开；
- `parallax`：同一张图分成模糊背景和清晰前景两个平面，以不同速度平移，形成视差纵深。

多图合成：

- `split_screen`：两张语义相关图片分屏并置；
- `photo_stack`：图片按真实节拍依次滑入并轻微旋转叠放；
- `double_exposure`：双图银幕混合，适合桥段、梦幻或抽象意境；
- `beat_montage`：最多四张图片在镜头内部按音乐节拍依次切换。

**合成里每一格都全出血，不用单图那套"留边 + 虚化副本"。** 单图处理（`_adapt_image`）是为了一张填不满画面的图片设计的：按 92% 缩放居中，周围用**同一张图的虚化副本**补满。它在合成里错两次——每张图各自带来一片虚化底，一帧里就有两个互相竞争的背景；而且同一张图会以两个不同尺度出现两遍。这正是"堆叠看起来很乱"的来源：底图是清晰的自己叠在自己的虚化副本上，再往上压卡片，等于三个尺度同框。并排的"割裂"也是同一个原因：一格走留边处理、另一格全出血，两边取景和曝光看起来就不一样，读起来像两张恰好挨着的图，而不是一张图被切成两半。现在 `_fill_frame` 对每一格做 `increase` 缩放 + 裁切，只有**单图**效果才保留留边处理。堆叠的底图另外压暗去饱和（`brightness=-.085` / `saturation=.74`），让卡片能读出来。

图片运镜使用 `perspective` 滤镜逐帧求值裁剪四边形，而不是 `zoompan`。`zoompan` 会把裁剪原点截断到输入帧的整数像素上：本项目的运镜速度通常只有每帧 0.03–0.1 像素，于是画面会连续多帧完全静止、再突然跳一个整像素，在任何直线边缘上都能看出规律性的顿挫。`perspective` 对每帧做真正的亚像素仿射重采样，单次重采样即可完成，既不会像超采样那样二次损失清晰度，也不会出现整像素跳变。裁剪框宽度是 `W/zoom`，可平移量程是 `W - W/zoom`，两者不可互换。

滚转和 3D 转向都复用**同一次** `perspective` 处理：四边形由同一个中心点和同一个半尺寸推导出来，再整体旋转或做梯形变形。这样既不额外增加一次重采样，也不会有第二次重采样带来的清晰度损失。代价是能平移的量程必须按四边形的**包围盒**而不是裁剪框来算——转过的角点会甩到裁剪框外，梯形较宽的那条边也会顶到画面边缘；`_required_zoom` 会把整条 zoom 曲线抬高到足够容纳包围盒的高度（抬高整条曲线而不是夹住某几帧，是为了保住运镜原本的推进幅度）。

运镜强度 `amount`（由艺术指导的 `camera_intensity`、该镜头的旋律活跃度和 `motion` 共同决定，实测区间约 0.25–3.2）会同时缩放推拉、滚转、手持抖动和梯形变形。估算包围盒的 `_quad_spread` 与生成四边形的 `_camera_quad` 必须用**同一个** `amount`：只把 `amount` 用在其中一边，估算就会与真实四边形脱节，`amount < 1` 的镜头会抬升不足而把角点甩出画面。这种偏差在 `amount == 1` 时完全看不出来，所以测试和探针都必须**扫过整个 `amount` 区间**，只测一个点等于没测。

关于 `perspective` 有三个容易踩空的地方，改动运镜时必须一并核对：`on` 是**从 1 开始**的输出帧序号，而且会**越过镜头长度**继续增长（静态图来自 `-loop 1`，是无限长的，下游的 `fps`、`trim` 会为了确定自己的时间戳多拉几帧），所以进度必须写成 `clip((on-1)/(frames-1),0,1)`——少了偏移会在末帧过冲，少了钳制则会让"推进到终点停住"的运镜在末尾反转，zoom 一旦掉到 1 以下，`W - W/zoom` 变负，滤镜会直接拒绝该帧。其次，每个运镜都额外抬高了 0.4% 的放大率，避免出现"裁剪框正好等于整帧"的退化映射。第三，表达式里的 `clip()` 不会交换上下界，它算的是 `min(max(x, min), max)`，所以上界一旦为负就会原样返回负数。

`scripts/quad_probe.py` 会逐帧求值全部运镜的四边形，扫过 10 个 `amount`，报出最紧的余量（负值即越界像素数）和对应的 `amount`，以及形变量——滚转角度或梯形强度调大时先跑它，比等 ffmpeg 在渲染中途拒绝某一帧要快得多。

多图辅助素材来自同一句歌词的视觉语义排序，同时受质量、色彩连续性、重复使用和分辨率惩罚约束。默认仅约 24% 图片镜头使用多图组合，副歌和 AI 导演标记的冲击段会提高概率，呼吸段会降低概率，避免整支 MV 变成模板化电子相册。`image_composites = false` 可只保留单图景深和方向性运镜。

多图门控和运镜轮转**互不共享计数器**：轮转按"已选出的单图镜头数"推进，而不是按镜头总序号。两者若共用序号，被合成吃掉的那些序号对应的轮转槽位就永远轮不到——实测同一首歌只用到 13 种运镜，改成独立计数后用满 21 种。

转场轮转也有同一类风险，但成因不同：转场分支很窄（`open` 只在梦幻歌曲的段落切换处出现，一首歌只有两三次），按镜头序号轮转会让这几次触发全落在同一个余数上。所以转场按"已分配的**可见**转场数"轮转，密度门控仍用镜头序号——两者回答的是不同问题。

`tests/test_planner.py` 里有两个跨歌曲扫描的测试，断言**每一个**运镜和**每一个**转场族都至少能被某首歌选中一次。写新的选择分支时要一起扩充它们，否则新加的名字可能根本轮不到；而单个样本歌曲测不出这种饿死，因为覆盖率取决于该曲的段落与能量分布。

### 转场

转场分两类。第一类走 `xfade`：规划器按"转场族"下发，渲染器再结合**进入镜头**的色调和**离开镜头**的运动方向，解析成具体动作。共 21 族、58 个具体转场（`fadewhite` 同时属于 `dip` 和 `flash`）：

| 族 | 具体转场 | 时长 | 读作 |
| --- | --- | --- | --- |
| `dissolve` | `dissolve` / `fade` / `fadegrays` | 0.44s | 平静的溶解；暗调段落改用去色溶解 |
| `dip` | `fadeblack` / `fadewhite` | 0.36s | 淡黑表示沉重，淡白表示释放 |
| `flash` | `fadewhite` / `fadefast` | 0.16s | 冲击；必须短才成立 |
| `wipe` | `wipeleft` / `wiperight` / `wipeup` / `wipedown` | 0.28s | 明确的方向性推进 |
| `slide` | `slideleft` / `slideright` / `slideup` / `slidedown` | 0.28s | 画面整体推移 |
| `circle` | `circleopen` / `circleclose` | 0.42s | 圆形开合 |
| `radial` | `radial` | 0.26s | 放射状擦除 |
| `zoom` | `zoomin` | 0.20s | 快速纵深冲击 |
| `blur` | `hblur` | 0.44s | 柔化的段落过渡 |
| `slice` | `hlslice` / `hrslice` / `vuslice` / `vdslice` | 0.26s | 条状切片 |
| `diag` | `diagtl` / `diagtr` / `diagbl` / `diagbr` | 0.30s | 斜向擦除 |
| `pixel` | `pixelize` | 0.22s | 数字感闪切 |
| `squeeze` | `squeezeh` / `squeezev` | 0.24s | 挤压变形 |
| `reveal` | `revealleft` … / `coverleft` … | 0.32s | 揭示与覆盖 |
| `mask` | `circlecrop` / `rectcrop` | 0.34s | 形状像蒙版一样在切点上张开 |
| `open` / `close` | `vertopen` / `horzopen` / `vertclose` / `horzclose` | 0.38s | 画面裂开，或合拢 |
| `smooth` | `smoothleft` / `smoothright` / `smoothup` / `smoothdown` | 0.30s | 像 `slide`，但进来的一侧是缓入而不是硬推 |
| `corner` | `wipetl` / `wipetr` / `wipebl` / `wipebr` | 0.28s | 从画面四角斜向擦除 |
| `wind` | `hlwind` / `hrwind` / `vuwind` / `vdwind` | 0.22s | 滚动切片，读起来像快门 |
| `soft` | `distance` / `fadeslow` | 0.50s | 无重量感的长过渡，留在一个段落的末尾 |

第二类是**效果型转场**，它们根本不用 `xfade`：滤镜直接烧进两个镜头各自的帧里，时间线上是硬切。这正是"溶解"和"冲击"的区别——冲击发生在切点**上**，而不是横跨切点。

| 族 | 时长 | 做了什么 |
| --- | --- | --- |
| `glitch` | 0.30s | RGB 通道分离 + 传感器噪点，切点上再压一记白闪 |
| `light_leak` | 0.34s | 暖色光晕漫过整个切点，像光从镜头边缘漏进来 |
| `film_burn` | 0.30s | 过曝白闪 + 颗粒，像接片处烧了一下 |

效果型转场需要的时间戳比看起来讲究：`fade` 在 `st + d` 才到达目标色，所以出点一侧要从结束前一帧起算——停在片段末尾的话最后一帧只走到三成，闪光就退化成一记轻微提亮。它们也**不需要 handle**（没有重叠要补footage），因此不会改变总时长。

`transition_density`（默认 `0.35`）控制**段落内部**使用可见转场的比例：`0` 表示段落内部一律硬切、只在段落切换时转场，`1` 表示每个切点都转场。段落切换、乐段边界和导演标记的冲击点不受该比例限制，始终使用可见转场。同一族不会连续出现两次（`cut`、`dissolve`、`blur` 这类含蓄转场除外），避免连续几个切点重复同一个动作；替代族也保持同等强度，把故障换成溶解会把这一刀彻底泄掉。

方向性族（`wipe`、`slide`、`diag`、`reveal`、`slice`、`smooth`、`corner`、`wind`）跟随镜头自身的漂移方向：画面本来就在向右平移时，转场也向右擦除，而不是逆着运动走。未知族名（例如手工编辑过的 `plan.json`）会回退成溶解，而不是硬切，让这个镜头仍然读得出原本的意图。

`scripts/cut_effect_probe.py` 会在真实切点上渲染三种效果型转场并测量切点那一帧：闪光要真的过曝，漏光要真的偏暖，而两者都不能渗到镜头其余部分。只测平均亮度是不够的——噪点和通道分离几乎不改变均值，会看起来"什么都没做"。

拼接整条时间线是**一次** ffmpeg 调用：全部镜头作为输入，转场写成一条滤镜图。卡点快剪会让这一条命令很长（253 个镜头约 27k 字符滤镜图 + 9k 字符输入路径），而 Windows 的单条命令行上限是 32767 字符，超了会在 ffmpeg 启动前就抛 `FileNotFoundError: [WinError 206]`——注意症状很误导：**253 个镜头全部渲染成功**，只在最后拼接时失败。所以滤镜图写进 `<cache>/picture.filter`，用 ffmpeg 通用的"从文件读选项值"形式 `-/filter_complex` 传入（专门的 `-filter_complex_script` 在 ffmpeg 9 已被移除）。输入列表仍是内联的，约 36 字符/镜头。

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
separate_vocals = true
separation_model = "vocals_mel_band_roformer.ckpt"
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
separate_vocals = true
separation_model = "vocals_mel_band_roformer.ckpt"
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

### 送进模型前先把素材压到合理范围

Qwen-VL 系的 processor 在视觉塔看到画面之前，会先把每张图缩放到它自己的像素预算——`max_pixels = 1280×28×28`，约 100 万像素。也就是说，把一张 2400 万像素的原图交给它，**换不来任何模型能看到的细节**，只换来一次全分辨率解码、一次全分辨率缩放，以及两者同时在内存里。而且重排阶段会为**每个候选**重复支付一次，同一张素材在候选表里通常还出现好几次。

所以 BeatForge 会先归一化再送进去：

- **图片素材**不再以原始文件路径喂给模型，而是先生成一份压到预算内的缓存副本（`model-input/`），编码和重排共用同一份；
- **视频抽帧**直接按素材实际尺寸解码到预算内，而不是固定 768 宽——固定宽度对竖屏是失效的，`768×2048` 听起来不大，实际仍然超标；
- 归一化是**上限而不是目标**：本来就在预算内的素材原样透传，连缓存目录都不会创建；
- JPEG 解码还会用 Pillow 的 `draft()` 走 1/2、1/4、1/8 的免解码降采样，大文件连全分辨率解码都省掉；
- 编码完成后立刻释放常驻的关键帧。重排阶段只回读它真正需要的那**一帧**，而不是把整套采样帧再持有到整轮结束。

`scripts/vision_input_probe.py` 用两个独立进程分别测两条路径（这样峰值读数各归各的）。8 张 6000×4000 照片的实测：

| | 耗时 | 峰值内存 | 送进模型的像素 |
| --- | --- | --- | --- |
| 原样喂 | 1.29s | 224 MiB | 192.00 MP |
| 归一化后 | 0.80s | 61 MiB | 8.02 MP |

这还只是**素材预处理**这一层。真正的显存收益要算上视觉编码器：它的 patch 数量和激活显存直接按像素数走，所以「模型看到的像素少 23.9 倍」才是显存那本账上的数字，而且重排阶段是乘以候选数的。预算可以通过 `vision_input_pixels` 调整，显存紧张时降到 `501760`（约 0.5MP）能再省一半。

## 专业剪辑策略

剪辑风格是一层**可选的剪辑手艺**。它不管调色、字体、颗粒——那些属于艺术指导，读的是同一首歌；它管的是关于**时间**的决定：一个镜头允许活多久、一刀允许落在哪个音乐网格上、两个相邻镜头之间允许不允许景别不变、一次转场允许有多响。这些才是让一支片子像"卡点快剪"而不是像"长镜 MV"的东西，哪怕两边剪的是同一批素材。

```toml
edit_style = "auto" # auto 按歌曲情绪自动选；也可固定某个风格名；manual = 用下面的手工值
```

八种风格，各自是一套**自洽的习惯**而不是一堆独立旋钮：

| 风格 | 镜头长度 | 切点网格 | 可见转场 | 转场取向 | 景别对比 | 适合 |
| --- | --- | --- | --- | --- | --- | --- |
| `beat` 卡点快剪 | 0.55–1.9s | 每一拍 | 55% | 冲击 | 高 | 副歌密集的流行、短影音 |
| `cinematic` 电影感长镜 | 3.0–6.5s | 小节线 | 18% | 含蓄 | 最高 | 抒情、叙事、想让画面自己发生 |
| `lyric` 歌词主导 | 1.6–4.6s | 歌词行 | 28% | 含蓄 | 中 | 歌词就是主线 |
| `montage` 蒙太奇叙事 | 1.2–3.6s | 乐句 | 30% | 均衡 | 最高 | 靠画面并置表意 |
| `documentary` 纪实手持 | 2.4–6.0s | 小节线 | 10% | 含蓄 | 中 | 真实感、情绪记录 |
| `dream` 梦幻叠化 | 2.8–6.0s | 乐句 | 42% | 含蓄 | 低 | 氛围、dreamy |
| `impact` 冲击碎剪 | 0.35–1.2s | 每一拍 | 70% | 冲击 | 高 | 副歌最后一段或 drop |
| `minimal` 极简留白 | 4.0–9.0s | 乐句 | 14% | 含蓄 | 低 | 极简美学、大量负空间 |

`scripts/edit_style_probe.py` 会把同一首歌喂给全部八种风格，打印出上面这些数字——改风格或加风格之后先跑它，因为"名字不同但剪出来一样"是这一层最容易犯的错。

### 每个字段背后是一条剪辑判断

- **`cut_alignment` 切点网格**——整个文件里最听得出来的一个决定。切在**每一拍**上，剪辑就变成打击乐的一部分；切在**小节线**上，一刀是一句标点；切在**乐句**（若干小节）上，一刀是一个段落；切在**歌词行**上，就是让词来推动画面。乐句网格会**按风格自己的镜长窗口自适应**取几小节：固定四小节的教科书答案在 120bpm 下等于 8 秒，而一个镜头最长 6 秒的风格永远找不到候选点，会静默退化成按小节切——那样这个字段就成了装饰。
- **`tempo` + `energy_gain`**——平均能量下想要多长的镜头，以及能量允许把它拉多远。想要长镜的风格必须把 `energy_gain` 压得很低：一个每段副歌都变短的长镜，已经不是长镜了。
- **`section_speedup` 段落内加速**——越接近本段的顶点，镜头越短。剪辑是往 drop 里收紧的；反过来（段落越推进镜头越长）读起来像在正该发力的时候没想法了。
- **`transition_density` 可见转场比例**——只管**段落内部**的普通切点。段落之间的接缝永远有标点，因为那是结构，不是节奏。这一条要真的生效，密度门控必须在"冲击/呼吸"这些意图分支**之前**：否则一个只想要 10% 可见转场的纪实风格，仍然会在大多数切点上转场。
- **`transition_flavour` 转场取向**——同一处音乐，三种音量。含蓄的剪辑仍然要标点，只是用淡黑和模糊而不是闪白；把故障换成溶解会把那一刀彻底泄掉，所以降级用的是**轮转的替代名**，避免所有安静转场塌成同一个。
- **`shot_size_contrast` 景别对比**——两个中景挨着是一张清单，一个远景接一个特写才是一句话。快剪尤其依赖它：一秒四刀的时候，观众只能靠景别分辨这一刀和下一刀的区别，没有景别变化，快剪会糊成一片。
- **`camera_intensity` / `composite_ratio`**——乘在情绪给出的运镜能量上，而不是取代它：一部纪实和一支冲击碎剪可以共享同一个情绪，但仍然想要完全不同的运动量。

### 其余编排规则

- 优先在 downbeat 和乐段边界切镜，而不是每句歌词机械切换；
- 歌词决定镜头内容，但不再强制每句换画面，避免歌词幻灯片感；
- 对视频的采样帧分别计算歌词相似度，从最相关画面附近开始取材；连续使用同一视频时顺接时间轴，避免随机跳段；AI 导演同时收到每句歌词的前四名候选及视频时间点；
- 素材够用时每个镜头都换新素材，同一素材只在素材数量不足时才再次出现，且复用会在所有素材之间均摊；
- 主歌保持色彩和主体连续，副歌提高视频镜头比例和切换强度；
- 副歌复现少量视觉母题，让成片有记忆点（素材充足时以不复用同一素材优先，母题主要在素材不足时体现）；
- 同景别连续出现会扣分，画质、曝光、清晰度和分辨率参与选镜；
- 相邻镜头主色差异过大时降低分数，冲击型剪辑除外；
- 常规节拍点以硬切为主；可见转场的密度由 `transition_density` 控制，段落切换和导演标记的冲击点必用转场，具体动作从十余个转场族里按进入镜头的色调与离开镜头的运动方向挑选；
- 原视频保留自身摄影运动，不再叠加周期性摇摆；静态图片根据 `impact/continuity/breathe` 在十余种运镜之间选择，不再按镜头序号机械横移；
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

把全部图片效果渲染成一张接触表，一眼看清每种运镜和复合动效长什么样（每行是起始/中间/结束三帧）：

```powershell
uv run python scripts/effects_preview.py
uv run python scripts/effects_preview.py --media demo/media --out .probe/effects
```

单图运镜由素材里**边缘能量最高**的那张图渲染，因为平坦的画面会把裁剪窗口完全藏起来；多图合成则需要目录里至少两张图。测试能证明裁剪框不越界、圆形遮罩确实在张开，但证明不了某个运镜是否读起来像它本该是的那个镜头——这张表就是给这件事用的。

调滚转角、梯形强度或新增运镜之前先跑四边形探针，它会逐帧求值每个运镜的裁剪四边形，报出越界像素和形变量：

```powershell
uv run python scripts/quad_probe.py
```

字幕特效与效果型转场各有自己的探针，因为它们的失效方式是**静默**的：libass 会直接忽略不认识的标签，切点上的滤镜也可能只是没生效，两种情况都渲染得完美无缺却什么也没做。

```powershell
uv run python scripts/edit_style_probe.py        # 同一首歌喂给全部剪辑风格，对比镜头长度/转场/景别
uv run python scripts/subtitle_effect_probe.py   # 逐帧比对，确认每种字幕动效真的在动
uv run python scripts/subtitle_layout_probe.py   # 渲染四种断句路径，看每片歌词落在哪里
uv run python scripts/subtitle_layout_probe.py --fill knockout   # 同上，镂空填充
uv run python scripts/cut_effect_probe.py        # 在真实切点上确认闪光/漏光/烧毁真的发生
```

字幕探针同时检查"有没有画出来"（墨量）和"有没有在动"（相邻帧差）；效果转场探针同时看均值亮度、空间标准差和暖度，因为噪点和通道分离几乎不改变均值。

版式探针不依赖 LRC——自由排版要的"歌手在哪换气"是 LRC 带不了的信息——所以它直接构造带逐字时间轴的歌词，走真实的 `render()`，再把每个镜头的三帧铺成一张图，一眼就能看出每片落在哪、什么时候出现。改落点或版式前先跑它。

调 `vision_input_pixels` 之前先量一遍，它会把两条路径放在各自的进程里测，峰值读数才各归各的：

```powershell
uv run python scripts/vision_input_probe.py
uv run python scripts/vision_input_probe.py --size 6000x4000 --count 16 --budget 501760
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
beatforge/models/ai_director.py       本地 Spark-X2.5 导演协议与校验
beatforge/models/separator.py         MelBand-RoFormer 人声分离（转录前置）
beatforge/models/transcriber.py       Qwen3-ASR 与强制对齐时间轴
beatforge/models/audio_semantics.py   CLAP 音乐语义
beatforge/models/music_structure.py   All-In-One/Beat This 结构分析
beatforge/models/vision_index.py      WeMM/Qwen3-VL-Embedding/SigLIP2 检索
beatforge/editing.py                  剪辑风格：把专业剪辑决策结构化为配置
beatforge/planner.py                  多目标镜头编排
beatforge/renderer.py                 FFmpeg 成片渲染
beatforge/pipeline.py                 分阶段模型生命周期
scripts/                              演示素材、复用/显存/抖动/四边形/效果/字幕/版式/视觉输入探针
tests/                                不下载模型的测试
```
