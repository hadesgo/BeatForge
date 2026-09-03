# BeatForge

BeatForge 是一个 Python + uv 的本地 AI 音乐视频剪辑器。输入音乐、图片/视频以及可选的 LRC 歌词，输出自动编排的 MV 和一份可审阅的镜头决策文件。

## AI 流程

```text
音乐 ── Qwen3-ASR + ForcedAligner ── 逐字时间轴 ──┐
  └── All-In-One + CLAP ── 旋律/节拍/章节/意境 ──┼── Qwen3.5 AI 导演 ── 确定性规划器 ── FFmpeg
图片/视频 ── Qwen3-VL Embedding + Reranker ───────┘           │
                                                      导演方案 JSON
```

- `Qwen/Qwen3-ASR-1.7B-hf`：Transformers 原生歌曲识别模型；
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`：Transformers 原生字符/单词级演唱时间对齐；
- `laion/clap-htsat-fused`：音乐情绪、质感和强度的零样本分类；
- `Qwen/Qwen3-VL-Embedding-2B`：中文歌词与图片/视频的跨模态检索；
- `Qwen/Qwen3-VL-Reranker-2B`：对初选画面进行歌词意境和叙事适配精排；
- `All-In-One-Infer`：识别 intro、verse、chorus、bridge、solo、outro 和强拍；
- `Beat This!`：可选的高精度 beat/downbeat 后备；
- `Qwen/Qwen3.5-4B`：本地 AI 导演，负责全片概念、叙事弧、视觉母题和分段剪辑策略；
- `librosa`：旋律变化、节拍密度、能量、音色亮度和章节边界；
- `FFmpeg`：裁切、图片运镜、调色、字幕和最终编码。

模型分阶段加载并释放，不会同时占用显存。当前无 NVIDIA 显卡的电脑可以用 CPU 完成开发和验证；目标 5070 机器使用 PyTorch 2.14.0 + CUDA 13.2 官方轮子。Qwen3-ASR 走 Transformers 5.13+ 原生接口，Qwen3-VL Embedding/Reranker 走 Sentence Transformers，不再安装会锁死旧版 PyTorch/Transformers 的专用包。

## 安装

基础开发环境，不包含任何模型框架：

```powershell
uv sync
```

当前 CPU 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cpu --extra qwen --extra music-ai
```

RTX 5070 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cuda --extra qwen --extra music-ai
```

CPU 和 CUDA profile 互斥，uv 会阻止二者同时安装。模型权重不会在 `uv sync` 时下载；第一次运行相应模型时才会进入 Hugging Face 本地缓存。若先手工准备权重并设置 `offline = true`，运行时只读取本地缓存。

## 使用

```powershell
uv run beatforge init my-mv
uv run beatforge run my-mv/project.toml
```

将音乐放到 `my-mv/music.mp3`，图片和视频放到 `my-mv/media/`。如果已有 LRC，保存为 `my-mv/lyrics.lrc`；如果要使用 Qwen3-ASR，删除该文件并删除或注释 `project.toml` 的 `lyrics` 配置。

只生成剪辑决策，不渲染：

```powershell
uv run beatforge run my-mv/project.toml --plan-only
```

禁用模型，验证基础分析和渲染：

```powershell
uv run beatforge run my-mv/project.toml --no-ai
```

输出包括 `output.mp4`、`.beatforge/plan.json`、`.beatforge/lyrics.ass` 和视频分析所用的缓存关键帧。`plan.json` 包含音乐结构、模型配置、素材信息、逐镜头语义得分和剪辑参数。

## 本地 AI 导演

导演模型由 BeatForge 直接通过 Transformers 加载，不需要 llama.cpp、Ollama、LM Studio 或额外服务。视觉检索结束并释放显存后才加载导演；导演方案完成后立即删除模型、执行垃圾回收并清空 CUDA allocator，再进入 FFmpeg 渲染。加载或输出校验失败时自动使用规则导演，流程不会中断。

```toml
[ai]
director_enabled = true
director_model = "Qwen/Qwen3.5-4B"
director_temperature = 0.25
director_max_new_tokens = 2048
director_gpu_memory_gb = 9.0
director_cpu_memory_gb = 20.0
director_offload = true
```

`director_gpu_memory_gb` 是 Accelerate 的显存上限；12GB 显卡默认只允许导演使用 9GB。超出部分在 `director_offload = true` 时卸载到内存和 `.beatforge/director-offload/`。导演接收歌曲统计、逐句歌词、乐段和最多 60 个高价值素材候选，输出经 Pydantic 校验的结构化方案；第一次 JSON 不合法会在同一次模型生命周期内自动修正一次。它不会生成时间码或直接执行 FFmpeg，具体剪辑点仍由节拍模型和确定性规划器控制。

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
```

- `karaoke`：逐字高亮，并带轻微缩放入场；
- `cinematic`：模糊消散和长淡入淡出；
- `bounce`：随句子出现的弹跳缩放。
- `float`：伴随舒缓旋律缓慢上浮；
- `glow`：适合浪漫和梦幻段落的柔光入场；
- `typewriter`：适合暗黑、叙事感段落的逐字出现。

可以把 `.ttf`/`.otf` 放入项目的 `fonts/`，然后为不同意境配置字体族名：

```toml
[render.subtitle_fonts]
energetic = "My Display Font"
uplifting = "My Sans Font"
melancholic = "My Serif Font"
dreamy = "My Light Font"
romantic = "My Handwriting Font"
dark = "My Condensed Font"
cinematic = "My Cinema Font"
```

图片和视频的推拉幅度同时参考局部能量、旋律变化率和歌曲意境。高能/高节奏密度段落使用锐化与亮色闪切，低能段落使用柔化与长淡入，梦幻和抒情歌曲降低镜头运动，所有镜头可选暗角和动态胶片颗粒。最终选择会写入 `plan.json` 的 `art_direction` 和每个 `shot.melody`。

## CPU 与 RTX 5070 配置

CPU 默认配置：

```toml
[ai]
device = "auto"
asr_backend = "qwen3"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "qwen3-vl-embedding"
vision_model = "Qwen/Qwen3-VL-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
music_structure_backend = "allin1"
frame_samples = 3
director_enabled = true
director_model = "Qwen/Qwen3.5-4B"
director_gpu_memory_gb = 9.0
```

RTX 5070 12GB 推荐配置：

```toml
[ai]
device = "cuda"
asr_backend = "qwen3"
qwen_asr_model = "Qwen/Qwen3-ASR-1.7B-hf"
qwen_aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
vision_backend = "qwen3-vl-embedding"
vision_model = "Qwen/Qwen3-VL-Embedding-2B"
vision_reranker_model = "Qwen/Qwen3-VL-Reranker-2B"
music_structure_backend = "allin1"
frame_samples = 5
director_enabled = true
director_model = "Qwen/Qwen3.5-4B"
director_gpu_memory_gb = 9.0
```

运行 `uv run beatforge doctor` 检查实际使用 CPU 还是 CUDA。Faster Whisper 和 SigLIP2 仍可通过 `asr_backend`/`vision_backend` 作为兼容后备。

## 素材语义

Qwen3-VL-Embedding 会直接比较歌词、图片和视频关键帧。文件名和 sidecar 标签作为模型关闭时的后备。例如 `海边_日落_回忆.jpg`，或创建 `portrait.jpg.json`：

```json
{
  "description": "女孩在海边回头，夕阳逆光",
  "tags": ["女孩", "海边", "日落", "回忆", "离别"],
  "mood": "melancholic",
  "shot_size": "medium_close_up",
  "camera_motion": "slow_push_in",
  "quality_score": 0.9,
  "dominant_color": [184, 112, 74]
}
```

视频默认均匀抽取 3 个关键帧并平均视觉向量。提高 `frame_samples` 会提升长视频覆盖率，也会增加分析时间。

## 专业剪辑策略

- 优先在 downbeat 和乐段边界切镜，而不是每句歌词机械切换；
- 主歌保持色彩和主体连续，副歌提高视频镜头比例和切换强度；
- 副歌复现少量视觉母题，让成片有记忆点；
- 同景别连续出现会扣分，画质、曝光、清晰度和分辨率参与选镜；
- 相邻镜头主色差异过大时降低分数，冲击型剪辑除外；
- 使用带 handle 的真实 xfade，避免每个镜头先黑场再出现；
- intro/outro 留呼吸，chorus 紧凑，bridge/solo 给旋律性镜头更长时间。
- AI 导演统一概念、叙事弧、调色倾向和视觉母题，并对各乐段给出剪辑强度、景别、素材偏好、字幕与转场意见；

这些决策会写入 `plan.json` 的 `section`、`edit_intent`、`melody`、`quality_score` 和 `art_direction`，方便人工复核。

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
beatforge/models/vision_index.py      Qwen3-VL-Embedding/SigLIP2 检索
beatforge/planner.py                  多目标镜头编排
beatforge/renderer.py                 FFmpeg 成片渲染
beatforge/pipeline.py                 分阶段模型生命周期
tests/                                不下载模型的测试
```
