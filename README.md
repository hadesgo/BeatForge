# BeatForge

BeatForge 是一个 Python + uv 的本地 AI 音乐视频剪辑器。输入音乐、图片/视频以及可选的 LRC 歌词，输出自动编排的 MV 和一份可审阅的镜头决策文件。

## AI 流程

```text
音乐 ── Whisper ── 歌词时间轴 ───────────────┐
  └── librosa + CLAP ── 节拍/章节/氛围 ─────┼── 镜头打分与编排 ── FFmpeg
图片/视频 ── 关键帧 ── SigLIP2 视觉向量 ─────┘
                         ↑
                    每句歌词向量
```

- `faster-whisper small`：CPU 默认转写模型，使用 INT8；
- `faster-whisper large-v3-turbo`：RTX 5070 推荐转写模型，使用 INT8/FP16；
- `laion/clap-htsat-fused`：音乐情绪、质感和强度的零样本分类；
- `google/siglip2-base-patch16-224`：中文歌词与图片/视频关键帧的跨模态匹配；
- `librosa`：节拍、能量、音色亮度和章节边界；
- `FFmpeg`：裁切、图片运镜、调色、字幕和最终编码。

模型分阶段加载并释放，不会同时占用显存。当前无 NVIDIA 显卡的电脑可以用 CPU 完成开发和验证；目标 5070 机器可切换 CUDA 12.8 环境。

## 安装

基础开发环境，不包含任何模型框架：

```powershell
uv sync
```

当前 CPU 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cpu
```

RTX 5070 电脑的 AI 环境：

```powershell
uv sync --extra ai --extra ai-cuda
```

CPU 和 CUDA profile 互斥，uv 会阻止二者同时安装。模型权重不会在 `uv sync` 时下载；第一次运行相应模型时才会进入 Hugging Face 本地缓存。

## 使用

```powershell
uv run beatforge init my-mv
uv run beatforge run my-mv/project.toml
```

将音乐放到 `my-mv/music.mp3`，图片和视频放到 `my-mv/media/`。如果已有 LRC，保存为 `my-mv/lyrics.lrc`；如果要使用 Whisper，删除该文件并删除或注释 `project.toml` 的 `lyrics` 配置。

只生成剪辑决策，不渲染：

```powershell
uv run beatforge run my-mv/project.toml --plan-only
```

禁用模型，验证基础分析和渲染：

```powershell
uv run beatforge run my-mv/project.toml --no-ai
```

输出包括 `output.mp4`、`.beatforge/plan.json`、`.beatforge/lyrics.srt` 和视频分析所用的缓存关键帧。`plan.json` 包含音乐结构、模型配置、素材信息、逐镜头语义得分和剪辑参数。

## CPU 与 RTX 5070 配置

CPU 默认配置：

```toml
[ai]
device = "auto"
whisper_model = "small"
whisper_compute_type = "int8"
frame_samples = 3
```

RTX 5070 12GB 推荐配置：

```toml
[ai]
device = "cuda"
whisper_model = "large-v3-turbo"
whisper_compute_type = "int8_float16"
frame_samples = 5
```

运行 `uv run beatforge doctor` 检查实际使用 CPU 还是 CUDA。Windows 上 Faster Whisper GPU 模式还需要 CUDA 12 的 cuBLAS 与 cuDNN 9 动态库。

## 素材语义

SigLIP2 会直接比较歌词和画面。文件名和 sidecar 标签作为模型关闭时的后备。例如 `海边_日落_回忆.jpg`，或创建 `portrait.jpg.json`：

```json
{
  "description": "女孩在海边回头，夕阳逆光",
  "tags": ["女孩", "海边", "日落", "回忆", "离别"],
  "mood": "melancholic"
}
```

视频默认均匀抽取 3 个关键帧并平均视觉向量。提高 `frame_samples` 会提升长视频覆盖率，也会增加分析时间。

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
beatforge/models/transcriber.py       Whisper 时间轴
beatforge/models/audio_semantics.py   CLAP 音乐语义
beatforge/models/vision_index.py      SigLIP2 歌词/画面检索
beatforge/planner.py                  多目标镜头编排
beatforge/renderer.py                 FFmpeg 成片渲染
beatforge/pipeline.py                 分阶段模型生命周期
tests/                                不下载模型的测试
```
