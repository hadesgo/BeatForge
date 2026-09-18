# BeatForge 代码梳理

扫描范围：`beatforge/` 21 个模块 6022 行、`scripts/` 17 个脚本、`tests/` 19 个文件。
结论先行：**代码整体是干净的**——零死代码残留（已修 1 处）、零裸 `except`、零 TODO、
70 个配置项全部被读取、`demo/project.toml` 无未知键。下面按"已修 / 建议修 / 观察"分层。

---

## 一、已修（本次）

### 1. `_section_info` 在两处重复，且兜底行为不一致 ⚠️ 真问题

`director.py` 和 `planner.py` 各定义了一份同名函数。查同一段时间戳时两者一致，但**越过最后一个
段落边界后不一致**：

| 模块 | `time >= duration` 时的返回 |
| --- | --- |
| `director.py` | `("unknown", index)` |
| `planner.py` | `(最后一个真实标签, index)` |

实测（`demo/music.wav`，时长 15.0s，边界 `[0.0, 0.023, 15.0]`，标签 `['intro','outro']`）：
在 0–17.2s 上以 0.01s 步长扫描，**225 个采样点两者不同**，最早出现在 `t = 15.0`。

为什么不是小事：段落名驱动**剪辑意图、效果族、色彩连续性、母题选择、切点比例**
（`planner.py:150/151/236/296/300`、`director.py:79/81/84`）。同一时刻被描述成两种，
末句就会拿到两套不同的处理。

**修法**：真源提到 `audio.py::section_at()`（`AudioAnalysis` 的归属地），取**正确的那版兜底**
——`sections` 永远覆盖全曲，所以越过最后边界仍属于最后一段，`"unknown"` 只应表示"整首没有段落"。
两处定义与两处 `_section_at` 包装一并删除。

### 2. `_section_at` 是死代码

两个模块各定义一次，**全仓库 0 处调用**。（第一版扫描漏掉了它——同名出现两次就被判为"被引用"；
改成"定义处与引用处分开计数"后才暴露。）

### 3. 5 处未使用的 import

`cli.py` / `composite_probe.py` / `subtitle_effect_probe.py` / `subtitle_layout_probe.py`。
已清除，复查为 0。

测试：**330 passed**（新增 3 条守 `section_at` 的边界行为）。

---

## 二、建议修

### P1 · `transcriber.py` 不夹紧时长，是上面那条分歧的触发路径

`_line_from_tokens()` 直接采用对齐器返回的 `tokens[-1].end`，**全程没有任何夹紧**：

```
beatforge/models/transcriber.py:93
    return LyricLine(tokens[0].start, tokens[-1].end, text, tokens)
```

对比 LRC 路径（`lyrics.py:38/42`）是夹紧的：`start < total_duration` 过滤 + `end = min(total, ...)`。
所以 LRC 路径不会产生 `time >= duration` 的歌词中点，**ASR 路径可能**——对齐器按自己的帧网格输出，
不感知音频时长，末词轻微过冲就会把 `line.end` 顶出时长之外。

`section_at` 的分歧已修，但**过冲本身还会传播到别处**：`write_srt` 不夹、`plan_placements`
拿它算淡入淡出。建议在 `transcriber.transcribe()` 出口统一夹到 `duration()`。

> 未实测对齐器是否真的过冲——按你的要求没有跑 ASR 验证。上面是读代码得出的可达性分析。

### P2 · `audio_semantics.py` 零测试覆盖

它是**真实路径**（`pipeline.py:91` 调 `classify_music`），却是唯一 0 覆盖的模块。
另外三个偏低：`cli.py` 25%、`transcriber.py` 50%、`pipeline.py` 50%。

`classify_music` 值得补的部分不需要真模型：**采样窗口逻辑**（>30s 时取 3 段 5 秒中心、
≤30s 时整段）、返回是否覆盖全部 7 个 `MOOD_LABELS`、分数是否归一。

### P3 · `_image_filter_graph` 的 4 个合成分支可抽出（167 行 → 约 90 行）

`renderer.py:592` 是个分派函数，4 个多图合成各占约 20 行且都早返回
（`split_screen`/`photo_stack`/`double_exposure`/`beat_montage`，601–679 行）。
抽成 `_composite_graph(...) -> (filters, label) | None` 能把主函数缩到约 90 行，
分支边界也更清楚。**低风险**：纯搬运，`test_all_still_image_effects_render` 覆盖。

另外 7 个超 80 行的函数（`create_plan` 187、`run_project` 151、`_camera_quad` 88 等）
大多是线性流程，**不建议动**——拆了反而更难跟。

### P4 · 同一个音频文件被解码 2–3 次

```
audio.py:50              librosa.load(file, sr=22_050)   ← 分析
models/audio_semantics.py:34  librosa.load(file, sr=48_000)   ← CLAP
models/music_structure.py      （All-In-One 自己再读一次）
```

三次全量解码。对 4 分钟的歌各约 20 MB PCM，**属于可接受但没必要**。
可改为解一次 48k、降采样给分析用。收益不大，改动会碰到三个模块的接口，**优先级低**。

### P5 · `ai_director.py:592` 的 `except Exception: pass`

在 `_remote_module_paths()` 里，是**可选的** transformers 模块缓存查找（本地目录那条是主路径），
所以吞掉是合理的，但 `Exception` 太宽——一个拼错的属性名也会被静默吃掉。
建议收窄到 `(ImportError, OSError, AttributeError)`。同类还有 `media.py:136`
（校验回退，建议直接 `return [.5,.5]` 而不是 `pass` 后落空）。

---

## 三、观察项（不算问题，但值得知道）

- **README 模块表列了 12 个模块，实际有 19 个**。未列出的 7 个：`cli.py`、`config.py`、
  `fonts.py`、`lyrics.py`、`media.py`、`models/downloader.py`、`runtime.py`。
- **`scripts/` 里 3 个相机探针重复同一段 ffmpeg 抽帧代码**
  （`camera_look_check.py:57`、`camera_move_probe.py:64`、`zoompan_probe.py:39`）。
  探针之间共享代码的收益低于耦合成本，**倾向保持现状**。
- **`__pycache__` 里同时有 `cpython-312` 和 `cpython-313` 的 `.pyc`**（已 gitignore，只是本地噪声）。
- `scripts/` 里 15 个 `main()` 同名、4 个 `report()` 同名——各脚本独立入口，正常。

---

## 四、确认过、没有问题的方面

- **无死代码残留**（除已修的 `_section_at`）、**无裸 `except`**、**无 TODO/FIXME/HACK**。
- **配置面干净**：70 个字段全部被读取；`demo/project.toml` 无未知键。
- **无重复的音频分析**：`analyze_music` 全仓库只调用一次。
- **异常处理成体系**：`runtime.py` 的两处静默兜底是有注释、有理由的（尽力而为的显存清理）。
- **测试与实现的绑定很紧**：`test_renderer.py` 161 条覆盖 21 种运镜 × 多种强度、
  `test_separator.py` 11 条覆盖缓存/选轨/降级。守门测试（可达性、越界、帧数精确相等）
  是这个项目最值得保留的部分。
