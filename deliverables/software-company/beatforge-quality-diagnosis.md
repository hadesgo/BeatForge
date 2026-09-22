# BeatForge「廉价感」问题诊断（交付总监齐活林，2026-09-22）

本文是**实测结论**，不是推测。所有数字来自 `my-mv/` 真实工程（198s 成片、255 个镜头、
341 个素材、`plan.json` 1.0 MB）与 `beatforge/` 源码逐行核对，画面结论来自
`output.mp4` 实际抽帧（`.probe/frames/out_*.png`）。

---

## 0. 结论先行

成片看起来廉价，**不是因为没有特效，而是因为特效密度、剪辑速度、画面处理方式三者互相拆台**。
项目在「运镜数学模型」「多图版式不重叠」这类**局部正确性**上投入极深（README 里几十条
用血换来的不变式），但**整片层面的编排意图没有一个地方负责收敛**：

- 剪辑风格只读一个情绪词（`art_direction.mood`），把 AI 导演写下的完整叙事弧、
  逐段 `cut_intensity`、用户手工设的镜长全部覆盖；
- 素材零复用，「视觉母题」在数学上不可能生效；
- 每一张异幅照片都被做成「虚化双胞胎 + 硬边内嵌」；
- 字幕在自由版式下用 46px 发丝描边字放在画面任意点上；
- 调色三层叠加 + 全局颗粒暗角 + 全范围像素格式交付。

---

## 1. 实测数据（`my-mv/`）

成片：1920×1080 / 30fps / 197.5s / CRF19 / **yuvj420p** / 17.8 Mbps / 446 MB。

| 指标 | 实测值 | 意味着什么 |
| --- | --- | --- |
| 镜头数 | **255** | 平均 **0.775s / 镜** |
| 时长分布 | <0.8s: **207**；1.2–1.8s: 39；1.8–3.0s: 9 | 81% 的镜头短于 0.8 秒 |
| 最短 / 最长 | 0.55s / 1.90s | 恰好是 `beat` 风格的窗口 `.55–1.9` |
| 素材 | 341 个，镜头 255 个 | |
| 复用 | **被用素材 255 个，全部只用 1 次，复用 0 个** | 零锚点 |
| 导演母题 | `motifs = [256,282,257,31,217]` | 实际出现次数 **1, 0, 1, 1, 1** |
| 镜头类型 | image 232 / video 23 | 91% 是静止照片，仅 9% 真实运动素材 |
| 转场 | **22 种** / 255 个切点 | 其中 flash 16、zoom 15、glitch 14、film_burn 13 = **58 个冲击型** |
| 多图合成 | 50 镜（19.6%），`image_effect` 共 **31 种** | 31 种运镜/版式摊在 0.775s 上 |
| 调色 | `grade_filter` = `eq=contrast=1.05:saturation=1.14:brightness=0.025,colorbalance=rs=.018:bs=-.012` | 全片同一条静态串 |
| 字幕版式 | 配置 `band`，实际 ASS 为 `\pos` 自由版式（98 处，y 中位 517，贴底占比 0%） | 配置被静默覆盖 |
| `camera_motion` | 255 个镜头全部 `unknown` | 死字段 |

---

## 2. 源码级确诊（每条都定位到行）

### P0-A　`edit_style` 无条件压过一切，且与 AI 导演的叙事意图完全脱节

- `planner.py:94-97`：
  ```python
  if style is not None:
      min_shot, max_shot = style.shot_min, style.shot_max
      transition_density = style.transition_density
      image_composite_ratio = style.composite_ratio
  ```
  用户的 `min_shot_seconds=1.8 / max_shot_seconds=5.5` 被**静默丢弃**（README 里承诺了这件事，
  但用户无法通过配置把节奏调回来，只能把 `edit_style` 设成 `manual` 才能恢复）。
- **未文档化的第四项接管**：`pipeline.py:151` 还把 `style.subtitle_layout` 写进 render 配置。
  README「关键配置速查」明确只列了 **镜长 / 转场密度 / 多图比例** 三项被接管。
  于是 `project.toml` 写 `subtitle_layout = "band"`，成片却是自由版式 —— 这是本工程
  「静默配置失效」家族的新成员（与 README 里字体、分离模型那条同一性质）。
- `editing.py:161-169, 184-189`：`auto` 只做 `_MOOD_STYLES[mood]` 一次查表。
  本曲 `mood = "energetic"` → `beat`。
- **两套子系统读同一首歌却给出相反答案，且粗的那套赢**：
  同一份 `plan.json` 里，AI 导演写了 `concept`「温情纪实短片」、`visual_style`「避免风格割裂」、
  逐段 `cut_intensity` 0.2→0.65 的渐进弧、以及 intro/outro 要「留呼吸」；
  而 `beat` 风格把这些全部无视，用 0.55–1.9s 的窗口切了 197 秒。
  `art_direction.mood` 是一个词，导演的 `sections[]` 是十段结构化意图 —— **一个词赢了十段**。
- 后果是连锁的：**21 种运镜、31 种版式在 0.775s 内全部不可读**。
  `perspective` 那条「亚像素而非整像素跳变」的深度优化，观众永远看不到，
  因为镜头在推进到 1/3 时就切走了。

### P0-B　零素材复用，使「视觉母题」在数学上不可能生效

- `planner.py:161-165`：`avoid_asset_repeats` 时优先取「使用次数最少」的一层
  （`tier = [item for item in candidates if usage.get(item[1].id, 0) == minimum_usage]`）。
  素材 341 > 镜头 255，于是每一镜都拿到一个没用过的素材，**全片零复用**。
- `planner.py:172-173`：`chorus_motifs` 只在副歌收录、且**上限 2 个**；
  `planner.py:152` 的 `motif = .12 if ... asset.id in chorus_motifs`。
  但当所有候选都同处「零使用」层时，这个加分**对排序毫无影响** ——
  母题资产一样会在别处被一次性用掉，等副歌再来时它已不是最低使用层。
  实测：导演点名的 5 个母题，1 个一次都没出现，其余各出现 1 次。
- 结果：一支 3 分 17 秒、255 张互不相同的画面，**没有任何东西回来过**。
  这正是「照片轮播」与「MV」的分界：MV 靠复现建立记忆，轮播靠数量堆满时长。

### P0-C　每一张异幅照片都被做成「虚化双胞胎 + 硬边内嵌」

- `renderer.py:864-884`（`_adapt_image_layers`）+ `887-895`（`_adapt_image`）：
  把**同一张图** split 成两路 —— 一路 `scale=…increase,crop` 满屏后 `gblur=sigma=26,
  eq=brightness=-.055:saturation=.88` 当背景；另一路 `scale=…decrease` 到
  `image_foreground_scale=0.92` 居中叠上。
- 三个可指认的画面缺陷（见 `.probe/frames/`）：
  1. **硬边矩形**：前景是 92% 的矩形，与虚化背景之间有可见的直线接缝
     （`out_00_12s.png` x≈150/940 处，`out_04_105s.png`、`out_05_130s.png` 同）；
  2. **同一主体出现两次、两个尺度**：`out_05_130s.png` 右侧是一张被放大 26px
     高斯糊掉的**同一张脸**，与画面正中的清晰主体争夺注意力，画面读成一片噪声；
  3. **叠加 `film_bars` 时出现三层框**：`out_04_105s.png` 同时有黑边遮幅 + 虚化底 +
     内嵌矩形，一个画面里三个「画框」。
- 对照 `renderer.py:898-917` 的 `_fill_frame`：满出血 `increase`+`crop` ——
  它**只用在多图合成里**。README §412 用大段篇幅论证「虚化底在合成里是错的」，
  并据此删掉了 `photo_stack` / `double_exposure`；但证据显示**单图才是更大的问题**：
  本片 182 个单图镜头全部走 `_adapt_image`。
- 更讽刺的是：项目**已经**为视频做了主体感知裁切（`renderer.py:191-194` 用 `focus_point`
  算 `x='clip(iw*focus_x-ow/2,0,iw-ow)'`），单图却退回「留边 + 虚化」。

### P0-D　字幕在自由版式下几乎不可读，且与配置不符

- 实测 ASS：`\pos` 98 处，y 从 72 到 834，中位 517 → 文字被撒在画面各处；
  配置要的是 `band`（底部居中一行）。
- `subtitle_size = 46`（1920×1080）。参考 `.probe/frames/out_07_180s.png`：
  白字落在明亮沙滩上，**基本读不出来**；`subtitle_outline = 1.1` 是发丝描边，
  在细节丰富的照片上没有分离度。
- `PrimaryColour = &H0000D7FF`（金黄），部分行内联 `\1c&H00FFFFFF&` 改白
  → 同一支片子里字幕**颜色不一致**（`out_00_12s.png` 白 vs `out_02_51s.png`/`out_04` 金黄）。
- 与 P0-A 的耦合：0.775s 一镜，而一行歌词往往只承载 2–4 个字的分片
  → **单字片在屏时间短于一次眼跳**，即使排版正确也读不到。

### P0-E　调色三层叠加 + 全范围像素格式交付

- 每个镜头依次叠三层色彩操作（`renderer.py:196-200` / `251-254`）：
  1. `_shot_match_filter`（`932-941`）：按素材代表色做 eq；
  2. `art.grade_filter`：`eq=contrast=1.05:saturation=1.14:brightness=0.025,colorbalance=…`；
  3. `_section_color_filter`（`944-967`）：按色彩弧再加 `colorbalance`。
- 素材本身多为 `微信图片_*` / `mmexport*`（微信二次压缩的 JPEG，1080px 级）。
  **在已经压过的 JPEG 上再叠 saturation 1.14 + contrast 1.05**，暗部与高光一起被推过界
  → 视觉上的「脆、脏、假」。这是「廉价滤镜」的量化定义。
- 其后还有 `noise=alls=1.6`（`renderer.py:213` / `261`）与 `vignette=PI/5`（`210` / `259`）
  **无条件**作用于全部 255 个镜头，包括画面本身就暗的镜头
  （`out_07_180s.png` 沙粒特写，暗角把四角吃到近乎纯黑）。
- 交付像素格式：`_video_encode_args`（`970-978`）写了 `-pix_fmt yuv420p`，
  但 **JPEG 源是 `yuvj420p`（全范围），`format=yuv420p` 并不改 range 标记**，
  于是 full-range 一路传到最终 H.264，实测 `pix_fmt: yuvj420p`。
  叠加对比度/饱和度提升后，全范围更容易**削顶**。专业交付应为 limited range。
- 17.8 Mbps / 446 MB 也说明颗粒正在吃掉码率 —— 编码器在给噪点花钱。

---

## 3. 次要问题（本轮记录，不一定实现）

- **转场词汇过载**：22 种 / 255 切点，58 个冲击型用在「温情纪实」上，与导演
  `transition_tone` 逐段的 soft/neutral/dark 意图相反。
- **视频素材被饿死**：24 个真实运动素材只拿到 23 个镜头（9%）。真实运动是素材里
  最强的部分，却在以 317 张静态图为主的候选池里按逐镜得分竞争。
- **素材质量无兜底**：`quality_score` 参与打分但无硬闸门；`out_05_130s.png`
  是一张「对着手机屏幕拍的屏幕照片」，仍拿到了正常镜头。
- **`camera_motion` 字段全为 `unknown`**：255/255，死字段，建议确认是否应从
  `plan.json` 移除或补全。

---

## 4. 必须守住的不变式（不得为了改画面而破坏）

以下都是 README 记录过、有测试钉住的既有约束，任何改动都必须继续满足：

1. 多图合成**不得把两张素材画在同一片像素上**（`test_a_composite_never_paints_one_picture_over_another`）；
2. 分割版式**各格宽度之和必须等于画布宽度**（`test_the_panels_of_a_division_add_up_to_the_canvas`）；
3. `_quad_spread` 与 `_camera_quad` 必须用**同一个 `amount`**，且测试/探针要扫过整个区间；
4. 门控 / 版式轮转按**镜头序号**、运镜轮转按**已选单图镜头数**，三个计数器不可合并；
5. `fontsdir` 只能一个目录；ASS 里的家族名必须是**实际存在的那个拼写**；
6. 滤镜图要能被 ffmpeg 接受，且**长片要用 `-/filter_complex` 从文件读**（Windows 命令行上限）；
7. 全量测试 `uv run pytest -q`（约 370 条，不下载模型）必须保持全绿；
   静态检查 `ruff` 必须干净。
