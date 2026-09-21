# BeatForge 项目长期笔记

## 目录与约定
- 工作区：`D:\code\BeatForge`；包在 `beatforge/`，探针在 `scripts/`，不加载模型的测试在 `tests/`。
- 环境：`uv run ...`。全量测试 `uv run pytest -q`（约 370 条，纯单元 + FFmpeg 渲染，不下载模型）。
- 本机 bash 工具链不可用（缺 coreutils），命令走 PowerShell；`uv run python -c "..."` 是跑临时脚本的可靠方式。
- `.probe/` 是探针输出目录，已 gitignore。`my-mv/` 是用户真实工程，也已 gitignore。

## 多图合成（image_effect）的硬约束
1. **不许把两张素材画在同一片像素上**。`blend=`、卡片压在图上、旋转卡片叠放都属于这一类，
   已整体删除（`photo_stack`、`double_exposure`）。每个版式要么给每张图一块独占区域，
   要么一次只显示一张。测试 `test_a_composite_never_paints_one_picture_over_another` 钉住这条。
2. **分割各格宽度之和必须等于画布宽度**，缝隙画在接缝之上而不是从宽度里挖出来。
   否则最后的 `normalize`（scale+pad）会给镜头加上黑边。
3. **分裂版式的每一格全出血**（`_fill_frame`，`increase` + `crop`）；只有一次只显示一张的
   `beat_montage` 保留单图那套"留边 + 虚化副本"。
4. 词表在 `planner.IMAGE_COMPOSITES`、`_COMPOSITE_SMALLER`、`_composite_rotation`；
   实现全部在 `renderer._composite_graph`。加版式要动这两处 + 测试 + 两个探针 + README。

## 三个计数器不能混用（踩过坑）
- **门控**（是否用多图）按**镜头序号 `index`**：挂在别处会自我强化 —— 计数器只在拒绝时
  前进，一个镜头变成多图就把它冻住，`image_composite_ratio` 会彻底失效。
- **版式轮转**同样按 `index`。
- **运镜轮转**按**已选出的单图镜头数** `single_image_cursor`，按镜头序号会让被合成吃掉的
  槽位永远轮不到。

## 窄分支的可达性
按 `cursor % n` 轮转的窄分支（impact 运镜、段落切换处的 `open`/`radial` 转场）只有两三次
触发机会，同长度的歌永远在同一处进入分支，相位一致 → 总有名字选不到。测试里要同时扫
**歌曲时长**（相位）与风格/能量（分支），只扫风格不够。

## 字体链路（libass）的三条硬约束
1. **`fontsdir` 只接受一个目录**。传两个路径（`;` 或 `:` 分隔）会让 libass **一个字体都找不到**。
   项目字体和仓库自带字体必须先被 `fonts.stage_fonts()` 收集进同一个目录。
2. **`subtitle_fonts_dir` 相对项目文件夹解析**，模板里的 `"fonts"` 指向 `my-mv/fonts`
   （`init` 建的空目录）。所以仓库自带的字体必须自己作为一个搜索路径（`PACKAGED_FONTS_DIR`）。
3. **libass 不应用可变字体的 `wght` 轴**（Windows/DirectWrite 实测）：它要到 700 的 face 就
   拿回默认实例，且因为家族"已有 Bold"而不再合成粗体。自带两个 Noto 的默认实例是
   Thin(100)/ExtraLight(200) → 不烘静态字重的话字幕永远是发丝细体，`\b` 完全无效。
   `stage_fonts()` 用 fontTools 烘 Light/Regular/Bold，缓存在整机级
   （`%LOCALAPPDATA%\beatforge\fonts`，`BEATFORGE_FONT_CACHE` 可覆盖，首次约 70 秒）。

字体失效是**静默**的：目录不可达、家族名不存在、字重没送达，都渲染出一帧合法但字体不对的画面。
判定要靠像素（墨量 / 与无字幕参考帧的差），读配置看不出来。探针：`scripts/font_probe.py`。
另外解析家族名时，模糊命中必须返回**实际存在的那个拼写**，否则 ASS 里会写进一个找不到的名字。
