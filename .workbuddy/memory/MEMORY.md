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
