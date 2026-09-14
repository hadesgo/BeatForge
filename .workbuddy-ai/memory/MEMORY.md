# BeatForge 项目长期记忆

## 设计约定
- 选镜（`beatforge/planner.py`）的素材复用策略：按"已出现次数"分层，优先最少使用层；
  素材够用（最少使用层数量 ≥ 剩余镜头数）时保证零复用。多图合成属于"奢侈品"，
  只在 `富余素材 = 最少使用层数量 - 剩余镜头数 >= 0` 时消耗富余素材。
  开关为 `render.avoid_asset_repeats`（默认 true）。细节见 README「素材复用规则」。
- 规划器保持确定性：不使用随机数，平票靠素材发现顺序（列表顺序）决出，便于测试与复现。
- 所有剪辑决策都要写进 `.beatforge/plan.json`，方便人工复核；新增 render 配置会自动出现在其中。

## 本机环境注意
- `uv run pytest` 默认临时目录无权限，需要 `--basetemp=<可写目录>`。
- 无 NVIDIA 显卡：AI 链路用 `--no-ai` 验证，模型相关测试用 mock。
- `tests/test_ai_director.py` 有 2 个已知失败（`CausalModel` 缺 `generation_config`）。
