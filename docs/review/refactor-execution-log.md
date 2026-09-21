# 重构执行记录 · BeatForge

> 编写人：主理人（交付总监）
> 日期：2026-09-22
> 边界：**行为不变纯重构** + 测试用例优化 + 静态检查引入
> 验收：重构前 `378 passed` → 重构后 `387 passed / 0 error / 0 failure / 0 skipped`

## 0. 为什么有这份文档

施工期间仓库的 `.git` 对象库损坏（见 §5），经用户裁定「先不管 git，专注代码」，所以**这批改动一开始是没有 commit 的**，这份文档 + 磁盘快照当时是唯一的变更凭证。

> **补记（同日）**：用户随后修好了 git（历史完整恢复，`git fsck` clean）。这批改动已按 4 个逻辑提交落到 `master`，落在 `760966e` 之上：
> | 提交 | 说明 |
> | --- | --- |
> | `00b2594` | `chore: 引入 ruff 静态检查，并清理失效的 noqa` |
> | `5d31f52` | `refactor: 全项目行为不变重构，并修复三处死代码与资源管理缺陷` |
> | `c7f736b` | `test: 优化测试用例：补覆盖盲区、修假断言并提速` |
> | `d4e69ed` | `docs: 补充代码审查报告、重构执行记录与项目记忆` |
>
> 提交后工作区与 `%TEMP%\bf_REFACTORED_20260922_014522` 快照**逐文件 SHA256 比对一致**（67/67），即提交内容与验收内容完全相同。

本文件记录**实际做了什么**；`architecture-review.md` 记录**计划做什么**。两份配合阅读。

## 1. 最终验收结果（均为主理人独立复跑）

| 项 | 结果 |
| --- | --- |
| `uv run pytest -q` | **387 passed / 0 failed / 0 error / 0 skipped** · 37.3s（冷跑 52.6s） |
| `uv run ruff check beatforge tests scripts` | **All checks passed!**（0 errors） |
| 基线（重构前） | 378 passed · 53.5s |
| 净变化 | 用例 **+9**，耗时 **-5 ~ -16s** |

## 2. 执行了哪些批次

每一批都由主理人独立复核（重算 diff / 目视高危补丁 / 复跑验收），不只采信执行者自述。

| 批次 | 内容 | 改动量 | 验证 |
| --- | --- | --- | --- |
| **T01** 护栏与基线 | 引入 ruff 配置；清理 25 条失效 `# noqa` | 8 文件 | pytest 378 ✅ · ruff 82→57 |
| **T02** 安全自动修 | `I001/UP037/F401/F541/SIM117/PIE808/C416` | 70 行 | pytest 378 ✅ · ruff 57→18 |
| **T03** 手改等价 | `UP035/RUF007/C405/RUF046/BLE001` | 16 行 | pytest 378 ✅ · ruff 18→3 |
| **T04** 真缺陷隔离 | 见 §3 | 14 行 | pytest 378 ✅ · ruff 3→0 |
| **T04b** 收尾 | `SIM115` 长注释重排（**AST 逐节点等价已实测**） | 3 行 | pytest 378 ✅ · ruff 0 |
| **T05** 测试优化 | 见 §4 | tests/ only | **pytest 387 ✅** · ruff 0 |

**总改动**：`beatforge/` 3 个文件 + `pyproject.toml` + `scripts/` 6 个文件 + `tests/` 6 个文件（新增 3、删除 1、修改 3）。

## 3. 修掉的真实缺陷（T04）

| # | 位置 | 问题 | 修法 |
| --- | --- | --- | --- |
| 1 | `beatforge/models/llama_server.py:409` | 清理条件 **写反**：`if log is None and sink is not subprocess.DEVNULL` 两个子句互斥（前者为真时 `sink` 必为 `DEVNULL`），`close()` 是**死代码** | 改为 `if log is not None`；并给 `except OSError` 路径补一次 `close()` |
| 2 | `beatforge/renderer.py:183` | `frames = max(1, round(render_duration * cfg.fps))` 赋值后从未读取（image 分支直接 return，video 分支全程不用） | 删除该行 |
| 3 | `beatforge/models/transcriber.py:74` | 每轮迭代新建闭包 `lambda key: getattr(item, key)`（非 bug，但脆弱且多一次分配） | 提为模块级 `_field(item, key)`，**保留** dict→`.get`（缺失返 `None`）/ 对象→`getattr`（缺失抛 `AttributeError`）的语义差异 |

> **对第 1 条的定性更正**：最初判断为「句柄永远关不掉、日志尾部会丢」**是夸大的**。架构师用 OS 级探针（`.probe/probe_sink.py`）实测证明 CPython 会在生成器帧销毁时回收并 flush，**无可观察行为差异**。故该修复属**潜在缺陷修复**，不是崩溃级问题。这也是它被隔离成单独提交/快照的原因。

## 4. 测试用例优化（T05，只动 `tests/`）

### 新增（+9 条、3 个新文件）

| 目标 | 文件 | 守住了什么 |
| --- | --- | --- |
| **video 分支盲区**（此前 **0 覆盖**） | `test_renderer.py` | `_render_shot` 的 video 路径（`renderer.py:189–221`）：真渲染输出存在 + 帧数正确；另一条 monkeypatch 断言用 `-stream_loop -1`/`-ss` 而非 `-loop`。**这正是死变量 `frames` 能长期存活的根因** |
| **假断言** | `test_renderer.py` | 原 `... or "st=" in ...` 析取项**几乎恒真**，断言被架空。改为解析真实 `fade=t=out:st=X:d=Y`，断言 `X>duration/2` 且 `X+Y ≈ duration-1/fps`（一帧容差），钉住 README「出点从结束前一帧起算」 |
| **beat-this 后端**（此前 0 覆盖） | `test_music_structure.py` | 构造参数 + 返回结构；`librosa` 后端须返回 `None` |
| **`classify_music`**（此前 0 覆盖） | `test_audio_semantics.py`（新） | >30s→3 个中心窗口 / ≤30s 整段（含 30.0s 边界）；7 个 `MOOD_LABELS`、softmax 归一、argmax |
| **`run_project` 主流程** | `test_pipeline.py`（新） | `--plan-only --no-ai` 轻量 e2e：plan.json 落盘、shots 非空、三处模型槽位为 `None` |
| **CLI** | `test_cli.py`（新） | `init` 冒烟 + 用 `load_project` 回读生成的模板 |

### 修改 / 删除（均不削弱覆盖）

- **删除 `tests/test_vision_rerank.py`**（仅 1 条）：其断言 `blended[1] > blended[0]`（2 元素）被 `test_vision_index.py::test_reranker_blend_preserves_shape_and_uses_pairwise_scores` 的 `np.argmax(result) == 1`（3 元素 + 形状断言）**完整包含且更强**。主理人已逐字比对确认。
- `test_llama_server.py`：删去重复的局部 `Response` 类，改用共享 `_FakeResponse`。
- `test_renderer.py`：提 `make_analysis(...)` 工厂替换 10 处内联 `AudioAnalysis(...)`，**断言一字未改**。

### 加速

| 项 | 前 | 后 | 说明 |
| --- | --- | --- | --- |
| `test_every_transition_in_the_library_actually_composes` | **13.66s** | **0.35s** | 59 个转场（21 族）从「每个各起一次 ffmpeg（59 次子进程）」合并为**一条 `filter_complex`**（1 次）。验证强度未降：整图成功即要求每个名字都能与两路真实流组合；另断言总时长，失效时自动回退逐个组合并点名出错转场 |

### 主动放弃项（经判断，理由记录在案）

- **Q4 去重**：`test_planner._varied_song` 与 `test_editing._song` **并非等价**（后者含 `downbeats`/`melody_times`/`melodic_motion`/`rhythmic_density`，前者刻意没有）。合并会改变 planner 测试输入语义 → **有真实回归风险，保持独立**。
- **Q17**：前提不成立——仓库里只有 1 处 `ffmpeg -h filter=xfade`，不存在重复。
- **Q18**：收益约 1s，却会把独立渲染测试耦合到 session 级共享可变媒体，**不值得**。
- Q9 / Q14 / Q15 / Q19：低价值或需真机（GPU）。

## 5. ⚠️ 施工期间的 git 损坏事件

- **时间线**：`00:52` 时 `git status --short` **仍成功**（返回 ` M .gitignore` + `master`）→ `01:07:48` `objects/pack` 目录被改动 → `01:08` 发现损坏。**损坏发生在会话进行中，非预先存在。**
- **损坏内容**：`pack-eec86aebe5e52478360b39af454f515259501e77.pack` 被删除（同名 `.idx` 残留）；`multi-pack-index` 还引用第三个不存在的 pack。5 个引用（`master` / `tmp-test` / `origin/HEAD` / `origin/master` / `HEAD`）全部指向丢失的 `760966e`。
- **可读提交仅剩**：`3d0ebdc`、`8a6f9696`（origin/master）。最近 4 个本地提交（`a6a6bc5` / `e5578f6` / `3347f2c` / `760966e`）是 loose objects，已丢失，**本地无从恢复**；`git fetch` 因网络禁用失败。
- **处置（用户裁定）**：不动 `.git`，冻结一切 git 写入，改用**文件级快照**回滚。
- **推测（非结论）**：损坏可能与宿主环境的文件删除机制（safe-delete 钩子）有关，且与 git 的写操作在时间上相关。**建议在恢复前避免对 `.git` 做任何写操作**，以保住尚可读的 `pack-15755e9b`（内含最后两个可读提交）。

## 6. 交付物位置与回滚方式

| 用途 | 路径 |
| --- | --- |
| 变更凭证（本文件） | `docs/review/refactor-execution-log.md` |
| 审查与方案（计划） | `docs/review/architecture-review.md` |
| **重构前**全量基线 | `%TEMP%\bf_baseline_20260922_011446`（65 文件） |
| **重构后**全量成果 | `%TEMP%\bf_REFACTORED_20260922_014522`（67 文件 + `MANIFEST.sha256`） |
| 各任务分部快照 | `%TEMP%\bf_snap_T01 … bf_snap_T04`、`bf_snap_T04_finalize`、`bf_snap_QA` |
| 验收证据（junit / ruff / diff） | `.probe\junit_*.xml`、`.probe\ruff_*.txt`、`.probe\t0*_diff.txt` |

**回滚**：git 已恢复，首选 `git reset --soft|--hard HEAD~4`（见 §7）。快照仍保留作为第二重保险：用上面的路径覆盖回工作区即可。`bf_baseline_*` = 重构前，`bf_REFACTORED_*` = 重构后（带 `MANIFEST.sha256`，可逐文件校验）。

## 7. 未做与已知偏差

- **git 提交**：已按 4 个逻辑提交落到 `master`（见 §0 补记）。原始计划是按 T01–T04b 边界拆成更细的提交；实际按「lint 护栏 / 行为不变重构+缺陷修复 / 测试优化 / 文档」四组切分，因为这四组可以**按文件干净分开**，而 T02/T03/T04 在 `renderer.py`、`llama_server.py`、`transcriber.py`、`vision_index.py` 上互相重叠，按任务边界切需要逐 hunk 手术，收益不抵风险。**若要回滚**：`git reset --soft HEAD~4` 只撤提交、保留工作区；`git reset --hard HEAD~4` 连工作区一起回退到 `760966e`。
- `beatforge/models/__pycache__/quantization.cpython-313.pyc` 是无主陈旧字节码（源码已删）。已 grep 确认对 `quantization` / `director_engine` / `contact_sheet` 等废弃配置**零引用**，属构建产物残留，未处理。
- `beatforge/models/vision_index.py:556` 的 `TRY004`（类型检查抛 `ValueError`）**保持原样**：其外层 `except` 会立刻接住并改抛 `RuntimeError`，异常永不外泄；且 `TRY` 规则组本次未启用，改它属新增规则集，超出"行为不变"边界。
- `ruff` 的 `E501`（长行）未启用：代码里 524 处长行多为中文文案与长 filter 字符串，截断有害。`RUF001/002/003`（全角标点 300+ 处）与 `B905` 已在配置中显式 `ignore` 并注明理由。
- **本轮不涉及任何模型推理与真实成片渲染**：验收标准是纯单元测试全绿。渲染正确性由既有守门测试 + `scripts/` 探针保证，本轮未改任何渲染逻辑。
