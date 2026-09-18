# 内置中文字体

这里放的是 BeatForge **随仓库分发**的中文字体。没有它们，一台干净的 Linux 机器或一个精简安装的
Windows 上，所有 `subtitle_font = "preset:*"` 都会落到同一个系统兜底字体，或者直接渲染成方框。

全部为 **SIL Open Font License 1.1**，允许再分发；每份字体旁边都放了对应的许可证原文。

| 文件 | 字族 | 用途 | 体积 |
| --- | --- | --- | --- |
| `NotoSansSC-VF.ttf` | Noto Sans SC | `modern` / `minimal`，以及 `dreamy`(300) / `dark`(700) / `energetic`(700) 的字重 | 16.9 MiB |
| `NotoSerifSC-VF.ttf` | Noto Serif SC | `cinematic`、`lyrical` 的衬线 | 24.0 MiB |
| `LXGWWenKai-Regular.ttf` | LXGW WenKai | `lyrical` 的楷体 | 23.6 MiB |
| `SmileySans-Oblique.ttf` | Smiley Sans | `energetic` / `dark` 的标题感黑体 | 2.5 MiB |

合计约 **67 MiB**，用 **git-lfs** 存储（见仓库根目录的 `.gitattributes`）。

## 为什么是可变字体，以及字重怎么传

前两个是**可变字体**：一个文件覆盖 100–900 的字重轴，比打包五份静态字重省得多。代价是
**字重不能写进族名**——fontconfig 不暴露命名实例，`Fontname: Noto Sans SC Light` 解析不到任何东西，
会**静默回退到系统字体**，而画面上只表现为"这个预设好像没生效"。

所以字重走 ASS 的 `\b` 标签：预设表里写成 `"Noto Sans SC@300"`，
`fonts.resolve_subtitle_font()` 把它拆成 `FontChoice(family="Noto Sans SC", weight=300)`，
`write_ass()` 再对每条字幕发出 `{\b300}`。实测该轴确实生效：

| 预设 | 字族 | 字重 | 实测墨量 |
| --- | --- | --- | --- |
| `modern` | Noto Sans SC | 默认(400) | 3735 |
| `dreamy` | Noto Sans SC | 300 | 2096 |
| `cinematic` | Noto Serif SC | 默认 | 3016 |
| `lyrical` | LXGW WenKai | 默认 | 3195 |
| `dark` | Smiley Sans | 默认 | 5494 |

（墨量 = 渲染后亮度 >40 的像素数，同一句同一字号；只用于横向比较粗细。）

## 加新字体

1. 确认许可证允许再分发（OFL / Apache / 公有领域），把许可证原文一起放进来；
2. 文件放到本目录，**族名要在 `beatforge/fonts.py` 的 `FONT_PRESETS` 里出现**，
   否则没有任何预设能选中它——`tests/test_fonts.py` 会检查这一条；
3. 如果是可变字体，在预设表里用 `族名@字重` 指定字重。

## 注意

`git clone` 之后如果看到这些 `.ttf` 是**几百字节的文本文件**，说明机器上没装 git-lfs。
`tests/test_fonts.py::test_bundled_fonts_are_real_font_files` 会直接报出来——它检查文件头魔数，
因为指针文件本身是能正常读的文本，不检查就会一路跑到渲染才以"方框字"的形式暴露。
