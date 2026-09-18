# 内置中文字体

这里放的是 BeatForge **随仓库分发**的中文字体。没有它们，一台干净的 Linux 机器或一个精简安装的
Windows 上，所有 `subtitle_font = "preset:*"` 都会落到同一个系统兜底字体，或者直接渲染成方框。

全部为 **SIL Open Font License 1.1**，允许再分发；每份字体旁边都放了对应的许可证原文。

| 文件 | 字族 | 预设 | 体积 |
| --- | --- | --- | --- |
| `NotoSansSC-VF.ttf` | Noto Sans SC | `modern` / `minimal`，以及 `dreamy`(300) / `dark`(700) / `energetic`(700) 的字重 | 16.9 MiB |
| `NotoSerifSC-VF.ttf` | Noto Serif SC | `cinematic`、`lyrical` 的衬线 | 24.0 MiB |
| `LXGWWenKai-Regular.ttf` | LXGW WenKai | `lyrical` 的楷体 | 23.6 MiB |
| `SmileySans-Oblique.ttf` | Smiley Sans | `energetic` / `dark` 的标题感黑体 | 2.5 MiB |
| `MaShanZheng-Regular.ttf` | Ma Shan Zheng 马善政毛笔楷书 | `brush` 毛笔 | 5.6 MiB |
| `ZhiMangXing-Regular.ttf` | Zhi Mang Xing 钟齐志莽行书 | `script` 行书 | 3.9 MiB |
| `ZCOOLKuaiLe-Regular.ttf` | ZCOOL KuaiLe 站酷快乐体 | `playful` 活泼 | 1.4 MiB |
| `ZCOOLQingKeHuangYou-Regular.ttf` | ZCOOL QingKe HuangYou 站酷庆科黄油体 | `poster` 海报粗圆 | 7.9 MiB |
| `ZCOOLXiaoWei-Regular.ttf` | ZCOOL XiaoWei 站酷小薇体 | `elegant` 秀气衬线 | 6.0 MiB |

合计约 **92 MiB**，用 **git-lfs** 存储（见仓库根目录的 `.gitattributes`）。

后五个是**艺术字体**，都不由情绪自动选中——毛笔或海报体用错歌比用普通字体更糟，
所以只能用 `subtitle_font = "preset:brush"` 这类方式显式指定。

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

- **族名读取不能只靠 PIL**。`PIL.ImageFont.getname()` 返回的是字体**默认语言**的族名，
  而中文商业字体的默认族名常常就是中文——PIL 解码失败会得到 `"?????"`，
  于是字体明明能用（fontconfig 认得它的英文名），却因为发现阶段读不到而永远选不中。
  `fonts.py` 现在直接读 name 表，收集**所有平台和语言**的记录，英文名和中文名都能寻址。
  `ZCOOL XiaoWei` 就是踩到这个的真实例子（默认族名 `站酷小薇体`，
  但 `Fontname: ZCOOL XiaoWei` 渲染正常、`Fontname: 站酷小薇体` 回退系统字体）。

## 字形覆盖：艺术字体是裁剪过的

五个艺术字体来自 Google Fonts，**码位只到 GB2312 常用字级别**。实测（65 个测试字符 =
demo 歌词用字 + 常见生僻字）：

| 字体 | demo 歌词用字 | 生僻字 |
| --- | --- | --- |
| 全部五个艺术字体 | **100% 覆盖（0 缺）** | 缺 7–13 个（囍垚彧惢掱燚犇瞐翀風飝骉龘 之类） |
| Noto Sans SC / Noto Serif SC | 100% | 0 缺 |

**结论：歌词正常用没问题，遇到真正的生僻字（人名、囍 之类）会缺字形。**
libass 会按字形回退到其它已安装字体，所以通常不是空白方框，但字形会跳变。
对字形完整性要求高的场合用 Noto 那两个。

## 加新字体

1. 确认许可证允许再分发（OFL / Apache / 公有领域），把许可证原文一起放进来；
2. 文件放到本目录，**族名要在 `beatforge/fonts.py` 的 `FONT_PRESETS` 里出现**，
   否则没有任何预设能选中它——`tests/test_fonts.py` 会检查这一条；
3. 如果是可变字体，在预设表里用 `族名@字重` 指定字重。

## 注意

`git clone` 之后如果看到这些 `.ttf` 是**几百字节的文本文件**，说明机器上没装 git-lfs。
`tests/test_fonts.py::test_bundled_fonts_are_real_font_files` 会直接报出来——它检查文件头魔数，
因为指针文件本身是能正常读的文本，不检查就会一路跑到渲染才以"方框字"的形式暴露。
