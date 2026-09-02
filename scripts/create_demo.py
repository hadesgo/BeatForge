"""Create deterministic synthetic media for an end-to-end test. Downloads nothing."""

from __future__ import annotations

import json
from pathlib import Path

from beatforge.runtime import command

ROOT = Path(__file__).resolve().parents[1] / "demo"
MEDIA = ROOT / "media"
MEDIA.mkdir(parents=True, exist_ok=True)

assets = [
    ("sunrise_天空_希望.jpg", "color=c=0xF4A261:s=1280x720:d=1", "drawbox=x=0:y=430:w=1280:h=290:color=0x264653:t=fill,drawbox=x=790:y=120:w=150:h=150:color=0xFFE66D:t=fill"),
    ("ocean_海_梦.jpg", "color=c=0x16324F:s=1280x720:d=1", "drawbox=x=0:y=360:w=1280:h=360:color=0x1D7874:t=fill,drawgrid=w=160:h=90:t=2:c=white@0.12"),
    ("city_城市_灯光.jpg", "color=c=0x111827:s=1280x720:d=1", "drawbox=x=0:y=360:w=1280:h=360:color=0x1D7874:t=fill,drawgrid=w=70:h=70:t=5:c=0xFACC15@0.5"),
]
for name, source, filter_graph in assets:
    command(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-vf", filter_graph, "-frames:v", "1", str(MEDIA / name)])

command([
    "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=15:sample_rate=44100",
    "-f", "lavfi", "-i", "sine=frequency=440:duration=15:sample_rate=44100",
    "-filter_complex", "[0:a]volume=.20,tremolo=f=2:d=.72[a0];[1:a]volume=.08,tremolo=f=2:d=.55[a1];[a0][a1]amix=inputs=2,afade=t=in:d=1,afade=t=out:st=14:d=1",
    str(ROOT / "music.wav"),
])
(ROOT / "lyrics.lrc").write_text("[00:00.00]黎明照亮天空\n[00:04.00]我们奔向希望\n[00:08.00]像海浪追逐着梦\n[00:12.00]城市亮起灯光\n", "utf-8")
print(json.dumps({"demo": str(ROOT), "next": "uv run beatforge run demo/project.toml --no-ai"}, ensure_ascii=False))
