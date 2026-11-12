"""서브스텝 안정성 '우리(육면체)' 같은 N 비교 영상 합성 (11-12 지시, 묶음 stabhex; 노드 CPU).

판 (왼쪽부터, 칸 위에 한국어 이름): 이미 있는 영상 (R2 result/anchorflow/stab_vid, 같은 렌더러 rep_ip 의 F 반영 3DGS -- 다시 안 돌림) + 이번 육면체 렌더.
  레고 N16 : GF N16 | 우리(사면체, rep_ip) N16 | 우리(육면체) N16
  레고 N64 : PG N64 | 우리(사면체, rep_ip) N64 | 우리(육면체) N64
  레고 N1  : 우리(사면체, ovs) N1 | 우리(육면체) N1 ;  mic·ficus N1 : 우리(사면체, rep_ip) N1 | 우리(육면체) N1
없는 영상은 그 칸을 빼고 남은 칸만. 출력 가로 ≤ 2400, yuv420p, faststart (H.264).

  python exe/stabhex_compose.py --hexvid <육면체 영상 폴더 (<형상>_N<k>_a.mp4)> --old <옛 영상 폴더> --font <otf> --out <폴더>
"""
import argparse
import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--hexvid", required=True)
ap.add_argument("--old", required=True)
ap.add_argument("--font", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--maxw", type=int, default=2400)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FONT = ImageFont.truetype(a.font, 30)
PLAN = [("lego_N16", [("GF N16", "old", "lego_gf_N16.mp4"), ("우리(사면체, rep_ip) N16", "old", "lego_ours_N16.mp4"), ("우리(육면체) N16", "hex", "lego_N16_a.mp4")]),
        ("lego_N64", [("PG N64", "old", "lego_pg_N64.mp4"), ("우리(사면체, rep_ip) N64", "old", "lego_ours_N64.mp4"), ("우리(육면체) N64", "hex", "lego_N64_a.mp4")]),
        ("lego_N1", [("우리(사면체, ovs) N1", "old", "lego_ovs_N1.mp4"), ("우리(육면체) N1", "hex", "lego_N1_a.mp4")]),
        ("mic_N1", [("우리(사면체, rep_ip) N1", "old", "mic_ours_N1.mp4"), ("우리(육면체) N1", "hex", "mic_N1_a.mp4")]),
        ("ficus_N1", [("우리(사면체, rep_ip) N1", "old", "ficus_ours_N1.mp4"), ("우리(육면체) N1", "hex", "ficus_N1_a.mp4")]),
        ("mic_N16", [("우리(사면체, rep_ip) N16", "old", "mic_ours_N16.mp4"), ("우리(육면체) N16", "hex", "mic_N16_a.mp4")]),
        ("ficus_N16", [("우리(사면체, rep_ip) N16", "old", "ficus_ours_N16.mp4"), ("우리(육면체) N16", "hex", "ficus_N16_a.mp4")])]


def frames(p):
    r = imageio.get_reader(p)
    out = [np.asarray(f)[..., :3] for f in r]
    r.close()
    return out


for name, cols in PLAN:
    panels = []
    for lab, src, fn in cols:
        p = os.path.join(a.hexvid if src == "hex" else a.old, fn)
        if os.path.exists(p):
            panels.append((lab, frames(p)))
        else:
            print(f"[합성] {name}: {fn} 없음 -> 칸 뺌", flush=True)
    if not any(lab.startswith("우리(육면체)") for lab, _ in panels):
        print(f"[합성] {name}: 육면체 영상 없음 -> 건너뜀", flush=True)
        continue
    H0 = min(f[0].shape[0] for _, f in panels)
    T = max(len(f) for _, f in panels)
    ws = [int(round(f[0].shape[1] * H0 / f[0].shape[0])) for _, f in panels]
    sc = min(1.0, a.maxw / sum(ws))
    Hh = int(H0 * sc) // 2 * 2; ws = [int(w * sc) // 2 * 2 for w in ws]
    raw = os.path.join(a.out, f"{name}_raw.mp4")
    W = imageio.get_writer(raw, fps=30, codec="libx264", quality=9, macro_block_size=2)
    for t in range(T):
        row = []
        for (lab, fr), w in zip(panels, ws):
            im = Image.fromarray(fr[min(t, len(fr) - 1)]).resize((w, Hh), Image.LANCZOS)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, w, 44], fill=(255, 255, 255))
            dr.text((10, 6), f"{lab}  프레임 {min(t, len(fr) - 1)}", fill=(0, 0, 0), font=FONT)
            row.append(np.asarray(im))
        W.append_data(np.concatenate(row, 1))
    W.close()
    dst = os.path.join(a.out, f"{name}.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw, "-vf", f"scale='min({a.maxw},iw)':-2", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-c:v", "libx264", "-crf", "18", dst], check=True)
    os.remove(raw)
    print(f"[합성] {name}: {[lab for lab, _ in panels]} -> {dst} ({sum(ws)}x{Hh}, {T} 프레임)", flush=True)
