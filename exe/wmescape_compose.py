"""수박 입자 날아감 비교 영상 로컬 합성 (exe/wmescape_vid.sh 결과; 읽기만).

칸 (왼쪽부터): 정답 | E | F = E + W̄ + R2 + A2 | F + 벌점 셋 γ̄ 0.1.  위 줄 = 3DGS 일반 렌더 (같은 렌더러·카메라), 아래 줄 = 멀리서 본 입자 점 그림
(모든 칸 같은 고정 범위, 바닥 포함; 몸 상자 4 칸 밖 이탈 입자 빨강 -- exe/wmescape_pts.py 의 정의 = 진단 D1).
사용: python exe/wmescape_compose.py --dir <R2 lbours/wmescape 받은 곳> --gt_plain gt_new_plain.mp4 --out out.mp4
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--gt_plain", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nbody", type=int, default=40000, help="칸마다 그릴 몸 점 수 (고정 표본에서 다시 고름)")
ap.add_argument("--frames", type=int, default=101)
ap.add_argument("--jobs", type=int, default=4)
a = ap.parse_args()
FONT = "/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf"
FONTB = "/home/wgsong/.fonts/NotoSansCJKkr-Bold.otf"
PW = 600                                                                                # 칸 한 변 (4 칸 = 2400)
FLOOR = 0.02

pick = json.load(open(os.path.join(a.dir, "pick.json")))
COLS = [("gt", "정답", None)]
NAME = {"E": "E (대조, 고침 없음)", "F": "F = E + W̄ + R2 + A2", "G": "F + 벌점 셋 γ̄ 0.1"}
for v in "EFG":
    n = pick[v]["pick"]
    r = [q for q in pick[v]["runs"] if q["run"] == n][0]
    COLS.append((n, NAME[v], r))
PTS = {c[0]: dict(np.load(os.path.join(a.dir, "pts", c[0] + ".npz"))) for c in COLS}


def far_at(P, f):
    return P["far_x"][P["far_o"][f]:P["far_o"][f + 1]]


# 고정 범위: 모든 칸·모든 프레임의 유한 점 (몸 표본 + 이탈 입자) 을 담는 정육면체 (바닥 포함)
lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
for k, P in PTS.items():
    s = P["sub"].astype(np.float32).reshape(-1, 3)
    s = s[np.isfinite(s).all(1)]
    fx = P["far_x"][np.isfinite(P["far_x"]).all(1)]
    allp = np.concatenate([s, fx]) if len(fx) else s
    lo = np.minimum(lo, allp.min(0)); hi = np.maximum(hi, allp.max(0))
lo[2] = min(lo[2], FLOOR); hi[2] = max(hi[2], FLOOR)
ctr = (lo + hi) / 2
half = float((hi - lo).max()) / 2 * 1.08
LIM = [(ctr[i] - half, ctr[i] + half) for i in range(3)]
print("고정 범위 (m)", [tuple(round(v, 2) for v in l) for l in LIM], "한 변", round(2 * half, 2), flush=True)
rng = np.random.default_rng(0)
SEL = {k: np.sort(rng.choice(P["sub"].shape[1], min(a.nbody, P["sub"].shape[1]), replace=False)) for k, P in PTS.items()}
os.makedirs(a.out + ".parts/pts", exist_ok=True)


def draw(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    fig = plt.figure(figsize=(4 * PW / 100, PW / 100), dpi=100)
    for i, (k, lab, r) in enumerate(COLS):
        P = PTS[k]
        ax = fig.add_axes([i / 4, 0, 1 / 4, 1], projection="3d")
        ax.set_proj_type("persp", focal_length=0.6)
        ff = min(f, P["sub"].shape[0] - 1)
        s = P["sub"][ff][SEL[k]].astype(np.float32)
        s = s[np.isfinite(s).all(1)]
        fx = far_at(P, ff)
        fx = fx[np.isfinite(fx).all(1)]
        gx, gy = np.meshgrid(np.linspace(*LIM[0], 2), np.linspace(*LIM[1], 2))
        ax.plot_surface(gx, gy, np.full_like(gx, FLOOR), color="#9ab", alpha=0.25, linewidth=0, shade=False)
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.15, c="#555", alpha=0.6, depthshade=False, linewidths=0)
        if len(fx):
            ax.scatter(fx[:, 0], fx[:, 1], fx[:, 2], s=6, c="#e01010", depthshade=False, linewidths=0)
        ax.set_xlim(*LIM[0]); ax.set_ylim(*LIM[1]); ax.set_zlim(*LIM[2]); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-60)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_xlabel("x m", fontsize=8, labelpad=-4); ax.set_ylabel("y m", fontsize=8, labelpad=-4); ax.set_zlabel("z m", fontsize=8, labelpad=-4)
        nf = int(P["nfar"][ff]); dm = float(P["dmax"][ff])
        fig.text(i / 4 + 0.008, 0.955, f"프레임 {f}   지금 이탈 {nf} 개 · 최대 {dm:.2f} m", fontsize=12, color="#c00" if nf else "#222")
        if i:
            fig.add_artist(plt.Line2D([i / 4, i / 4], [0, 1], color="#999", lw=1))
    p = f"{a.out}.parts/pts/{f:04d}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


with ProcessPoolExecutor(a.jobs) as ex:
    list(ex.map(draw, range(a.frames)))

# 머리 (칸 이름 + 이탈 수·거리)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
HH = 132
im = Image.new("RGB", (4 * PW, HH), "white"); dr = ImageDraw.Draw(im)
fB, fR, fS = ImageFont.truetype(FONTB, 26), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 17)
for i, (k, lab, r) in enumerate(COLS):
    dr.text((i * PW + 14, 8), lab, font=fB, fill="black")
    if r is None:
        P = PTS["gt"]
        t = f"이탈 최대 {int(P['nfar'].max())} 개 · 최대 거리 {float(P['dmax'].max()):.2f} m"
    else:
        t = f"이탈 최대 {r['nfar']} 개 · 최대 거리 {r['dmax']:.2f} m"
        t2 = f"덤프 판 {k} (2 판 중 이탈 많은 판), 오차 {100 * r['err']:.2f} %" if r.get("err") is not None else f"덤프 판 {k}"
        dr.text((i * PW + 14, 72), t2, font=fS, fill="#444")
    dr.text((i * PW + 14, 44), t, font=fR, fill="#c00000")
    if i:
        dr.line([(i * PW, 0), (i * PW, HH)], fill="#999", width=1)
dr.text((14, HH - 26), "위: 3DGS 렌더 (같은 카메라)   아래: 멀리서 본 입자 점 (모든 칸 같은 고정 범위, 파란 면 = 바닥, 빨강 = 몸 상자에서 4 칸 넘게 벗어난 이탈 입자)", font=fS, fill="#222")
im.save(a.out + ".parts/head.png")

vids = [a.gt_plain] + [os.path.join(a.dir, k, "vj", "plain.mp4") for k, _, _ in COLS[1:]]
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "32", "-i", a.out + ".parts/head.png"]
for v in vids:
    cmd += ["-i", v]
cmd += ["-framerate", "32", "-i", a.out + ".parts/pts/%04d.png"]
fc = "".join(f"[{j + 1}:v]scale={PW}:{PW},setsar=1,trim=end_frame={a.frames},setpts=PTS-STARTPTS[v{j}];" for j in range(4))
fc += "[v0][v1][v2][v3]hstack=inputs=4[top];[0:v]trim=end_frame=" + str(a.frames) + ",setpts=PTS-STARTPTS[h];[5:v]setpts=PTS-STARTPTS[b];"
fc += "[h][top][b]vstack=inputs=3,format=yuv420p[o]"
cmd += ["-filter_complex", fc, "-map", "[o]", "-frames:v", str(a.frames), "-r", "32", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True)
print("영상", a.out, flush=True)
