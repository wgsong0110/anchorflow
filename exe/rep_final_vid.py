"""rep_final (11-15) 표현력 비교 영상 도구.
  to_h5:   rep_track_sg 결과 (--save_traj --traj32 --save_F K) -> lb_render 가 읽는 궤적 디렉토리 (정답 디렉토리의 meta.json·0 프레임을 그대로 쓰고,
           프레임 t = 정규화 위치 traj[t-1] -> 시뮬 좌표 (y - off)·s + lo, F = Fs[t-1] (phystwin 은 공식 누적 회전 R)).
           python exe/rep_final_vid.py to_h5 --res R.npz --aux A.npz --gt_dir GTDIR --out DIR
  grid:    같은 카메라로 그린 PNG 폴더들을 3x2 격자 (칸마다 한국어 이름표) 로 합쳐 mp4 (libx264, yuv420p).
           python exe/rep_final_vid.py grid --png D1 ... --label L1 ... --title T --font F.otf --fps 60 --out V.mp4
"""
import argparse, glob, json, os, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["to_h5", "grid"])
ap.add_argument("--res"); ap.add_argument("--aux"); ap.add_argument("--gt_dir"); ap.add_argument("--out", required=True)
ap.add_argument("--png", nargs="*", default=[]); ap.add_argument("--label", nargs="*", default=[])
ap.add_argument("--title", default=""); ap.add_argument("--font", default=""); ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--cols", type=int, default=3)
a = ap.parse_args()

if a.cmd == "to_h5":
    import h5py
    Z = np.load(a.res); AX = np.load(a.aux, allow_pickle=True)
    lo, s_, off = AX["lo"].astype(np.float64), float(AX["s"]), AX["off"].astype(np.float64)
    TR, FS = Z["traj"], Z["Fs"]
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(f"{a.gt_dir}/meta.json")); json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    g0 = sorted(glob.glob(f"{a.gt_dir}/sim_*.h5"))[0]
    with h5py.File(g0) as h, h5py.File(f"{a.out}/sim_0000000000.h5", "w") as o:
        x0 = np.array(h["x"]); o.create_dataset("x", data=x0); o.create_dataset("F", data=np.array(h["F"]))
    x0 = x0.T if x0.shape[0] == 3 and x0.shape[1] != 3 else x0
    # 검산: 방법 궤적의 시작 (정규화 X0 -> 시뮬) 이 정답 0 프레임과 같은 입자 순서인가
    G = (AX["G"].astype(np.float64) - off) * s_ + lo
    e0 = float(np.abs(G - x0[:G.shape[0]]).max())
    print(f"[대응] 입자 {G.shape[0]} / 정답 {x0.shape[0]}  0 프레임 위치 차이 최대 {e0:.2e}  F 저장 점 {FS.shape[1]}  프레임 {TR.shape[0]}", flush=True)
    assert e0 < 1e-4 and TR.shape[1] == x0.shape[0], "정답 0 프레임과 입자 대응이 맞지 않는다"
    for t in range(TR.shape[0]):
        x = (TR[t].astype(np.float64) - off) * s_ + lo
        with h5py.File(f"{a.out}/sim_{t + 1:010d}.h5", "w") as o:
            o.create_dataset("x", data=x.astype(np.float32))
            o.create_dataset("F", data=FS[t].astype(np.float32).reshape(-1, 9))
    print(f"[저장] {a.out}  {TR.shape[0] + 1} 프레임", flush=True)
    sys.exit(0)

# grid
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
fs = [sorted(glob.glob(f"{d}/*.png")) for d in a.png]
n = min(len(q) for q in fs)
print("[격자] 칸", len(fs), "프레임", [len(q) for q in fs], "->", n, flush=True)
im0 = np.asarray(Image.open(fs[0][0]).convert("RGB")); H, Wd = im0.shape[:2]
cols = a.cols; rows = (len(fs) + 1 + cols - 1) // cols if a.title else (len(fs) + cols - 1) // cols
F1 = ImageFont.truetype(a.font, int(H * 0.055)); F2 = ImageFont.truetype(a.font, int(H * 0.06))
WR = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=16)
for i in range(n):
    can = Image.new("RGB", (Wd * cols, H * rows), (255, 255, 255)); dr = ImageDraw.Draw(can)
    for k, q in enumerate(fs):
        r, c = divmod(k, cols); can.paste(Image.open(q[i]).convert("RGB"), (c * Wd, r * H))
        dr.text((c * Wd + int(Wd * 0.03), r * H + int(H * 0.02)), a.label[k], fill=(0, 0, 0), font=F1)
    if a.title:
        k = len(fs); r, c = divmod(k, cols)
        y = r * H + int(H * 0.3)
        for ln in a.title.split("|"):
            dr.text((c * Wd + int(Wd * 0.06), y), ln, fill=(0, 0, 0), font=F2); y += int(H * 0.1)
        dr.text((c * Wd + int(Wd * 0.06), y + int(H * 0.05)), f"프레임 {i}", fill=(120, 120, 120), font=F2)
    WR.append_data(np.asarray(can))
WR.close()
print(f"[영상] {a.out}  {Wd * cols}x{H * rows}  {n} 프레임", flush=True)
