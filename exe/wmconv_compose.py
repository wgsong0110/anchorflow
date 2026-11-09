"""절 CM90 묶음 wmconv 영상 합성 (로컬 ffmpeg; exe/wolfh10_compose.py 방식): 정답 | H9 그대로 | 수렴 판, 위 줄 일반 3DGS, 아래 줄 손상 색 (logJp viridis −0.2~0.4).
사용: python exe/wmconv_compose.py <자료 디렉토리> <H9 판> <수렴 판>
  자료: gt_new_plain.mp4 · gt_lj.mp4 (R2 lbours/fem_cm/wmJd), <판>/vj/plain.mp4 · lj.mp4 · summary.json · wmconv_stat.json (R2 lbours/wmconv)
출력: fem_wmconv_cmp.mp4 (칸 800, 2400 폭, yuv420p, faststart, 정답 프레임 수), wmconv_f5.png · wmconv_f96.png"""
import json
import os
import subprocess
import sys

D, A, C = sys.argv[1], sys.argv[2], sys.argv[3]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
T = f"{D}/txt"
os.makedirs(T, exist_ok=True)


def lab(r):
    s = json.load(open(f"{D}/{r}/summary.json"))
    st = json.load(open(f"{D}/{r}/wmconv_stat.json"))
    e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
    f = st["fr"]
    return f"오차 {100 * e:.2f}% · 손상 f5 {100 * f['5']['dmg']:.2f} · 끝 {100 * f[str(st['L'])]['dmg']:.2f}%" + (f" (f{s['diverged']} 발산)" if s.get("diverged") else "")


cols = [("수박 · 정답 (GF)", "손상 f5 4.41 · 끝 6.51%", f"{D}/gt_new_plain.mp4", f"{D}/gt_lj.mp4"),
        ("수박 · H9 그대로", lab(A), f"{D}/{A}/vj/plain.mp4", f"{D}/{A}/vj/lj.mp4"),
        ("수박 · 수렴 판 (예산 끔·뉴턴 20·PCG 40)", lab(C), f"{D}/{C}/vj/plain.mp4", f"{D}/{C}/vj/lj.mp4")]
fps, nf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0", cols[0][2]],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")


def tf(name, text):
    p = f"{T}/{name}.txt"
    open(p, "w").write(text)
    return p


def dt(f, font, size, x, y, color="black"):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}"


Wc, H = 800, 110


def build(sel=None):
    ins, fl = [], []
    for k, (a, b, p, l) in enumerate(cols):
        for j, path in enumerate((p, l)):
            ins += ["-i", path]
            i = 2 * k + j
            g = f"[{i}:v]fps={fps},tpad=stop=-1:stop_mode=clone,scale={Wc}:{Wc},setsar=1"
            if j == 0:
                g += f",pad={Wc}:{Wc + H}:0:{H}:white," + dt(tf(f"a{k}", a), FB, 30, 14, 10) + "," + dt(tf(f"b{k}", b), FB, 24, 14, 62, "0x1f4fbf" if k else "0x444444")
            fl.append(g + f"[v{i}]")
        fl.append(f"[v{2 * k}][v{2 * k + 1}]vstack=inputs=2[c{k}]")
    fl.append("[c0][c1][c2]hstack=inputs=3," + ",".join(f"drawbox=x={Wc * j - 1}:y=0:w=2:h={2 * Wc + H}:color=0xdddddd:t=fill" for j in (1, 2))
              + f",pad={3 * Wc}:{2 * Wc + H + 44}:0:44:white," + dt(tf("title", "수박 · 위 일반 3DGS · 아래 손상 색 (logJp viridis −0.2~0.4)"), FR, 22, 14, 10, "0x444444")
              + (f",select='eq(n\\,{sel})'" if sel is not None else "") + "[o]")
    return ins, ";".join(fl)


ins, fl = build()
out = f"{D}/fem_wmconv_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fl, "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", fps, "-frames:v", nf, out], check=True)
print("[저장]", out)
for f in (5, 96):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-vf", f"select='eq(n\\,{f})'", "-frames:v", "1", "-update", "1", f"{D}/wmconv_f{f}.png"], check=True)
    print("[저장]", f"{D}/wmconv_f{f}.png")
