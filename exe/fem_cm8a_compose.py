"""절 CM8 (A) lego 영상 합성 (로컬 ffmpeg, 절 CM4 합성 방식): 정답 | 우리 (A) 가우스 점 고착 바닥.
우리 칸은 멈춘 프레임 뒤 정지 화면 + 발산 프레임 표시. 사용: python exe/fem_cm8a_compose.py L000|0016
입력 R2 lbours/fem_cm/cm8av/v_<씬>/{summary.json, vj/fem_lego.mp4}; 출력 ~/workspace/result/anchorflow/v20261009/fem_cm8a_lego_<씬>_cmp.mp4"""
import json, os, subprocess, sys
L = os.path.expanduser("~/r2/result/anchorflow/lbours")
V = f"{L}/fem_cm/cm8av"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
TMP = "/tmp/fem_cm8a_compose"
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf")
FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
t = sys.argv[1]
gt = {"L000": f"{L}/fem2_lego/gt_lego.mp4", "0016": f"{L}/fem_af_lego/gt_0016/gt_lego.mp4"}[t]
s = json.load(open(f"{V}/v_{t}/summary.json"))
d = s.get("diverged"); nf = s["frames"]
e = s.get("err_mean") if s.get("err_mean") is not None else s.get("err_mean_done")
er = f"오차 {100 * e:.2f}%" + (f" (프레임 {s.get('frames_done')} 까지)" if d else "")
ours = f"{TMP}/ours_{t}.mp4"  # R2 FUSE 목록이 늦게 갱신되어 rclone 으로 직접 받음
subprocess.run(["rclone", "copyto", f"r2:storage/result/anchorflow/lbours/fem_cm/cm8av/v_{t}/vj/fem_lego.mp4", ours, "-q"], check=True)
n_ours = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", ours]).split()[0])
title = f"lego {t} {nf} 프레임 · (A) 가우스 점 고착 바닥 · HHT ρ 0.8, K 2 x 뉴턴 2 x PCG 6, εN 1e-2 · εP 1e-1 · 결정적"


def tf(name, text):
    p = f"{TMP}/{t}_{name}.txt"; open(p, "w").write(text); return p


def dt(f, font, size, x, y, color="black", extra=""):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}{extra}"


# 우리 칸: 마지막 프레임에서 정지 (tpad clone), 발산 이후 프레임에는 붉은 경고
f1 = f"[1:v]scale=800:800,tpad=stop_mode=clone:stop={max(nf - n_ours, 0)},pad=800:1000:0:200:white," + dt(tf("c1a", "우리 (A) 가우스 점 고착"), FB, 32, 16, 70) + "," + dt(tf("c1b", er), FB, 32, 16, 124, "0x1f4fbf")
if d:
    f1 += "," + dt(tf("c1c", f"프레임 {d} 에서 발산 · 이후 정지 화면"), FB, 24, 16, 170, "0xcc0000", f":enable='gte(n,{n_ours})'")
    f1 += "," + dt(tf("c1d", f"프레임 {d} 에서 발산 (그 전까지 계산)"), FR, 24, 16, 170, "0x888888", f":enable='lt(n,{n_ours})'")
f1 += "[c1]"
f0 = "[0:v]scale=800:800,pad=800:1000:0:200:white," + dt(tf("c0a", "정답 (MPM)"), FB, 32, 16, 70) + "[c0]"
fc = f0 + ";" + f1 + ";[c0][c1]hstack=inputs=2,drawbox=x=799:y=60:w=2:h=940:color=0xdddddd:t=fill," + dt(tf("title", title), FR, 20, 16, 16, "0x444444") + "[o]"
out = f"{OUT}/fem_cm8a_lego_{t}_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", gt, "-i", ours, "-filter_complex", fc, "-map", "[o]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-shortest", out], check=True)
for k, ss in (("a", 0.3), ("b", 3.0)):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(ss), "-i", out, "-frames:v", "1", f"{TMP}/chk_{t}_{k}.png"], check=True)
print(out)
