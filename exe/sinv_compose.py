"""절 CM128 wolf 영상 합성 (로컬 ffmpeg, exe/convabl_compose.py 복사): 정답 | H9 (지금) | H9 + --xsinv 1 | cvb + --xsinv 1, 칸 600, 2400 폭, yuv420p, faststart, 51 프레임.
사용: python exe/sinv_compose.py <자료 디렉토리>   (gt_wolf.mp4, an.json (extsolve_an 꼴: a9·b1·c1), a9·b1·c1/vj/fem_wolf.mp4)"""
import json, os, subprocess, sys

D = sys.argv[1]
FB = os.path.expanduser("~/.fonts/NotoSansCJKkr-Bold.otf"); FR = os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")
T = f"{D}/txt"; os.makedirs(T, exist_ok=True)
A = json.load(open(f"{D}/an.json"))
NF = 51


def tf(name, text):
    p = f"{T}/{name}.txt"; open(p, "w").write(text); return p


def dt(f, font, size, x, y, color="black", en=None):
    return f"drawtext=fontfile={font}:textfile={f}:fontsize={size}:fontcolor={color}:expansion=none:x={x}:y={y}" + (f":enable='{en}'" if en else "")


def er(k):
    a = A[k]; dv = f" · f{a['frames_done']} 발산" if a.get("diverged") else ""
    return f"오차 {100 * a['err']:.2f}%{dv}"


cells = [("정답 (PG 모래)", "", f"{D}/gt_wolf.mp4", None),
         ("H9 (지금, 뒤집힌 점 = 거울상)", er("a9"), f"{D}/a9/vj/fem_wolf.mp4", "a9"),
         ("H9 + 뒤집힌 점 = 압축", er("b1"), f"{D}/b1/vj/fem_wolf.mp4", "b1"),
         ("수렴 판 (뉴턴 60·PCG 100) + 압축", er("c1"), f"{D}/c1/vj/fem_wolf.mp4", "c1")]
W = 600; ins, fl = [], []
for k, (a, b, path, key) in enumerate(cells):
    ins += ["-i", path]
    f_ = f"[{k}:v]fps=30,tpad=stop=-1:stop_mode=clone,scale={W}:{W},setsar=1,pad={W}:{W + 120}:0:120:white," + dt(tf(f"c{k}a", a), FB, 26, 12, 10)
    if b:
        f_ += "," + dt(tf(f"c{k}b", b), FB, 24, 12, 52, "0x1f4fbf")
    for n in range(NF):
        h = A["a9"]["h"].get(str(max(n, 1)))
        if key is None:
            txt = f"프레임 {n} · 질량중심 {h[1]:.0f} mm (바닥 위)" if h else f"프레임 {n}"
        else:
            hk = A[key]["h"].get(str(max(n, 1)))
            rr = A[key]["r07"].get(str(n), 0) if n > 0 else 0
            txt = f"프레임 {n} · 질량중심 {hk[0]:.0f} mm · r<0.7 칸 {rr}" if hk else f"프레임 {n}"
        f_ += "," + dt(tf(f"c{k}t{n}", txt), FR, 19, 12, 90, "0x333333", f"eq(n,{n})")
    fl.append(f_ + f"[c{k}]")
fl.append("[c0][c1][c2][c3]hstack=inputs=4," + ",".join(f"drawbox=x={W * j - 1}:y=0:w=2:h={W + 120}:color=0xdddddd:t=fill" for j in range(1, 4))
          + f",pad={4 * W}:{W + 170}:0:50:white," + dt(tf("title", "wolf 모래 · 뒤집힌 가우스 점을 압축으로 (부호 있는 SVD, 절 CM128, 새 플래그, 기본 판 안 바꿈), 각 1 판; 질량중심 = 바닥 위, r<0.7 = 그 프레임 칸 세로 두께 비"), FR, 20, 16, 14, "0x444444") + "[o]")
out = f"{D}/fem_wolf_sinv_cmp.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", ";".join(fl), "-map", "[o]", "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", "-frames:v", str(NF), out], check=True)
print("[저장]", out)
