"""절 AH.17 기준 통과 16 씬 영상 합성 (로컬): 정답 | 우리 롤아웃 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
ms·오차 = 절 AH.17 표 값 (단독 측정 ah_bwg_<씬>_c1..3 평균). 영상 = 같은 명령 + --dump_f32 비측정 회 (lbours/fem_pass_*/pd_<씬>) 의 3DGS 알파블렌딩 렌더.
사용: python exe/fem_pass_compose.py [씬 ...]  (없으면 16 개 전부 + fem_pass_all.mp4)"""
import json
import os
import subprocess
import sys

R = "r2:storage/result/anchorflow/lbours"
SRC = "/tmp/fem_pass_src"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
ALL = ["0000", "0002", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0019", "L000", "F"]
SC = sys.argv[1:] or ALL


def get(rel, f):
    p = f"{SRC}/{rel}/{f}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        subprocess.run(["rclone", "copyto", f"{R}/{rel}/{f}", p], check=True)
    return p


rows = []
for t in SC:
    if t == "F":
        sub, shape, N, gt = "fem_ah_ficus", "ficus", 640, get("fem3_ficus", "gt_ficus.mp4")
        name, gtn, frames = "ficus F_pg_v0", "정답 (PG)", 125
        pd = "fem_pass_ficus/pd_F"
    else:
        sub, shape, N = "fem_ah_lego", "lego", 267
        gt = get("fem2_lego", "gt_lego.mp4") if t == "L000" else get(f"fem_af_lego/gt_{t}", "gt_lego.mp4")
        name, gtn, frames = (f"lego {t}" if t == "L000" else f"lego 홀드아웃 {t}"), "정답 (MPM)", 80
        pd = f"fem_pass_lego/pd_{t}"
    T = [json.load(open(get(f"{sub}/ah_bwg_{t}_c{r}", "summary.json"))) for r in (1, 2, 3)]
    assert not any(q["diverged"] for q in T)
    s = dict(T[0])
    s["ms_mean"] = sum(q["ms_mean"] for q in T) / 3
    s["fps"] = 1000 / s["ms_mean"]
    s["err_mean"] = sum(q["err_mean"] for q in T) / 3
    sj = f"{SRC}/tab_{t}.json"
    json.dump(s, open(sj, "w"))
    d = json.load(open(get(pd, "summary.json")))
    nv = get(pd, f"vj/fem_{shape}.mp4")
    title = f"{name} · {frames} 프레임 · 수치 = A6000 단독 3 회 평균 · 영상 회 오차 {100 * d['err_mean']:.2f}%"
    o = f"{OUT}/fem_pass_{'ficus' if t == 'F' else 'lego' + t}_cmp.mp4"
    subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, f"{gtn}|-|{gt}", f"우리 롤아웃 (N{N})|{sj}|{nv}"], check=True)
    rows.append((t, 100 * s["err_mean"], 100 * d["err_mean"], s["ms_mean"], s["fps"], o))
for r in rows:
    print(f"[씬] {r[0]} 표 오차 {r[1]:.2f}% · 영상 회 오차 {r[2]:.2f}% · {r[3]:.2f} ms · {r[4]:.1f} fps -> {r[5]}")
if not sys.argv[1:]:
    lst = f"{SRC}/concat.txt"
    open(lst, "w").write("".join(f"file '{r[5]}'\n" for r in rows))
    o = f"{OUT}/fem_pass_all.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", o], check=True)
    print("[저장]", o)
