"""절 AH 영상 합성 (로컬): 정답 | 절 AF 공통 설정 | 절 AH 새 판 -- exe/fem_ex_compose.py (칸마다 ms·fps·오차, % 는 textfile + expansion=none).
ms 는 단독 측정 3 회 평균 (GPU0), 오차는 같은 판 완주 회 평균. 영상은 비측정 덤프 회 (F f32) 의 3DGS 알파블렌딩 렌더.
사용: python exe/fem_ah_compose.py <장면: 0000~0019 | L000 | F> [새 판 스윕 이름 gkr_d025_vc75] [측정 이름 fin]"""
import json
import os
import subprocess
import sys

R = "r2:storage/result/anchorflow/lbours"
SRC = "/tmp/fem_ah_src"
OUT = os.path.expanduser("~/workspace/result/anchorflow/v20261009")
here = os.path.dirname(os.path.abspath(__file__))
t = sys.argv[1]
V = sys.argv[2] if len(sys.argv) > 2 else "gkr_d025_vc75"
FN = sys.argv[3] if len(sys.argv) > 3 else "fin"


def get(rel, f, must=True):
    d = f"{SRC}/{rel}"
    os.makedirs(os.path.dirname(f"{d}/{f}"), exist_ok=True)
    if not os.path.exists(f"{d}/{f}"):
        r = subprocess.run(["rclone", "copyto", f"{R}/{rel}/{f}", f"{d}/{f}"])
        if r.returncode and must:
            raise SystemExit(f"없음 {rel}/{f}")
    return f"{d}/{f}" if os.path.exists(f"{d}/{f}") else None


def avg(base, timed, runs, name):
    """base summary 에 ms = 단독 평균, 오차 = 완주 평균을 덮어쓴 json."""
    s = dict(json.load(open(base)))
    T = [json.load(open(p)) for p in timed if p]
    A = [json.load(open(p)) for p in runs if p]
    if T:
        s["ms_mean"] = sum(q["ms_mean"] for q in T) / len(T)
        s["fps"] = 1000 / s["ms_mean"]
    fin = [q for q in A + T if not q["diverged"]]
    s["err_mean"] = sum(q["err_mean"] for q in fin) / len(fin) if fin else None
    o = f"{SRC}/{name}.json"
    json.dump(s, open(o, "w"))
    return o, len(T), len(fin), len(A + T)


if t == "F":
    sub, shape = "fem_ah_ficus", "ficus"
    gt = get("fem3_ficus", "gt_ficus.mp4")
    afv = get("fem_ad_ficus/f_f_vc05_n640_F1", "vj/fem_ficus.mp4")
    afs = get("fem_ad_ficus/f_f_vc05_n640_F1", "summary.json")
    afn = "절 AF 공통 설정 (= 절 AD 판, N640)"
    title0 = "ficus F_pg_v0 125 프레임"
    N = 640
else:
    sub, shape = "fem_ah_lego", "lego"
    N = 267
    if t == "L000":
        gt = get("fem2_lego", "gt_lego.mp4")
        afv = get(f"{sub}/v_af_L000", "vj/fem_lego.mp4")
        afs, *_ = avg(get(f"{sub}/v_af_L000", "summary.json"), [get(f"{sub}/ah_af_L000_t{r}", "summary.json", False) for r in (1, 2, 3)],
                      [get(f"{sub}/v_af_L000", "summary.json")], "af_L000")
        title0 = "lego L000 80 프레임"
    else:
        gt = get(f"fem_af_lego/gt_{t}", "gt_lego.mp4")
        afv = get(f"fem_af_lego/af_{t}_c267_r1", "vj/fem_lego.mp4")
        afs, *_ = avg(get(f"fem_af_lego/af_{t}_c267_r1", "summary.json"), [get(f"fem_af_lego/af_{t}_c267_t{r}", "summary.json") for r in (1, 2, 3)],
                      [get(f"fem_af_lego/af_{t}_c267_r{r}", "summary.json") for r in (1, 2, 3)], f"af_{t}")
        title0 = f"lego 홀드아웃 {t} 80 프레임"
    afn = "절 AF 공통 설정 (N267)"
nv = get(f"{sub}/v_{FN}_{t}", f"vj/fem_{shape}.mp4")
ns, nT, nf, na = avg(get(f"{sub}/v_{FN}_{t}", "summary.json"), [get(f"{sub}/ah_{FN}_{t}_t{r}", "summary.json", False) for r in (1, 2, 3)],
                     [get(f"{sub}/ah_{V}_{t}_r{r}", "summary.json", False) for r in (1, 2, 3)], f"new_{t}")
title = (f"{title0} · A6000 단독 GPU0 · ms 단독 {nT} 회 평균 · 오차 완주 평균 · 새 판 = 절 AF + 정답 sticky 격자 가중 바닥·단위 시간 고착·비관통 깊이 5 mm·부피 상한 θ0.75")
o = f"{OUT}/fem_ah_{shape}{'' if t == 'F' else t}_cmp.mp4"
subprocess.run([sys.executable, f"{here}/fem_ex_compose.py", o, title, f"정답 ({'PG' if t == 'F' else 'MPM'})|-|{gt}", f"{afn}|{afs}|{afv}",
                f"절 AH 새 판 (N{N})|{ns}|{nv}"], check=True)
print("[저장]", o)
