"""절 AF 공통 설정 표: 궤적마다 비측정 3 회 (af_<t>_c<N>_r*) + 단독 측정 3 회 (af_<t>_c<N>_t*) -- 완주 (6 회), 오차 (완주 평균; 회별), 단독 ms·fps, 기준 (≤ 16.7 ms · < 10 %),
진단 (미달 궤적): 첫 바닥 접촉 프레임, 프레임별 오차가 처음 10 % 를 넘는 프레임, 마지막·최대 프레임 오차, 발산 프레임, 클램프 작동 프레임 (r1 기준), 측정 중 카드 1~6 최대 MiB.
사용: python exe/fem_af_ctable.py <루트> <N>"""
import json
import os
import sys

root, N = sys.argv[1], int(sys.argv[2])


def load(tag):
    d = os.path.join(root, tag)
    f = os.path.join(d, "summary.json")
    return (json.load(open(f)), d) if os.path.exists(f) else (None, d)


def other(d):
    v = []
    for f in ("gpu_at_start.txt", "gpu_at_end.txt"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            v += [int(q.split(",")[1].split()[0]) for q in open(p) if q.split(",")[0].strip() != "0"]
    return max(v) if v else None


def diag(d):
    f = os.path.join(d, "fem.jsonl")
    if not os.path.exists(f):
        return None
    R = [json.loads(q) for q in open(f)]
    e = [r["err"] for r in R]
    return dict(c=next((r["t"] + 1 for r in R if r.get("n_contact", 0) > 0), None), x10=next((r["t"] + 1 for r in R if r["err"] >= 0.10), None),
                last=e[-1], mx=max(e), lim=sum(1 for r in R if r.get("lim_n", 0) > 0))


print(f"| 궤적 | N | 완주 (비측정 + 단독) | 오차 (완주 평균; 회별 비측정 / 단독) | 단독 ms/프레임 (평균; 회별) | fps | 기준 | 진단 (r1): 접촉 · 10 % 넘는 프레임 · 끝/최대 오차 · 클램프 작동 프레임 | 측정 중 카드 1~6 최대 MiB |")
print("|---|---|---|---|---|---|---|---|---|")
ng = 0; nt = 0
for i in range(20):
    t = f"{i:04d}"
    A = [load(f"af_{t}_c{N}_r{r}") for r in (1, 2, 3)]
    B = [load(f"af_{t}_c{N}_t{r}") for r in (1, 2, 3)]
    S = [s for s, _ in A + B if s]
    if not S:
        print(f"| {t} | {N} | 요약 없음 |"); continue
    fin = [s for s in S if not s["diverged"]]
    err = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
    fmt = lambda s: f"{100 * (s['err_mean'] if s['err_mean'] is not None else s['err_mean_done']):.2f}" + (f" 발산 {s['diverged']}" if s["diverged"] else "")
    BT = [s for s, _ in B if s]
    ms = sum(s["ms_mean"] for s in BT) / len(BT) if BT else None
    good = err is not None and len(fin) == 6 and ms is not None and ms <= 1000 / 60 and err < 0.10
    nt += 1; ng += good
    dg = diag(A[0][1])
    dgs = f"{dg['c']} · {dg['x10'] if dg['x10'] else '-'} · {100 * dg['last']:.1f}/{100 * dg['mx']:.1f} % · {dg['lim']}" if dg else "-"
    om = max((other(d) or 0) for _, d in B)
    print(f"| {t} | {N} | {len(fin)}/{len(A + B)} ({sum(1 for s, _ in A if s and not s['diverged'])}+{sum(1 for s in BT if not s['diverged'])}) | "
          + (f"{100 * err:.2f} % " if err is not None else "- ") + "(" + ", ".join(fmt(s) for s, _ in A if s) + " / " + ", ".join(fmt(s) for s in BT) + ")"
          + (f" | {ms:.2f} (" + ", ".join(f"{s['ms_mean']:.2f}" for s in BT) + f") | {1000 / ms:.1f}" if ms else " | - | -")
          + f" | {'만족' if good else '미달'} | {dgs} | {om} |")
print(f"\n기준 만족 {ng}/{nt} 궤적")
