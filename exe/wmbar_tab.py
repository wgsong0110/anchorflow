"""절 CM21 묶음 wmbar 표 (결과 불변 후처리): python exe/wmbar_tab.py OUT > OUT/wmbar_table.md

판 s0 (지금 SNH 판, 미끄럼 바닥) · b1 (1) 장벽 바닥 · b2 (2) + logJp 보간 · b3 (3) + F 보간, 각 a·b; 단독 시간 z_<판>.
지표: 발산·프레임, 평균 오차 (끝난 프레임), 오차 곡선 (f4·10·30·50·70·100), 손상 몫 logJp > 0 (마지막 유한 프레임, 정답 f100 6.51 %),
바닥 아래 최대 깊이 (입자 평면 부호 거리 최소의 음수, 프레임 최대 / 그 프레임 아래 입자 수 최대), 장벽 걸린 표본점 (프레임 평균 · 최대 / 표본점 수),
입자 det F ≤ 0 최대 (jdiag), 단독 시간 (재생 GPU 시간 첫 2 프레임 뺀 평균 · 최대 ms).
정답 바닥 아래 깊이는 wm.pt 정답 위치에서 같은 식 (OUT/gt_fd.json 에 한 번 계산해 둠)."""
import json
import math
import os
import sys

O = sys.argv[1]
WM = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dkta/ag_data/wm.pt"
PN, NR = 0.02, (0.0, 0.0, 1.0)
GT_LJ0 = 0.0651
NAME = {"s0": "지금 SNH 판 (미끄럼 바닥)", "b1": "(1) 장벽 바닥", "b2": "(2) (1) + logJp 보간", "b3": "(3) (2) + F 보간"}


def jl(p):
    try:
        return [json.loads(q) for q in open(p)]
    except Exception:
        return []


def js(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def gt_fd():
    p = f"{O}/gt_fd.json"
    g = js(p)
    if g is not None:
        return g
    try:
        import torch
        x = torch.load(WM, map_location="cpu", weights_only=False, mmap=True)["x"]
        dep, nb = [], []
        for i in range(1, min(101, x.shape[0])):
            z = x[i][:, 2].double() - PN
            z = z[torch.isfinite(z)]
            dep.append(max(0.0, -float(z.min()))); nb.append(int((z < 0).sum()))
        g = dict(depth_max=max(dep), depth_f100=dep[-1], nbelow_max=max(nb), nbelow_f100=nb[-1])
        json.dump(g, open(p, "w"))
        return g
    except Exception as e:
        return dict(err=repr(e))


def row(n):
    s = js(f"{O}/{n}/summary.json")
    if s is None:
        return None
    F = [q for q in jl(f"{O}/{n}/fem.jsonl")]
    err = {q["t"] + 1: q["err"] for q in F}
    curve = " · ".join(("x" if (t not in err or not (err[t] == err[t])) else f"{100 * err[t]:.1f}") for t in (4, 10, 30, 50, 70, 100))
    dv = s.get("diverged")
    fd = [q["fd"] for q in F if "fd" in q]
    dep = max((q["depth"] or 0.0) for q in fd) if fd else None
    nbm = max(q["nbelow"] for q in fd) if fd else None
    bc = [q["bc"] for q in F if "bc" in q]
    bcs = (f"{sum(q['hit'] for q in bc) / len(bc):.0f} · {max(q['hit'] for q in bc)} / {max(q['nsamp'] for q in bc)}") if bc else "-"
    jd = [q["jd"] for q in F if "jd" in q]
    ple = max((q["p_le0"] for q in jd), default=None)
    ds = js(f"{O}/{n}/dmg_stats.json")
    lj = None
    if ds and ds.get("frames"):
        fr = ds["frames"]; k = ds.get("n_fin_frames") or len(fr)
        lj = fr[max(k, 1) - 1]["lj0"]
    return dict(dv=dv, done=s.get("frames_done"), em=100 * s["err_mean_done"], curve=curve, dep=dep, nbm=nbm, bcs=bcs, ple=ple, lj=lj)


def zrow(n):
    s = js(f"{O}/{n}/summary.json")
    if s is None:
        return "-"
    ms = [q["ms"] for q in jl(f"{O}/{n}/fem.jsonl")][2:]
    ms = [q for q in ms if q == q]
    pr = []
    try:
        pr = [int(l.split("카드")[1].split()[0]) for l in open(f"{O}/{n}/gpu_procs.txt")]
    except Exception:
        pass
    dv = s.get("diverged")
    return (f"{s['ms_mean']:.1f} · {max(ms):.1f}" if ms else "-") + (f" (x{dv})" if dv else "") + (f", 카드 프로세스 최대 {max(pr)}" if pr else "")


print("# wmbar -- 절 CM21 수박 장벽 바닥 + logJp 보간 (+ F 보간)\n")
print("| 판 | 실행 | 발산 | 평균 오차 % | 오차 곡선 f4·10·30·50·70·100 % | logJp>0 % (정답 6.51) | 바닥 아래 최대 깊이 mm (아래 입자 최대) | 장벽 걸린 표본점 평균 · 최대 / 표본점 | det F ≤ 0 입자 최대 | 단독 시간 평균 · 최대 ms |")
print("|---|---|---|---|---|---|---|---|---|---|")
for v in ("s0", "b1", "b2", "b3"):
    for r in "ab":
        x = row(f"{v}_{r}")
        if x is None:
            print(f"| {NAME[v]} | {r} | 도는 중 / 없음 | | | | | | | {zrow('z_' + v) if r == 'a' else ''} |")
            continue
        ljs = "-" if x["lj"] is None else f"{100 * x['lj']:.2f}"
        dps = "-" if x["dep"] is None else f"{1e3 * x['dep']:.2f}"
        dvs = f"x{x['dv']}" if x["dv"] else "완주"
        zs = zrow("z_" + v) if r == "a" else ""
        print(f"| {NAME[v]} | {r} | {dvs} | {x['em']:.2f} | {x['curve']} | {ljs} | {dps} ({x['nbm']}) | {x['bcs']} | {x['ple']} | {zs} |")
g = gt_fd()
if "err" in g:
    print(f"\n정답 바닥 아래 깊이: 계산 실패 {g['err']}")
else:
    print(f"\n정답 (wm.pt, 프레임 1~100): 바닥 아래 최대 깊이 {1e3 * g['depth_max']:.2f} mm (f100 {1e3 * g['depth_f100']:.2f}), 아래 입자 최대 {g['nbelow_max']} (f100 {g['nbelow_f100']}); 손상 logJp>0 f100 6.51 % (절 CM13.4 다시 돌린 GF).")
print("\n단위: 깊이는 시뮬 공간 (m 로 표기, 바닥 z 0.02); 단독 시간 = 재생 GPU 시간 첫 2 프레임 뺀 평균·최대 (발산 판은 발산 전).")
