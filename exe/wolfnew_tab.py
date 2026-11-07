"""절 CM22 표: python exe/wolfnew_tab.py <묶음 출력 디렉토리> <정답 wolf.pt>  -> 표 (stdout), 곡선 원자료 <출력>/wolfnew_curves.json
판: (A) 가우스 점 고착 = 절 CM12 barstick a_wolf 재사용 (R2, 덤프에서 입자 det·폭), (B) 장벽 = 이 묶음 b_off (끝 해시를 절 CM8 cm8j b5_wolf 와 대조), (B)+보간 = b_on.
높이 = 바닥 법선 (z) 방향, 질량 중심 z 는 질량 가중 (실행기 --hgt), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (절 CM11 표와 같은 정의).
뒤집힌 가우스 점 = 프레임 끝 X_q = F_q + Σ U ⊗ G_qn 의 J ≤ 0 (--qjd; 끔 판은 마지막 걸음 RM 전 시험값, 켬 판은 마지막 걸음 커밋 RM 뒤 -- 모래 RM 은 부호 보존),
뒤집힌 입자 = 프레임 끝 입자 det F ≤ 0 (--pdiag), 장벽 걸린 표본점 = --bar_cnt neng."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

O, GTP = sys.argv[1:3]
R2 = "r2:storage/result/anchorflow/lbours/fem_cm"
REF_B = "fa318a68b4"                                                                  # 절 CM8 cm8j b5_wolf (잡 130567, aa5a7ef) 끝 해시


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fetch(src, dst, pred=False):
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(os.path.join(dst, "summary.json")):
        subprocess.run(["rclone", "copy", src, dst, "-q", "--include", "summary.json", "--include", "fem.jsonl", "--include", "run_info.txt"], check=False)
    if pred and not os.path.exists(os.path.join(dst, "pred.pt")):
        subprocess.run(["rclone", "copy", src, dst, "-q", "--include", "pred.pt"], check=False)


def spread(x):
    x = x[np.isfinite(x).all(1)]
    if not len(x):
        return float("nan")
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


fetch(f"{R2}/barstick/a_wolf", os.path.join(O, "ref_A"), pred=True)
fetch(f"{R2}/cm8j/b5_wolf", os.path.join(O, "ref_B"))
RUNS = [("(A) 가우스 점 고착 [절 CM12 a_wolf 재사용]", os.path.join(O, "ref_A")), ("(B) 장벽 [b_off]", os.path.join(O, "b_off")), ("(B)+보간 교체 [b_on]", os.path.join(O, "b_on"))]
G = torch.load(GTP, map_location="cpu", weights_only=False)["x"]
cur = {"gt": {"com": [], "zmax": [], "w": []}}
info = {}
for nm, d in RUNS:
    J, S = jl(d), sm(d)
    P = Fd = None
    pp = os.path.join(d, "pred.pt")
    if os.path.exists(pp):
        _d = torch.load(pp, map_location="cpu", weights_only=False)
        P, Fd = _d.get("pred"), _d.get("F")
    c = {"com": [], "zmax": [], "w": [], "qneg": [], "pneg": [], "pdet": [], "beng": [], "err": []}
    for i, r in enumerate(J):
        t = r["t"]
        hp = r.get("hp")
        c["err"].append(r.get("err"))
        c["com"].append(hp[0] if hp else None); c["zmax"].append(hp[1] if hp else None)
        c["w"].append(spread(P[t].numpy()) if P is not None and t < P.shape[0] else None)
        if "qj" in r:
            c["qneg"].append(r["qj"].get("nneg"))
        if "pd" in r:
            c["pneg"].append(r["pd"].get("nJneg")); c["pdet"].append(r["pd"].get("Jmin"))
        elif Fd is not None and t < Fd.shape[0]:
            _J = torch.linalg.det(Fd[t].double())
            c["pneg"].append(int((_J <= 0).sum())); c["pdet"].append(float(torch.nan_to_num(_J, nan=-1e30).min()))
        if "bc" in r:
            c["beng"].append(r["bc"].get("neng"))
    cur[nm] = c
    info[nm] = (S, J)
for i in range(1, G.shape[0]):
    z = G[i].numpy()
    cur["gt"]["w"].append(spread(z))
J0 = next((J for S, J in info.values() if J and "hg" in J[0]), [])
cur["gt"]["com"] = [r["hg"][0] for r in J0]; cur["gt"]["zmax"] = [r["hg"][1] for r in J0]
json.dump(cur, open(os.path.join(O, "wolfnew_curves.json"), "w"), ensure_ascii=False)


def fmt_cnt(a):
    a = [q for q in a if q is not None]
    if not a:
        return "-"
    nz = [(i + 1, q) for i, q in enumerate(a) if q > 0]
    if not nz:
        return "0"
    mx = max(nz, key=lambda q: q[1])
    return f"{len(nz)} 프레임 · 최대 {mx[1]} (f{mx[0]}) · 처음 f{nz[0][0]}"


print("# 절 CM22 wolf 새 기본 판 -- (B) 장벽 바닥 + --fp_interp (F·재질 상태 가우스 점 보간 교체) -- 묶음 wolfnew (exe/stab_wolfnew.sh)")
print()
print("모두 결정적 --det, 50 프레임. 오차 = 끝난 프레임 평균 %. (A) 는 절 CM12 barstick a_wolf (vast, 덤프에서 입자 det·폭; 가우스 점 J·장벽은 기록 없음).")
print()
print("## 1. 판정")
print()
print("| 판 | 결과 · % | 끝 해시 | 뒤집힌 가우스 점 (J ≤ 0) | 뒤집힌 입자 (det F ≤ 0) | 입자 det F 최소 | 장벽 걸린 표본점 프레임 합 · 최대 | N·P /프레임 |")
print("|---|---|---|---|---|---|---|---|")
for nm, d in RUNS:
    S, J = info[nm]
    c = cur[nm]
    if S is None:
        print(f"| {nm} | {'도는 중 (' + str(len(J)) + ' 프레임)' if J else '없음'} | | | | | | |")
        continue
    e = S.get("err_mean") if S.get("err_mean") is not None else S.get("err_mean_done")
    res = (f"발산 {S['diverged']} · " if S.get("diverged") else "완주 · ") + f"{100 * e:.2f}"
    pd = [q for q in c["pdet"] if q is not None]
    be = [q for q in c["beng"] if q is not None]
    it = f"{S['it_nN']['mean']:.1f}·{S['it_nP']['mean']:.1f}" if "it_nN" in S else "-"
    print(f"| {nm} | **{res}** | {S.get('x_sha1')} | {fmt_cnt(c['qneg']) if c['qneg'] else '-'} | {fmt_cnt(c['pneg'])} | {('%.3g' % min(pd)) if pd else '-'} | {(str(sum(be)) + ' · ' + str(max(be))) if be else '-'} | {it} |")
Sb, Sr = info["(B) 장벽 [b_off]"][0], sm(os.path.join(O, "ref_B"))
if Sb:
    print()
    print(f"- 끄면 같음: b_off 끝 해시 {Sb.get('x_sha1')} vs 절 CM8 cm8j b5_wolf {(Sr or {}).get('x_sha1', REF_B)} -> **{'같음' if Sb.get('x_sha1') == (Sr or {}).get('x_sha1', REF_B) else '다름'}** (오차 {100 * Sb['err_mean_done']:.3f} vs {100 * (Sr or {}).get('err_mean_done', float('nan')):.3f} %)")
print()
print("## 2. 높이·폭 곡선 (mm; 정답 / (A) / (B) / (B)+보간)")
print()
print("| 프레임 | 질량중심 z 정답 | (A) | (B) | (B)+보간 | 최대 z 정답 | (A) | (B) | (B)+보간 | 퍼짐 폭 정답 | (A) | (B) | (B)+보간 | 오차 % (A) | (B) | (B)+보간 | 장벽 걸린 점 (B) | (B)+보간 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
f = lambda v, k=1000: ("%.1f" % (k * v)) if v is not None and np.isfinite(v) else "-"
g = lambda a, i: a[i] if i < len(a) else None
names = [nm for nm, _ in RUNS]
for i in range(len(cur["gt"]["w"])):
    if (i + 1) % 5:
        continue
    row = [str(i + 1)]
    for key in ("com", "zmax", "w"):
        row.append(f(g(cur["gt"][key], i)))
        row += [f(g(cur[nm][key], i)) for nm in names]
    row += [("%.2f" % (100 * g(cur[nm]["err"], i))) if g(cur[nm]["err"], i) is not None else "-" for nm in names]
    row += [str(g(cur[nm]["beng"], i)) if g(cur[nm]["beng"], i) is not None else "-" for nm in names[1:]]
    print("| " + " | ".join(row) + " |")
print()
print("## 3. 프레임별 뒤집힌 가우스 점 / 입자 수 (0 아닌 프레임만)")
print()
for nm in names:
    c = cur[nm]
    nz = [f"f{i + 1} {g(c['qneg'], i) if c['qneg'] else '-'}/{g(c['pneg'], i)}" for i in range(max(len(c["qneg"]), len(c["pneg"]))) if (g(c["qneg"], i) or 0) > 0 or (g(c["pneg"], i) or 0) > 0]
    print(f"- {nm}: {' '.join(nz) if nz else '없음'}")
print()
print("## 4. 단독 시간 (비결정 = 판정 명령에서 --det·--hgt·진단·덤프 뺌; 카드 하나에 차례로 끔·켬 번갈아 두 번; 프레임당 재생 GPU 시간, 첫 2 프레임 뺌)")
print()
print("| 판 | ms 평균 | ms 최대 | 결과 · % | 실행 중 그 카드 계산 프로세스 수 최대 (자기 포함) | 노드 전체 최대 |")
print("|---|---|---|---|---|---|")
for n in ("z_off_a", "z_on_a", "z_off_b", "z_on_b"):
    S = sm(os.path.join(O, n))
    if S is None:
        print(f"| {n} | 없음 | | | | |")
        continue
    gp = os.path.join(O, n, "gpu_procs.txt")
    cm = nm_ = "-"
    if os.path.exists(gp):
        L = [l.split() for l in open(gp) if l.strip()]
        try:
            cm = max(int(q[2]) for q in L); nm_ = max(int(q[4]) for q in L)
        except (ValueError, IndexError):
            pass
    e = S.get("err_mean") if S.get("err_mean") is not None else S.get("err_mean_done")
    print(f"| {n} | {S['ms_mean']:.2f} | {S['ms_max']:.2f} | {'발산 ' + str(S['diverged']) + ' · ' if S.get('diverged') else '완주 · '}{100 * e:.2f} | {cm} | {nm_} |")
