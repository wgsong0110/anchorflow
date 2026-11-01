"""절 AG.19 (사용자 지시): 정답 입자 운동에서 같은 Kuhn 사면체 안 입자 순서가 실제로 뒤집히는지 -- 최소제곱 없이 직접.
프레임 t 입자를 우리 격자 (simplex.lattice(node_h, node_h/√6), 원점 0, 격자 좌표 정수 칸 + 소수부 순열 = Kuhn 사면체) 에 배정하고,
사면체마다 안의 입자 네 개 묶음 (nC4 전부, 많으면 무작위 최대 500, 시드 0) 의 부호 있는 부피 V_t, V_{t+1} 의 부호가 바뀌는지 센다.
거의 한 평면인 묶음은 |V_t| ≥ 10 x (float32 좌표 반올림으로 생길 수 있는 부피 상한) 인 것만: 상한 = (1/6)·3·(2δ)·L², δ = max|x|·2^-24 (좌표 하나의 반올림 오차),
L = 사면체 꼭짓점 사이 최대 거리 (칸 대각선 √3 a, a = 칸 변). CPU 만 (GPU 안 씀).
사용: python -I exe/fem_ag19_flip.py <정답 .pt> <node_h> <프레임,... 또는 all> <바닥 평면 z (없으면 none)> <출력 json>"""
import itertools
import json
import math
import sys

import torch

torch.set_num_threads(8)
G = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
nh = float(sys.argv[2])
X = G["x"]
T = X.shape[0] - 1
frames = list(range(T)) if sys.argv[3] == "all" else [int(q) for q in sys.argv[3].split(",")]
zfl = None if sys.argv[4] == "none" else float(sys.argv[4])
r = nh / math.sqrt(3.0); c, s3 = 0.5, math.sqrt(3.0) / 2.0; hz = nh / math.sqrt(6.0)
u = ((1.0, 0.0), (-c, s3), (-c, -s3))
A = torch.tensor([[r * u[i][0] for i in range(3)], [r * u[i][1] for i in range(3)], [hz] * 3], dtype=torch.float64)
Ai = torch.linalg.inv(A)
a_edge = float(A[:, 0].norm())                                                  # 칸 변 (정육면체, 변 = node_h/√2)
Vtet = a_edge ** 3 / 6.0
g = torch.Generator().manual_seed(0)
combos = {n: torch.tensor(list(itertools.combinations(range(n), 4)), dtype=torch.long) for n in range(4, 12)}
out = []
for f in frames:
    x0 = X[f].double(); x1 = X[f + 1].double()
    dl = float(x0.abs().max()) * 2.0 ** -24
    thr = 10.0 * (1.0 / 6.0) * 3.0 * (2.0 * dl) * (3.0 * a_edge ** 2)
    y = x0 @ Ai.T
    b = torch.floor(y).long(); fr = y - b
    perm = torch.argsort(-fr, dim=1)
    pid = perm[:, 0] * 3 + perm[:, 1]
    b = b - b.min(0).values
    nb = b.max(0).values + 1
    key = ((b[:, 0] * nb[1] + b[:, 1]) * nb[2] + b[:, 2]) * 9 + pid
    ks, order = torch.sort(key)
    uk, cnt = torch.unique_consecutive(ks, return_counts=True)
    starts = torch.cumsum(cnt, 0) - cnt
    sel = (cnt >= 4).nonzero().flatten()
    zmin = float(x0[:, 2].min())
    n_t = n_flip_any = n_flip_half = 0; nb_t = nb_any = nb_half = 0; n_q = n_qf = 0
    for s_, c_ in zip(starts[sel].tolist(), cnt[sel].tolist()):
        idx = order[s_:s_ + c_]
        if c_ in combos:
            cb = combos[c_]
        else:
            cb = torch.stack([torch.randperm(c_, generator=g)[:4] for _ in range(500)])
        if cb.shape[0] > 500:
            cb = cb[torch.randperm(cb.shape[0], generator=g)[:500]]
        P0 = x0[idx][cb]; P1 = x1[idx][cb]
        V0 = torch.det(P0[:, 1:] - P0[:, :1]) / 6.0
        V1 = torch.det(P1[:, 1:] - P1[:, :1]) / 6.0
        ok = V0.abs() >= thr
        if not bool(ok.any()):
            continue
        fl = (torch.sign(V0[ok]) != torch.sign(V1[ok]))
        n_t += 1; n_q += int(ok.sum()); n_qf += int(fl.sum())
        anyf = bool(fl.any()); half = float(fl.float().mean()) >= 0.5
        n_flip_any += anyf; n_flip_half += half
        if zfl is not None and float(x0[idx, 2].min()) <= max(zmin, zfl) + 3.0 * nh:
            nb_t += 1; nb_any += anyf; nb_half += half
    rr = dict(frame=f, tets_ge4=int(sel.numel()), tets_judged=n_t, frac_any=n_flip_any / max(n_t, 1), frac_half=n_flip_half / max(n_t, 1),
              quads=n_q, frac_quads_flipped=n_qf / max(n_q, 1), thr_over_Vtet=thr / Vtet,
              bottom_tets=nb_t, bottom_frac_any=(nb_any / nb_t) if nb_t else None, bottom_frac_half=(nb_half / nb_t) if nb_t else None, zmin=zmin)
    out.append(rr)
    print(json.dumps(rr, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[5], "w"), indent=1)
