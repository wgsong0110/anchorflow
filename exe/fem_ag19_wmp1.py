"""절 AG.19 수박 판별 진단 (진단 전용): 정답 프레임 t -> t+1 입자 이동을 프레임 t 의 Kuhn 사면체 (칸 크기 node_h, 칸마다 6 사면체) 에 배정하고,
입자 4 개 이상인 사면체마다 아핀 사상 x1 = A x0 + b 를 최소제곱으로 맞춰 det A ≤ 0 (방향 보존 P1 로 못 따라감) 인 사면체를 센다 -- 바닥 층 (아래 3 칸) 과 전체.
사용: python -I exe/fem_ag19_wmp1.py <정답 .pt> <출력 json> <node_h> <프레임,...>"""
import json
import sys

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
G = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
h = float(sys.argv[3])
frames = [int(q) for q in sys.argv[4].split(",")]
X = G["x"]
out = []
for f in frames:
    x0 = X[f].float().to(dev); x1 = X[f + 1].float().to(dev)
    o = torch.floor(x0.min(0).values / h) * h
    q = (x0 - o) / h
    bi = torch.floor(q).long(); u = q - bi
    perm = torch.argsort(-u, dim=1)                                             # Kuhn: 국소 좌표 내림차순 = 사면체 6 개
    pid = perm[:, 0] * 3 + perm[:, 1]                                           # 순열 6 가지를 0..8 범위 키로
    nb = bi.max(0).values + 1
    key = ((bi[:, 0] * nb[1] + bi[:, 1]) * nb[2] + bi[:, 2]) * 9 + pid
    uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    K = uk.numel()
    # 사면체별 최소제곱 아핀: [x0 1] -> x1 (정규 방정식, 4x4)
    Xa = torch.cat([x0 - x0.mean(0), torch.ones_like(x0[:, :1])], 1).double()
    AtA = torch.zeros(K, 4, 4, device=dev, dtype=torch.float64).index_add_(0, inv, Xa[:, :, None] * Xa[:, None, :])
    AtB = torch.zeros(K, 4, 3, device=dev, dtype=torch.float64).index_add_(0, inv, Xa[:, :, None] * (x1 - x0.mean(0)).double()[:, None, :])
    ok = cnt >= 4
    sol = torch.linalg.lstsq(AtA[ok] + 1e-14 * torch.eye(4, device=dev, dtype=torch.float64), AtB[ok]).solution
    A = sol[:, :3, :].transpose(1, 2)                                           # x1 = A x0 + b
    dA = torch.det(A)
    zb = bi[:, 2]
    tz = torch.zeros(K, device=dev, dtype=torch.long).scatter_reduce(0, inv, zb, "amin", include_self=False)[ok]
    bottom = tz <= int(zb.min()) + 2
    r = dict(frame=f, n_part=int(x0.shape[0]), n_tets=K, n_tets_ge4=int(ok.sum()), det_le0=int((dA <= 0).sum()), det_lt01=int((dA < 0.1).sum()),
             bottom_ge4=int(bottom.sum()), bottom_det_le0=int((dA[bottom] <= 0).sum()), bottom_det_lt01=int((dA[bottom] < 0.1).sum()),
             det_q=[float(torch.quantile(dA.float(), qq)) for qq in (0.001, 0.01, 0.5)], bottom_det_q=[float(torch.quantile(dA[bottom].float(), qq)) for qq in (0.001, 0.01, 0.5)] if bool(bottom.any()) else None,
             max_disp=float((x1 - x0).norm(dim=1).max()), z_min=[float(x0[:, 2].min()), float(x1[:, 2].min())])
    out.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
