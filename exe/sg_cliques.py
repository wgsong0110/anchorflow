"""Spring-Gaus 질량점 그래프(공식: 앵커 N_SAMPLE=2048, 스프링 K_NEIGHBORS=256 최근접)의
4-클릭(서로 모두 연결된 질량점 4 개 = 사면체 후보) 수를 센다.

스프링은 i -> knn(i) 방향으로 만들어지므로 무방향 그래프는 합집합(어느 한쪽이라도 이웃)과
교집합(서로 이웃) 둘 다 센다. K4 = Σ_i tri(N(i) 유도 부분그래프) / 4.

  python exe/sg_cliques.py
"""
import random

import numpy as np
import torch
from plyfile import PlyData

SG = "/home/dkta/work/Spring-Gaus"
dev = "cuda"


def anchors_wolf(seed=0):
    v = PlyData.read(f"{SG}/checkpoints/wolf/static_gaussians/point_cloud.ply")["vertex"]
    x = torch.as_tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
    u = torch.unique(torch.floor(x / 0.01).int() + 0.5, dim=0) * 0.01   # uniform_sampling(0.01)
    random.seed(seed)
    return u[random.sample(range(u.shape[0]), 2048)]


def count(X, K=256):
    n = X.shape[0]
    idx = torch.cdist(X, X).topk(K + 1, largest=False).indices[:, 1:]
    A = torch.zeros(n, n, device=dev)
    A[torch.arange(n, device=dev)[:, None], idx] = 1.0
    out = {}
    for name, M in (("합집합", ((A + A.T) > 0).float()), ("서로 이웃", A * A.T)):
        e = int(M.sum() / 2)
        tri = float(torch.trace(M @ M @ M) / 6)
        k4 = 0.0
        for i in range(n):
            nb = torch.nonzero(M[i]).squeeze(1)
            B = M[nb][:, nb]
            k4 += float(torch.trace(B @ B @ B) / 6)
        out[name] = (e, int(round(tri)), int(round(k4 / 4)), float(M.sum(1).mean()))
    return out


sets = {"wolf(재현)": anchors_wolf()}
for obj, ex in (("lego", "af2_lego_2026_1006_0508_14"), ("mic", "af2_mic_2026_1006_0445_12"),
                ("bread", "af2_bread_2026_1006_0519_15")):
    sets[obj] = torch.load(f"{SG}/exp/{ex}/checkpoints_dynamic/checkpoint/anchors.pt",
                           map_location=dev).float()
for k, X in sets.items():
    for g, (e, tri, k4, deg) in count(X).items():
        print(f"{k:10s} 질량점 {X.shape[0]}  {g}: 평균 차수 {deg:.0f}  변 {e}  삼각형 {tri}  4-클릭 {k4}",
              flush=True)
print("SGC_DONE")
