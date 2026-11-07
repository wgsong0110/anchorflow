"""절 CM28 (a): 수박 씨 영역 β = 3e9 판정 -- GF 러너 gs_simulation_watermelon.py (set_parameters_dict 바로 뒤 블록) 를 numpy 로 그대로.
  1) 3DGS 가우시안 중 불투명도 sigmoid(opacity) > opacity_threshold (wm2 config 0.02) 만 남김 (러너의 mask -- gaussians._xyz·_features_dc 도 같이 거름)
  2) 씨 후보 = f_dc 세 채널 모두 (−2.5, −0.5) 안 (러너: (_features_dc < −0.5).all & (_features_dc > −2.5).all)
  3) 남긴 가우시안 전체에 반경 0.03 (원래 3DGS 좌표, 회전·이동 전) 이웃 -- 후보마다 거리 ≤ 0.03 인 점 모두 (sklearn radius_neighbors 와 같은 ≤)
  4) 마스크 (남긴 가우시안 순번) -> 러너는 beta[new_mask] = 3e9 로 입자 배열에 순번 그대로 씀 (가우시안 수 = 입자 수일 때만 IndexError 없음)
사용: python exe/wm_seedmask.py <point_cloud.ply> <out.npz> [opacity_threshold 0.02] [radius 0.03]
출력 npz: seed (bool, 남긴 가우시안 순번), xyz (남긴 가우시안 원래 좌표 float32), cand (후보 bool), n_all, n_kept
"""
import sys

import numpy as np

ply, out = sys.argv[1], sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
R = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
with open(ply, "rb") as fh:
    props, n = [], None
    while True:
        ln = fh.readline().decode("ascii").strip()
        if ln.startswith("element vertex"):
            n = int(ln.split()[-1])
        elif ln.startswith("property"):
            t, nm = ln.split()[1:3]
            assert t == "float", ln
            props.append(nm)
        elif ln == "end_header":
            break
    a = np.frombuffer(fh.read(n * 4 * len(props)), dtype="<f4").reshape(n, len(props))
col = {k: i for i, k in enumerate(props)}
op = 1.0 / (1.0 + np.exp(-a[:, col["opacity"]].astype(np.float64)))
keep = op > thr
xyz = a[keep][:, [col["x"], col["y"], col["z"]]].astype(np.float32)
dc = a[keep][:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]
cand = (dc < -0.5).all(1) & (dc > -2.5).all(1)
# 반경 이웃: 칸 크기 R 격자, 후보마다 27 칸의 점과 거리 비교 (float64 거리, ≤ R)
P = xyz.astype(np.float64)
lo = P.min(0)
ci = np.floor((P - lo) / R).astype(np.int64)
dim = ci.max(0) + 3
key = ((ci[:, 0] + 1) * dim[1] + (ci[:, 1] + 1)) * dim[2] + (ci[:, 2] + 1)
order = np.argsort(key, kind="stable")
ks = key[order]
seed = np.zeros(len(P), dtype=bool)
C = np.nonzero(cand)[0]
for ch in range(0, len(C), 20000):
    cc = C[ch:ch + 20000]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k2 = ((ci[cc, 0] + 1 + dx) * dim[1] + (ci[cc, 1] + 1 + dy)) * dim[2] + (ci[cc, 2] + 1 + dz)
                s0 = np.searchsorted(ks, k2, "left"); s1 = np.searchsorted(ks, k2, "right")
                cnt = s1 - s0
                if cnt.sum() == 0:
                    continue
                rep = np.repeat(np.arange(len(cc)), cnt)
                off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                j = order[np.repeat(s0, cnt) + off]
                d2 = ((P[j] - P[cc[rep]]) ** 2).sum(1)
                seed[j[d2 <= R * R]] = True
np.savez_compressed(out, seed=seed, xyz=xyz, cand=cand, n_all=n, n_kept=int(keep.sum()))
print(f"가우시안 {n}, 불투명도 > {thr} 남김 {int(keep.sum())}, 씨 후보 {int(cand.sum())}, 반경 {R} 이웃 합집합 (씨 영역) {int(seed.sum())} ({100 * seed.mean():.2f} %)")
