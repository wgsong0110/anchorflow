import sys, torch, h5py, numpy as np
# 절 CM185 (vast 인스턴스 안에서 실행, 로컬 사본)
pred, gt = sys.argv[1], sys.argv[2]
d = torch.load(pred, map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
L = P.shape[0]
for t in [5, L // 2, L - 1]:
    f = h5py.File(f"{gt}/sim_{t+1:010d}.h5"); G = np.array(f["f_tensor"]); x = np.array(f["x"])
    G = G.reshape(-1, 3, 3) if G.shape[-1] == 9 or G.ndim == 3 else G.T.reshape(-1, 3, 3)
    if G.shape[0] != P.shape[1]: G = np.array(f["f_tensor"]).T.reshape(-1, 3, 3)
    jg = np.linalg.det(G); jo = torch.linalg.det(F[t].double()).numpy()
    sg = np.linalg.svd(G, compute_uv=False); so = torch.linalg.svdvals(F[t].double()).numpy()
    q = lambda a: np.percentile(a, [1, 10, 50, 90, 99]).round(3).tolist()
    print(f"frame {t+1}: detF 정답 {q(jg)} 우리 {q(jo)} | 최소 특이값 정답 {q(sg[:,2])} 우리 {q(so[:,2])} | 최대 정답 {q(sg[:,0])} 우리 {q(so[:,0])}")
