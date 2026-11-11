import glob, sys, numpy as np, h5py
fs = sorted(glob.glob(sys.argv[1] + "/sim_*.h5"))
for i in (0, 1, 10, 25, 50):
    with h5py.File(fs[i], "r") as h:
        for k in h.keys():
            a = np.array(h[k]); print(i, k, a.shape, a.dtype, "zero-frac(all-zero rows)", float((np.abs(a.reshape(a.shape[0], -1) if a.shape[0] != 3 and a.shape[0] != 9 else a.T.reshape(a.shape[1], -1)).max(1) == 0).mean()), flush=True)
        F = np.array(h["F"]); F = F.T.reshape(-1, 3, 3) if F.shape[0] == 9 else F.reshape(-1, 3, 3)
        z = (np.abs(F).reshape(len(F), -1).max(1) == 0)
        print(i, "F 전부 0 인 입자 몫", z.mean(), "비영 입자 det 분위", np.percentile(np.linalg.det(F[~z]), [0, 10, 50, 90, 100]) if (~z).any() else None, "예시", F[~z][:2].tolist() if (~z).any() else None, flush=True)
