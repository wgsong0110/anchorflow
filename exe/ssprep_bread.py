"""[substep_fix] bread 낙하 장면 입력: PG 공식 bread (tear_bread) 채우기 0 프레임 입자 (R2 _vastmirror/topo/pg_bread_plastic sim_0) 를 z 로 평행이동해
바닥 (0.48) 위 0.02 에 맨 아래 입자가 오게 하고 (다른 PG 장면 vasedeck·plane 의 z 최소 0.5 와 같음), 물리 = ficus 낙하 config (E 1e6 jelly) 그대로.
사용: python exe/ssprep_bread.py <ficus 낙하 config.json> <sim_0.h5> <출력 stab 루트>"""
import json, os, shutil, sys
import h5py, numpy as np
cfg, src, out = sys.argv[1:4]
os.makedirs(f"{out}/bread/init", exist_ok=True)
c = json.load(open(cfg)); c["drop_note"] = "PG 공식 bread 0 프레임 입자 z 평행이동 (min z 0.5) + 낙하 물리 (ficus 기존 stab config)"
json.dump(c, open(f"{out}/bread/config.json", "w"), indent=1)
dst = f"{out}/bread/init/sim_0000000000.h5"
shutil.copyfile(src, dst)
with h5py.File(dst, "r+") as h:
    x = np.array(h["x"]); T = x.shape[0] == 3 and x.shape[1] != 3
    X = x.T if T else x
    X = X.copy(); X[:, 2] += 0.5 - X[:, 2].min()
    h["x"][...] = (X.T if T else X).astype(x.dtype)
    if "v" in h:
        h["v"][...] = 0
print("[ssprep_bread] 입자", X.shape, "z", X[:, 2].min(), X[:, 2].max())
