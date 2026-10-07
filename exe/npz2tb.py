"""TB 를 붙이기 전에 띄운 추적 결과(npz)를 TensorBoard 로 옮긴다 (한 번만, 끝난 결과에).

rep_track.py / rep_track2.py 가 저장한 metrics [t, RMSE, CD, EMD, det 최소, 뒤집힘 비율] 과
curve [t, 반복, L2, 장벽, ...] 를 --tb auto 와 같은 이름(/home/dkta/work/tbrf/<폴더>_<파일>)에 쓴다.

  python exe/npz2tb.py repflow/old/wolf_ours.npz repflow/r7/wolf_gaussim.npz
"""
import glob
import os
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

for p in sys.argv[1:]:
    Z = np.load(p, allow_pickle=True)
    d = os.path.join("/home/dkta/work/tbrf", os.path.basename(os.path.dirname(os.path.abspath(p)))
                     + "_" + os.path.splitext(os.path.basename(p))[0])
    for f in glob.glob(os.path.join(d, "events.out.tfevents.*")):   # txt2tb 의 중간본을 덮어쓴다
        os.remove(f)
    w = SummaryWriter(d)
    for t, rmse, cd, emd, dmin, inv in Z["metrics"]:
        t = int(t)
        w.add_scalar("frame/RMSE_pct", 100 * rmse, t)
        w.add_scalar("frame/CD_pct", 100 * cd, t)
        if emd == emd:
            w.add_scalar("frame/EMD_pct", 100 * emd, t)
        w.add_scalar("frame/det_min", dmin, t)
        w.add_scalar("frame/inverted_pct", 100 * inv, t)
    if "dof" in Z:
        for t, v in enumerate(Z["dof"], 1):
            w.add_scalar("frame/dof", v, t)
    for i, c in enumerate(Z["curve"]):
        w.add_scalar("iter/l2", c[2], i)
        w.add_scalar("iter/barrier", c[3], i)
    w.close()
    print(f"[TB] {p} -> {d}", flush=True)
