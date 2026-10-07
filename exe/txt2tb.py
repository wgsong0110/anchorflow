"""돌고 있는 추적의 텍스트 로그([t=  N] 줄, 10 프레임마다)를 지금까지 분량만 TensorBoard 로 쓴다.

같은 이름(/home/dkta/work/tbrf/<폴더>_<파일>)을 쓰고, 끝나면 npz2tb.py 가 전체로 덮어쓴다.

  python exe/txt2tb.py repflow/old/wolf_ours.txt repflow/r7/wolf_gaussim.txt
"""
import glob
import os
import re
import sys

from torch.utils.tensorboard import SummaryWriter

PAT = re.compile(r"\[t=\s*(\d+)\]\s+RMSE ([\d.]+)%\s+CD ([\d.]+)%(?:\s+EMD ([\d.]+)%)?\s+"
                 r"det 최소 (-?[\d.]+) \(≤0 ([\d.]+)%\)\s+자유도 (\d+)"
                 r"(?:.*?이웃 det 최소 (-?[\d.]+)\s+되돌림 (\d+))?")
KEYS = ["RMSE_pct", "CD_pct", "EMD_pct", "det_min", "inverted_pct", "dof", "knn_det_min",
        "backtracks"]
for p in sys.argv[1:]:
    d = os.path.join("/home/dkta/work/tbrf", os.path.basename(os.path.dirname(os.path.abspath(p)))
                     + "_" + os.path.splitext(os.path.basename(p))[0])
    for f in glob.glob(os.path.join(d, "events.out.tfevents.*")):
        os.remove(f)
    w = SummaryWriter(d)
    n = 0
    txt = open(p, errors="ignore").read()
    knn = "[결합]" in txt                     # rep_track.py(예전 설정판): det 는 이웃 8 개 최소제곱
    for line in txt.replace("\r", "\n").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        t = int(m.group(1))
        for k, g in zip(KEYS, m.groups()[1:]):
            if g is not None:
                if knn and k in ("det_min", "inverted_pct"):
                    k = "knn_" + k
                w.add_scalar("frame/" + k, float(g), t)
        n += 1
    w.close()
    print(f"[TB] {p} -> {d}  {n} 점", flush=True)
