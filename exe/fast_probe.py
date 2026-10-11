"""롤아웃 최적화용 탐색: 체크포인트 설정·크기(N, M, E, 사면체 수) 출력."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import deform_model as DM                        # noqa: E402

M = DM.EvalModel(sys.argv[1])
print({k: v for k, v in vars(M.a).items() if len(str(v)) < 200})
print("sub", M.sub, "h", M.h, "ext0", M.ext0, "n_mat", M.n_mat, M.n_film)
print(M.net)
d = torch.load(sys.argv[2], map_location="cuda", weights_only=False)
print("cfg", d["cfg"])
x, v = M.init(d, 0)
print("N", x.shape, x.dtype)
with torch.no_grad():
    for i in range(3):
        e = DM.node_feats(M.a, d["cfg"], d["_X0"], d["_mass"], x, v, M.vel_scale)
        print("M", e[0].shape, "E", e[6][0].shape, "nn", e[2][2].tolist(), "h", e[2][1].s,
              "T", torch.unique(e[3], dim=0).shape, "rows", e[3].shape)
        x, v, _ = M.frame(d, x, v)
print("PROBE_DONE")
