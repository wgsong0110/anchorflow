"""말림 감시 정밀도 시험: ficus 와 같은 h 격자에서 float32/float64, 항등·작은 무작위(계수>0)·큰 무작위."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import simplex as SX, wf as WF
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
x = (0.7 + 0.4 * torch.rand(30000, 3)).to(dev)
lo, lat, nn = SX.grid_for_nodes(x, 32, hz_ratio=SX.HZ_CUBE, h_fix=0.056569)
idx, lam, aux = SX.locate(x, lo, lat, nn); rows, uq = SX.active_nodes(idx)
ws = WF.particle_struct(x, lo, lat, nn, SX.Aux(aux), rows); wt = WF.tet_struct(rows, ws["ty"], lat)
M = uq.numel()
for amp in (0.0, 1e-4, 1e-3, 3e-3, 3e-2):
    d = amp * torch.randn(M, WF.C1_NOUT, device=dev)
    c = WF.det_coefs(d.double(), {k: v for k, v in wt.items()}, type(lat)(lat.A.double(), lat.Ai.double(), lat.s, lat.hz))
    print(f"amp {amp:g}: 계수 최소 {float(c.min()):.3f}, 말림(float32 입력) {WF.curl_check(d, wt, lo, lat, nn, uq)}, 사면체 {wt['rows'].shape[0]}", flush=True)
print("CURL_TEST_DONE")
