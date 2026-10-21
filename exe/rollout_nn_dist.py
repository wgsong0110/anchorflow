"""lego 다섯 궤적 모델 롤아웃 상태가 학습 상태(400 개)에서 얼마나 벗어났나.

학습 상태 = AF_STATE_FIT_MULTI 와 같은 정의: 궤적 j(L000·0000~0003)마다 프레임 0 = 정답 시작(x[0], --v0_traj 의 v[0]),
프레임 i>=1 = 출력만 최적화 롤아웃 덤프 pred[i-1]·v[i-1] (i=0..79) -> 5 x 80 = 400 상태.
롤아웃(기존 덤프만, 재실행 없음): 프레임 t(1..80) = pred[t-1]. 정답 = lbdata h5 프레임 t (입자 순서 검산).
거리 d = mean_p ‖X_t,p − P_S,p‖ / EXT, EXT = L000 프레임 0 바운딩박스 대각선 (학습 코드 traj_scope 의 _ext 와 같은 식, 고정 상수).
참고: 질량중심(traj_mass 와 같은 질량)을 맞춘 거리, 위치 최근접 짝의 속도 차 mean ‖v−v_S‖ / (EXT/h), h = 1/60.
기준선: 학습 상태마다 (가) 다른 궤적 상태 중 최소, (나) 자기 자신만 뺀 전부(같은 궤적 이웃 포함) 중 최소.

  python exe/rollout_nn_dist.py --out DIR
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow.deform_model import traj_mass, extent  # noqa: E402

W = "/home/dkta/work"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = a.dev
H = 1.0 / 60.0
TRAJ = ["L000", "0000", "0001", "0002", "0003"]
RUNS = [("c18", "18채널 (8500 반복)", f"{W}/lbours/ipfeatonly/mid/roll_ipfeatonly_mid_L000.pt", f"{W}/lbours/ipfeatonly/mid/iter.txt"),
        ("c51", "51채널 (7000 반복)", f"{W}/lbours/ipfeat/mid/roll_ipfeat_mid_L000.pt", f"{W}/lbours/ipfeat/mid/iter.txt"),
        ("old", "기존 다섯 궤적 (15000 반복)", f"{W}/lbours/lego_psi_spstick/roll_mm_L000.pt", None)]
LOG = open(f"{a.out}/log.txt", "w")


def say(*s):
    m = " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def ld(p):
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)


# ---- 학습 상태 400 개
data = {t: ld(f"{W}/multi5/{t}.pt") for t in TRAJ}
for t in TRAJ:
    say(f"[데이터] {t}: {os.path.realpath(f'{W}/multi5/{t}.pt')} x {tuple(data[t]['x'].shape)} v {'있음' if 'v' in data[t] else '없음'}")
dL = data["L000"]
x0L = dL["x"][0].float()
EXT = extent(x0L)
N = x0L.shape[0]
exts = {t: extent(data[t]["x"][0].float()) for t in TRAJ}
say(f"[EXT] L000 {EXT:.6f} (고정 상수로 전부 이것으로 나눈다); 궤적별 {exts}")
mass = traj_mass(dict(x=dL["x"], cfg=dL["cfg"]), dev="cpu").float()
say(f"[질량] 합 {float(mass.sum()):.6g}, 최소/최대 {float(mass.min()):.3g}/{float(mass.max()):.3g}")
w = (mass / mass.sum()).to(dev)

SP = torch.empty(len(TRAJ) * 80, N, 3, dtype=torch.float32, device=dev)
SV = torch.empty_like(SP)
lab = []
for j, t in enumerate(TRAJ):
    D = ld(f"{W}/lbours/spstick/roll_ov_{t}.pt")
    assert D["pred"].shape[0] >= 79 and D["pred"].shape[1] == N, (t, D["pred"].shape)
    d = data[t]
    SP[j * 80] = d["x"][0].float().to(dev)
    SV[j * 80] = (d["v"][0].float() if "v" in d else (d["x"][0] - d["x"][0]).float()).to(dev)
    SP[j * 80 + 1:(j + 1) * 80] = D["pred"][:79].float().to(dev)
    SV[j * 80 + 1:(j + 1) * 80] = D["v"][:79].float().to(dev)
    if "gt" in D:
        say(f"[검산] {t} 덤프 gt[0] vs 데이터 x[1] 최대차 {float((D['gt'][0].float() - d['x'][1].float()).abs().max()):.3g}, "
            f"덤프 EXT {D.get('EXT')}")
    lab += [(t, i) for i in range(80)]
    del D
say(f"[학습 상태] {SP.shape[0]} 개, N {N}")


def com(X):  # X [B,N,3]
    return (X * w[None, :, None]).sum(1, keepdim=True)


SPc = SP - com(SP)


def dist_all(x, Y, chunk=100):
    """x [N,3] 와 Y [S,N,3] 각각 mean_p ‖·‖ / EXT -> [S]."""
    out = []
    for s in range(0, Y.shape[0], chunk):
        out.append((Y[s:s + chunk] - x[None]).norm(dim=-1).mean(1))
    return torch.cat(out) / EXT


def vel(v, i):
    return float((v - SV[i]).norm(dim=-1).mean()) / (EXT / H)


def nn_frames(X, V):
    """X,V [T,N,3] (프레임 1..T) -> 프레임별 기록."""
    rec = []
    for t in tqdm(range(X.shape[0]), desc="최근접", leave=False):
        x = X[t].float().to(dev)
        d = dist_all(x, SP); k = int(d.argmin())
        xc = x - com(x[None])[0]
        dc = dist_all(xc, SPc); kc = int(dc.argmin())
        r = dict(frame=t + 1, d=float(d[k]), traj=lab[k][0], sframe=lab[k][1],
                 d_com=float(dc[kc]), traj_com=lab[kc][0], sframe_com=lab[kc][1],
                 d_same_frame_L000=float(d[t + 1]) if t + 1 < 80 else None)
        if V is not None:
            r["dv"] = vel(V[t].float().to(dev), k)
        rec.append(r)
    return rec


RES = {}
# ---- 롤아웃
for key, nm, p, itp in RUNS:
    D = ld(p)
    assert D["pred"].shape[1] == N, (p, D["pred"].shape)
    it = open(itp).read().strip() if itp and os.path.exists(itp) else None
    say(f"[롤아웃] {key} {nm}: {p} 프레임 {D['pred'].shape[0]} v {'있음' if D.get('v') is not None else '없음'} 반복 {it} EXT {D.get('EXT')}")
    if "x0" in D:
        say(f"  x0 vs 데이터 L000 x[0] 최대차 {float((D['x0'].float() - x0L).abs().max()):.3g}")
    RES[key] = nn_frames(D["pred"], D.get("v"))
    del D
ck = f"{W}/lbours/spstick/multi/ck.pt"
if os.path.exists(ck):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    say(f"[기존 모델] {ck} 반복 k = {c.get('k')}")
    del c
# ---- 정답
gt_dir = f"{W}/lbdata/test_low/L000"
fs = sorted(glob.glob(f"{gt_dir}/sim_*.h5"))
with h5py.File(fs[0]) as h:
    say(f"[정답] {gt_dir} 파일 {len(fs)}, 키 {list(h.keys())}")
    xg0 = torch.as_tensor(np.array(h["x"], dtype=np.float32))
    hasv = "v" in h
assert xg0.shape[0] == N, (xg0.shape, N)
say(f"[정답 순서 검산] h5 프레임 0 vs 데이터 L000 x[0] 최대차 {float((xg0 - x0L).abs().max()):.3g}")
TG = min(80, len(fs) - 1)
XG = torch.stack([torch.as_tensor(np.array(h5py.File(fs[t], "r")["x"], dtype=np.float32)) for t in range(1, TG + 1)])
VG = torch.stack([torch.as_tensor(np.array(h5py.File(fs[t], "r")["v"], dtype=np.float32)) for t in range(1, TG + 1)]) if hasv else None
say(f"[정답 순서 검산] h5 프레임 1 vs 데이터 L000 x[1] 최대차 {float((XG[0] - dL['x'][1].float()).abs().max()):.3g}, "
    f"h5 프레임 {TG} vs x[{TG}] {float((XG[-1] - dL['x'][TG].float()).abs().max()):.3g}")
RES["gt"] = nn_frames(XG, VG)
del XG, VG

# ---- 기준선: 학습 상태끼리
B_other, B_self = [], []
for i in tqdm(range(SP.shape[0]), desc="기준선"):
    d = dist_all(SP[i], SP)
    ti = lab[i][0]
    same = torch.tensor([lab[k][0] == ti for k in range(SP.shape[0])], device=dev)
    do = d.masked_fill(same, float("inf")); ko = int(do.argmin())
    ds = d.clone(); ds[i] = float("inf"); ks = int(ds.argmin())
    B_other.append(dict(traj=ti, frame=lab[i][1], d=float(do[ko]), nn=lab[ko]))
    B_self.append(dict(traj=ti, frame=lab[i][1], d=float(ds[ks]), nn=lab[ks]))
dc_self = []
for i in tqdm(range(SPc.shape[0]), desc="기준선 질량중심"):
    d = dist_all(SPc[i], SPc); d[i] = float("inf")
    ti = lab[i][0]
    same = torch.tensor([lab[k][0] == ti for k in range(SP.shape[0])], device=dev)
    dc_self.append((float(d.masked_fill(same, float("inf")).min()), float(d.min())))
for i in range(len(B_other)):
    B_other[i]["d_com"], B_self[i]["d_com"] = dc_self[i]
out = dict(EXT=EXT, h=H, N=N, exts=exts, runs={k: dict(name=nm, path=p) for k, nm, p, _ in RUNS},
           gt=gt_dir, res=RES, base_other=B_other, base_self=B_self, labels=lab)
json.dump(out, open(f"{a.out}/rollout_nn_dist.json", "w"), ensure_ascii=False, indent=1)
for nm_, B in (("다른 궤적만", B_other), ("자기 제외 전부", B_self)):
    v = np.array([b["d"] for b in B])
    say(f"[기준선 {nm_}] 최소 {v.min():.4%} 중앙 {np.median(v):.4%} 최대 {v.max():.4%} (프레임 0 제외 중앙 {np.median([b['d'] for b in B if b['frame'] > 0]):.4%})")
say("ROLLOUTNN_JSON_OK")
