"""사용자 10-30 "발산이 일어난 셀을 영상에서 표시 -- 발산한 셀에 들어 있던 입자가 어떤 입자인지":
fem_ex_sim.py --lagdiag --divtrack 의 divtrack/fNNN.pt (계산 불변 진단) 에서 발산 프레임·처음 터진 셀들을 찾아, 그 프레임 P2G 때 그 셀에 들어 있던 입자 번호를 고른다.
덤프만 읽는다 (계산 경로 불변).

정의 (프레임 번호 t 는 0 부터 = 롤아웃 루프 i; 영상의 상태 번호 k 는 k 프레임 뒤 상태, 상태 t = 프레임 t 시작):
  발산 프레임 td = divtrack.jsonl 에서 처음으로 노드 |v| 가 비유한 또는 발산 판정 속도 (0.25 EXT / h, 롤아웃 루프 판정과 같은 값) 를 넘은 프레임.
  발산 서브스텝 s_div = 그 프레임에서 처음 넘은 서브스텝 (0 부터).
  처음 터진 노드 = s_div ~ s_div + W 서브스텝에 처음 넘은 노드 (W = --win, 기본 2).
  처음 뒤집힌 셀 = 처음 뒤집힘 서브스텝 s_flip ~ s_flip + W 에 det A_T ≤ 0 (또는 비유한) 이 된 점유 사면체 (s_flip ≤ s_div 일 때만 -- 발산 뒤 뒤집힘은 결과).
  발산 셀 = (처음 터진 노드를 꼭짓점으로 갖는 점유 사면체) ∪ (처음 뒤집힌 셀). 입자 = 그 프레임 pinv (P2G 때 입자 -> 사면체) 가 그 셀인 입자.
  폭주 시작 = 처음 터진 노드들의 '5 x q90 연속 구간' 시작 서브스텝 (절 U 규칙; 프레임 안에서만).
  사건 순서 = 점유 사면체마다 min(뒤집힘, 꼭짓점 노드의 발산 판정 넘음·비유한) 서브스텝 -> 오름차순 앞 --nlist 개.
위치 (프레임 td 시작 입자 x): 바닥 높이 s = (x − p)·n, 표면 깊이 = 정답 dx (node_h/√2) 복셀 점유에서 빈 칸까지 거리 (복셀, 6-이웃 침식 횟수),
  분류 바닥 접촉 (s < node_h) / 표면 (깊이 ≤ 1, 바닥 접촉 아님) / 내부.
F 특이값·이웃 퍼짐 (발산 전 --pre 프레임 + 발산 프레임 시작): 우리 = roll_fem.pt 의 F·위치 (상태 k), 정답 = --data 의 F·x (상태 k).
  이웃 퍼짐 = 0 상태 16-NN 까지 평균 거리 (상태 k) / (상태 0), 표시 입자마다 -> 중앙·최대.
출력: <run>/dc/{dc.json, mark_idx.npy (발산 셀 입자 번호, int64), events.txt}
  python exe/fem_dc_cells.py --run DIR --data X.pt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True, help="정답 .pt (fem_ex_sim --data 와 같은 파일: x·F [L+1, Np, ...])")
ap.add_argument("--win", type=int, default=2)
ap.add_argument("--pre", type=int, default=5)
ap.add_argument("--knn", type=int, default=16)
ap.add_argument("--nlist", type=int, default=30)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
OD = os.path.join(a.run, "dc")
os.makedirs(OD, exist_ok=True)
J = [json.loads(q) for q in open(os.path.join(a.run, "divtrack.jsonl"))]
td = next((r["t"] for r in J if r["n_nf"] or r["n_abs"]), None)
assert td is not None, "발산 (비유한·발산 판정 속도 넘음) 프레임이 없다"
fl_first = next((r["t"] for r in J if r["n_flip"]), None)
D = torch.load(os.path.join(a.run, "divtrack", f"f{td:03d}.pt"), map_location="cpu", weights_only=False)
N, W = int(D["N"]), a.win
tets, occ = D["tets"].long(), D["occ"]
flip, nf, ab, rs, rf = D["flip"].long(), D["nf"].long(), D["abs"].long(), D["run_start"].long(), D["run_first"].long()
BIG = 10 ** 9
nodev = torch.where(ab >= 0, ab, torch.full_like(ab, BIG))
nodev = torch.minimum(nodev, torch.where(nf >= 0, nf, torch.full_like(nf, BIG)))
s_div = int(nodev.min())
s_nf = int(nf[nf >= 0].min()) if bool((nf >= 0).any()) else None
first_nodes = torch.nonzero(nodev <= s_div + W).squeeze(1)
T = tets.shape[0]
tnode = nodev[tets].min(1).values                                                   # 사면체 꼭짓점 노드 사건
tflip = torch.where(flip >= 0, flip, torch.full_like(flip, BIG))
s_flip = int(tflip[occ].min()) if bool((tflip[occ] < BIG).any()) else None
sel_node = occ & torch.isin(tets, first_nodes).any(1)
sel_flip = (occ & (tflip <= s_flip + W)) if (s_flip is not None and s_flip <= s_div) else torch.zeros(T, dtype=torch.bool)
cells = torch.nonzero(sel_node | sel_flip).squeeze(1)
pinv = D["pinv"].long()
mark = torch.nonzero(torch.isin(pinv, cells)).squeeze(1)
np.save(os.path.join(OD, "mark_idx.npy"), mark.numpy().astype(np.int64))
# 사건 순서
tev = torch.minimum(tflip, tnode)
tev = torch.where(occ, tev, torch.full_like(tev, BIG))
order = torch.argsort(tev)
lines = []
cnt_p = torch.bincount(pinv, minlength=T)
for q in order[:a.nlist].tolist():
    if int(tev[q]) >= BIG:
        break
    kind = []
    if int(tflip[q]) == int(tev[q]):
        kind.append("뒤집힘")
    if int(tnode[q]) == int(tev[q]):
        kind.append("노드 발산 판정 넘음" + ("·비유한" if bool((nf[tets[q]] == int(tev[q])).any()) else ""))
    pp = torch.nonzero(pinv == q).squeeze(1).tolist()
    lines.append(f"서브스텝 {int(tev[q])}: 사면체 {q} ({'·'.join(kind)}), 노드 {tets[q].tolist()}, 입자 {len(pp)} 개 {pp[:8]}{' ...' if len(pp) > 8 else ''}")
open(os.path.join(OD, "events.txt"), "w").write("\n".join(lines) + "\n")
# 폭주 시작 (처음 터진 노드의 연속 구간 시작)
rs_f = rs[first_nodes]
run_on = int(rs_f[rs_f >= 0].min()) if bool((rs_f >= 0).any()) else None
# 위치
x = D["xp"].float()
mk = mark
pt, nr = (D["plane"] if D["plane"] is not None else (None, None))
nh = float(D["node_h"])
if pt is not None:
    s = ((x - torch.tensor(pt)) * torch.tensor(nr)).sum(-1)
else:
    s = x[:, 2] - x[:, 2].min()
vx = nh / 2 ** 0.5
lo = x.min(0).values - 2 * vx
ijk = ((x - lo) / vx).floor().long()
dims = (ijk.max(0).values + 3).tolist()
occg = torch.zeros(dims, dtype=torch.bool)
occg[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
depth = torch.full(dims, -1, dtype=torch.int32)
cur = occg.clone()
k = 0
while bool(cur.any()) and k < 200:                                                  # 6-이웃 침식: 깊이 k = k 번째 침식에서 사라진 칸
    nb = cur.clone()
    for ax in range(3):
        for sh in (1, -1):
            nb &= torch.roll(cur, sh, ax)
    gone = cur & ~nb
    depth[gone] = k
    cur = nb
    k += 1
dp = depth[ijk[mk, 0], ijk[mk, 1], ijk[mk, 2]]
sm = s[mk]
floor_c = sm < nh
surf = (~floor_c) & (dp <= 1)
inner = (~floor_c) & (dp > 1)
dp_all = depth[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
loc = dict(n=int(mk.numel()), floor_contact=int(floor_c.sum()), surface=int(surf.sum()), interior=int(inner.sum()),
           height_med=float(sm.median()), height_min=float(sm.min()), height_max=float(sm.max()), depth_vox_med=float(dp.float().median()), depth_vox_max=int(dp.max()),
           all_depth_vox_med=float(dp_all.float().median()), all_height_med=float(s.median()), voxel=vx,
           center=x[mk].mean(0).tolist(), bbox=[x[mk].min(0).values.tolist(), x[mk].max(0).values.tolist()],
           n_cells=int(cells.numel()), n_cells_node=int(sel_node.sum()), n_cells_flip=int(sel_flip.sum()), n_first_nodes=int(first_nodes.numel()),
           cell_npart=[int(q) for q in cnt_p[cells].tolist()][:200], cell_occ_vt=[float(q) for q in D["VT"][cells].tolist()][:200])
# 셀 질량 vs 전체
mT = torch.zeros(T).index_add_(0, pinv, D["mp"].float())
loc.update(cell_mT_med=float(mT[cells].median()), all_mT_med=float(mT[occ].median()), cell_npart_med=float(cnt_p[cells].float().median()),
           all_npart_med=float(cnt_p[occ].float().median()))
# F 특이값·이웃 퍼짐
R = torch.load(os.path.join(a.run, "roll_fem.pt"), map_location="cpu", weights_only=False)
G = torch.load(a.data, map_location="cpu", weights_only=False)
x0 = G["x"][0].float()
P, FP = R["pred"].float(), R["F"].float()                                           # 상태 k+1
xk = lambda k: x0 if k == 0 else P[k - 1]
Fk = lambda k: torch.eye(3).expand(mk.numel(), 3, 3) if k == 0 else FP[k - 1][mk].reshape(-1, 3, 3)
x0d = x0.to(dev)
qm = x0d[mk.to(dev)]
nbr = []
for c0 in range(0, qm.shape[0], 256):
    dd = torch.cdist(qm[c0:c0 + 256], x0d)
    nbr.append(dd.topk(a.knn + 1, largest=False).indices[:, 1:].cpu())
nbr = torch.cat(nbr) if nbr else torch.zeros(0, a.knn, dtype=torch.long)


def spread(X):
    d_ = (X[nbr] - X[mk][:, None]).norm(dim=-1).mean(1)
    d0 = (x0[nbr] - x0[mk][:, None]).norm(dim=-1).mean(1)
    r_ = d_ / d0
    return dict(med=float(r_.median()), max=float(r_.max()), q90=float(torch.quantile(r_, 0.9)))


def svs(F_):
    F_ = F_.double()
    ok = torch.isfinite(F_).all(-1).all(-1)
    sv = torch.linalg.svdvals(F_[ok])
    return dict(s1_med=float(sv[:, 0].median()), s1_max=float(sv[:, 0].max()), s3_med=float(sv[:, 2].median()), s3_min=float(sv[:, 2].min()),
                ratio_med=float((sv[:, 0] / sv[:, 2]).median()), ratio_max=float((sv[:, 0] / sv[:, 2]).max()), n_nonfinite=int((~ok).sum()))


hist = []
ndone = P.shape[0]
for kk in range(max(0, td - a.pre), td + 2):
    if kk > ndone:
        break
    X_ = xk(kk)
    if not bool(torch.isfinite(X_[mk]).all()):
        hist.append(dict(state=kk, ours="비유한"))
        continue
    Gx = G["x"][kk].float()
    GF = G["F"][kk].float()[mk].reshape(-1, 3, 3)
    hist.append(dict(state=kk, ours=dict(F=svs(Fk(kk)), spread=spread(X_)), gt=dict(F=svs(GF), spread=spread(Gx))))
# 같은 상태의 전체 입자 F 특이값 (비교)
allF = svs(FP[td - 1].reshape(-1, 3, 3)) if td >= 1 else None
ser = D["series"]
out = dict(run=a.run, td=td, frame_1based=td + 1, s_div=s_div, s_nf=s_nf, s_flip=s_flip, first_flip_frame=fl_first, run_onset=run_on, N=N, win=W,
           vthr=float(D["vthr"]), loc=loc, hist=hist, all_F_state_td=allF,
           frames=[dict(t=r["t"], n_flip=r["n_flip"], first_flip=r["first_flip"], n_nf=r["n_nf"], first_nf=r["first_nf"], n_abs=r["n_abs"], first_abs=r["first_abs"],
                        n_run_end=r["n_run_end"]) for r in J],
           series_td=[q for q in ser if q[0] >= max(0, (run_on if run_on is not None else s_div) - 5)][:80])
json.dump(out, open(os.path.join(OD, "dc.json"), "w"), ensure_ascii=False, indent=1)
print(f"[발산 셀] 프레임 {td} (1 부터 {td + 1}) 서브스텝 {s_div}/{N} (비유한 {s_nf}, 처음 뒤집힘 {s_flip}, 폭주 시작 {run_on}); 처음 터진 노드 {first_nodes.numel()}, "
      f"셀 {cells.numel()} (노드 {int(sel_node.sum())}, 뒤집힘 {int(sel_flip.sum())}), 입자 {mk.numel()}", flush=True)
print(f"[위치] 바닥 접촉 {loc['floor_contact']} · 표면 {loc['surface']} · 내부 {loc['interior']}, 높이 중앙 {loc['height_med']:.4f} (전체 {loc['all_height_med']:.4f}), "
      f"깊이 (복셀 {vx:.4f}) 중앙 {loc['depth_vox_med']} 최대 {loc['depth_vox_max']} (전체 중앙 {loc['all_depth_vox_med']}), 셀 입자 수 중앙 {loc['cell_npart_med']} (전체 {loc['all_npart_med']}), "
      f"셀 질량 중앙 {loc['cell_mT_med']:.3g} (전체 {loc['all_mT_med']:.3g})", flush=True)
for hh in hist:
    print(f"[이력] 상태 {hh['state']}: {json.dumps(hh, ensure_ascii=False)}", flush=True)
print(f"[전체 입자 F 상태 {td}] {allF}", flush=True)
print("\n".join(lines[:15]), flush=True)
