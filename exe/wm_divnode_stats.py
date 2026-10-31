"""수박 발산 프레임 서브스텝별 노드 속도·힘 요약 (사용자 10-31 '발산 노드 서브스텝별 속도 그래프').

입력: fem_ex_sim.py --divhist 덤프 (<run>/divhist/fNNN.pt -- 서브스텝 x 전 노드 |v|·|f|·바닥 접촉, 노드 질량).
출력 (작은 npz, 그림은 exe/wm_divnode_plot.py):
  E = 프레임 끝 |v| > 발산 판정 속도인 노드 (끝까지 넘어 있음), R = 넘었다 돌아온 노드, Q = 한 번도 안 넘은 점유 노드 (질량 > 0)
  - E 전부와 R 전부의 서브스텝별 |v|·|f|·접촉, 질량, 처음 넘은 서브스텝
  - Q 의 서브스텝별 |v|·|f|·|f|/m 평균·표준편차, 질량 분포
  - 대표 노드: E 를 처음 넘은 서브스텝 순으로 늘어놓고 등간격 분위 7 개, R 에서 같은 식 3 개
"""
import argparse
import json
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_e", type=int, default=7)
ap.add_argument("--n_r", type=int, default=3)
a = ap.parse_args()

D = torch.load(a.dump, map_location="cpu", weights_only=False)
N, vthr = int(D["N"]), float(D["vthr"])
V = D["speed"].float()            # [N, M]
Fo = D["force"].float()
C = D["contact"]
mn = D["mn"].float()
live = D["live"].bool()
ab = D["abs"].long()
M = V.shape[1]
E = (live & (V[-1] > vthr)).nonzero().flatten()
ever = live & (ab >= 0)
endm = torch.zeros(M, dtype=torch.bool); endm[E] = True
R = (ever & ~endm).nonzero().flatten()
Q = (live & ~ever).nonzero().flatten()
# 처음 넘은 서브스텝은 기록에서 다시 계산해 덤프의 abs 와 대조
first = torch.where((V > vthr).any(0), (V > vthr).float().argmax(0), torch.full((M,), -1))
mism = int((first[ever] != ab[ever]).sum())
E = E[torch.argsort(ab[E], stable=True)]
R = R[torch.argsort(ab[R], stable=True)]


def pick(idx, n):
    if len(idx) == 0:
        return idx
    q = np.unique(np.round(np.linspace(0, len(idx) - 1, n)).astype(int))
    return idx[q]


selE, selR = pick(E, a.n_e), pick(R, a.n_r)
mmed = float(mn[live].median())


def ms(X):
    return X.mean(1).numpy(), X.std(1).numpy()


vQm, vQs = ms(V[:, Q])
fQm, fQs = ms(Fo[:, Q])
aQm, aQs = ms(Fo[:, Q] / mn[Q])
EXTRA = [k for k in ("f_floor", "f_total", "f_clamp", "tau_max", "tau_mean") if k in D]   # 10-31 추가 기록 (셀 응력·힘 성분)
xo = {}
for k in EXTRA:
    X = D[k].float()
    xo[k + "_E"] = X[:, E].numpy().T
    xo[k + "_R"] = X[:, R].numpy().T
    XQ = X[:, Q]
    ok = torch.isfinite(XQ)
    n_ = ok.sum(1).clamp_min(1)
    mu_ = torch.where(ok, XQ, torch.zeros_like(XQ)).sum(1) / n_
    sd_ = (torch.where(ok, (XQ - mu_[:, None]) ** 2, torch.zeros_like(XQ)).sum(1) / n_).sqrt()
    xo[k + "_Qm"], xo[k + "_Qs"] = mu_.numpy(), sd_.numpy()
    del X, XQ, ok
dq = np.quantile(mn[Q].numpy(), [0.01, 0.1, 0.5, 0.9, 0.99])
out = dict(N=N, vthr=vthr, h=float(D["h"]), hs=float(D["hs"]), t=int(D["t"]), m_med=mmed,
           E=E.numpy(), R=R.numpy(), selE=selE.numpy(), selR=selR.numpy(),
           vE=V[:, E].numpy().T, fE=Fo[:, E].numpy().T, cE=C[:, E].numpy().T, mE=mn[E].numpy(), sE=ab[E].numpy(),
           vR=V[:, R].numpy().T, fR=Fo[:, R].numpy().T, cR=C[:, R].numpy().T, mR=mn[R].numpy(), sR=ab[R].numpy(),
           vQm=vQm, vQs=vQs, fQm=fQm, fQs=fQs, aQm=aQm, aQs=aQs, mQ_q=dq, nQ=len(Q),
           mQ_hist=np.histogram(np.log10(mn[Q].numpy() / mmed), bins=np.linspace(-3, 1.5, 91)),
           occE=D["occ_node"][E].numpy(), occR=D["occ_node"][R].numpy(), occQ_q=np.quantile(D["occ_node"][Q].numpy(), [0.1, 0.5, 0.9]),
           zE=D["npos"][E, 2].numpy(), zR=D["npos"][R, 2].numpy(), extra=np.array(EXTRA), **xo)
np.savez_compressed(a.out, **{k: (np.asarray(v, dtype=object) if k == "mQ_hist" else v) for k, v in out.items()})
s = dict(N=N, vthr=vthr, M=M, n_live=int(live.sum()), n_E=len(E), n_R=len(R), n_Q=len(Q), first_any=int(ab[ever].min()) if bool(ever.any()) else None,
         first_E=int(ab[E].min()) if len(E) else None, first_mismatch=mism, selE=selE.tolist(), selR=selR.tolist(),
         sE_sel=ab[selE].tolist(), sR_sel=ab[selR].tolist(), m_med=mmed,
         mE_rel_q=np.quantile(mn[E].numpy() / mmed, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist() if len(E) else None,
         mR_rel_q=np.quantile(mn[R].numpy() / mmed, [0.0, 0.1, 0.5, 0.9, 1.0]).tolist() if len(R) else None,
         mQ_rel_q=(dq / mmed).tolist(), contact_E_any=int(C[:, E].any(0).sum()), contact_R_any=int(C[:, R].any(0).sum()),
         force_nan=int(torch.isnan(Fo[:, live]).sum()), extra=EXTRA,
         **{f"{k}_at_div_E_med": float(np.median([xo[k + "_E"][q, int(ab[E][q])] for q in range(len(E))])) for k in EXTRA if len(E)})
print(json.dumps(s, ensure_ascii=False))
json.dump(s, open(a.out.replace(".npz", ".json"), "w"), ensure_ascii=False, indent=1)
