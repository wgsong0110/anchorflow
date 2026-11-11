"""절 CM106.8 (사용자 11-11): 레고 TC 뒤집힌 사면체 전부 기록 (결과 불변, 프레임 끝 버퍼 읽기만).
fem_tetc._divdiag (프레임마다 한 번, 한 프레임 = 한 걸음) 를 감싸 det X <= 0 (V_q > 0) 인 사면체 전부의 무게중심을 inv_<기록번호 3자리>.npz 로 (출력 디렉토리 = fem_tetc.OUT[0]).
  ref = 꼭짓점 4 개 격자 위치 평균 (x = A y, 노드 ijk 는 y = ijk), cur = ref + 꼭짓점 U 평균 (프레임 끝 모양), pmean = 프레임 시작에 그 사면체에 속한 입자의 프레임 끝 위치 평균 (없으면 nan), npart, det.
기록 번호 = 호출 순번 (tetc.jsonl 줄 번호와 같음).
"""
import os

import numpy as np
import torch

from . import fem_tetc as TC
from .fem_tetc import _q_nodes

_CNT = [0]


def _inv_all(sv, i):
    IP = sv.IP.detach().cpu().numpy().astype(np.int64)
    Q, nN, nb1, nb2 = int(IP[12]), int(IP[9]), int(IP[7]), int(IP[8])
    if Q > 4_000_000:
        return dict(skip=Q)
    dev = sv.Vq.device
    nid, bg = _q_nodes(sv, Q, nb1, nb2)
    Vq = sv.Vq[:Q].float(); Fq = sv.Fq[:Q].float().view(Q, 3, 3)
    U = sv.U[:nN].float()
    ok = (bg.abs().sum(-1) > 0) & (nid >= 0) & (nid < nN)
    nidc = nid.clamp(0, nN - 1)
    Gd = torch.einsum("qoa,qob->qab", U[nidc] * ok[..., None], bg)
    X = (torch.eye(3, device=dev) + Gd) @ Fq
    dX = torch.linalg.det(X.double()).float()
    inv = (Vq > 0) & (dX <= 0)
    qi = torch.nonzero(inv).flatten()
    K = int(qi.numel())
    out = dict(ninv=K, nok_min=int(ok.sum(-1)[qi].min()) if K else 0, nok_max=int(ok.sum(-1)[qi].max()) if K else 0)
    A = torch.linalg.inv(sv.Ai.double().reshape(3, 3))
    ref = np.zeros((K, 3), np.float32); cur = np.zeros((K, 3), np.float32); pm = np.full((K, 3), np.nan, np.float32)
    npart = np.zeros(K, np.int32); det = np.zeros(K, np.float32)
    if K:
        w = ok[qi].double()                                                                 # [K,27]
        n_ = nid[qi]
        ijk = torch.stack([n_ // (nb1 * nb2), (n_ // nb2) % nb1, n_ % nb2], -1).double()     # [K,27,3]
        cnt = w.sum(1, keepdim=True).clamp(min=1)
        mi = (ijk * w[..., None]).sum(1) / cnt
        mu = (U[nidc[qi]].double() * w[..., None]).sum(1) / cnt
        rf = mi @ A.T
        ref = rf.float().cpu().numpy(); cur = (rf + mu).float().cpu().numpy(); det = dX[qi].cpu().numpy()
        N = int(sv.N)
        fx = sv.fx[:N].float(); cidx = sv.cidx.long(); pcd = sv.pcd[:N].long()
        srt = torch.sort(fx, dim=1, descending=True, stable=True).indices
        tt = 2 * srt[:, 0] + (srt[:, 1] > srt[:, 2]).long()
        pq = cidx[pcd] * 8 + tt
        pq = torch.where((pq >= 0) & (pq < Q), pq, torch.full_like(pq, -1))
        x = sv.x[:N].float()
        slot = torch.full((Q,), -1, dtype=torch.long, device=dev); slot[qi] = torch.arange(K, device=dev)
        sp = torch.where(pq >= 0, slot[pq.clamp(min=0)], torch.full_like(pq, -1))
        m = (sp >= 0) & torch.isfinite(x).all(1)
        if bool(m.any()):
            s_ = sp[m]; xs = x[m].double()
            sm = torch.zeros(K, 3, dtype=torch.float64, device=dev).index_add_(0, s_, xs)
            c_ = torch.bincount(s_, minlength=K)
            npart = c_.cpu().numpy().astype(np.int32)
            pmv = (sm / c_.clamp(min=1)[:, None]).float().cpu().numpy(); pmv[c_.cpu().numpy() == 0] = np.nan
            pm = pmv
    np.savez(os.path.join(TC.OUT[0], f"inv_{i:03d}.npz"), ref=ref, cur=cur, pmean=pm, npart=npart, det=det)
    return out


def install():
    orig = TC._divdiag

    def dd(sv, x0):
        r = orig(sv, x0)
        i = _CNT[0]; _CNT[0] += 1
        try:
            r["inv_all"] = _inv_all(sv, i)
        except Exception as e_:
            r["inv_all"] = dict(err=repr(e_)[:300])
        return r
    TC._divdiag = dd
