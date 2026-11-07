"""절 CM39 진단 (읽기만): wolf 떨어져 나간 입자가 --ue_loc 경계 (덜 풀린 걸음의 S / S 밖) 에서 생기는지.

사용: python exe/speed3_wolfdiag.py <셀 m> <판 디렉토리...>   (판 = AF_S3_UL=1 감싸개 + --dump 판: pred.pt, ul.jsonl, ul_f*.npz)
이탈 입자 E = 어느 프레임이든 몸 상자 (예측 0.1~99.9 백분위) 밖 4 칸 넘게 나간 입자 (speed1_bbox 와 같은 정의).
갈라짐 프레임 f_sep = 그 입자가 처음 몸 상자 밖 1 칸 넘게 나간 프레임. 그 앞 (f_sep−3 .. f_sep) 덜 풀린 걸음 프레임 f_u 에서 입자의 S 몫 sfrac·둘레 노드 속도 퍼짐·속도를
같은 프레임 모든 입자 분포와 비교.
"""
import glob
import json
import os
import sys

import numpy as np
import torch


def outd(x):
    q = torch.quantile(x[::7].double(), torch.tensor([0.001, 0.999], dtype=torch.float64), dim=0)
    return torch.clamp(torch.maximum(q[0] - x.double(), x.double() - q[1]), min=0).norm(dim=-1)


cell = float(sys.argv[1])
for rd in sys.argv[2:]:
    o = dict(run=os.path.basename(rd.rstrip("/")))
    try:
        P = torch.load(os.path.join(rd, "pred.pt"), map_location="cpu", weights_only=False)["pred"].float()
    except Exception as ex:
        print(json.dumps(dict(o, err=str(ex)[:200]), ensure_ascii=False)); continue
    L, N = P.shape[0], P.shape[1]
    ok = torch.isfinite(P).all(-1)
    D = torch.stack([outd(torch.where(ok[f][:, None], P[f], P[f][ok[f]].mean(0))) for f in range(L)])     # [L, N] 몸 상자 밖 거리 (비유한 = 몸 중심)
    far = (D > 4 * cell).any(0)
    E = torch.nonzero(far).flatten()
    o["nE"] = int(E.numel()); o["dmax"] = round(float(D.max()), 3)
    ul = {}
    for fn in glob.glob(os.path.join(rd, "ul_f*.npz")):
        ul[int(os.path.basename(fn)[4:7])] = dict(np.load(fn))
    o["ul_frames"] = sorted(ul)
    if E.numel() == 0:
        print(json.dumps(o, ensure_ascii=False)); continue
    sep = torch.argmax((D[:, E] > cell).int(), 0) + 1                                 # 프레임 번호 (1 부터; pred[f] = 프레임 f+1 끝)
    o["f_sep_hist"] = {int(k): int(v) for k, v in zip(*np.unique(sep.numpy(), return_counts=True))}
    rows, hit = [], 0
    for e_, fs in zip(E.tolist(), sep.tolist()):
        fu = [f for f in ul if fs - 3 <= f <= fs]
        if not fu:
            continue
        hit += 1
        f = max(fu)
        u = ul[f]
        rows.append((f, fs, float(u["sfrac"][e_]), float(u["vspr"][e_]), float(u["vnmax"][e_]), float(u["vp"][e_])))
    o["E_with_ul_frame_within3"] = [hit, int(E.numel())]
    if rows:
        R = np.array(rows)
        sf = R[:, 2]
        o["E_sfrac"] = dict(zero=int((sf <= 1e-3).sum()), mixed=int(((sf > 1e-3) & (sf < 1 - 1e-3)).sum()), full=int((sf >= 1 - 1e-3).sum()), median=round(float(np.median(sf)), 3))
        pop = {}
        for f in sorted(set(R[:, 0].astype(int))):
            u = ul[f]; s_ = u["sfrac"].astype(np.float32); vp = u["vp"].astype(np.float32); vs = u["vspr"].astype(np.float32); vn = u["vnmax"].astype(np.float32)
            m = (R[:, 0] == f)
            pop[int(f)] = dict(nE=int(m.sum()), all_mixed_frac=round(float(((s_ > 1e-3) & (s_ < 1 - 1e-3)).mean()), 4), all_full_frac=round(float((s_ >= 1 - 1e-3).mean()), 4),
                          E_mixed_frac=round(float(((R[m, 2] > 1e-3) & (R[m, 2] < 1 - 1e-3)).mean()), 3), E_full_frac=round(float((R[m, 2] >= 1 - 1e-3).mean()), 3),
                          vp_p50_p99_all=[round(float(np.percentile(vp, 50)), 3), round(float(np.percentile(vp, 99)), 3)], vp_med_E=round(float(np.median(R[m, 5])), 3),
                          vspr_p99_all=round(float(np.percentile(vs, 99)), 3), vspr_med_E=round(float(np.median(R[m, 3])), 3),
                          vnmax_p99_all=round(float(np.percentile(vn, 99)), 3), vnmax_med_E=round(float(np.median(R[m, 4])), 3))
        o["by_ul_frame"] = pop
    # 갈라짐 프레임 앞뒤 E 속도 (위치 차 / 프레임 시간 비율 대신 프레임당 m)
    V = torch.zeros(L, E.numel())
    V[1:] = (P[1:, E] - P[:-1, E]).norm(dim=-1)
    o["E_step_m_med_by_frame"] = [round(float(V[f].median()), 4) for f in range(L)]
    print(json.dumps(o, ensure_ascii=False), flush=True)
