"""절 CM39 묶음 speed3 (조정자 위임 11-08): 공통 후보 D 60 fps -- 진단·가속·후보 모듈 (두 실행기 훅 없이 감싸개 exe/speed3_wrap.py 가 부름).

(진단 1, 결과 불변, 호스트 읽기만) wolf 입자 이탈 -- --ue_loc 국소 갱신 경계: 덜 풀린 걸음이 있는 프레임마다 (K 1 마지막 시도 = 그 프레임 결과)
  입자마다 B-스플라인 27 노드 가중으로 S 몫 sfrac = Σ w ulm (S = 후퇴 오일러 노드 집합), 둘레 노드 속도 퍼짐 vspr = Σ w |v_n − v_p|,
  둘레 노드 속도 최대 vnmax (w > 1e-3), 입자 속도 |v_p| 를 원래 입자 순서로 ul_f<프레임>.npz 에, 노드 묶음 (S · S 밖 S 이웃 · 나머지) 속도·질량 통계를 ul.jsonl 에.
"""
import json
import os

import numpy as np
import torch


def _bw(fx):
    """이차 B-스플라인 가중 (fx ∈ [0.5, 1.5), 노드 0·1·2) -- rt_coord 의 fx = y − floor(y − 0.5)."""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return torch.stack([w0, w1, w2], -1)                                              # [N, 3(축), 3(노드)]


class ULDiag:
    def __init__(s, out):
        s.out = out
        s.f = open(os.path.join(out, "ul.jsonl"), "w")

    def frame(s, S, fi):
        cjl = getattr(S, "cjl", None)
        K = int(getattr(S, "Kcur", 1))
        fail = [int(q) for q in cjl[:K].cpu()] if cjl is not None else []
        rec = dict(f=fi + 1, K=K, fail=fail)
        if not any(fail) or getattr(S, "ulm", None) is None or K != 1:
            s.f.write(json.dumps(rec) + "\n"); s.f.flush()
            return
        IP = S.IP.cpu()
        nb0, nb1, nb2, nn = int(IP[6]), int(IP[7]), int(IP[8]), int(IP[9])
        ulm = S.ulm[:nn].float()
        mn = S.mn[:nn].float()
        vk = S.vk.reshape(-1, 3)[:nn].float()
        sp = vk.norm(dim=-1)
        N = S.x.shape[0]
        pb = S.Np.reshape(-1, 3)[:N].long()
        fx = S.fx.reshape(-1, 3)[:N]
        w = _bw(fx.float())
        vp = S.v[:N].float()
        sfrac = torch.zeros(N, device=vp.device); vspr = torch.zeros(N, device=vp.device); vnmax = torch.zeros(N, device=vp.device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    n = ((pb[:, 0] + i) * nb1 + (pb[:, 1] + j)) * nb2 + (pb[:, 2] + k)
                    ok = (n >= 0) & (n < nn)
                    n = n.clamp(0, nn - 1)
                    ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k] * ok
                    sfrac += ww * ulm[n]
                    vspr += ww * (vk[n] - vp).norm(dim=-1)
                    vnmax = torch.where(ww > 1e-3, torch.maximum(vnmax, sp[n]), vnmax)
        # 노드 묶음: S, S 밖이면서 27 이웃에 S 가 있는 노드 (경계 띠), 나머지 -- 질량 > 0 노드만
        m3 = ulm.view(nb0, nb1, nb2)
        dil = torch.nn.functional.max_pool3d(m3[None, None], 3, stride=1, padding=1)[0, 0].reshape(-1)
        has = mn > 0
        grp = dict(S=has & (ulm > 0), B=has & (ulm == 0) & (dil > 0), O=has & (dil == 0))
        med = float(mn[has].median()) if bool(has.any()) else 1.0
        for g, R in grp.items():
            if not bool(R.any()):
                rec[g] = dict(n=0)
                continue
            m_, s_ = mn[R], sp[R]
            i_ = int(torch.argmax(s_))
            rec[g] = dict(n=int(R.sum()), m=float(m_.sum()), vmean=float((m_ * s_).sum() / m_.sum()), vmax=float(s_.max()), m_at_vmax=float(m_[i_]) / med,
                          n_v3=int((s_ > 3.0).sum()), n_v10=int((s_ > 10.0).sum()))
        perm = S.perm[:N].long()
        o = lambda t: (lambda a: (a.__setitem__(perm, t), a)[1])(torch.empty_like(t))
        np.savez_compressed(os.path.join(s.out, f"ul_f{fi + 1:03d}.npz"), sfrac=o(sfrac).cpu().numpy().astype(np.float16), vspr=o(vspr).cpu().numpy().astype(np.float16),
                            vnmax=o(vnmax).cpu().numpy().astype(np.float16), vp=o(vp.norm(dim=-1)).cpu().numpy().astype(np.float16))
        rec["p"] = dict(mixed=int(((sfrac > 1e-3) & (sfrac < 1 - 1e-3)).sum()), inS=int((sfrac >= 1 - 1e-3).sum()), vp_max=float(vp.norm(dim=-1).max()), vspr_max=float(vspr.max()))
        s.f.write(json.dumps(rec) + "\n"); s.f.flush()
