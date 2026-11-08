"""절 CM42 묶음 escape1 감싸개: 진단 훅 (결과 불변, 읽기만) 을 붙인 뒤 exe/speed3_wrap.py 로 넘김 (후보 T2·I 등은 그쪽 인자 그대로).

사용: python exe/escape1_wrap.py [speed3_wrap 인자...] <실행기.py> <실행기 인자...>
  환경 AF_E1_DG=1 이면 프레임 (모든 시도 끝) 마다 <out>/e1.jsonl 에:
    노드: m_n > 0 노드 수, 그 프레임 m_n / 중앙값 최소 (rmin), < 1e-2·1e-4·1e-6 노드 수, 속도 > 3 m/s 노드 수와 그 질량 비 중앙·최소
    입자 이탈 (exe/speed1_bbox.py 와 같은 정의, 프레임마다): 몸 상자 (0.1~99.9 백분위, 입자 7 개마다 표본) 에서 4 칸 넘게 떨어진 입자 수 nfar·거리 최대 dmax
    새 이탈 입자 (이 프레임에 처음 4 칸 밖): 수, 그 입자 B-스플라인 27 노드 (프레임 시작 칸, 가중 > 1e-3) 의 질량 비 최소의 중앙·최소·< 1e-4 몫,
      둘레 노드 속도 최대의 중앙, 입자 속도 중앙; 그 프레임 걸음별 판정 (덜 풀림 표시·‖g‖²/F²; 보폭은 fem.jsonl uf.smin)
  진단 실패는 풀이에 영향 없음 (<out>/e1_err.txt 에 기록).
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)

if os.environ.get("AF_E1_DG", "0") == "1":
    import torch
    from anchorflow import fem_an_rt as ART
    from anchorflow import fem_ao_rt as AORT
    from anchorflow import fem_wmrx as WRX

    LOG = open(os.path.join(out, "e1.jsonl"), "w")
    ST = dict(dep=0, far=None)

    def _bw(fx):
        # 2 차 B-스플라인 가중 (fem_speed3._bw 와 같은 식): fx = 칸 안 위치 [N,3] -> [N,3,3]
        return torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)

    @torch.no_grad()
    def diag(s, fi):
        IP = s.IP.cpu()
        nb0, nb1, nb2, nn = int(IP[6]), int(IP[7]), int(IP[8]), int(IP[9])
        mn = s.mn[:nn].float()
        has = mn > 0
        med = float(mn[has].median()) if bool(has.any()) else 1.0
        r = mn / med
        rh = r[has]
        sp = s.vk.reshape(-1, 3)[:nn].float().norm(dim=-1)
        fast = has & (sp > 3.0)
        rec = dict(f=fi + 1, nn=nn, nact=int(has.sum()), rmin=float(rh.min()) if rh.numel() else None,
                   n2=int((rh < 1e-2).sum()), n4=int((rh < 1e-4).sum()), n6=int((rh < 1e-6).sum()),
                   nfast=int(fast.sum()), fast_rmed=float(r[fast].median()) if bool(fast.any()) else None, fast_rmin=float(r[fast].min()) if bool(fast.any()) else None)
        N = int(getattr(s, "N", s.x.shape[0]))
        x = s.x[:N].float()
        ok = torch.isfinite(x).all(-1)
        cell = float(s.A[:, 0].norm())
        xs = x[ok][::7].double()
        q = torch.quantile(xs, torch.tensor([0.001, 0.999], dtype=torch.float64, device=xs.device), dim=0).float()
        d = torch.clamp(torch.maximum(q[0] - x, x - q[1]), min=0).norm(dim=-1)
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        far = d > 4 * cell
        rec.update(nfar=int(far.sum()), dmax=float(d[ok].max()) if bool(ok.any()) else None, nonfin=int((~ok).sum()))
        prev = ST["far"]
        new = far & (~prev if prev is not None and prev.numel() == far.numel() else torch.ones_like(far))
        ST["far"] = far
        nnew = int(new.sum())
        rec["new"] = dict(n=nnew)
        if 0 < nnew:
            idx = torch.nonzero(new).squeeze(1)
            pb = s.Np.reshape(-1, 3)[:N][idx].long()
            w = _bw(s.fx.reshape(-1, 3)[:N][idx].float())
            rmin_p = torch.full((idx.numel(),), float("inf"), device=x.device)
            vn_p = torch.zeros(idx.numel(), device=x.device)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        n = ((pb[:, 0] + i) * nb1 + (pb[:, 1] + j)) * nb2 + (pb[:, 2] + k)
                        okn = (n >= 0) & (n < nn)
                        n = n.clamp(0, nn - 1)
                        ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                        use = okn & (ww > 1e-3)
                        rr = torch.where(r[n] > 0, r[n], torch.full_like(r[n], float("inf")))
                        rmin_p = torch.where(use, torch.minimum(rmin_p, rr), rmin_p)
                        vn_p = torch.where(use, torch.maximum(vn_p, sp[n]), vn_p)
            fin = torch.isfinite(rmin_p)
            vp = s.v[:N][idx].float().norm(dim=-1)
            if bool(fin.any()):
                rf = rmin_p[fin]
                rec["new"].update(rmin_med=float(rf.median()), rmin_min=float(rf.min()), frac_lt1e4=float((rf < 1e-4).float().mean()), frac_lt1e2=float((rf < 1e-2).float().mean()))
            rec["new"].update(vn_med=float(vn_p.median()), vp_med=float(vp[torch.isfinite(vp)].median()) if bool(torch.isfinite(vp).any()) else None, d_med=float(d[idx].median()))
        K = int(getattr(s, "Kcur", 1))
        for fl, gg in (("cjl", "cjg"), ("ufl", "ufg")):
            L_, G_ = getattr(s, fl, None), getattr(s, gg, None)
            if L_ is not None and G_ is not None and L_.numel() >= K and G_.numel() >= K and L_.numel() < 4096:
                rec["step"] = dict(K=K, fail=[int(v) for v in L_[:K].cpu()], g2=[float(v) for v in G_[:K].cpu()])
                break
        LOG.write(json.dumps(rec) + "\n"); LOG.flush()

    def outer(fn):
        def w(s, t, h, *a, **kw):
            ST["dep"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                ST["dep"] -= 1
            if ST["dep"] == 0:
                try:
                    diag(s, int(round(t / h)))
                except Exception:
                    import traceback
                    with open(os.path.join(out, "e1_err.txt"), "a") as f_:
                        f_.write(traceback.format_exc()[-2000:] + "\n")
            return res
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    print("[절 CM42 진단] 프레임마다 노드 질량 비·입자 이탈 -> e1.jsonl (결과 불변)", flush=True)

sys.argv = [os.path.join(HERE, "speed3_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
