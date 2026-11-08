"""절 CM45 묶음 escape3 감싸개: 장벽 표본점 W̄ 고침·새 실행기 G (lib/anchorflow/fem_escape3.py) 와 진단 D3 를 붙인 뒤 exe/escape2_wrap.py 로 넘김
(그쪽 --e2* 후보·진단 D2, escape1·speed3 인자는 그대로 넘어감).

사용: python exe/escape3_wrap.py [--e3w 0|1|2] [--e3g 0|1] <escape2_wrap 인자...> <실행기.py> <실행기 인자...>
  --e3w 1|2  장벽 표본점 이동 기저 W̄ (1 = 원래 G2P 기저, 2 = G 판 노드 질량 가중 G2P 기저; 옛 실행기는 --e2g 1, 새 실행기는 --e3g 1 과 함께)
  --e3g 1    새 실행기 G2P 를 노드 질량 가중 보간으로 (절 CM43 G 정의)
  환경 AF_E3_DG=1 (옛 실행기만, 결과 불변, 읽기만): 프레임 (모든 시도 끝) 마다 <out>/e3.jsonl 에
    표본점 q (팔분 칸, 프레임 시작 입자 질량 중심 x̄_q) 의 걸음 끝 바닥 높이 세 가지:
      d_bw  = 장벽이 쓰는 가우스 점 자리 기저 BW 로 옮긴 x̄_q + Σ BW U (원래 정의),
      d_wq  = W̄ 로 옮긴 높이 (--e3w 판만),
      d_pm  = 그 칸 입자들의 실제 G2P 끝 높이 질량 평균 (Σ m_p z_p / Σ m_p),
    그 차 |d_bw − d_pm| 의 최대·중앙·99.9 백분위, |d_wq − d_pm| 최대 (W̄ 확인), d_pm < 0 (질량 평균이 바닥 밑) 인 표본점 수와 그중 d_bw ≥ 0 (장벽이 바닥 위로 본) 수,
    입자: 바닥 밑 (z < 바닥) 수·이 프레임 새로 바닥 밑, 새로 바닥 밑 입자 (최대 24 개) 의 z 시작·끝, 자기 q 의 d_bw·d_pm·d_wq·활성 AF_BACT·칸 입자 질량 비,
    입자 G2P 재구성 확인 (x_z − (x0_z + Σ ŵ U_z) 최대).
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
CAND = {}
while argv and argv[0].startswith("--e3"):                                              # 감싸개 인자 (맨 앞에만)
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
W = int(float(CAND.get("e3w", 0)))
G = int(float(CAND.get("e3g", 0)))
OLD_G = False
if "--e2g" in argv:
    i_ = argv.index("--e2g")
    OLD_G = int(float(argv[i_ + 1])) == 1
if W == 2:
    assert OLD_G or G == 1, "--e3w 2 (G 기저) 는 --e2g 1 (옛 실행기) 또는 --e3g 1 (새 실행기) 과 함께"
if W == 1:
    assert not OLD_G and G == 0, "--e3w 1 은 원래 G2P 기저 판 (G 없음)"
if W or G:
    from anchorflow import fem_escape3 as E3
    E3.install(w=W, g=W, newg=bool(G))
    print(f"[절 CM45] 장벽 표본점 기저 W̄ {W} (0 끔 1 원래 G2P 2 G 판), 새 실행기 G {G}", flush=True)

if os.environ.get("AF_E3_DG", "0") == "1":
    import torch
    from anchorflow import fem_ao_rt as AORT
    from anchorflow import fem_wmrx as WRX

    LOG = open(os.path.join(out, "e3.jsonl"), "w")
    ST = dict(dep=0, x0=None)
    ZF = float(os.environ.get("AF_E2_FLOORZ", "0.02"))                                 # 수박 바닥 평면 z (--floor_plane 1,1,0.02,0,0,1)

    def _bw(fx):
        return torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)

    def _w27(fx):
        w = _bw(fx)
        o = torch.arange(27, device=fx.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return w[:, 0, i] * w[:, 1, j] * w[:, 2, k]

    def _nid(b, nb1, nb2):
        o = torch.arange(27, device=b.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        return ((b[:, 0:1] + i) * nb1 + (b[:, 1:2] + j)) * nb2 + (b[:, 2:3] + k)

    @torch.no_grad()
    def post(s, fi):
        IP = s.IP.cpu()
        if int(IP[13]):
            LOG.write(json.dumps(dict(f=fi + 1, overflow=1)) + "\n"); LOG.flush()
            return
        nb1, nb2, nn, nq = int(IP[7]), int(IP[8]), int(IP[9]), int(IP[12])
        N = int(getattr(s, "N", s.x.shape[0]))
        x = s.x[:N].float()
        x0 = ST["x0"]
        m = s.m[:N].float()
        fx = s.fx.reshape(-1, 3)[:N].float()
        pb = s.Np.reshape(-1, 3)[:N].long()
        c = s.cidx.long()[s.pcd[:N].long()]
        oc = (fx[:, 0] >= 1).long() * 4 + (fx[:, 1] >= 1).long() * 2 + (fx[:, 2] >= 1).long()
        q = c * 8 + oc
        bx = s.bx.reshape(-1, 4)[:nq].float()
        bm = bx[:, 3]
        act = bm > 0
        U = s.U.reshape(-1, 3)[:nn].float()
        Uz = U[:, 2]
        cb = s.cb.reshape(-1, 3).long()
        qi = torch.arange(nq, device=x.device)
        ids = _nid(cb[qi // 8], nb1, nb2).clamp(0, nn - 1)
        BW = s.BW.reshape(8, 27).float()[qi % 8]
        zbar = torch.where(act, bx[:, 2] / bm.clamp_min(1e-30), torch.zeros_like(bm))
        d_bw = zbar + (BW * Uz[ids]).sum(1) - ZF
        zs = torch.zeros(nq, device=x.device, dtype=torch.float64).index_add_(0, q, (m * x[:, 2]).double())
        ms = torch.zeros(nq, device=x.device, dtype=torch.float64).index_add_(0, q, m.double())
        d_pm = (zs / ms.clamp_min(1e-30)).float() - ZF
        has = act & (ms > 0)
        rec = dict(f=fi + 1, nq=int(has.sum()))
        dd = (d_bw - d_pm).abs()[has]
        if dd.numel():
            rec.update(dbw_max=float(dd.max()), dbw_med=float(dd.median()), dbw_p999=float(torch.quantile(dd[::max(1, dd.numel() // 200000)].double(), 0.999)))
        wq = getattr(s, "e3wq", None)
        d_wq = None
        if wq is not None:
            d_wq = zbar + (wq[:nq * 27].reshape(nq, 27).float() * Uz[ids]).sum(1) - ZF
            if bool(has.any()):
                rec["dwq_max"] = float((d_wq - d_pm).abs()[has].max())
                sw = wq[:nq * 27].reshape(nq, 27).float().sum(1)[has]
                rec.update(wsum_min=float(sw.min()), wsum_max=float(sw.max()), wsum_bad=int(((sw - 1).abs() > 1e-3).sum()), trig=list(getattr(s, "_e3trig", (None, 0))))
        below_q = has & (d_pm < 0)
        rec.update(nq_pm_below=int(below_q.sum()), nq_pm_below_bw_above=int((below_q & (d_bw >= 0)).sum()))
        # 입자 G2P 재구성 확인 (G 판이면 노드 질량 가중)
        w = _w27(fx)
        nid = _nid(pb, nb1, nb2).clamp(0, nn - 1)
        if OLD_G or G:
            w = w * s.mn[:nn].float()[nid]
            w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        zr = x0[:, 2] + (w * Uz[nid]).sum(1) if x0 is not None else None
        if wq is not None and bool(has.any()):                                           # W̄ 를 파이썬으로 다시 모아 장치 값과 비교 (디버그)
            wp = torch.zeros(nq, 27, device=x.device, dtype=torch.float64).index_add_(0, q, (w * m[:, None]).double())
            wp = (wp / ms.clamp_min(1e-30)[:, None]).float()
            rec["wq_py_err"] = float((wp - wq[:nq * 27].reshape(nq, 27).float())[has].abs().max())
            rec["dpy_max"] = float((zbar + (wp * Uz[ids]).sum(1) - ZF - d_pm).abs()[has].max())
        ok = torch.isfinite(x).all(-1)
        if zr is not None:
            rec["g2p_err"] = float((x[:, 2] - zr).abs()[ok].max())
        zp = x[:, 2]
        bel = ok & (zp < ZF)
        rec.update(nbelow=int(bel.sum()), zmin=float(zp[ok].min()))
        if x0 is not None:
            new = bel & (x0[:, 2] >= ZF)
            rec["nnew_below"] = int(new.sum())
            idx = torch.nonzero(new).squeeze(1)
            if idx.numel():
                idx = idx[torch.argsort(zp[idx])[:24]]
                qq = q[idx]
                bact = s.bact[:nq][qq] if getattr(s, "bact", None) is not None else None
                med = float(ms[has].median())
                rec["new"] = [dict(z0=round(float(x0[p_, 2]), 5), z1=round(float(zp[p_]), 5), d_bw=round(float(d_bw[q_]), 5), d_pm=round(float(d_pm[q_]), 5),
                                   d_wq=(round(float(d_wq[q_]), 5) if d_wq is not None else None), bact=(int(bact[k_]) if bact is not None else None),
                                   mq=round(float(ms[q_]) / med, 4), vz=round(float(s.v[p_, 2]), 3))
                              for k_, (p_, q_) in enumerate(zip(idx.tolist(), qq.tolist()))]
        LOG.write(json.dumps(rec) + "\n")
        LOG.flush()

    def outer(fn):
        def w(s, t, h, *a, **kw):
            if ST["dep"] == 0:
                ST["x0"] = s.x[:int(getattr(s, "N", s.x.shape[0]))].float().clone()
            ST["dep"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                ST["dep"] -= 1
            if ST["dep"] == 0:
                try:
                    post(s, int(round(t / h)))
                except Exception:
                    import traceback
                    with open(os.path.join(out, "e3_err.txt"), "a") as f_:
                        f_.write(traceback.format_exc()[-2000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    print("[절 CM45 진단 D3] 표본점 바닥 높이 (BW 기저 · W̄ · 입자 질량 평균) 와 바닥 밑 입자 -> e3.jsonl (결과 불변)", flush=True)

sys.argv = [os.path.join(HERE, "escape2_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
