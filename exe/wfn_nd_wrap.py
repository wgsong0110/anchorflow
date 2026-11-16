"""절 CM193 진단 감싸개 (결과 불변): 프레임 끝 (counts 호출) 마다 env WND_F0..WND_F1 (1 부터) 프레임에서
입자 속도 상위 20 개와 그 27 노드 (질량·고정 pm·고착 몫 an/mqn·노드 속도·높이), 전체 노드 중 |v| > 1 m/s 노드 수·질량 분포를 <out>/nd.jsonl 에.
사용: python exe/wfn_nd_wrap.py <다음 감싸개.py> <그 인자...> (실행기 앞 어디든)."""
import json, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
F0, F1 = int(os.environ.get("WND_F0", "1")), int(os.environ.get("WND_F1", "0"))
_orig = runpy.run_path; _done = [False]


def _install():
    import torch
    from anchorflow import fem_an_rt as ART
    oc = ART.RTSolver.counts; n = [0]; LOG = open(os.path.join(out, "nd.jsonl"), "a")

    def counts(s):
        o = oc(s); n[0] += 1
        if F0 <= n[0] <= F1:
            torch.cuda.synchronize()
            ip = s.IP.cpu().tolist(); nn = ip[9]; nb = ip[6:9]
            mn = s.mn[:nn].double(); vk = s.vk[:nn].reshape(-1, 3).double(); pm = s.pm[:nn]; an = s.an[:nn].double(); mqn = s.mqn[:nn].double()
            act = mn > 0; med = float(mn[act].median())
            sp = vk.norm(dim=1); fast = act & (sp > 1.0)
            pv = s.v.double().norm(dim=1); top = torch.topk(pv, 20).indices
            pb = s.Np[top].long()
            rows = []
            for t_, p in enumerate(top.tolist()):
                ids = [((pb[t_, 0] + i) * nb[1] + (pb[t_, 1] + j)) * nb[2] + (pb[t_, 2] + k) for i in range(3) for j in range(3) for k in range(3)]
                ids = torch.tensor([int(q) for q in ids]).clamp(0, nn - 1).to(mn.device)
                m_ = mn[ids]; a_ = mn[ids] > 0
                rows.append(dict(p=p, v=round(float(pv[p]), 3), z=round(float(s.x[p, 2]), 3), J=round(float(torch.linalg.det(s.F[p].reshape(3, 3).double())), 3),
                                 m_rel=[round(float(q) / med, 4) for q in m_[a_].sort().values[:4]], npm2=int((pm[ids] == 2).sum()), nact=int(a_.sum()),
                                 stick=round(float((an[ids] / mqn[ids].clamp(min=1e-30))[a_].max()), 3), vn=round(float(sp[ids][a_].max()), 3)))
            LOG.write(json.dumps(dict(f=n[0], K=int(s.Kcur), nn=nn, med=med, nfast=int(fast.sum()), fast_mrel=[round(float(q) / med, 4) for q in mn[fast].sort().values[:10]],
                                      fast_pm2=int((fast & (pm == 2)).sum()), vmax=float(sp[act].max()), top=rows)) + "\n"); LOG.flush()
        return o
    ART.RTSolver.counts = counts


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True; _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
