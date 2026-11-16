"""절 CM193 진단 감싸개 (결과 불변): 걸음 끝 갱신 커널 (rt_ga_end*) 뒤에 그래프 안 torch 연산으로 걸음마다 노드 최대 속도·그 노드 (번호·질량/평균·pm·고착 몫·걸음 변위)·
속도 > 1 m/s 노드 수와 그 노드들의 질량 합을 버퍼에 쓰고, 프레임 끝 (counts) 에 env WST_F0..WST_F1 프레임의 걸음별 기록을 <out>/st.jsonl 에.
사용: python exe/wfn_st_wrap.py <다음 감싸개.py> <그 인자...>"""
import json, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
F0, F1 = int(os.environ.get("WST_F0", "1")), int(os.environ.get("WST_F1", "0"))
_orig = runpy.run_path; _done = [False]


def _install():
    import torch
    from anchorflow import fem_an_rt as ART
    ok = ART.RTSolver.k; oc = ART.RTSolver.counts; n = [0]; LOG = open(os.path.join(out, "st.jsonl"), "a"); S = dict(k=0)

    def k(s, name, nthr, args, b=128):
        r = ok(s, name, nthr, args, b)
        if name == "rt_reset":
            S["k"] = 0
        if name.startswith("rt_ga_end"):
            if getattr(s, "_stb", None) is None or s._stb.shape[0] < 512 or s._stb_cap != s.capN:
                s._stb = torch.zeros(512, 10, device=s.x.device, dtype=torch.float32); s._stb_cap = s.capN
            kk = min(S["k"], 511)
            v = s.vk.reshape(-1, 3).norm(dim=1); act = (s.mn > 0).float(); v = v * act
            i = torch.argmax(v)
            med = s.mn.sum() / act.sum().clamp(min=1.0)                      # 활성 노드 평균 질량 (그래프 안 동기화 없는 연산만)
            fast = (v > 1.0).float()
            du = (s.U.reshape(-1, 3)[i]).norm()
            s._stb[kk, 0] = v[i]; s._stb[kk, 1] = i.float(); s._stb[kk, 2] = s.mn[i] / med; s._stb[kk, 3] = s.pm[i].float()
            s._stb[kk, 4] = s.an[i] / s.mqn[i].clamp(min=1e-30); s._stb[kk, 5] = du; s._stb[kk, 6] = fast.sum(); s._stb[kk, 7] = (fast * s.mn).sum() / med
            s._stb[kk, 8] = (v * v * s.mn).sum() / med; s._stb[kk, 9] = s.sk[i]
            S["k"] += 1
        return r

    def counts(s):
        o = oc(s); n[0] += 1
        if F0 <= n[0] <= F1 and getattr(s, "_stb", None) is not None:
            torch.cuda.synchronize()
            K = int(s.Kcur); B = s._stb[:K].cpu().tolist()
            LOG.write(json.dumps(dict(f=n[0], K=K, st=[[round(q, 5) for q in r] for r in B])) + "\n"); LOG.flush()
        return o
    ART.RTSolver.k = k; ART.RTSolver.counts = counts


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True; _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
