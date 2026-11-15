"""[substep_fix 11-15] 걸음·뉴턴 반복별 진단 감싸개 (결과 불변): 프레임 --sd_f0..--sd_f1 (1 부터, counts 순서) 마다 s.DG (반복 시작 |g|², 선탐색 보폭 SC14,
관성·탄성 E, g·δ) 와 s.Y0K (걸음별 시작점 k) 를 읽어 <out>/ssdiag.jsonl 에. 사용: python exe/ssdiag_wrap.py [--sd_f0 N --sd_f1 M] <다음 감싸개.py 인자...> (stabhex_wrap 앞에)."""
import json, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
C = dict(f0="0", f1="0")
while argv and argv[0].startswith("--sd_"):
    C[argv[0][5:]] = argv[1]; del argv[:2]
F0, F1 = int(C["f0"]), int(C["f1"])
out = argv[argv.index("--out") + 1]
_orig = runpy.run_path; _done = [False]


def _install():
    import torch
    from anchorflow import fem_an_rt as ART
    oc = ART.RTSolver.counts; n = [0]
    LOG = open(os.path.join(out, "ssdiag.jsonl"), "a")

    def counts(s):
        o = oc(s); n[0] += 1
        if F0 <= n[0] <= F1:
            torch.cuda.synchronize()
            D = s.DG.cpu().numpy(); Y = s.Y0K.cpu().numpy(); nt = int(s.nt); K = int(s.Kcur)
            rows = []
            for k in range(K):
                its = []
                for it in range(nt):
                    d0, d1 = D[2 * (k * nt + it)], D[2 * (k * nt + it) + 1]
                    if d0[1] == 0 and d1[14] == 0:
                        break
                    its.append([float(d0[1]), float(d1[14]), float(d1[9]), float(d1[20] + d1[10])])
                rows.append(dict(k=k, y0k=int(Y[k, 0]), its=its))
            LOG.write(json.dumps(dict(f=n[0], K=K, rows=rows)) + "\n"); LOG.flush()
        return o
    ART.RTSolver.counts = counts


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True; _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
