"""[substep_fix 11-15] 걸음·뉴턴 반복별 진단 감싸개 (결과 불변): 프레임 --sd_f0..--sd_f1 (1 부터, counts 순서) 마다 s.DG (반복 시작 |g|², 선탐색 보폭 SC14,
관성·탄성 E, g·δ) 와 s.Y0K (걸음별 시작점 k) 를 읽어 <out>/ssdiag.jsonl 에. 사용: python exe/ssdiag_wrap.py [--sd_f0 N --sd_f1 M] <다음 감싸개.py 인자...> (stabhex_wrap 앞에)."""
import json, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
C = dict(f0="0", f1="0", kh="0", pen="0", nd="0", load="")
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
                    its.append([float(d0[1]), float(d1[14]), float(d1[9]), float(d1[20] + d1[10]), [float(d1[21 + j]) for j in range(19)], [float(d1[40 + j]) for j in range(19)], [float(x) for x in d0[5:9]]])
                rows.append(dict(k=k, y0k=int(Y[k, 0]), its=its, pen=(s.sspen[k].tolist() if hasattr(s, "sspen") else None)))
            LOG.write(json.dumps(dict(f=n[0], K=K, rows=rows)) + "\n"); LOG.flush()
        return o
    ART.RTSolver.counts = counts
    if C["load"]:                                                                    # --sd_load 경로: 첫 step 호출에서 저장해 둔 프레임 시작 입자 상태 (ssfix --ssf_dumpfail) 로 바꿔 그 프레임을 재현
        ostep = ART.RTSolver.step; LD = dict(done=False)

        def step(s, t):
            if not LD["done"]:
                LD["done"] = True
                d = torch.load(C["load"], map_location="cpu")
                s.ga_first = False
                s._rs([q.to(s.x.device) for q in d["bk"]])
                s.alloc(None)
                print(f"[ssdiag] 프레임 시작 상태 불러옴: 원래 프레임 {d['fi']} K0 {d['K0']}", flush=True)
            return ostep(s, t)
        ART.RTSolver.step = step
    if C.get("nd", "0") != "0":                                                      # --sd_nd 1 (--sh_direct 1 판): 걸음 첫 뉴턴 반복 시작 (rt_grad it0) 마다 잔차 상위 노드·√k 분포 -> ssnd.jsonl (결과 불변)
        import numpy as np
        okn = ART.RTSolver.k; NS = dict(fr=0, stp=0, it=0); LG = open(os.path.join(out, "ssnd.jsonl"), "a")

        def kn(s, name, n, args, b=128):
            r = okn(s, name, n, args, b)
            if name == "rt_reset":
                NS["fr"] += 1; NS["stp"] = 0
            if name == "rt_floor_n":
                NS["it"] = 0
            if name == "rt_grad" and NS["it"] == 0:
                NS["it"] = 1
                torch.cuda.synchronize()
                nn_ = int(s.IP[9]); r0 = s.g0buf[:nn_ * 3].view(-1, 3).double(); g2n = (r0 * r0).sum(1)
                sk = s.sk[:nn_].double(); pm = s.pm[:nn_]; mn = s.mn[:nn_].double(); et = s.eta[:nn_ * 3].view(-1, 3).double()
                top = torch.topk(g2n, 5).indices.tolist()
                itau2 = float(args[2]) if not hasattr(args[2], "item") else float(args[2])
                LG.write(json.dumps(dict(fr=NS["fr"], stp=NS["stp"], g2=float(g2n.sum()), nsk=[int((sk < 1e-2).sum()), int(((sk >= 1e-2) & (sk < 0.5)).sum()), int(((sk >= 0.5) & (sk < 1)).sum())],
                                         etamax=float(et.abs().max()), top=[[i, float(g2n[i]), float(sk[i]), int(pm[i]), float(mn[i]), float(et[i].abs().max())] for i in top], itau2=itau2)) + "\n"); LG.flush()
                NS["stp"] += 1
            return r
        ART.RTSolver.k = kn
    if C.get("pen", "0") != "0":                                                     # --sd_pen 1: 걸음 끝 위치 U 에서 표본점 d<=0 (관통)·d<d̂ (장벽 활성) 수를 걸음마다 (그래프 안 추가 커널, 결과 불변)
        import numpy as np
        ok2 = ART.RTSolver.k; PS = dict(k=0)

        def k2(s, name, n, args, b=128):
            r = ok2(s, name, n, args, b)
            if name == "rt_reset":
                PS["k"] = 0
            if name in ("rt_ga_end", "rt_ga_end_uf", "rt_ga_end_ufl") and getattr(s, "bar", False):
                if not hasattr(s, "sspen"):
                    s.sspen = torch.zeros(4 * 4096, 4, device=s.x.device, dtype=torch.int32)
                    s.ssbuf = torch.zeros(4, device=s.x.device, dtype=torch.int32)
                FL = s.floor; nr = FL["NR"]; f32 = np.float32
                BA = (*[f32(q) for q in nr], f32(FL["PN"]), f32(FL["DH"]), f32(FL["KS"]))
                P = ART.P
                ok2(s, "rt_bar_q", s.capQc * 8, (P(s.IP), P(s.cb), P(s.BW), P(s.Vq), P(s.U), P(s.U), f32(1.0), *BA, np.int32(0), P(s.bGn), P(s.bKn), P(s.SC[10:])))
                occ = s.bx[: s.capQc * 8 * 4].view(-1, 4)[:, 3] > 0
                s.sspen[PS["k"], 0] = (occ & (s.bact[: occ.numel()] == 0)).sum().to(torch.int32)
                s.sspen[PS["k"], 1] = occ.sum().to(torch.int32)
                PS["k"] += 1
            return r
        ART.RTSolver.k = k2
    if C.get("kh", "0") != "0":                                                      # --sd_kh 1: 직접 (--sh_direct 1) 판에서 y0e 후보 에너지·k 를 걸음마다 (진단)
        ok = ART.RTSolver.k; ST = dict(fr=0, stp=0)

        def k(s, name, n, args, b=128):
            r = ok(s, name, n, args, b)
            if name == "rt_reset":
                ST["fr"] += 1; ST["stp"] = 0
            if F0 <= ST["fr"] <= F1 and name in ("rt_y0epick", "rt_bar_y0e"):
                torch.cuda.synchronize()
                if name == "rt_bar_y0e":
                    LOG.write(json.dumps(dict(fr=ST["fr"], stp=ST["stp"], E=[float(x) for x in s.SC[:6].cpu().numpy()])) + "\n")
                else:
                    LOG.write(json.dumps(dict(fr=ST["fr"], stp=ST["stp"], kk=int(s.kk[0].item()))) + "\n"); ST["stp"] += 1
                LOG.flush()
            return r
        ART.RTSolver.k = k


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True; _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
