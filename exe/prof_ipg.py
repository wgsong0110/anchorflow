"""[final_fix_mpmbl] i-PG Newton-GMRES 프로파일 (exe/pareto_mpmbl.py 를 그대로 돌리며 계측, 결과 무관).

pareto_mpmbl.py 인자를 그대로 받아 runpy 로 돌린다. 앞에 붙는 계측 인자:
  --prof_steps K      : K 서브스텝 뒤 멈춤 (요약 출력)
  --prof_skip S       : 처음 S 서브스텝은 계측 합계에서 뺌 (첫 걸음 컴파일·캐시)
  --prof_torch A:B    : 서브스텝 A..B-1 동안 torch.profiler (CUDA 활동 = 프로세스 전체 커널: cupy·warp 포함) -> 커널 표
  --prof_out PATH     : 요약 json
계측: 서브스텝 벽시계, eval (_af_eval_forces_gpu / _eval_forces_and_kdiag) 수·벽시계, GMRES 벽시계, numpy lstsq 수·시간,
      cupy 메모리 풀 할당 수·바이트, wp.launch 수·호스트 시간, cupy 동기화 (float()/asnumpy) 는 py-spy 로 따로.
  python exe/prof_ipg.py --prof_steps 6 --prof_skip 2 --prof_torch 3:5 --prof_out /tmp/p.json -- --method ipg ... (pareto_mpmbl 인자)
"""
import json
import os
import runpy
import sys
import time

argv = sys.argv[1:]
sep = argv.index("--")
pa, rest = argv[:sep], argv[sep + 1:]
P = dict(steps=6, skip=2, torch="", out="/tmp/prof_ipg.json")
i = 0
while i < len(pa):
    k = pa[i].lstrip("-").replace("prof_", ""); P[k] = pa[i + 1]; i += 2
K, SKIP = int(P["steps"]), int(P["skip"])
TA, TB = (int(q) for q in P["torch"].split(":")) if P["torch"] else (-1, -1)
HERE = os.path.dirname(os.path.abspath(__file__))
pg = rest[rest.index("--pg") + 1] if "--pg" in rest else "/home/dkta/work/i-physgaussian"

import numpy as np                       # noqa: E402
import cupy as cp                        # noqa: E402
import torch                             # noqa: E402
import warp as wp                        # noqa: E402
wp.init()
cwd = os.getcwd(); sys.path.insert(0, pg); os.chdir(pg)
import implicit_mpm_solver as ims        # noqa: E402
os.chdir(cwd)
import cupyx.scipy.sparse.linalg as csl  # noqa: E402

S = dict(step_ms=[], eval_n=0, eval_ms=0.0, gmres_n=0, gmres_ms=0.0, lstsq_n=0, lstsq_ms=0.0, launch_n=0, launch_ms=0.0,
         alloc_n=0, alloc_bytes=0, evals_logged=[])
ON = [False]


def _sync():
    cp.cuda.get_current_stream().synchronize(); wp.synchronize()


class _Hook(cp.cuda.MemoryHook):
    name = "afcount"

    def malloc_postprocess(self, **kw):
        if ON[0]:
            S["alloc_n"] += 1; S["alloc_bytes"] += int(kw.get("size", 0))


_hook = _Hook(); _hook.__enter__()

_lstsq = np.linalg.lstsq


def lstsq(*a, **k):
    t = time.perf_counter(); r = _lstsq(*a, **k)
    if ON[0]:
        S["lstsq_n"] += 1; S["lstsq_ms"] += 1e3 * (time.perf_counter() - t)
    return r


np.linalg.lstsq = lstsq
_launch = wp.launch


def launch(*a, **k):
    t = time.perf_counter(); r = _launch(*a, **k)
    if ON[0]:
        S["launch_n"] += 1; S["launch_ms"] += 1e3 * (time.perf_counter() - t)
    return r


wp.launch = launch
_gm = csl.gmres


def gmres(*a, **k):
    _sync(); t = time.perf_counter(); r = _gm(*a, **k); _sync()
    if ON[0]:
        S["gmres_n"] += 1; S["gmres_ms"] += 1e3 * (time.perf_counter() - t)
    return r


csl.gmres = gmres
for nm in ("_af_eval_forces_gpu", "_eval_forces_and_kdiag", "_af_eval_forces_fused"):
    if hasattr(ims.ImplicitMPMSolver, nm):
        def mk(f):
            def w(self, *a, **k):
                t = time.perf_counter(); r = f(self, *a, **k)
                if ON[0]:
                    _sync(); S["eval_n"] += 1; S["eval_ms"] += 1e3 * (time.perf_counter() - t)
                return r
            return w
        setattr(ims.ImplicitMPMSolver, nm, mk(getattr(ims.ImplicitMPMSolver, nm)))

_step = ims.ImplicitMPMSolver.p2g2p_newton_gmres
PROF = [None]


def step_w(self, step, dt):
    if step == TA:
        PROF[0] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        PROF[0].__enter__()
    ON[0] = step >= SKIP
    _sync(); t = time.perf_counter()
    r = _step(self, step, dt)
    _sync()
    if ON[0]:
        S["step_ms"].append(1e3 * (time.perf_counter() - t))
    if step == TB - 1 and PROF[0] is not None:
        PROF[0].__exit__(None, None, None)
        ka = PROF[0].key_averages()
        tab = ka.table(sort_by="cuda_time_total", row_limit=40)
        print(tab, flush=True)
        S["torch_table"] = tab
        try:
            PROF[0].export_chrome_trace(P["out"].replace(".json", "_trace.json"))
        except Exception as e:
            print("trace 실패", e)
        PROF[0] = None
    if step + 1 >= K:
        n = len(S["step_ms"])
        S["summary"] = {k: (S[k] / max(n, 1)) for k in ("eval_n", "eval_ms", "gmres_n", "gmres_ms", "lstsq_n", "lstsq_ms", "launch_n", "launch_ms",
                                                         "alloc_n", "alloc_bytes")}
        S["summary"]["step_ms_mean"] = float(np.mean(S["step_ms"])) if n else None
        S["summary"]["per_eval_ms"] = S["eval_ms"] / max(S["eval_n"], 1)
        S["summary"]["per_eval_alloc_n"] = S["alloc_n"] / max(S["eval_n"], 1)
        S["summary"]["per_eval_launch_n"] = S["launch_n"] / max(S["eval_n"], 1)
        print("[프로파일 요약 / 서브스텝당]", json.dumps(S["summary"], ensure_ascii=False), flush=True)
        json.dump(S, open(P["out"], "w"), ensure_ascii=False, indent=1)
        raise SystemExit(0)
    return r


if os.environ.get("AF_IPG_FAST") == "1":                      # pareto_mpmbl 가 ipg_fast.newton_gmres_fast 를 끼우므로 그것을 감싼다
    sys.path.insert(0, HERE)
    import ipg_fast                      # noqa: E402
    _step = ipg_fast.newton_gmres_fast
    ipg_fast.newton_gmres_fast = step_w
else:
    ims.ImplicitMPMSolver.p2g2p_newton_gmres = step_w
sys.argv = [os.path.join(HERE, "pareto_mpmbl.py")] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
