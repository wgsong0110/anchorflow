"""프레임 시간 분해 (paper_remeasure2): 감싸개. torch.profiler 로 몇 프레임의 CUDA 커널 시간을 커널 이름별로 모아 <out>/prof.txt 에 씀 (결과 불변, 시간 판 아님).

사용: python -u exe/prof_wrap.py --prof_from 10 --prof_n 4 [<다음 감싸개.py> ...] exe/fem_an_rt_sim.py <실행기 인자 ...>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))


def install(f0, n, out):
    import torch
    from torch.profiler import ProfilerActivity, profile
    from anchorflow import fem_an_rt as ART
    orig = ART.RTSolver.step
    st = dict(i=0, prof=None)

    def step(s, t):
        st["i"] += 1
        if st["i"] == f0:
            st["prof"] = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]); st["prof"].__enter__()
        r = orig(s, t)
        if st["prof"] is not None and st["i"] == f0 + n - 1:
            torch.cuda.synchronize(); st["prof"].__exit__(None, None, None)
            ka = [e for e in st["prof"].key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
            tot = sum(e.device_time_total for e in ka) if ka and hasattr(ka[0], "device_time_total") else sum(e.cuda_time_total for e in ka)
            rows = sorted(ka, key=lambda e: -(getattr(e, "device_time_total", None) or e.cuda_time_total))
            with open(os.path.join(out, "prof.txt"), "w") as fh:
                fh.write(f"프레임 {f0}..{f0 + n - 1} (프레임 {n} 개) CUDA 커널 시간 합 {tot / 1e3:.2f} ms (프레임당 {tot / 1e3 / n:.2f} ms)\n")
                for e in rows[:60]:
                    tt = getattr(e, "device_time_total", None) or e.cuda_time_total
                    fh.write(f"{tt / 1e3 / n:8.3f} ms/프레임  호출 {e.count / n:7.1f}/프레임  {e.key[:90]}\n")
            st["prof"] = None
        return r

    ART.RTSolver.step = step


if __name__ == "__main__":
    argv = sys.argv[1:]
    f0, n = 10, 4
    while argv and argv[0] in ("--prof_from", "--prof_n"):
        if argv[0] == "--prof_from":
            f0 = int(argv[1])
        else:
            n = int(argv[1])
        del argv[:2]
    out = argv[argv.index("--out") + 1] if "--out" in argv else "."
    os.makedirs(out, exist_ok=True)
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
            _done[0] = True
            install(f0, n, out)
        return _orig(path, *a, **kw)

    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name="__main__")
