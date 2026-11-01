"""절 AM: mpm_fast.py 판들을 **한 파이썬 프로세스 안에서** 차례로 (클러스터 NFS 위 conda 시작·import 가 판마다 10 분 넘게 걸려서 -- 11-02 실측).
runpy 로 같은 스크립트를 판마다 다시 실행한다 (import 된 torch·warp·cupy 는 그대로, 솔버·상태는 판마다 새로).

  python exe/mpm_fast_multi.py --out DIR --vars "pg c0_port ..." [--extra "--frames 3"] [--timed] [--r2 r2:.../DIR]
--timed: 판마다 카드 1~6 이 모두 10 MiB 이하가 될 때까지 기다린 뒤 /home/dkta/work/gpu0.lock (flock) 을 잡고 GPU0 이 빌 때까지 기다려 실행, TIMED 표식.
아니면 NOT_TIMED 표식 (카드는 바깥 스크립트가 정함).
"""
import argparse
import contextlib
import fcntl
import gc
import os
import runpy
import shlex
import subprocess
import sys
import time

V = {
    "pg": "--mode pg",
    "c0_port": "--mode fast",
    "c1_sort": "--mode fast --sort",
    "c2_fuse": "--mode fast --sort --fuse",
    "c3_sparse": "--mode fast --sort --fuse --sparse",
    "c4_wred": "--mode fast --sort --fuse --sparse --wred",
    "c5_graph": "--mode fast --sort --fuse --sparse --wred --graph",
    "c5_graph_r2": "--mode fast --sort --fuse --sparse --wred --graph",
    "c6_nosort": "--mode fast --fuse --sparse --wred --graph",
    "c7_nowred": "--mode fast --sort --fuse --sparse --graph",
    "c8_shared": "--mode fast --sort --fuse --sparse --wred --shared --graph --block 256",
    "c8_shared_nowred": "--mode fast --sort --fuse --sparse --shared --graph --block 256",
    "c8_shared_r2": "--mode fast --sort --fuse --sparse --wred --shared --graph --block 256",
    "c5_b256": "--mode fast --sort --fuse --sparse --wred --graph --block 256",
    "floor": "--mode floor --graph --no_dump",
}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--vars", default=" ".join(V))
ap.add_argument("--extra", default="")
ap.add_argument("--timed", action="store_true")
ap.add_argument("--r2", default="")
ap.add_argument("--lock", default="/home/dkta/work/gpu0.lock")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "mpm_fast.py")
os.makedirs(a.out, exist_ok=True)
C = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout


def idle():
    for ln in smi().strip().splitlines():
        i, m = ln.split(",")[:2]
        if int(i) >= 1 and int(m.split()[0]) > 10:
            return False
    return True


for v in a.vars.split():
    o = os.path.join(a.out, v)
    if os.path.isfile(os.path.join(o, "res.json")):
        print(f"[건너뜀] {v}", flush=True)
        continue
    os.makedirs(o, exist_ok=True)
    lk = None
    if a.timed:
        w = 0
        while not idle():
            time.sleep(20)
            w += 20
        lk = open(a.lock, "w")
        fcntl.flock(lk, fcntl.LOCK_EX)
        while int(smi().splitlines()[0].split(",")[1].split()[0]) >= 500:
            time.sleep(10)
        open(os.path.join(o, "TIMED"), "w").write(f"gpu0.lock {time.strftime('%T')} 대기 {w} s\n")
    else:
        open(os.path.join(o, "NOT_TIMED"), "w").write("NOT_TIMED (경합 상태 값, 비율만)\n")
    open(os.path.join(o, "gpu_at_start.txt"), "w").write(smi())
    args = shlex.split(V[v]) + shlex.split(a.extra)
    open(os.path.join(o, "cmd.txt"), "w").write(f"커밋 {C} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} mpm_fast.py --out {o} {' '.join(args)}\n")
    t0 = time.time()
    with open(os.path.join(o, "log.txt"), "w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        sys.argv = [SCRIPT, "--out", o] + args
        try:
            runpy.run_path(SCRIPT, run_name="__main__")
        except BaseException as e:                                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[실패] {v}: {e!r}", flush=True)
    open(os.path.join(o, "gpu_at_end.txt"), "w").write(smi())
    if lk is not None:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()
    res = [ln for ln in open(os.path.join(o, "log.txt"), errors="replace") if "절 AM 결과" in ln or "[실패]" in ln]
    print(f"{time.strftime('%T')} {v} ({time.time() - t0:.0f} s) {res[-1].strip()[:300] if res else '결과 없음'}", flush=True)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:                                                                 # noqa: BLE001
        pass
    if a.r2:
        subprocess.run(["rclone", "copy", o, f"{a.r2}/{v}", "-q"])
