"""명령 목록을 GPU 에 하나씩 배정해 돌린다 (한 GPU 에 한 칸).

GASP 칸은 꼭짓점이 가우시안의 3 배라 한 칸이 GPU 를 거의 다 쓴다. 그래서
`for` 로 순차 실행하지 않고 **GPU 수만큼 동시에** 돌리고, 끝나는 자리에 다음
칸을 밀어 넣는다.

목록 파일은 한 줄에 한 명령. 빈 줄과 `#` 주석은 건너뛴다.
각 명령은 `CUDA_VISIBLE_DEVICES=<배정>` 와 함께 `cwd` 에서 돈다. 명령 안의
`{gpu}` 는 배정된 **물리 카드 번호**로 바뀐다 -- 스스로 CUDA_VISIBLE_DEVICES 를
덮어쓰는 스크립트(Spring-Gaus 의 train.py 가 `-g` 값을 그대로 넣는다)는 가림이
통하지 않으므로 번호를 직접 받아야 한다.

  python exe/run_queue.py --jobs /home/dkta/work/q.txt --gpus 0,1,2,3,4,5,6 \
      --logdir /home/dkta/work/qlog
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)
ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
ap.add_argument("--logdir", default="/home/dkta/work/qlog")
ap.add_argument("--cwd", default="/home/dkta/work/anchorflow")
ap.add_argument("--env", default="", help="k=v,k=v 로 추가 환경변수")
ap.add_argument("--free_mib", type=int, default=2000,
                help="이만큼 아래로 비어 있는 GPU 에만 띄운다 (0 이면 검사 안 함)")
a = ap.parse_args()

jobs = [q.strip() for q in open(a.jobs)
        if q.strip() and not q.strip().startswith("#")]
gpus = [q.strip() for q in a.gpus.split(",") if q.strip()]
os.makedirs(a.logdir, exist_ok=True)
extra = dict(q.split("=", 1) for q in a.env.split(",") if "=" in q)
print(f"[큐] 명령 {len(jobs)} 개, GPU {gpus}", flush=True)

def busy():
    """다른 잡이 쓰고 있는 GPU 번호 (내가 띄운 것은 뺀다)."""
    if a.free_mib <= 0:
        return set()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=60).stdout
    except Exception:
        return set()
    bad = set()
    for ln in out.strip().splitlines():
        q = [w.strip() for w in ln.split(",")]
        if len(q) == 2 and int(q[1]) > a.free_mib:
            bad.add(q[0])
    return bad


run, nxt, done = {}, 0, 0
while nxt < len(jobs) or run:
    bad = busy()
    for g in gpus:
        if g in run or nxt >= len(jobs) or g in bad:
            continue
        cmd = jobs[nxt].replace("{gpu}", g)
        tag = f"{nxt:02d}"
        lp = os.path.join(a.logdir, f"q{tag}.log")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUTF8="1",
                   PYTHONPATH="/home/dkta/work/anchorflow/lib", **extra)
        p = subprocess.Popen(cmd, shell=True, cwd=a.cwd, env=env,
                             stdout=open(lp, "w"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL)
        run[g] = (p, tag, cmd, time.time())
        print(f"[띄움] gpu{g} <- {tag}: {cmd}  (로그 {lp})", flush=True)
        nxt += 1
        break               # 한 바퀴에 하나씩 (방금 띄운 것이 메모리를 잡을 때까지)
    time.sleep(60)
    for g in list(run):
        p, tag, cmd, t0 = run[g]
        if p.poll() is None:
            continue
        done += 1
        print(f"[끝] gpu{g} {tag} rc={p.returncode} "
              f"{(time.time() - t0) / 60:.1f} 분 ({done}/{len(jobs)}): {cmd}",
              flush=True)
        del run[g]
print("QUEUE_DONE", flush=True)
