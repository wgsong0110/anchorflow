"""칸 단위 잠금 -- 같은 칸을 두 번 돌리지 않는다.

큐를 여러 개 띄우면(비는 GPU 가 생길 때마다 범위를 넓히게 된다) 같은 칸이
두 번 뜰 수 있다. 큐의 목록은 시작할 때 한 번만 읽히므로 목록을 고쳐도
막을 수 없어, **칸 쪽에서** 잠근다.

  import joblock
  joblock.take("gasp_wolf_fracture", out_exists=os.path.exists(J))

- 살아있는 다른 프로세스가 같은 칸을 잡고 있으면 조용히 종료(rc=0)
- `out_exists` 가 참이면 이미 끝난 칸이므로 종료
- 죽은 프로세스가 남긴 잠금은 빼앗는다
"""
from __future__ import annotations

import atexit
import os
import sys

DIR = "/home/dkta/work/bench/.locks"


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def take(name, out_exists=False, force=False):
    if out_exists and not force:
        print(f"[건너뜀] {name}: 결과가 이미 있다", flush=True)
        sys.exit(0)
    os.makedirs(DIR, exist_ok=True)
    p = os.path.join(DIR, name)
    if os.path.exists(p):
        try:
            old = int(open(p).read().split()[0])
        except Exception:
            old = -1
        if _alive(old) and old != os.getpid():
            print(f"[건너뜀] {name}: pid {old} 가 돌리고 있다", flush=True)
            sys.exit(0)
    with open(p, "w") as f:
        f.write(f"{os.getpid()} {os.uname().nodename}\n")
    atexit.register(lambda: os.path.exists(p) and os.remove(p))
    return p
