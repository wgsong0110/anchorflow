"""[paper_remeasure 11-15] 파레토 프런트의 우리 점 다시 재기: 현재 기본 판 (R2 cmd.txt 원본, exe/pr_ours_tmpl.json) 에서 걸음 수 K x PCG 상한 (--cg) 만 바꿈.
오차 = 롤아웃 평균 |x−x_gt|/EXT (summary.json err_mean), 시간 = 재생 GPU 시간 (summary.json ms_mean, 첫 2 프레임 뺀 평균), 카드 0 단독 (gpu0.lock, 시작 때 다른 프로세스 0 대기).
정답 MPM·VR-GS 점은 안 바뀌므로 R2 lbours/pareto/points.json (A6000 단독, 같은 정의) 을 그대로 붙임.
사용: python exe/pr_pareto.py --out <dir> [--old points.json] [--smoke N]"""
import argparse
import fcntl
import glob
import json
import os
import shlex
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--old", default="")
ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--only", default="", help="씬 하나만 (L000|ficus)")
a = ap.parse_args()
D = os.getcwd()
PY = "/home/dkta/.conda/envs/af/bin/python"
W = "/home/dkta/work"
TMPL = json.load(open(f"{D}/exe/pr_ours_tmpl.json"))
GRID = dict(L000=[4, 1, 2, 3, 5, 10], ficus=[24, 5, 10, 20, 40])     # PCG 상한 (원본 기본: 레고 4, ficus 24 -- 옛 격자에 기본점 추가)
KS = [1, 2, 4]
os.makedirs(a.out, exist_ok=True)


def cmd(t, K, cg, O):
    q = shlex.split(TMPL[t]["cmd"])
    env = {}
    while q and "=" in q[0] and not q[0].startswith("-"):
        k_, v_ = q.pop(0).split("=", 1); env[k_] = v_
    q[q.index("--out") + 1] = O
    q[q.index("--cg") + 1] = str(cg)
    if K > 1:
        q[q.index("--K") + 1] = str(K)
        for f_ in ("--ksplit", "--ks_big"):
            j = q.index(f_); q.pop(j)
            if f_ == "--ksplit":
                q.pop(j)
    for f_ in ("--dump", "--dump_F", "--dump_part"):       # 시간 판: 덤프 없이 (재생 GPU 시간에는 안 들어가지만 디스크 절약)
        while f_ in q:
            q.remove(f_)
    if a.smoke:
        q[q.index("--frames") + 1] = str(a.smoke); env["AF_SP_FR"] = f"1,{a.smoke}"
    q[0] = PY
    return q, env


def smi(x):
    return subprocess.run(["nvidia-smi", f"--query-{x}", "--format=csv,noheader"], capture_output=True, text=True).stdout


def others():
    u0 = smi("gpu=index,uuid").splitlines()[0].split(",")[1].strip()
    return [ln.split(",")[1].strip() for ln in smi("compute-apps=gpu_uuid,pid").splitlines() if ln.split(",")[0].strip() == u0]


for t in ("L000", "ficus"):
    if a.only and t != a.only:
        continue
    for K in KS:
        for cg in GRID[t]:
            O = f"{a.out}/t_ours_{t}_K{K}c{cg}"
            if os.path.exists(f"{O}/summary.json") or os.path.exists(f"{O}/END"):
                continue
            os.makedirs(O, exist_ok=True)
            q, env = cmd(t, K, cg, O)
            fd = open(f"{W}/gpu0.lock", "w"); fcntl.flock(fd, fcntl.LOCK_EX)
            while others():
                time.sleep(5)
            e = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUTF8="1", OMP_NUM_THREADS="8", WARP_CACHE_PATH="/dev/shm/dkta/warp_cache_0", **env)
            os.makedirs(e["WARP_CACHE_PATH"], exist_ok=True)
            open(f"{O}/card.txt", "w").write(f"시작 때 카드 0 의 다른 계산 프로세스 {len(others())} 개 {time.strftime('%F %T')} {' '.join(shlex.quote(x) for x in q)}\n")
            seen = set()
            with open(f"{O}/log.txt", "w") as lg:
                p = subprocess.Popen(q, cwd=D, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT)
                t0 = time.time()
                while p.poll() is None:
                    seen |= set(others())
                    if time.time() - t0 > 5400:
                        p.kill()
                    time.sleep(5)
            nf = max(len(seen) - 1, 0)
            open(f"{O}/card.txt", "a").write(f"rc={p.returncode} 끝 {time.strftime('%F %T')} 실행 중 카드 0 의 다른 계산 프로세스 {nf} 개\n")
            open(f"{O}/END", "w").write(f"rc={p.returncode}\n")
            fcntl.flock(fd, fcntl.LOCK_UN); fd.close()
            print(time.strftime("%H:%M:%S"), t, K, cg, "rc", p.returncode, flush=True)


def card_others(d):
    import re
    try:
        m = re.search(r"끝 .* 다른 계산 프로세스 (\d+) 개", open(os.path.join(d, "card.txt")).read())
        return int(m.group(1)) if m else None
    except OSError:
        return None


P = []
for d in sorted(glob.glob(f"{a.out}/t_ours_*")):
    import re
    m = re.match(r"t_ours_(\w+?)_K(\d+)c(\d+)$", os.path.basename(d))
    f = os.path.join(d, "summary.json")
    if not m:
        continue
    if not os.path.exists(f):
        P.append(dict(scene=m.group(1), method="ours", setting=f"K{m.group(2)} x PCG {m.group(3)}", err=None, err_done=None, ms=None, ms_max=None, div="중단", others=card_others(d), K=int(m.group(2)), cg=int(m.group(3))))
        continue
    s = json.load(open(f))
    ss = []
    try:
        ss = [json.loads(x).get("sec") for x in open(os.path.join(d, "fem.jsonl")) if x.strip()][2:]
        ss = [x for x in ss if isinstance(x, (int, float))]
    except OSError:
        pass
    P.append(dict(scene=m.group(1), method="ours", setting=f"K{m.group(2)} x PCG {m.group(3)}",
                  err=(100 * s["err_mean"] if not s.get("diverged") and s.get("err_mean") is not None else None),
                  err_done=(100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None), ms=s.get("ms_mean"),
                  ms_max=s.get("ms_max") or (1e3 * max(ss) if ss else None), div=s.get("diverged"), others=card_others(d), K=int(m.group(2)), cg=int(m.group(3))))
old = json.load(open(a.old)) if a.old and os.path.exists(a.old) else []
allp = [p for p in old if p["method"] != "ours"] + P
json.dump(P, open(f"{a.out}/points_ours.json", "w"), indent=1, ensure_ascii=False)
json.dump(allp, open(f"{a.out}/points.json", "w"), indent=1, ensure_ascii=False)
open(f"{a.out}/pareto.END", "w").write(time.strftime("%F %T") + "\n")
print("끝", len(P), "점")
