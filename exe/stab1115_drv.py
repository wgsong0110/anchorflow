"""[stab1115] 한 씬 한 카드 드라이버 (vast 인스턴스 안): 방법 PG · i-PG · GF · 우리(육면체) 의 최소 안정 N (2 의 거듭제곱 -> 정수 스캔 -> 3 회 확정), N 사다리 (오차 곡선용),
시각 지표용 덤프·렌더, 우리 단독 시간. 이어 돌리기: 결과 파일이 있으면 건너뜀 (R2 에서 받아 온 뒤 다시 띄우면 이어 감).
  python exe/stab1115_drv.py --scene ficus --par 4 --stages main,vis,speed
"""
import argparse, glob, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stab1115_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--par", type=int, default=4)
ap.add_argument("--stages", default="main,vis,speed")
ap.add_argument("--methods", default="pg,ipg,gf,ours")
ap.add_argument("--cap", default="pg:1024,ipg:256,gf:512,ours:256", help="방법별 사다리 상한 N")
ap.add_argument("--hexfix", default="", help="우리 판 추가 플래그 (발산 수정 플래그가 나오면)")
ap.add_argument("--hextag", default="", help="우리 결과 폴더 접미 (수정판 구분)")
ap.add_argument("--tifrac", type=float, default=0.2)
ap.add_argument("--timeout", type=int, default=10800)
a = ap.parse_args()
S = a.scene; ST = L.ST; W = os.path.dirname(ST); PY = sys.executable
PG = f"{W}/i-physgaussian"
CAP = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.cap.split(",")}
SEM = threading.BoundedSemaphore(a.par); GSEM = threading.BoundedSemaphore(2)
os.makedirs(f"{ST}/{S}", exist_ok=True)
LOGF = open(f"{ST}/{S}/drv.log", "a"); LK = threading.Lock()
FR = 60


def say(*s):
    with LK:
        m = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(q) for q in s); print(m, flush=True); LOGF.write(m + "\n"); LOGF.flush()


def gpu_free_cmd():
    return ""


def sh(cmd, log, timeout=None):
    with open(log, "w") as f:
        try:
            p = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout or a.timeout, stdin=subprocess.DEVNULL)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write("\n[timeout] 시간 초과\n"); return -9


def old_cmd(m, n, out, dump=""):
    c = f"{ST}/{S}"
    if m in ("pg", "ipg"):
        c2 = (f"cd {PG} && {PY} -u {WT}/exe/stab_mpm.py --method {m} --h5 {c}/init/sim_0000000000.h5 --config {c}/config.json "
              f"--substeps {n} --out {out}.npz --ckpt_every 0")
        return c2 + (f" --dump {dump}" if dump else "")
    c2 = (f"cd {WT} && {PY} -u exe/gf_mpm.py --config {c}/config.json --h5 {c}/init/sim_0000000000.h5 "
          f"--flip on --substeps {n} ")
    return c2 + (f"--out {dump} --resume" if dump else f"--out {out}_tmp --stab_out {out}.npz")


def hex_cmd(n, d, dump=False, stats=1, direct=None, frames=FR):
    base = open(f"{HERE}/stab1115_base.txt").read().strip()
    dr = 1 if (n >= 512 or direct) else 0
    du = "--dump --dump_F " if dump else ""
    env = f"AF_SP_FR=1,{frames} AF_E1_DG=1 AF_G1_DG=1 " + ("" if stats else "")
    for k, v in (("@ENV@", env), ("@OUT@", d), ("@ST@", str(stats)), ("@DATA@", f"{ST}/{S}/hex_in.pt"), ("@FR@", str(frames)), ("@TAG@", f"{S}_N{n}"),
                 ("@N@", str(n)), ("@S@", S), ("@RR@", f"{ST}/{S}/rr"), ("@CFG@", f"{ST}/{S}/config.json"), ("@DUMP@", du), ("@DIR@", str(dr))):
        base = base.replace(k, v)
    return f"cd {WT} && {base} {a.hexfix}"


def run(m, n, rep):
    """한 실행 (있으면 건너뜀). 결과는 L.load(S, m, n, rep)."""
    if m == "ours":
        d = L.hex_dir(S, n, rep) + a.hextag
        if os.path.exists(f"{d}/FIN"):
            return
        os.makedirs(d, exist_ok=True)
        with SEM:
            t0 = time.time(); rc = sh(f"CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d)}", f"{d}/log.txt")
        open(f"{d}/FIN", "w").write(f"rc={rc} {time.time() - t0:.0f}s")
        shutil.rmtree(f"{d}/pred.pt", ignore_errors=True)
        if os.path.exists(f"{d}/pred.pt"):
            os.remove(f"{d}/pred.pt")
    else:
        rd = L.res_dir(S, rep); os.makedirs(rd, exist_ok=True)
        out = f"{rd}/{m}_N{n}"
        if os.path.exists(out + ".fin"):
            return
        with SEM:
            if m == "gf": GSEM.acquire()
            try:
                t0 = time.time(); rc = sh(f"CUDA_VISIBLE_DEVICES=0 TI_DEVICE_MEMORY_FRACTION={a.tifrac} {old_cmd(m, n, out)}", out + ".log")
            finally:
                if m == "gf": GSEM.release()
        for t in glob.glob(out + "_tmp*"):
            shutil.rmtree(t, ignore_errors=True) if os.path.isdir(t) else os.remove(t)
        open(out + ".fin", "w").write(f"rc={rc} {time.time() - t0:.0f}s")
    say("실행", S, m, n, rep, "rc", rc, f"{time.time() - t0:.0f}s")


def stable(m, n, rep):
    ok, mt = L.judge(L.load(S, m, n, rep), S)
    return ok


def pmap(fn, items):
    with ThreadPoolExecutor(max_workers=max(len(items), 1)) as ex:
        return list(ex.map(fn, items))


def plan_path(m):
    return f"{ST}/{S}/plan_{m}{a.hextag}.json"


def method_main(m):
    if os.path.exists(plan_path(m)):
        return
    cap = CAP[m]; pows = [p for p in L.POW if p <= cap]
    pmap(lambda n: run(m, n, "a"), sorted(pows, reverse=True))                   # 큰 N 먼저 (긴 것부터)
    st = {n: stable(m, n, "a") for n in pows}
    say(S, m, "사다리", {n: st[n] for n in pows})
    good = [n for n in pows if st[n]]
    plan = dict(method=m, ladder=st, N_min=None)
    if good:
        hi = good[0]
        cand = [hi]
        if hi > 1:
            lo = hi // 2
            ints = list(range(lo + 1, hi))
            pmap(lambda n: run(m, n, "a"), ints)
            cand = [n for n in ints if stable(m, n, "a")] + [hi]
        n = cand[0]; tried = []
        while n <= 2 * hi:
            pmap(lambda r: run(m, n, r), list(L.REPS))
            res = [stable(m, n, r) for r in L.REPS]; tried.append((n, res))
            if all(res):
                plan["N_min"] = n; break
            n += 1
        plan["confirm_tried"] = tried
    json.dump(plan, open(plan_path(m), "w"))
    say(S, m, "N_min", plan["N_min"])


def stage_main():
    ms = a.methods.split(",")
    ts = [threading.Thread(target=method_main, args=(m,)) for m in ms]
    [t.start() for t in ts]; [t.join() for t in ts]


# --------------------------------------------------------------------------------------------- 시각 지표
def vis_one(name, m, n):
    V = f"{ST}/{S}/vis"; os.makedirs(V, exist_ok=True)
    png = f"{V}/png_{name}"
    if os.path.exists(f"{png}/DONE"):
        return png
    dump = f"{V}/dump_{name}"; shutil.rmtree(dump, ignore_errors=True)
    with SEM:
        if m == "ours":
            d = f"{V}/hexdump_{name}"; os.makedirs(d, exist_ok=True)
            sh(f"CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d, dump=True, stats=1)}", f"{d}/log.txt")
            sh(f"""cd {WT} && {PY} - <<'PYEOF'
import torch, numpy as np
q = torch.load('{d}/pred.pt', map_location='cpu', weights_only=False)
np.savez('{V}/{name}_full.npz', traj_full=q['pred'].float().numpy(), F_full=q['F'].float().numpy())
PYEOF""", f"{V}/{name}_conv.log")
            rc = sh(f"cd {WT} && CUDA_VISIBLE_DEVICES=0 {PY} -u exe/rep_ip.py --sim {ST}/{S}/init --cfg {ST}/{S}/config.json --shape {S} --method ours --frames {FR} --no_ref "
                    f"--render_npz {V}/{name}_full.npz --video {V}/{name}.mp4 --png_dir {png} --out {V}/{name}.r.npz", f"{V}/{name}_render.log")
            shutil.rmtree(d, ignore_errors=True); os.path.exists(f"{V}/{name}_full.npz") and os.remove(f"{V}/{name}_full.npz")
        else:
            sh(f"CUDA_VISIBLE_DEVICES=0 TI_DEVICE_MEMORY_FRACTION=0.5 {old_cmd(m, n, f'{V}/{name}', dump=dump)}", f"{V}/{name}_dump.log")
            rc = sh(f"cd {WT} && CUDA_VISIBLE_DEVICES=0 {PY} -u exe/rep_ip.py --sim {dump} --cfg {ST}/{S}/config.json --shape {S} --method ours --frames {FR} --render_ref "
                    f"--video {V}/{name}.mp4 --png_dir {png} --out {V}/{name}.r.npz", f"{V}/{name}_render.log")
            shutil.rmtree(dump, ignore_errors=True)
    if os.path.isdir(png) and len(glob.glob(f"{png}/*.png")) >= FR:
        open(f"{png}/DONE", "w").write(str(rc))
    else:
        say("렌더 실패", S, name, "rc", rc)
    return png


def stage_vis():
    V = f"{ST}/{S}/vis"; os.makedirs(V, exist_ok=True)
    items = [("pg_N1024", "pg", 1024)]
    for m in a.methods.split(","):
        if os.path.exists(plan_path(m)):
            nm = json.load(open(plan_path(m)))["N_min"]
            if nm:
                items.append((f"{m}_N{nm}", m, nm))
    for nm, m, n in items:                                    # 직렬 (렌더가 GPU 메모리 큼)
        vis_one(nm, m, n)
    ref = f"{V}/png_pg_N1024"
    runs = " ".join(f"{nm}={V}/png_{nm}" for nm, m, n in items if nm != "pg_N1024" and os.path.exists(f"{V}/png_{nm}/DONE"))
    if runs and os.path.exists(f"{ref}/DONE"):
        sh(f"cd {WT} && CUDA_VISIBLE_DEVICES=0 {PY} -u exe/stab_vis.py --ref {ref} --runs {runs} --out {V}/vis.json", f"{V}/vis.log")
    say(S, "vis 끝", open(f"{V}/vis.json").read()[:300] if os.path.exists(f"{V}/vis.json") else "vis.json 없음")


# --------------------------------------------------------------------------------------------- 단독 시간
def stage_speed():
    SP = f"{ST}/{S}/speed"; os.makedirs(SP, exist_ok=True)
    for n in (1, 2, 4, 8, 16):
        d = f"{SP}/t_N{n}"
        if os.path.exists(f"{d}/FIN"):
            continue
        os.makedirs(d, exist_ok=True)
        # 단독: 이 카드에 다른 계산 프로세스 0 (이 드라이버는 직렬, 다른 단계는 끝난 뒤)
        out = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
        open(f"{d}/apps_at_start.txt", "w").write(out)
        cmd = hex_cmd(n, d, stats=0).replace("AF_SP_FR=1,60 AF_E1_DG=1 AF_G1_DG=1 ", "")
        rc = sh(f"CUDA_VISIBLE_DEVICES=0 {cmd}", f"{d}/log.txt")
        open(f"{d}/FIN", "w").write(f"rc={rc}")
        say("단독 시간", S, n, "rc", rc)


if "ours" in a.methods.split(",") or "speed" in a.stages:
    os.makedirs(f"{ST}/{S}/rr", exist_ok=True)
    json.dump({"h5": f"{ST}/{S}/init/sim_0000000000.h5", "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, open(f"{ST}/{S}/rr/meta.json", "w"))
    if not os.path.exists(f"{ST}/{S}/hex_in.pt"):
        subprocess.run([PY, f"{HERE}/stabhex_data.py", f"{ST}/{S}", f"{ST}/{S}/hex_in.pt", "60"], check=True)
for st in a.stages.split(","):
    {"main": stage_main, "vis": stage_vis, "speed": stage_speed}[st]()
open(f"{ST}/{S}/DRV_DONE_{a.stages.replace(',', '_')}{a.hextag}", "w").write(time.strftime("%F %T"))
say("끝", S, a.stages)
