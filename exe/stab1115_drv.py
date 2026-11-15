"""[stab1115] 한 씬 한 카드 드라이버 (vast 인스턴스 안): 방법 PG · i-PG · GF · 우리(육면체) 의 최소 안정 N (2 의 거듭제곱 -> 정수 스캔 -> 3 회 확정), N 사다리 (오차 곡선용),
시각 지표용 덤프·렌더, 우리 단독 시간. 이어 돌리기: 결과 파일이 있으면 건너뜀 (R2 에서 받아 온 뒤 다시 띄우면 이어 감).
  python exe/stab1115_drv.py --scene ficus --par 4 --stages main,vis,speed
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); WT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stab1115_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--par", type=int, default=4)
ap.add_argument("--stages", default="main,vis,speed")
ap.add_argument("--methods", default="pg,ipg,gf,omni,ours")
ap.add_argument("--cap", default="pg:1024,ipg:16,gf:512,omni:256,ours:256", help="방법별 사다리 상한 N")
ap.add_argument("--hexfix", default="", help="우리 판 추가 플래그 (발산 수정 플래그가 나오면)")
ap.add_argument("--hextag", default="", help="우리 결과 폴더 접미 (수정판 구분)")
ap.add_argument("--hexpre", default="", help="우리 판 바깥 감싸개 (stabhex_wrap 앞; 예 'exe/ssfix_wrap.py --ssf_bec 1 ...' 는 'python -u' 뒤에 그대로 끼움)")
ap.add_argument("--extra_runs", default="", help="main 단계에서 사다리 없이 돌릴 판 (예 pg:1024 = 기준 궤적만)")
ap.add_argument("--tifrac", type=float, default=0.2)
ap.add_argument("--timeout", type=int, default=10800)
a = ap.parse_args()
S = a.scene; ST = L.ST; W = os.path.dirname(ST); PY = sys.executable
PG = f"{W}/i-physgaussian"
CAP = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.cap.split(",")}
SEM = threading.BoundedSemaphore(a.par); GSEM = threading.BoundedSemaphore(2); OSEM = threading.BoundedSemaphore(int(os.environ.get("S1115_OPAR", "1")))  # [stab4] 우리 판 동시 수 (메모리 크면 1)
os.makedirs(f"{ST}/{S}", exist_ok=True)
LOGF = open(f"{ST}/{S}/drv{('_' + os.environ['S1115_TAG']) if os.environ.get('S1115_TAG') else ''}.log", "a"); LK = threading.Lock()   # [stabfinal] 인스턴스별 로그 (한 장면을 여러 인스턴스가 나눠 돌릴 때 R2 에서 서로 안 덮게)
FR = L.nframes(S)                                    # [stabfinal] config frame_num (낙하 장면 60, 새 형상 80)
NS = S in L.NSHAPE                                   # [stabfinal] 새 형상 장면: 우리 판 명령 = exe/nshape_cmd.py (BA_1 사슬, --cg 4) + 같은 감싸개
DIAG_DROP = "--pdiag --qjd --fhash --hgt --bar_cnt"  # 새 형상 단독 시간 판에서 빼는 진단 플래그 (vast_elastic_new.sh 시간 판과 같음)


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
    if m == "omni":
        c2 = (f"cd {WT} && {PY} -u exe/stab_omni.py --repo {W}/OmniPhysGS --h5 {c}/init/sim_0000000000.h5 --config {c}/config.json --substeps {n} --out {out}.npz")
        return c2 + (f" --dump {dump}" if dump else "")
    if m in ("pg", "ipg"):
        c2 = (f"cd {PG} && AF_IPG_FUSED=1 OMP_NUM_THREADS=2 {PY} -u {WT}/exe/stab_mpm.py --method {m} --h5 {c}/init/sim_0000000000.h5 --config {c}/config.json "
              f"--substeps {n} --out {out}.npz --ckpt_every 0")
        return c2 + (f" --dump {dump}" if dump else "")
    c2 = (f"cd {WT} && {PY} -u exe/gf_mpm.py --config {c}/config.json --h5 {c}/init/sim_0000000000.h5 "
          f"--flip on --substeps {n} ")
    return c2 + (f"--out {dump} --resume" if dump else f"--out {out}_tmp --stab_out {out}.npz")


def hex_cmd_ns(n, d, dump=False, stats=1, direct=None, frames=FR, speed=False):
    """[stabfinal] 새 형상: nshape_cmd.py (BA_1 = 레고·wolf·bread 와 같은 H9 사슬, 장면 의존 부분만 바꿈, --cg 4) -> stabhex 기록 감싸개 + --hexpre 를 실행 사슬 맨 앞에."""
    env = dict(os.environ, NS_NODUMP="0" if dump else "1", NS_DROP=DIAG_DROP if speed else "")
    c = subprocess.run([PY, f"{HERE}/nshape_cmd.py", f"{W}/BA_1.txt", f"{ST}/{S}/hex_in.pt", str(frames), S, d, f"{ST}/{S}/gt", f"{ST}/{S}/gt/config.json", str(n)],
                       env=env, capture_output=True, text=True, check=True).stdout.strip()
    dr = 1 if (n >= 512 or direct) else 0
    head, sep, tail = c.partition(" -u exe/startpt_wrap.py ")
    assert sep, "nshape 명령에 startpt_wrap 이 없음"
    pre = (a.hexpre + " ") if a.hexpre else ""
    c = f"{head} -u {pre}exe/stabhex_wrap.py --sh_npz {d}/res.npz --sh_stats {stats} --sh_direct {dr} exe/startpt_wrap.py {tail}"
    if speed:
        c = re.sub(r"AF_SP_FR=\S+ |AF_E1_DG=1 |AF_G1_DG=1 ", "", c)
    return f"cd {WT} && {c} {a.hexfix}"


def hex_cmd(n, d, dump=False, stats=1, direct=None, frames=FR, speed=False):
    if NS:
        return hex_cmd_ns(n, d, dump=dump, stats=stats, direct=direct, frames=frames, speed=speed)
    base = open(f"{HERE}/stab1115_base.txt").read().strip()
    dr = 1 if (n >= 512 or direct) else 0
    du = "--dump --dump_F " if dump else ""
    env = f"AF_SP_FR=1,{frames} AF_E1_DG=1 AF_G1_DG=1 " + ("" if stats else "")
    for k, v in (("@ENV@", env), ("@OUT@", d), ("@ST@", str(stats)), ("@DATA@", f"{ST}/{S}/hex_in.pt"), ("@FR@", str(frames)), ("@TAG@", f"{S}_N{n}"),
                 ("@N@", str(n)), ("@S@", S), ("@RR@", f"{ST}/{S}/rr"), ("@CFG@", f"{ST}/{S}/config.json"), ("@DUMP@", du), ("@DIR@", str(dr))):
        base = base.replace(k, v)
    if a.hexpre:
        base = base.replace("python -u exe/stabhex_wrap.py", f"python -u {a.hexpre} exe/stabhex_wrap.py", 1)
    if speed:
        base = base.replace(env, "")
    return f"cd {WT} && {base} {a.hexfix}"


def oom_artifact(log):
    """다른 프로세스와 카드 메모리를 나눠 쓰다 난 메모리 부족 (발산 아님) -> 다시. 우리 판의 20 GiB 넘는 할당 시도는 발산 (실행기 관례)."""
    try:
        t = open(log, errors="ignore").read()
    except Exception:
        return False
    if not re.search(r"out of memory|OutOfMemory|MemoryError", t, re.I):
        return False
    fr = [int(x) for x in re.findall(r"프레임 (\d+)/%d" % FR, t)]
    if fr and max(fr) >= 5:                          # 몇 프레임 진행한 뒤 메모리가 커져 터짐 = 노드 상자 폭증 (발산), 시작 때 부족과 다름
        return False
    big = [float(x) for x in re.findall(r"Tried to allocate ([0-9.]+) GiB", t)]
    return not (big and max(big) >= 20)


def run(m, n, rep):
    for k in range(4):
        run1(m, n, rep)
        lg = (L.hex_dir(S, n, rep) + a.hextag + "/log.txt") if m == "ours" else f"{L.res_dir(S, rep)}/{m}_N{n}.log"
        fin = (L.hex_dir(S, n, rep) + a.hextag + "/FIN") if m == "ours" else f"{L.res_dir(S, rep)}/{m}_N{n}.fin"
        if os.path.exists(fin) and oom_artifact(lg) and "rc=0" not in open(fin).read():
            say("메모리 부족 (다른 프로세스와 카드 공유) 다시", S, m, n, rep, k); os.remove(fin); time.sleep(40); continue
        break


def run1(m, n, rep):
    """한 실행 (있으면 건너뜀). 결과는 L.load(S, m, n, rep)."""
    if m == "ours":
        d = L.hex_dir(S, n, rep) + a.hextag
        if os.path.exists(f"{d}/FIN"):
            return
        os.makedirs(d, exist_ok=True)
        with SEM, OSEM:                                  # 우리 판은 카드 메모리를 크게 써서 (플레인 17 GB) 한 번에 하나
            t0 = time.time(); rc = sh(f"CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d, direct=True)}", f"{d}/log.txt")
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
                t0 = time.time(); rc = sh(f"CUDA_VISIBLE_DEVICES=0 AF_TI_MEM={a.tifrac} {old_cmd(m, n, out)}", out + ".log")
            finally:
                if m == "gf": GSEM.release()
        for t in glob.glob(out + "_tmp*"):
            shutil.rmtree(t, ignore_errors=True) if os.path.isdir(t) else os.remove(t)
        open(out + ".fin", "w").write(f"rc={rc} {time.time() - t0:.0f}s")
    say("실행", S, m, n, rep, "rc", rc, f"{time.time() - t0:.0f}s")


def stable(m, n, rep):
    r = L.hex_load(L.hex_dir(S, n, rep) + a.hextag, FR) if m == "ours" else L.load(S, m, n, rep)     # [stab4] 우리 판은 hextag 폴더 (L.load 는 접미 없이 읽어 판정이 None 이 됐음)
    ok, mt = L.judge(r, S)
    return ok


def pmap(fn, items):
    with ThreadPoolExecutor(max_workers=max(len(items), 1)) as ex:
        return list(ex.map(fn, items))


def plan_path(m):
    return f"{ST}/{S}/plan_{m}{a.hextag if m == 'ours' else ''}.json"


def method_main(m):
    if os.path.exists(plan_path(m)) and json.load(open(plan_path(m))).get("N_min") is not None:
        return                                       # [stab4] N_min 없는 계획은 다시 판정 (있는 판은 건너뛰므로 싸다)
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
            # [stabfinal] 아래부터 동시 수 (--par) 만큼씩 돌리고 안정한 정수가 나오면 멈춤 (처음 안정한 N 은 전부 돌린 판과 같음; 계산 줄이기)
            st_i = []
            for i in range(0, len(ints), max(a.par, 1)):
                bt = ints[i:i + max(a.par, 1)]
                pmap(lambda n: run(m, n, "a"), bt)
                st_i = [n for n in bt if stable(m, n, "a")]
                if st_i:
                    break
            cand = st_i + [hi]
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
    for kv in [x for x in a.extra_runs.split(",") if x]:
        run(kv.split(":")[0], int(kv.split(":")[1]), "a")
    ts = {m: threading.Thread(target=method_main, args=(m,)) for m in ms if m != "gf"}
    [t.start() for t in ts.values()]
    if "gf" in ms:                                   # GF 는 taichi 가 GPU 메모리를 크게 잡아 (우리 판·PG 가 끝난 뒤) 따로 -- 동시 실행은 메모리 부족 (ficus 시험)
        for m in ("ours", "pg"):
            if m in ts: ts[m].join()
        method_main("gf")
    [t.join() for t in ts.values()]


# --------------------------------------------------------------------------------------------- 시각 지표
def up_vec():
    """렌더러는 회전 전 모델 좌표로 그림 -> 세계 위 (중력 반대) = config 회전 R 의 셋째 행 (vast_elastic_new.sh 와 같은 식; 회전 없는 장면은 0,0,1)."""
    import math
    import numpy as np
    c = json.load(open(f"{ST}/{S}/config.json")); R = np.eye(3)
    for dg, ax in zip(c.get("rotation_degree") or [], c.get("rotation_axis") or []):
        cs, sn = math.cos(dg / 180 * 3.1415926), math.sin(dg / 180 * 3.1415926)
        R = np.array({0: [[1, 0, 0], [0, cs, -sn], [0, sn, cs]], 1: [[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]], 2: [[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]}[int(ax)]) @ R
    return ",".join("%.6f" % q for q in R[2])


def traj0():
    """lb_render·lb_roll_to_h5 용 0 프레임 궤적 폴더 (meta.json: h5 = 처음 입자, R = I; sim_0 = 처음 x, F = I)."""
    T0 = f"{ST}/{S}/traj0"
    if not os.path.exists(f"{T0}/sim_0000000000.h5"):
        os.makedirs(T0, exist_ok=True)
        sh(f"""cd {WT} && {PY} - <<'PYEOF'
import h5py, json, numpy as np
with h5py.File('{ST}/{S}/init/sim_0000000000.h5') as h:
    x = np.array(h['x']); x = x.T if x.shape[0] == 3 else x
with h5py.File('{T0}/sim_0000000000.h5', 'w') as o:
    o.create_dataset('x', data=x.astype(np.float32)); o.create_dataset('F', data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (len(x), 1)))
json.dump(dict(h5='{ST}/{S}/init/sim_0000000000.h5', R=np.eye(3).tolist()), open('{T0}/meta.json', 'w'))
PYEOF""", f"{ST}/{S}/traj0.log")
    return T0


# [stabfinal] 렌더 = lb_render.py (F 반영 3DGS, 네 장면 같은 렌더러), 위에서 내려다보는 카메라 (고도 40 도, 위 = 세계 위 -> 중력이 화면 아래), vast_elastic_new.sh 의 기본값
CAM = "--cam_back 1.6 --elev 40 --back2 1.1"


def vis_one(name, m, n):
    V = f"{ST}/{S}/vis"; os.makedirs(V, exist_ok=True)
    png = f"{V}/png_{name}"
    if os.path.exists(f"{png}/DONE"):
        return png
    dump = f"{V}/dump_{name}"; shutil.rmtree(dump, ignore_errors=True)
    T0 = traj0(); rc = -1
    with SEM:
        if m == "ours":
            d = f"{V}/hexdump_{name}"; os.makedirs(d, exist_ok=True)
            sh(f"CUDA_VISIBLE_DEVICES=0 {hex_cmd(n, d, dump=True, stats=1, direct=True)}", f"{d}/log.txt")
            sh(f"cd {WT} && {PY} -u exe/lb_roll_to_h5.py --dump {d}/pred.pt --traj {T0} --out {dump}", f"{V}/{name}_conv.log")
            shutil.rmtree(d, ignore_errors=True)
        else:
            sh(f"CUDA_VISIBLE_DEVICES=0 AF_TI_MEM=0.5 {old_cmd(m, n, f'{V}/{name}', dump=dump)}", f"{V}/{name}_dump.log")
            shutil.copyfile(f"{T0}/meta.json", f"{dump}/meta.json") if os.path.isdir(dump) else None
        if os.path.exists(f"{dump}/meta.json"):
            rc = sh(f"cd {WT} && CUDA_VISIBLE_DEVICES=0 {PY} -u exe/lb_render.py --sim {dump} --config {ST}/{S}/config.json --shape {S} --out {V}/{name}.mp4 "
                    f"--png {png} {CAM} --up {up_vec()} --fps 30", f"{V}/{name}_render.log")
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
                items.append((f"{m}{a.hextag if m == 'ours' else ''}_N{nm}", m, nm))
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
    ns = [1, 2, 4, 8, 16]
    if os.path.exists(plan_path("ours")):                              # [stabfinal] 확정 N 도 (16 넘으면)
        nm = json.load(open(plan_path("ours"))).get("N_min")
        if nm and nm not in ns:
            ns.append(int(nm))
    for n in ns:
        d = f"{SP}/t_N{n}{a.hextag}"
        if os.path.exists(f"{d}/FIN"):
            continue
        os.makedirs(d, exist_ok=True)
        # 단독: 이 카드에 다른 계산 프로세스 0 (이 드라이버는 직렬, 다른 단계는 끝난 뒤) -- [stabfinal] 다른 프로세스가 있으면 빌 때까지 기다림 (최대 2 시간)
        for _w in range(240):
            out = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
            if not out:
                break
            time.sleep(30)
        open(f"{d}/apps_at_start.txt", "w").write(out)
        cmd = hex_cmd(n, d, stats=0, speed=True)
        if os.path.exists(plan_path("ours")):                          # 사다리에서 발산한 N 은 시간 안 잼
            lad = json.load(open(plan_path("ours"))).get("ladder", {})
            if lad.get(str(n)) is False:
                open(f"{d}/FIN", "w").write("skip=발산"); continue
        rc = sh(f"CUDA_VISIBLE_DEVICES=0 {cmd}", f"{d}/log.txt")
        open(f"{d}/FIN", "w").write(f"rc={rc}")
        say("단독 시간", S, n, "rc", rc)


if "ours" in a.methods.split(",") or "speed" in a.stages:
    os.makedirs(f"{ST}/{S}/rr", exist_ok=True)
    json.dump({"h5": f"{ST}/{S}/init/sim_0000000000.h5", "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, open(f"{ST}/{S}/rr/meta.json", "w"))
    if not os.path.exists(f"{ST}/{S}/hex_in.pt"):
        assert not NS, "새 형상 장면의 hex_in.pt 는 부팅이 pg_h5_to_pt 로 만듦"
        subprocess.run([PY, f"{HERE}/stabhex_data.py", f"{ST}/{S}", f"{ST}/{S}/hex_in.pt", str(FR)], check=True)
for st in a.stages.split(","):
    {"main": stage_main, "vis": stage_vis, "speed": stage_speed}[st]()
open(f"{ST}/{S}/DRV_DONE_{a.stages.replace(',', '_')}{a.hextag}", "w").write(time.strftime("%F %T"))
say("끝", S, a.stages)
