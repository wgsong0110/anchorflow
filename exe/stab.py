"""안정 최소 서브스텝 표 (doc/pareto.md 절 S): 방법별·씬별 '끝까지 발산하지 않는 최소 프레임당 서브스텝 수' 와 그 서브스텝의 지표.

잡 안에서 (exe/stab.sh 가 부름; cwd = 이 커밋 코드). 단계 (방법 x 씬 하나마다 스레드 하나):
  1. 사다리 N = 1, 2, 4, ... (방법별 상한 hi 까지, 상한은 마지막에 그대로 한 번) -- 처음 안정한 N_s 와 그 앞 N_s/2 (발산) 사이를 정수 이분 -> 최소 N*.
     발산 = 각 실행기의 기존 BD 정의: 비유한·오차 > 2 (롤아웃 평균 아님, 프레임 오차), 정답 MPM 은 서브스텝마다 격자 배열 밖 (--sub_chk). 실행기가 결과 없이 죽으면 '중단' (발산 쪽으로 셈, 표에 따로 적음).
     단조 확인: N* 를 찾은 뒤 2N*, 4N* (상한 이하) 도 돌려 발산이 있으면 표에 '비단조'.
  2. N* 에서 덤프 판 (입자 위치 + F) -> exe/lb_roll_to_h5.py -> exe/lb_render.py (3DGS 알파블렌딩, 정답 렌더와 같은 카메라·같은 3DGS) PNG -> exe/stab_vis.py (PSNR·SSIM·LPIPS alex, 프레임 1.. 평균).
  3. N* 에서 카드 0 단독 시간 판 (/home/dkta/work/gpu0.lock; 시작 때 카드 0 의 다른 계산 프로세스 0 을 기다리고 실행 중 5 초마다 기록).
비측정 판 (1·2) 은 빈 카드 1..7 (카드 0 제외; 계산 프로세스 0 이고 /dev/shm/dkta/anerr_claim/<카드> 의 pid 가 살아 있지 않은 카드 -- 다른 에이전트와 같은 규약).
판 결과는 <out>/p/<방법>_<씬>_n<N>/res.json (있으면 다시 안 돎 -- 잡이 바뀌어도 R2 에서 내려받으면 이어 감).
출력: <out>/stab_table.md, <out>/stab.json, 끝 표식 <out>/stab.END. R2 동기화는 exe/stab.sh.

  python exe/stab.py --out /dev/shm/dkta/lbours/stab --until <epoch> --hard <epoch> [--methods mpm,vrgs,bnv,ours] [--scenes L000,ficus,wolf,bread]
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--until", type=float, required=True, help="이 epoch 뒤에는 새 판을 띄우지 않음")
ap.add_argument("--hard", type=float, required=True, help="이 epoch 에 돌던 판이 있어도 표·끝 표식")
ap.add_argument("--methods", default="mpm,vrgs,bnv,ours")
ap.add_argument("--scenes", default="L000,ficus,wolf,bread")
ap.add_argument("--cards", default="1 2 3 4 5 6 7")
ap.add_argument("--par", type=int, default=6, help="동시 비측정 판 상한")
ap.add_argument("--smoke", type=int, default=0, help="연기 시험: 프레임 수를 이 값으로 (표는 smoke 표시)")
a = ap.parse_args()
PY = "/home/dkta/.conda/envs/af/bin/python"
W = "/home/dkta/work"
D = os.getcwd()
OUT = a.out
os.makedirs(f"{OUT}/p", exist_ok=True)
CL = "/dev/shm/dkta/anerr_claim"
os.makedirs(CL, exist_ok=True)
LOGF = open(f"{OUT}/stab.log", "a")
LK = threading.Lock()


def say(*s):
    with LK:
        m = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(q) for q in s)
        print(m, flush=True)
        LOGF.write(m + "\n"); LOGF.flush()


def sh_out(cmd):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=D).stdout.strip()


# ---------- 씬 ----------
ELASTIC = ("L000", "ficus")                     # 탄성 씬 (모든 방법); wolf·bread = 재질 씬 (정답 MPM·우리만)
FRAMES = dict(L000=80, ficus=125, wolf=50, bread=300)
SD = "/dev/shm/dkta/anerr_data"
AGD = sh_out("source exe/fem_cm8_scn.sh >/dev/null 2>&1; for t in " + " ".join(a.scenes.split(",")) + "; do cm_data $t; done >/dev/null 2>&1; echo $AG")
DATA = dict(L000=f"{SD}/L000.pt", ficus=f"{SD}/F_pg_v0.pt", wolf=f"{AGD}/wolf.pt", bread=f"{AGD}/bread.pt")
GT = dict(   # 정답 h5 (gt_sens --gt), 정답 config, 3DGS 이름 (lb_render --shape), 렌더용 정답 궤적 폴더 (meta.json·sim_*.h5)
    L000=(f"{W}/lbdata/test_low/L000", f"{W}/lbdata/config.json", "lego"),
    ficus=(f"{W}/lbours/ficus_v0/gt", f"{W}/lbours/ficus_v0/gt/config.json", "ficus"),
    wolf=(f"{W}/topo/pg_wolf_sand", f"{W}/PhysGaussian/config/wolf_config.json", "wolf"),
    bread=(f"{W}/topo/pg_bread_plastic", f"{W}/topo/bread_plastic_config.json", "bread"))


def gt_render_dir(t):
    """렌더용 정답 궤적 폴더 (exe/fem_cm4.sh gtd·mkgt 와 같음)."""
    if t in ("L000", "ficus"):
        return GT[t][0], GT[t][1]
    for b in ("/tmp/dkta/ag_data", "/dev/shm/dkta/ag_data"):
        g = f"{b}/gt_{t}"
        if os.path.exists(f"{g}/meta.json"):
            return g, f"{g}/config.json"
    return None, None


# ---------- 방법 ----------
# 우리 = 채택 판 (B) (imp_wt doc 절 CM8): 절 CM4 공통 설정 (exe/fem_cm3_scn.sh BA, HHT ρ 0.8, 뉴턴 2, εN 1e-2 · εP 1e-1, 시작점 관성 예측) + 장벽 바닥 + --y0eb,
#   걸음당 PCG 상한 = cm8 (B) 고른 값 (marks/cm8_sel_b.txt: lego 4 · wolf 5; ficus 24 · bread 2 는 cm4 와 같은 명령). 서브스텝 = 프레임당 걸음 수 K (K 만 바꿈).
OURS_CAP = dict(L000=4, ficus=24, wolf=5, bread=2)
HI = dict(mpm=dict(L000=833, ficus=400, wolf=1999, bread=100),       # 공식 config 서브스텝 (PG gs_simulation int(frame_dt/substep_dt))
          vrgs=dict(L000=167, ficus=400),                             # 논문 PG 비교 설정 dt 1e-4 s
          bnv=dict(L000=1024, ficus=1024),
          ours=dict(L000=16, ficus=16, wolf=16, bread=16))
NAME = dict(mpm="정답 MPM (PG warp, 공식 config)", vrgs="VR-GS (직접 구현, XPBD 케이지, 반복 11)", bnv="명시적 B-스플라인 (같은 표현, 명시적 적분)",
            ours="우리 (채택 판 (B): 장벽 + --y0eb)")
VR_IT = 11                                                             # 논문 PG 비교 설정의 반복 수


def da_elastic(t):
    if t == "L000":
        return ["--data", DATA[t], "--tag", "L000", "--frames", str(a.smoke or 80), "--floor", "vel"]
    return ["--data", DATA[t], "--tag", "ficus", "--frames", str(a.smoke or 125), "--floor", "none", "--cfg_bc", "--region_density", "--pg_vdamp"]


def mpm_args(t):
    g, c, _ = GT[t]
    L = str(a.smoke or FRAMES[t])
    if t == "L000":
        return ["--frames", L]
    return ["--gt", g, "--config", c, "--pt", DATA[t], "--frames", L, "--pg_cfg"]


def ours_cmd(t, K, mode):
    """mode: probe (결정적) | dump (결정적 + 덤프) | time (비결정적)."""
    xa = f"--ga hht --rho 0.8 --K {K} --newton 2 --cg {OURS_CAP[t]} --tolN 1e-2 --tolP 1e-1"
    if mode != "time":
        xa += " --det"
    if mode == "dump":
        xa += " --dump --dump_F"
    line = sh_out(f"source exe/fem_cm8_scn.sh >/dev/null 2>&1; cm_scene {t} --v=b {xa} >/dev/null 2>&1; echo \"$SIM $DA $XA\"")
    q = line.split()
    if a.smoke:
        i = q.index("--frames"); q[i + 1] = str(a.smoke)
    return [PY, "-u", q[0], "--tag", t] + q[1:]


def build(m, t, N, mode, O, E=None):
    """(명령, cwd, 추가 환경)."""
    if m == "mpm":
        x = ["--sub_chk", "--rpic_cfg", "--nsub", str(N)]
        x += ["--time"] if mode == "time" else []
        x += ["--dump_pred"] if mode == "dump" else ["--no_dump"]
        return [PY, "-u", f"{D}/exe/gt_sens.py", "--out", O] + mpm_args(t) + x, f"{W}/i-physgaussian", {}
    if m == "vrgs":
        if mode == "probe":
            return [PY, "-u", "exe/pareto_vrgs.py", "--mode", "fit", "--out", O] + da_elastic(t) + ["--set", f"{N}x{VR_IT}"], D, {}
        return [PY, "-u", "exe/pareto_vrgs.py", "--mode", "run", "--out", O, "--E", repr(E)] + da_elastic(t) + ["--set", f"{N}x{VR_IT}"] \
            + (["--dump_F"] if mode == "dump" else []), D, {}
    if m == "bnv":
        nh = {"L000": "0.028284", "ficus": "0.056569"}[t]
        x = ["--dump", "--dump_F"] if mode == "dump" else []
        return [PY, "-u", "exe/fem_bsel_cuda_sim.py", "--out", O] + da_elastic(t)[:2] + da_elastic(t)[4:] + ["--tag", t, "--node_h", nh, "--theta", "0", "--vfl_out", "--N", str(N)] + x, D, {}
    if m == "ours":
        return ours_cmd(t, N, mode)[:3] + ["--out", O] + ours_cmd(t, N, mode)[3:], D, {}
    raise KeyError(m)


def parse(m, O):
    """판 결과 -> dict(stable, err %, div, reason, ms, ms_max, E). 결과 파일 없으면 stable None (중단)."""
    r = dict(stable=None, err=None, div=None, reason=None, ms=None, ms_max=None, E=None)
    try:
        if m == "mpm":
            s = json.load(open(f"{O}/err.json"))
            dv = s.get("div")
            r.update(stable=dv is None, err=(100 * s["err_mean"] if dv is None else None), err_done=(100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None),
                     div=(dv or {}).get("frame"), reason=(dv or {}).get("reason"), ms=s.get("ms_mean"), ms_max=s.get("ms_max"))
        elif m == "vrgs":
            fj = f"{O}/fit.json" if os.path.exists(f"{O}/fit.json") else f"{O}/run.json"
            q = json.load(open(fj))["res"][0]
            st = q["err"] is not None
            r.update(stable=st, err=(100 * q["err"] if st else None), div=q.get("div"), E=q.get("E_best", q.get("E")), ms=q.get("ms_mean"), ms_max=q.get("ms_max"),
                     err_done=(100 * q["err_done"] if q.get("err_done") is not None else None))
            if not st and "evals" in q:
                r["reason"] = "맞춤 평가 %d 회 모두 발산" % len(q["evals"]) if all(e["err"] is None for e in q["evals"]) else "가장 나은 E 에서 발산"
        else:   # bnv · ours: summary.json
            s = json.load(open(f"{O}/summary.json"))
            dv = s.get("diverged")
            mx = s.get("ms_max")
            if mx is None and os.path.exists(f"{O}/fem.jsonl"):
                ss = [json.loads(q).get("sec") for q in open(f"{O}/fem.jsonl") if q.strip()][2:]
                ss = [q for q in ss if isinstance(q, (int, float))]
                mx = 1e3 * max(ss) if ss else None
            r.update(stable=not dv, err=(100 * s["err_mean"] if not dv else None), err_done=100 * s["err_mean_done"] if s.get("err_mean_done") is not None else None,
                     div=dv, ms=s.get("ms_mean"), ms_max=mx)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as ex:
        r["reason"] = f"결과 없음 ({type(ex).__name__}) -- 실행기 중단"
    return r


# ---------- 카드 ----------
CLK = threading.Lock()


def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader"], capture_output=True, text=True).stdout


def claim_card():
    while True:
        with CLK:
            apps = smi("compute-apps=gpu_uuid")
            for ln in smi("gpu=index,uuid").splitlines():
                i, u = [q.strip() for q in ln.split(",")]
                if i == "0" or i not in a.cards.split() or u in apps:
                    continue
                p = f"{CL}/{i}"
                if os.path.exists(p):
                    try:
                        os.kill(int(open(p).read().split()[0]), 0)
                        continue
                    except (OSError, ValueError, IndexError):
                        pass
                open(p, "w").write(f"{os.getpid()} stab {time.time()}\n")
                return i
        time.sleep(30)


def release_card(i):
    with CLK:
        try:
            if open(f"{CL}/{i}").read().split()[1] == "stab":
                os.remove(f"{CL}/{i}")
        except (OSError, IndexError):
            pass


SEM = threading.Semaphore(a.par)


def run_free(m, t, N, mode, O, E=None, tmo=7200):
    """비측정 판 하나 (빈 카드). 이미 res.json 있으면 그대로."""
    if os.path.exists(f"{O}/res.json"):
        return json.load(open(f"{O}/res.json"))
    if time.time() >= a.until:
        return None
    with SEM:
        G = claim_card()
        try:
            os.makedirs(O, exist_ok=True)
            cmd, cwd, env = build(m, t, N, mode, O, E)
            e = dict(os.environ, CUDA_VISIBLE_DEVICES=G, PYTHONUTF8="1", OMP_NUM_THREADS="8", WARP_CACHE_PATH=f"/dev/shm/dkta/warp_cache_{G}", **env)
            os.makedirs(e["WARP_CACHE_PATH"], exist_ok=True)
            open(f"{O}/card.txt", "w").write(f"카드 {G} {time.strftime('%F %T')} {' '.join(cmd)}\n{smi('compute-apps=gpu_uuid,pid,used_memory')}")
            t0 = time.time()
            with open(f"{O}/log.txt", "w") as lg:
                try:
                    rc = subprocess.run(cmd, cwd=cwd, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT, timeout=tmo).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            r = parse(m, O)
            r.update(rc=rc, N=N, sec=time.time() - t0, card=G)
            if r["stable"] is None:
                r["reason"] = (r["reason"] or "") + f" rc={rc}"
            json.dump(r, open(f"{O}/res.json", "w"), ensure_ascii=False, indent=1)
            say(f"[판] {m} {t} N {N} {mode} 카드 {G}: " + ("안정 %.3f%%" % r["err"] if r["stable"] else ("발산 프레임 %s %s" % (r["div"], r["reason"] or "") if r["stable"] is False else "중단 " + str(r["reason"]))))
            return r
        finally:
            release_card(G)


# ---------- 카드 0 단독 시간 ----------
TQ = threading.Lock()


def run_excl(m, t, N, O, E=None, tmo=7200):
    if os.path.exists(f"{O}/res.json"):
        return json.load(open(f"{O}/res.json"))
    with TQ:
        if time.time() >= a.until:
            return None
        os.makedirs(O, exist_ok=True)
        fd = open(f"{W}/gpu0.lock", "w")
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            u0 = smi("gpu=index,uuid").splitlines()[0].split(",")[1].strip()

            def others():
                return [ln.split(",")[1].strip() for ln in smi("compute-apps=gpu_uuid,pid").splitlines() if ln.split(",")[0].strip() == u0]
            while others():
                time.sleep(5)
            cmd, cwd, env = build(m, t, N, "time", O, E)
            e = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUTF8="1", OMP_NUM_THREADS="8", WARP_CACHE_PATH="/dev/shm/dkta/warp_cache_0", **env)
            os.makedirs(e["WARP_CACHE_PATH"], exist_ok=True)
            open(f"{O}/card.txt", "w").write(f"시작 때 카드 0 의 다른 프로세스 {len(others())} 개 {time.strftime('%F %T')} {' '.join(cmd)}\n")
            seen = set()
            with open(f"{O}/log.txt", "w") as lg:
                p = subprocess.Popen(cmd, cwd=cwd, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT)
                t0 = time.time()
                while p.poll() is None:
                    seen |= set(others())
                    if time.time() - t0 > tmo:
                        p.kill()
                    time.sleep(5)
            nf = max(len(seen) - 1, 0)                                         # 하나는 이 실행 자신
            open(f"{O}/card.txt", "a").write(f"rc={p.returncode} 끝 {time.strftime('%F %T')} 실행 중 카드 0 의 다른 계산 프로세스 {nf} 개 (pid {sorted(seen)})\n")
            r = parse(m, O)
            r.update(rc=p.returncode, N=N, others=nf)
            json.dump(r, open(f"{O}/res.json", "w"), ensure_ascii=False, indent=1)
            say(f"[시간] {m} {t} N {N}: 평균 {r['ms']} ms 최대 {r['ms_max']} (다른 {nf})")
            return r
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN); fd.close()


# ---------- 렌더·시각 품질 ----------
GLK = {}


def gt_png(t):
    """정답 궤적 PNG (같은 렌더러·카메라). 씬마다 한 번."""
    with GLK.setdefault(t, threading.Lock()):
        P = f"{OUT}/gt_{t}"
        if os.path.exists(f"{P}/done"):
            return f"{P}/png"
        gd, cf = gt_render_dir(t)
        if gd is None:
            G = claim_card()
            try:                                                               # exe/fem_cm4.sh mkgt 와 같은 변환
                h, c = GT[t][0], GT[t][1]
                os.makedirs("/dev/shm/dkta/ag_data", exist_ok=True)
                subprocess.run([PY, "-u", "exe/pg_h5_to_pt.py", "--h5_dir", h, "--config", c, "--out_pt", f"/dev/shm/dkta/ag_data/{t}_rend_tmp.pt",
                                "--out_dir", f"/dev/shm/dkta/ag_data/gt_{t}"], cwd=D, env=dict(os.environ, CUDA_VISIBLE_DEVICES=G),
                               stdout=open(f"/dev/shm/dkta/ag_data/cv_{t}_rend.log", "w"), stderr=subprocess.STDOUT)
            finally:
                release_card(G)
            gd, cf = gt_render_dir(t)
        os.makedirs(P, exist_ok=True)
        G = claim_card()
        try:
            subprocess.run([PY, "-u", "exe/lb_render.py", "--sim", gd, "--config", cf, "--shape", GT[t][2], "--out", f"{P}/gt_{t}.mp4", "--png", f"{P}/png",
                            "--frames", str(a.smoke or FRAMES[t])], cwd=D, env=dict(os.environ, CUDA_VISIBLE_DEVICES=G), stdin=subprocess.DEVNULL,
                           stdout=open(f"{P}/r.log", "w"), stderr=subprocess.STDOUT)
        finally:
            release_card(G)
        if os.path.exists(f"{P}/png/0001.png"):
            open(f"{P}/done", "w").write("ok\n")
        return f"{P}/png"


def visual(m, t, O):
    """덤프 판 O 의 pred.pt -> h5 -> PNG -> PSNR·SSIM·LPIPS (vis.json)."""
    if os.path.exists(f"{O}/vis.json"):
        return json.load(open(f"{O}/vis.json")).get(m)
    if not os.path.exists(f"{O}/pred.pt"):
        say(f"[렌더 없음] {m} {t}: 덤프 pred.pt 없음")
        return None
    ref = gt_png(t)
    gd, cf = gt_render_dir(t)
    G = claim_card()
    try:
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=G)
        with open(f"{O}/render.log", "w") as lg:
            if not os.path.exists(f"{O}/h5/meta.json"):
                subprocess.run([PY, "-u", "exe/lb_roll_to_h5.py", "--dump", f"{O}/pred.pt", "--traj", gd, "--out", f"{O}/h5"], cwd=D, env=e, stdout=lg, stderr=subprocess.STDOUT)
            subprocess.run([PY, "-u", "exe/lb_render.py", "--sim", f"{O}/h5", "--config", cf, "--shape", GT[t][2], "--out", f"{O}/{m}_{t}.mp4", "--png", f"{O}/png"],
                           cwd=D, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT)
            subprocess.run([PY, "-u", "exe/stab_vis.py", "--ref", ref, "--runs", f"{m}={O}/png", "--out", f"{O}/vis.json"], cwd=D, env=e, stdout=lg, stderr=subprocess.STDOUT)
    finally:
        release_card(G)
    try:
        return json.load(open(f"{O}/vis.json")).get(m)
    except (OSError, ValueError):
        say(f"[시각 품질 실패] {m} {t} (render.log)")
        return None


# ---------- 방법 x 씬 하나 ----------
RES = {}


def pdir(m, t, N, mode="probe"):
    return f"{OUT}/p/{m}_{t}_n{N}" + ("" if mode == "probe" else "_" + mode)


def one(m, t):
    try:
        _one(m, t)
    except Exception as ex:                                                            # noqa: BLE001
        import traceback
        RES.setdefault(f"{m}_{t}", dict(method=m, scene=t, probes={}))["state"] = f"오류 {type(ex).__name__}: {ex}"
        say(f"[오류] {m} {t}: {traceback.format_exc()}")


def _one(m, t):
    key = f"{m}_{t}"
    rec = RES.setdefault(key, dict(method=m, scene=t, probes={}, state="도는 중"))
    hi = HI[m][t]
    probe = lambda N: run_free(m, t, N, "probe", pdir(m, t, N))                     # noqa: E731
    ladder = []
    N = 1
    while True:
        ladder.append(min(N, hi))
        if N >= hi:
            break
        N *= 2
    if a.smoke:                                                                         # 연기 시험: 상한 한 판만 (명령·덤프·렌더·시간 경로 확인)
        ladder = [hi]
    lo, ns = None, None
    for N in ladder:
        r = probe(N)
        if r is None:
            rec["state"] = "마감 (사다리 중)"; return
        rec["probes"][N] = r
        if r["stable"]:
            ns = N; break
        lo = N
    if ns is None:
        rec.update(state=f"상한 {hi} 까지 발산", nmin=None, lo=lo); return
    if lo is not None:                                                                  # (lo 발산, ns 안정) 정수 이분
        while ns - lo > 1:
            mid = (lo + ns) // 2
            r = probe(mid)
            if r is None:
                rec.update(state="마감 (이분 중)", nmin=ns, lo=lo); return
            rec["probes"][mid] = r
            if r["stable"]:
                ns = mid
            else:
                lo = mid
    rec.update(nmin=ns, lo=lo)
    mono = []
    for k in (() if a.smoke else (2, 4)):                                              # 단조 확인
        if ns * k <= hi:
            r = probe(ns * k)
            if r is not None:
                rec["probes"][ns * k] = r
                mono.append((ns * k, r["stable"]))
    rec["mono"] = mono
    best = rec["probes"][ns]
    E = best.get("E")
    O = pdir(m, t, ns, "dump")
    if os.path.exists(f"{O}/res.json") and not os.path.exists(f"{O}/pred.pt") and not os.path.exists(f"{O}/vis.json"):
        os.remove(f"{O}/res.json")                                                       # 이어 돌리기: 덤프가 없으면 덤프 판 다시
    rd = run_free(m, t, ns, "dump", O, E)
    rec["dump"] = rd
    rec["vis"] = visual(m, t, pdir(m, t, ns, "dump")) if rd else None
    rec["time"] = run_excl(m, t, ns, pdir(m, t, ns, "time"), E)
    rec["state"] = "끝"
    say(f"[끝] {m} {t}: 최소 N {ns} (아래 {lo} 발산), 오차 {best['err']}, 시간 {(rec['time'] or {}).get('ms')}, 시각 {rec['vis']}")


# ---------- 표 ----------
ALL = [("mpm", "정답 MPM (PG, 공식 설정)"), ("vrgs", "VR-GS (직접 구현)"), ("bnv", "명시적 B-스플라인"), ("phystwin", "PhysTwin"), ("simplicits", "Simplicits"),
       ("springgaus", "Spring-Gaus"), ("gaussim", "GausSim"), ("ours", "우리 (B)")]
ELONLY = {"vrgs", "bnv", "phystwin", "simplicits", "springgaus", "gaussim"}           # 탄성 전용 (bnv 는 실행기에 재질 없음)
TODO = dict(phystwin="미실행 -- 실행기 준비 중 (doc/pareto.md 절 S.3)", simplicits="미실행 -- 실행기 준비 중 (절 S.3)", springgaus="미실행 -- 정적·속도·동역학 학습 필요 (절 S.3)",
            gaussim="실행 안 함 -- 공식 학습 코드 미공개 (절 S.4)")


def f2(x, k=2):
    return "" if x is None else f"%.{k}f" % x


def table():
    L = ["# 안정 최소 서브스텝 표 (doc/pareto.md 절 S)" + (f" -- 연기 시험 {a.smoke} 프레임" if a.smoke else ""), "",
         "최소 N = 끝까지 발산하지 않는 최소 프레임당 서브스텝 수 (사다리 1·2·4·… 다음 정수 이분; 발산 = 비유한·프레임 오차 > 2·정답 MPM 격자 밖). 경계 = 바로 아래 시험한 N (발산 프레임).",
         "오차 = 롤아웃 평균 |x−x_gt|/EXT (N* 판정 판). 시간 = A6000 카드 0 단독, 첫 2 프레임 뺀 평균 (최대). 시각 품질 = 3DGS 알파블렌딩, 정답 렌더와 같은 카메라, 프레임 1.. 평균 PSNR·SSIM·LPIPS (alex).", ""]
    for t in a.scenes.split(","):
        L += [f"## {t}", "", "| 방법 | 최소 N | 경계 (발산 쪽) | 오차 % | 평균 ms (최대) | PSNR | SSIM | LPIPS | 비고 |", "|---|---|---|---|---|---|---|---|---|"]
        for m, nm in ALL:
            if t not in ELASTIC and m in ELONLY:
                L.append(f"| {nm} | 해당 없음 | | | | | | | 탄성 전용{' (실행기에 재질 없음)' if m == 'bnv' else ''} |"); continue
            if m in TODO:
                L.append(f"| {nm} | {TODO[m]} | | | | | | | |"); continue
            q = RES.get(f"{m}_{t}")
            if q is None:
                L.append(f"| {nm} | 안 돎 | | | | | | | |"); continue
            ns, lo = q.get("nmin"), q.get("lo")
            if ns is None:
                L.append(f"| {nm} | {q['state']} | | | | | | | |"); continue
            b = q["probes"][ns]
            bd = ""
            if lo is not None:
                rl = q["probes"][lo]
                bd = f"N {lo}: " + (f"x{rl['div']}" if rl["stable"] is False else "중단") + (f" ({rl['reason']})" if rl.get("reason") else "")
            else:
                bd = "N 1 안정 (아래 없음)"
            tm = q.get("time") or {}
            v = q.get("vis") or {}
            nt = []
            if q.get("mono") and any(s is False for _, s in q["mono"]):
                nt.append("비단조: " + ", ".join(f"N {n} {'안정' if s else '발산'}" for n, s in q["mono"]))
            if b.get("E"):
                nt.append(f"E {b['E']:.3g}")
            if tm.get("others"):
                nt.append(f"시간 판 카드 0 다른 프로세스 {tm['others']}")
            if tm and tm.get("err") is not None and b.get("err") is not None and abs(tm["err"] - b["err"]) > 0.01:
                nt.append(f"시간 판 오차 {tm['err']:.2f}")
            if q["state"] != "끝":
                nt.append(q["state"])
            ms = f"{f2(tm.get('ms'))} ({f2(tm.get('ms_max'))})" if tm.get("ms") is not None else ""
            L.append(f"| {nm} | {ns} | {bd} | {f2(b['err'])} | {ms} | {f2(v.get('psnr'))} | {f2(v.get('ssim'), 4)} | {f2(v.get('lpips'), 4)} | {'; '.join(nt)} |")
        L.append("")
    L += ["## 판 전부 (N: 안정 오차 % / x발산 프레임 / 중단)", ""]
    for k, q in sorted(RES.items()):
        L.append(f"- {k}: " + ", ".join(f"{n}: " + (f2(r['err']) if r["stable"] else (f"x{r['div']}" if r["stable"] is False else "중단")) for n, r in sorted(q["probes"].items())))
    open(f"{OUT}/stab_table.md", "w").write("\n".join(L) + "\n")
    tex()
    json.dump(RES, open(f"{OUT}/stab.json", "w"), ensure_ascii=False, indent=1, default=str)


def tx(x):
    return str(x).replace("\\", "/").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&").replace("#", "\\#").replace("−", "$-$")


def tex():
    """논문용 표 (writing/anchorflow/fem_results.tex 가 R2 마운트에서 \\input)."""
    T = ["\\begin{center}\\footnotesize\\begin{tabular}{@{}llrlrrrrr@{}}", "\\toprule",
         "씬 & 방법 & 최소 $N$ & 발산 쪽 경계 & 오차 \\% & ms (최대) & PSNR & SSIM & LPIPS \\\\ \\midrule"]
    for j, t in enumerate(a.scenes.split(",")):
        if j:
            T.append("\\midrule")
        for m, nm in ALL:
            c = [tx(t), tx(nm)]
            q = RES.get(f"{m}_{t}")
            if t not in ELASTIC and m in ELONLY:
                c += ["\\multicolumn{7}{l}{해당 없음 (탄성 전용)}"]
            elif m in TODO:
                c += ["\\multicolumn{7}{l}{" + tx(TODO[m].split(" -- ")[0] if m != "gaussim" else "실행 안 함 (공식 학습 코드 미공개)") + "}"]
            elif q is None or q.get("nmin") is None:
                c += ["\\multicolumn{7}{l}{" + tx("측정 중" if q is None else q.get("state", "")) + "}"]
            else:
                ns, lo = q["nmin"], q.get("lo")
                b = q["probes"][ns]
                bd = "--" if lo is None else (f"{lo} ($\\times${q['probes'][lo]['div']})" if q["probes"][lo]["stable"] is False else f"{lo} (중단)")
                tm, v = q.get("time") or {}, q.get("vis") or {}
                ms = "" if tm.get("ms") is None else f"{tm['ms']:.2f} ({tm['ms_max']:.2f})"
                c += [str(ns) + ("$^\\dagger$" if q.get("mono") and any(s_ is False for _, s_ in q["mono"]) else ""), bd, f2(b["err"]), ms,
                      f2(v.get("psnr")), f2(v.get("ssim"), 3), f2(v.get("lpips"), 3)]
            T.append(" & ".join(c) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}\\end{center}",
          "{\\footnotesize 최소 $N$ = 끝까지 발산하지 않는 최소 프레임당 서브스텝 수 (사다리 다음 정수 이분), 발산 쪽 경계 = 바로 아래 시험한 $N$ (발산 프레임). "
          "$\\dagger$ = $2N$ 또는 $4N$ 에서 발산 (비단조). 오차 = 롤아웃 평균 $|x-x_{gt}|/\\mathrm{EXT}$, ms = A6000 단독 프레임 시간, 시각 품질 = 3DGS 알파블렌딩, 정답 렌더와 같은 카메라."
          + (f" 연기 시험 {a.smoke} 프레임." if a.smoke else "") + "}"]
    open(f"{OUT}/stab_table.tex", "w").write("\n".join(T) + "\n")


say(f"[stab 시작] {D} 방법 {a.methods} 씬 {a.scenes} 마감 {time.strftime('%F %T', time.localtime(a.until))} / {time.strftime('%F %T', time.localtime(a.hard))}")
TH = []
for m in a.methods.split(","):
    for t in a.scenes.split(","):
        if m not in HI or t not in HI[m]:
            continue
        th = threading.Thread(target=one, args=(m, t), daemon=True)
        th.start(); TH.append(th)
        time.sleep(5)
while any(th.is_alive() for th in TH) and time.time() < a.hard:
    table()
    time.sleep(120)
table()
left = [f"{m}_{t}" for (m, t), th in zip([(m, t) for m in a.methods.split(",") for t in a.scenes.split(",") if m in HI and t in HI[m]], TH) if th.is_alive()]
open(f"{OUT}/stab.END", "w").write(f"[stab 끝] {time.strftime('%F %T')} 남은 것 {len(left)} {' '.join(left)}\n")
say(f"[stab 끝] 남은 것 {left}")
os._exit(0)
