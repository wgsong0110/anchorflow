"""[paper_remeasure2 11-15: pr_stab.py 사본 -- 설정 격자 x 3 회 반복 (중앙값) 측정기. 아래는 원본 설명]
[paper_remeasure 11-15: stab.py 사본 -- 우리 = 현재 기본 판 (R2 cmd.txt 원본 명령, exe/pr_ours_tmpl.json), 카드 1 장 (모든 판 직렬, 카드 0 단독)]
안정 최소 서브스텝 표 (doc/pareto.md 절 S): 방법별·씬별 '끝까지 발산하지 않는 최소 프레임당 서브스텝 수' 와 그 서브스텝의 지표.

잡 안에서 (exe/stab.sh 가 부름; cwd = 이 커밋 코드). 단계 (방법 x 씬 하나마다 스레드 하나):
  1. 사다리 N = 1, 2, 4, ... (방법별 상한 hi 까지, 상한은 마지막에 그대로 한 번) -- 처음 안정한 N_s 와 그 앞 N_s/2 (발산) 사이를 정수 이분 -> 최소 N*.
     발산 = 각 실행기의 기존 BD 정의: 비유한·오차 > 2 (롤아웃 평균 아님, 프레임 오차), 정답 MPM 은 서브스텝마다 격자 배열 밖 (--sub_chk). 실행기가 결과 없이 죽으면 '중단' (발산 쪽으로 셈, 표에 따로 적음).
     단조 확인: N* 를 찾은 뒤 2N*, 4N* (상한 이하) 도 돌려 발산이 있으면 표에 '비단조'.
  2. N* 에서 덤프 판 (입자 위치 + F) -> exe/lb_roll_to_h5.py -> exe/lb_render.py (3DGS 알파블렌딩, 정답 렌더와 같은 카메라·같은 3DGS) PNG -> exe/stab_vis.py (PSNR·SSIM·LPIPS alex, 프레임 1.. 평균).
  3. N* 에서 카드 0 단독 시간 판 (/home/dkta/work/gpu0.lock; 시작 때 카드 0 의 다른 계산 프로세스 0 을 기다리고 실행 중 5 초마다 기록).
  공개 코드 베이스라인 (PhysTwin · Simplicits · Spring-Gaus, 절 S.3) 은 사다리 앞에 씬마다 한 번 준비 판 (p/<방법>_<씬>_prep/: 논문 절차 물성 맞춤·
  가중치 학습·다시점 영상 + 3 단계 학습) -- 상한 hi 는 PhysTwin = 맞춤 때 서브스텝 N0 (공식 dt 5e-5), Spring-Gaus = 학습 끝 n_step, Simplicits = 64.
  환경은 exe/stab_bl_setup.sh (stab.sh 가 먼저 부름) 가 ST_* 환경 변수로 넘김.
비측정 판 (1·2) 은 빈 카드 1..7 (카드 0 제외; 계산 프로세스 0 이고 /dev/shm/dkta/anerr_claim/<카드> 의 pid 가 살아 있지 않은 카드 -- 다른 에이전트와 같은 규약).
판 결과는 <out>/p/<방법>_<씬>_n<N>/res.json (있으면 다시 안 돎 -- 잡이 바뀌어도 R2 에서 내려받으면 이어 감).
출력: <out>/stab_table.md, <out>/stab.json, 끝 표식 <out>/stab.END. R2 동기화는 exe/stab.sh.

  python exe/stab.py --out /dev/shm/dkta/lbours/stab --until <epoch> --hard <epoch> [--methods mpm,vrgs,bnv,ours,phystwin,simplicits,springgaus] [--scenes L000,ficus,wolf,bread]
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
ap.add_argument("--methods", default="ours")
ap.add_argument("--scenes", default="L000,ficus,wolf,bread")
ap.add_argument("--cards", default="1 2 3 4 5 6 7")
ap.add_argument("--par", type=int, default=1, help="동시 비측정 판 상한")
ap.add_argument("--jobs", default="")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--smoke", type=int, default=0, help="연기 시험: 프레임 수를 이 값으로 (표는 smoke 표시)")
ap.add_argument("--prep_only", action="store_true", help="[final_pa_bl] 준비 판 (공개 코드 베이스라인 물성·학습, VR-GS E 맞춤) 만 하고 끝 (시간 안 재는 카드용)")
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
          ours=dict(L000=16, ficus=16, wolf=16, bread=16),
          phystwin=dict(L000=None, ficus=None),                       # 준비 판 fit.json 의 N0 (공식 dt 5e-5 를 프레임 간격에 맞춘 서브스텝)
          simplicits=dict(L000=64, ficus=64),                         # 공식 서브스텝 개념 없음 (암시 뉴턴) -- 사다리 끝
          springgaus=dict(L000=None, ficus=None))                     # 준비 판 sg_train.json 의 학습 끝 n_step
NAME = dict(mpm="정답 MPM (PG warp, 공식 config)", vrgs="VR-GS (직접 구현, XPBD 케이지, 반복 11)", bnv="명시적 B-스플라인 (같은 표현, 명시적 적분)",
            ours="우리 (채택 판 (B): 장벽 + --y0eb)", phystwin="PhysTwin (공식 코드, CMA-ES + 경사 학습 물성)",
            simplicits="Simplicits (kaolin 0.18 공식, 기본 가중치 학습)", springgaus="Spring-Gaus (공식 코드, 다시점 영상 3 단계 학습)")
BL = ("phystwin", "simplicits", "springgaus")                          # 공개 코드 베이스라인 (절 S.3)
PYS = os.environ.get("ST_SIMP_PY", "/home/dkta/.conda/envs/simp/bin/python")
BLENV = dict(phystwin=dict(PYTHONPATH=os.environ.get("ST_PT_PYPATH", "")), simplicits=dict(PYTHONPATH=os.environ.get("ST_SIMP_PYPATH", "")),
             springgaus=dict(PYTHONPATH=os.environ.get("ST_SG_PYPATH", "")))
PT_REPO = os.environ.get("ST_PT_REPO", f"{W}/PhysTwin")
SG_REPO = os.environ.get("ST_SG_REPO", f"{W}/Spring-Gaus")
RETRY = int(os.environ.get("ST_RETRY", "4"))                                         # 덤프·시간 판 다시 시도 수 (절 S.8)
VR_IT = 11   # 설정 격자에서는 작업마다 바꿈                                                             # 논문 PG 비교 설정의 반복 수


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


TMPL = json.load(open(f"{D}/exe/pr_ours_tmpl.json"))


def ours_cmd(t, K, mode, O="@@OUT@@"):
    """현재 기본 판 = R2 cmd.txt 원본 (레고·ficus·bread = 절 CM157 A 판, wolf = 절 CM139 Kacdhnz). 걸음 수 K: 1 = 원본 그대로 (적응 쪼개기 --ksplit 3 --ks_big 포함),
    2 이상 = --K n 으로 바꾸고 쪼개기 뺌 (절 CM140 토큰 k2·k4 와 같음). mode: probe|time = 원본, dump = + --dump --dump_F (wolf 는 원본에 이미 있음)."""
    import shlex
    c = TMPL[t]["cmd"]
    c = c.replace(" --out /home/dkta/work/cm157/", " --out @@X@@/").replace(" --out /home/dkta/work/cm139/", " --out @@X@@/")
    q = shlex.split(c)
    env = {}
    while q and "=" in q[0] and not q[0].startswith("-"):
        k_, v_ = q.pop(0).split("=", 1); env[k_] = v_
    i = q.index("--out"); q[i + 1] = O
    if K > 1:
        i = q.index("--K"); q[i + 1] = str(K)
        for f_ in ("--ksplit", "--ks_big"):
            if f_ in q:
                j = q.index(f_); q.pop(j)
                if f_ == "--ksplit":
                    q.pop(j)                                                   # 값
    if mode == "dump":
        for f_ in ("--dump", "--dump_F"):
            if f_ not in q:
                q.append(f_)
    if a.smoke:
        i = q.index("--frames"); q[i + 1] = str(a.smoke)
        env["AF_SP_FR"] = f"1,{a.smoke}"
    q[0] = PY
    return q, env


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
        q_, env_ = ours_cmd(t, N, mode, O)
        return q_, D, env_
    if m in BL:
        F = pdir(m, t, 0, "prep")
        x = ["--dump", "--dump_F"] if mode == "dump" else []
        if m == "phystwin":
            return [PY, "-u", f"{D}/exe/stab_phystwin.py", "--mode", "run", "--phystwin", PT_REPO, "--fit", F, "--N", str(N), "--out", O] + da_elastic(t) + x, D, BLENV[m]
        if m == "simplicits":
            return [PYS, "-u", f"{D}/exe/stab_simplicits.py", "--mode", "run", "--fit", F, "--N", str(N), "--out", O] + da_elastic(t) + x, D, BLENV[m]
        return [PY, "-u", f"{D}/exe/stab_sg.py", "--mode", "run", "--sg", SG_REPO, "--fit", F, "--N", str(N), "--out", O] + da_elastic(t) + x, D, BLENV[m]
    raise KeyError(m)


def prep_cmds(m, t, F):
    """공개 코드 베이스라인 준비 판 명령들 (차례로; 각 실행기가 끝난 단계는 건너뜀). 연기 시험은 반복 수만 줄임."""
    if m == "phystwin":
        sm = ["--cma_iter", "1", "--train_iters", "1"] if a.smoke else []
        return [[PY, "-u", f"{D}/exe/stab_phystwin.py", "--mode", "fit", "--phystwin", PT_REPO, "--out", F] + da_elastic(t) + sm]
    if m == "simplicits":
        return [[PYS, "-u", f"{D}/exe/stab_simplicits.py", "--mode", "weights", "--out", F] + da_elastic(t)]
    gd, cf = gt_render_dir(t)
    sm = ["--smoke", "1"] if a.smoke else []
    return [[PY, "-u", f"{D}/exe/stab_mv_render.py", "--sim", gd, "--config", cf, "--shape", GT[t][2], "--frames", str(a.smoke or FRAMES[t]), "--out", f"{F}/mv"],
            [PY, "-u", f"{D}/exe/stab_sg.py", "--mode", "prep", "--sg", SG_REPO, "--fit", F, "--mv", f"{F}/mv", "--out", F] + da_elastic(t) + sm,
            [PY, "-u", f"{D}/exe/stab_sg.py", "--mode", "train", "--sg", SG_REPO, "--fit", F, "--out", F] + da_elastic(t)]


def prep(m, t, tmo=float(os.environ.get("PR2_PREP_TMO_H", "16")) * 3600):
    """씬마다 한 번 (빈 카드 하나). 반환: 상한 hi (실패·마감이면 None). 결과 p/<방법>_<씬>_prep/res.json."""
    F = pdir(m, t, 0, "prep")
    if os.path.exists(f"{F}/res.json"):
        h = json.load(open(f"{F}/res.json")).get("hi")
        if h is not None:
            return h                                                                   # 실패 판 (hi 없음) 은 다시 띄울 때 다시 (환경 고친 뒤)
    if time.time() >= a.until:
        return None
    with SEM:
        G = claim_card()
        try:
            os.makedirs(F, exist_ok=True)
            e = dict(os.environ, CUDA_VISIBLE_DEVICES=G, PYTHONUTF8="1", OMP_NUM_THREADS="8", WANDB_MODE="disabled",
                     WARP_CACHE_PATH=f"/dev/shm/dkta/warp_cache_{G}_{m}")
            if BLENV[m]["PYTHONPATH"]:
                e["PYTHONPATH"] = BLENV[m]["PYTHONPATH"]
            os.makedirs(e["WARP_CACHE_PATH"], exist_ok=True)
            t0, rc = time.time(), 0
            for k, cmd in enumerate(prep_cmds(m, t, F)):
                open(f"{F}/card.txt", "a").write(f"카드 {G} {time.strftime('%F %T')} {' '.join(cmd)}\n")
                with open(f"{F}/log_{k}.txt", "a") as lg:
                    try:
                        rc = subprocess.run(cmd, cwd=D, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT,
                                            timeout=max(60, tmo - (time.time() - t0))).returncode
                    except subprocess.TimeoutExpired:
                        rc = "timeout"
                if rc != 0:
                    break
            hi = None
            try:
                if m == "phystwin":
                    hi = int(json.load(open(f"{F}/fit.json"))["N0"])
                elif m == "springgaus":
                    hi = int(json.load(open(f"{F}/sg_train.json"))["n_step"])
                elif os.path.exists(f"{F}/weights.json"):
                    hi = HI[m][t]
            except (OSError, ValueError, KeyError):
                pass
            r = dict(rc=rc, hi=(hi if rc == 0 else None), sec=time.time() - t0, card=G)
            if rc == 0 and hi is not None:
                json.dump(r, open(f"{F}/res.json", "w"), ensure_ascii=False, indent=1)
            say(f"[준비] {m} {t} 카드 {G}: rc={rc} 상한 {hi} ({r['sec'] / 3600:.2f} h)")
            return r["hi"]
        finally:
            release_card(G)


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


_FD = {}


def claim_card():
    """카드 1 장 (0): gpu0.lock flock 으로 직렬 -- 비측정 판·렌더도 카드 0 을 단독으로 씀 (시간 판과 겹치지 않음)."""
    fd = open(f"{W}/gpu0.lock", "w")
    fcntl.flock(fd, fcntl.LOCK_EX)
    _FD[threading.get_ident()] = fd
    return "0"


def release_card(i):
    fd = _FD.pop(threading.get_ident(), None)
    if fd:
        fcntl.flock(fd, fcntl.LOCK_UN); fd.close()


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



def pdir(m, t, N, mode="probe"):
    if mode == "prep":
        return f"{OUT}/p/{m}_{t}_prep"
    return f"{OUT}/p/{m}_{t}_n{N}" + ("" if mode == "probe" else "_" + mode)


# ---------- paper_remeasure2: PhysGaussian 논문 장면 추가 (vasedeck 탄성 · plane 금속 소성 · pillow2sofa 충돌) ----------
# 정답 = 공식 config 그대로 만든 궤적 (R2 lbours/paper_remeasure2/scenes/<씬>/, exe/pr2_gt.sh). 우리 판 명령 = 가장 가까운 기존 씬의 현재 기본 판 명령에서 씬 의존 부분만 바꿈:
#   vasedeck (jelly, 중력 0, 회전 구동기) <- ficus 명령 (region_density 뺌) · plane (metal) <- bread 명령 · pillow2sofa (jelly, cuboid 충돌) <- ficus 명령 (region_density 뺌)
NS = dict(vasedeck=dict(fr=75, base="ficus", shape="vasedeck", drop=["--region_density"]),
          plane=dict(fr=400, base="bread", shape="plane", drop=[]),
          pillow2sofa=dict(fr=300, base="ficus", shape="pillow2sofa", drop=["--region_density"]))
NW = "/home/dkta/work/newsc"
for _t, _s in NS.items():
    FRAMES[_t] = _s["fr"]
    DATA[_t] = f"{NW}/{_t}.pt"
    GT[_t] = (f"{NW}/{_t}_h5", f"{NW}/{_t}_config.json", _t)
    import re as _re
    _c = TMPL[_s["base"]]["cmd"]
    _c = _c.replace("exe/startpt_wrap.py --sp 1", "exe/rotbc_wrap.py --rot_bc 1 exe/startpt_wrap.py --sp 1", 1)             # 회전 구동기·정지 영역 구속 (exe/rotbc_wrap.py; 이 두 구속이 없는 씬은 결과 불변)
    _c = _re.sub(r"AF_SP_FR=1,\d+", f"AF_SP_FR=1,{_s['fr']}", _c)
    _c = _re.sub(r"--rr_shape \S+", f"--rr_shape {_t}", _c)
    _c = _re.sub(r"--rr_traj \S+", f"--rr_traj {NW}/gt_{_t}", _c)
    _c = _re.sub(r"--rr_cfg \S+", f"--rr_cfg {NW}/gt_{_t}/config.json", _c)
    _c = _re.sub(r"--data \S+", f"--data {NW}/{_t}.pt", _c)
    _c = _re.sub(r"--frames \d+", f"--frames {_s['fr']}", _c)
    _c = _re.sub(r"--tag \S+", f"--tag {_t}", _c)
    for _f in _s["drop"]:
        _c = _c.replace(" " + _f, "")
    TMPL[_t] = dict(cmd=_c)

_g_old = gt_render_dir


def gt_render_dir(t):                                                                # noqa: F811
    if t in NS:
        return f"{NW}/gt_{t}", f"{NW}/gt_{t}/config.json"
    return _g_old(t)


HI["simplicits"]["vasedeck"] = 64
_da_old = da_elastic


def da_elastic(t):                                                                   # noqa: F811
    if t in NS:
        return ["--data", DATA[t], "--tag", t, "--frames", str(a.smoke or FRAMES[t]), "--floor", "none", "--cfg_bc", "--pg_vdamp"]
    return _da_old(t)


# ---------- paper_remeasure2: 설정 격자 x 반복 ----------
VR_E = dict(L000=245525.5146265931, ficus=164393654.74356765)   # E = 논문 PG 비교 설정 (L000 167 x 11, ficus 400 x 11) 에서 논문 절차 (황금분할 10 회) 로 맞춘 값, 격자 내내 고정
JOBS = []
GRID = f"{OUT}/g"
os.makedirs(GRID, exist_ok=True)


def one_run(m, t, N, rep):
    O = f"{GRID}/{m}_{t}_n{N}_r{rep}"
    if os.path.exists(f"{O}/res.json"):
        return json.load(open(f"{O}/res.json"))
    os.makedirs(O, exist_ok=True)
    fd = open(f"{W}/gpu0.lock", "w")
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        u0 = smi("gpu=index,uuid").splitlines()[0].split(",")[1].strip()

        def others():
            return [ln.split(",")[1].strip() for ln in smi("compute-apps=gpu_uuid,pid").splitlines() if ln.split(",")[0].strip() == u0]
        while others():
            time.sleep(5)
        cmd, cwd, env = build(m, t, N, "time", O, VR_E.get(t) if m == "vrgs" else None)
        e = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUTF8="1", OMP_NUM_THREADS="8", WARP_CACHE_PATH="/dev/shm/dkta/warp_cache_0", **env)
        os.makedirs(e["WARP_CACHE_PATH"], exist_ok=True)
        open(f"{O}/card.txt", "w").write(f"시작 때 다른 프로세스 {len(others())} 개 {time.strftime('%F %T')} {' '.join(cmd)}\n")
        seen = set()
        with open(f"{O}/log.txt", "w") as lg:
            p = subprocess.Popen(cmd, cwd=cwd, env=e, stdin=subprocess.DEVNULL, stdout=lg, stderr=subprocess.STDOUT)
            t0 = time.time()
            while p.poll() is None:
                seen |= set(others())
                if time.time() - t0 > 7200:
                    p.kill()
                time.sleep(5)
        nf = max(len(seen) - 1, 0)
        r = parse(m, O)
        r.update(rc=p.returncode, N=N, rep=rep, others=nf, sec=time.time() - t0)
        json.dump(r, open(f"{O}/res.json", "w"), ensure_ascii=False, indent=1)
        say(f"[격자] {m} {t} N {N} r{rep}: err {r['err']} ms {r['ms']} 발산 {r['div']} (다른 {nf})")
        return r
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); fd.close()


def collect():
    import statistics as S
    out = []
    for j in JOBS:
        rs = [one_run_cached(j["m"], j["t"], j["N"], k) for k in range(1, a.reps + 1)]
        rs = [r for r in rs if r]
        ok = [r for r in rs if r.get("stable") and r.get("ms") is not None]
        out.append(dict(method=j["m"], scene=j["t"], N=j["N"], reps=len(rs), ok=len(ok),
                        err=S.median([r["err"] for r in ok]) if ok else None, err_all=[r["err"] for r in ok],
                        ms=S.median([r["ms"] for r in ok]) if ok else None, ms_all=[r["ms"] for r in ok],
                        div=[r.get("div") for r in rs if not r.get("stable")]))
    json.dump(out, open(f"{OUT}/grid.json", "w"), ensure_ascii=False, indent=1)


def one_run_cached(m, t, N, k):
    f = f"{GRID}/{m}_{t}_n{N}_r{k}/res.json"
    return json.load(open(f)) if os.path.exists(f) else None


VR_REF = dict(vasedeck=400, pillow2sofa=200)                                         # VR-GS E 를 맞추는 기준 서브스텝 (프레임 간격 / 논문 비교 dt 1e-4), 반복 11


def ensure_E(t):
    if t in VR_E:
        return
    N = VR_REF[t]
    O = f"{GRID}/vrgs_{t}_fit"
    if not os.path.exists(f"{O}/res.json"):
        r = run_free("vrgs", t, N, "probe", O)
    r = json.load(open(f"{O}/res.json"))
    VR_E[t] = r["E"]
    say(f"[VR-GS E] {t}: 논문 절차 (황금분할 10 회, N {N} x 반복 11) E = {r['E']}")


if a.jobs:
    for m, t, Ns in json.load(open(a.jobs)):
        if Ns == "hi":                                                     # [final_pa_bl] 대표 점 하나: 준비 판의 공식 서브스텝 (PhysTwin N0 · Spring-Gaus 학습 끝 n_step)
            hi = prep(m, t)
            if hi is None:
                say(f"[준비 실패] {m} {t}"); continue
            Ns = [hi]
        if Ns == "auto":                                                   # 공개 코드 베이스라인: 준비 판 (논문 절차 물성) 뒤 공식 서브스텝 hi 의 1/4, 1/2, 1 배
            hi = prep(m, t)
            if hi is None:
                say(f"[준비 실패] {m} {t}"); continue
            Ns = [1, 4, 16, 64] if m == "simplicits" else sorted({max(1, hi // 4), max(1, hi // 2), hi})
        if m == "vrgs":
            ensure_E(t)
        JOBS += [dict(m=m, t=t, N=N) for N in Ns]
    if a.prep_only:
        json.dump(JOBS, open(f"{OUT}/prep_jobs.json", "w"), ensure_ascii=False, indent=1)
        open(f"{OUT}/prep.END", "w").write(time.strftime("%F %T") + "\n")
        say("[준비만 끝]")
        os._exit(0)
    for k in range(1, a.reps + 1):                                     # 반복을 바깥으로: 같은 설정이 시간상 떨어져 측정됨 (워밍업·열 효과 분산)
        for j in JOBS:
            if time.time() >= a.until:
                break
            try:
                one_run(j["m"], j["t"], j["N"], k)
            except Exception as ex:                                    # noqa: BLE001
                say(f"[오류] {j} r{k}: {ex}")
            collect()
    collect()
    open(f"{OUT}/grid.END", "w").write(time.strftime("%F %T") + "\n")
    say("[격자 끝]")
os._exit(0)
