"""A5 서브스텝 안정성 (PG 공식 씬 plane·pillow2sofa) 완전 자동 이어 돌리기 드라이버 (사용자 10-23).

계산 노드 잡 안에서만 돈다 (exe/stab_a5_go.sh -> exe/job_run.sh). 잡이 끝나 죽어도 같은 명령으로 다시 띄우면 이어진다.
상태는 전부 파일에서 다시 읽는다 (드라이버 자체는 상태를 들고 있지 않다):
  실행 단위 하나 = 결과 폴더의 접두사 P (예: stab/plane/scan/pg_N4). 끝 표식 = P.done (json: rc, 시각).
  표식이 없으면 다시 띄운다: PG·i-PG 는 <out>.ckpt.pt 에서, 우리는 P.parts/ 조각(AF_ROLL_PART)에서 이어 돌고,
  GF 는 짧아서(분 단위) 처음부터 다시 돈다.
  잡이 끝나 죽은 실행(신호로 종료)·메모리 부족은 끝 표식을 남기지 않는다 -> 다음 기동 때 이어 돈다.
  그 밖의 비정상 종료는 발산으로 끝 표식 (stab_metrics.load 가 로그의 오류를 발산으로 읽는다 -- A1~A3 와 같다).

절차 (A1~A3 와 같은 규칙, stab_scan2.sh·stab_robust.sh·stab_vis2.sh):
  0. 기준 궤적 PG N1024 (공식 config 전 프레임: plane 400, pillow2sofa 300) -- 남아 있는 체크포인트에서 이어 돈다.
  1. 2 의 거듭제곱 N = 1, 2, 4, ..., 128 을 작은 N 부터 차례로, 처음 안정한 N(hi) 에서 멈춘다 (128 까지 안정 없으면 "> 128").
  2. 직전 발산한 거듭제곱 lo 와 hi 사이 정수를 아래부터 올라가며 처음 안정한 N (스캔 최소 N). 결과가 같은 범위에서
     위쪽 N 을 SPEC 개까지 미리 같이 돌린다 (빈 카드가 있을 때만 -- 판정은 "아래부터 처음 안정" 그대로).
  3. 반복 확정: 같은 N 을 3 번 모두 안정이면 확정 (스캔 실행 1 회 포함, robust/r1..r3). 하나라도 발산하면 N+1.
  4. 시각 지표: 확정 N 의 전체 입자 덤프를 같은 렌더러(rep_ip, F 반영 3DGS)로 -> PSNR/SSIM/LPIPS (기준 PG N1024 렌더).
     i-PG 는 반복 확정 r1 실행이 덤프를 남긴다 (다시 안 돌림). 우리는 스캔/반복 실행의 롤아웃 덤프를 그대로 렌더.
  5. 보고: exe/stab_a5_report.py -> stab/a5/ (json, LaTeX 조각, 그림). 끝나면 stab/a5/STAB_A5_DONE.
  판정 = exe/stab_is_stable.py (바닥 없는 씬: 총에너지 (E-E_ref)/E_ref ≤ 1, E_ref = PG N1024 최대 총에너지, 전 프레임 완주).
  시작 전 이어 돌리기 자가 시험 (실제로 끊고 이어 돌려 끊지 않은 실행과 비교): 우리·i-PG. 통과해야 그 방법을 띄운다.

카드: 0~6 중 메모리 < 500 MiB (다른 실행이 없는 카드) 만 새로 잡는다. GPU 0 부터, 그 다음 6, 5, ... (대기열 러너는
GPU 0 을 안 쓰고 1 부터 고른다 -- 겹칠 일을 줄인다). 잡은 카드에는 가벼운 실행을 둘까지 같이 올린다.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 이 커밋의 워크트리
W = "/home/dkta/work"
ST = f"{W}/stab"
A5 = f"{ST}/a5"
PGDIR = f"{W}/i-physgaussian"
PY = sys.executable
SCENES = {"plane": 400, "pillow2sofa": 300}
METHODS = ("pg", "gf", "ipg", "ovs")
POW = (1, 2, 4, 8, 16, 32, 64, 128)
SPEC = int(os.environ.get("A5_SPEC", "2"))                # 정수 스캔에서 미리 같이 돌릴 개수
PART_EVERY = 10                                           # 우리 롤아웃 조각 간격 (프레임)
FREE_MIB = 500
GPU_ORDER = (0, 6, 5, 4, 3, 2, 1)
STALL_S = 3 * 3600                                        # 로그가 이만큼 안 바뀌면 멈춘 것으로 (발산 처리)
ENV = dict(PYTHONUTF8="1", AF_IPG_FUSED="1", AF_TI_MEM="0.2", OMP_NUM_THREADS="2")
OVS_FLAGS = ("--out_var --ov_init predict --ov_opt lbfgs --ov_roll 500 --no_mat --arch sgnn --gnn_layers 1 --hidden 128 "
             "--obj pts --bc_level particle --bc_mode barrier --bc_kappa 1.0 --bc_ext linear --bc_mpm --inv_barrier 1.0 "
             "--inv_jhat 0.3 --inv_ext linear --node_h 0.028284 --dt_cond --dt_scale --v_from_dt --v0_traj --det_eps 0.1 "
             "--det_w 100 --lr 3e-4 --batch 8 --det_every 200 --pool_drop 0.05,0.30,0.0,1.5 --pool_frames 80 "
             "--phase2 --phys_w 1.0 --iters 0 --eval_t0 0 --save_every 100000 --tb none --cfg_bc")
os.makedirs(A5, exist_ok=True)
LOG = open(f"{A5}/driver.log", "a", buffering=1)


def log(*s):
    LOG.write(time.strftime("%m-%d %H:%M:%S ") + " ".join(str(q) for q in s) + "\n")


# ------------------------------------------------------------------------------------------------ 실행 단위
class Unit:
    """key: 고유 이름, pre: 결과 접두사 (P.done 표식), cmd(gpu, resume_info) -> 셸 명령, heavy: 카드 하나 통째."""

    def __init__(self, key, pre, kind, S, m=None, N=None, heavy=False, prio=5):
        self.key, self.pre, self.kind, self.S, self.m, self.N, self.heavy, self.prio = key, pre, kind, S, m, N, heavy, prio

    @property
    def done(self):
        return os.path.exists(self.pre + ".done")

    @property
    def log(self):
        return self.pre + ".log"


def O(S):
    return f"{ST}/{S}"


def mpm_cmd(S, m, N, out_npz, g, dump="", ckpt=5, frames=0):
    o = O(S)
    # i-PG 통합 커널(AF_IPG_FUSED)은 탄성(재질 0)만 지원 -> metal(plane)은 공식 Warp 경로 (10-23 자가 시험 AssertionError)
    fz = " AF_IPG_FUSED=0" if (m == "ipg" and S == "plane") else ""
    c = (f"cd {PGDIR} && CUDA_VISIBLE_DEVICES={g}{fz} {PY} -u {WT}/exe/stab_mpm.py --method {m} --h5 {o}/init/sim_0000000000.h5 "
         f"--config {o}/config.json --substeps {N} --out {out_npz} --ckpt_every {ckpt}")
    if dump:
        c += f" --dump {dump}"
    if frames:
        c += f" --frames {frames}"
    return c


def gf_cmd(S, N, pre, g, vis=False):
    o = O(S)
    if vis:   # 전체 입자 h5 (렌더용), 10 프레임마다 이어 돌릴 상태
        return (f"cd {WT} && CUDA_VISIBLE_DEVICES={g} {PY} -u exe/gf_mpm.py --config {o}/config.json --h5 {o}/init/sim_0000000000.h5 "
                f"--out {pre} --flip on --substeps {N} --resume")
    return (f"cd {WT} && CUDA_VISIBLE_DEVICES={g} {PY} -u exe/gf_mpm.py --config {o}/config.json --h5 {o}/init/sim_0000000000.h5 "
            f"--out {pre}_tmp --flip on --substeps {N} --stab_out {pre}.npz")


def parts_count(pdir, k=PART_EVERY):
    """조각 폴더에서 처음부터 이어진 프레임 수."""
    n = 0
    while os.path.exists(f"{pdir}/c{n + k:05d}.pt"):
        n += k
    return n


def ovs_cmd(S, N, pre, g, frames=None, part_every=PART_EVERY, parts=True):
    o, R = O(S), f"{W}/stab_ovs/{S}"
    F = frames or SCENES[S]
    s = parts_count(pre + ".parts", part_every) if parts else 0
    cont = f"AF_OV_CONT={pre}.parts:{s} " if s > 0 else ""
    pp = f"AF_ROLL_PART={pre}.parts AF_ROLL_PART_EVERY={part_every} " if parts else ""
    return (f"cd {WT} && {cont}{pp}AF_ROLL_DUMP={pre}.dump.pt "
            f"AF_ROLL_DUMP_F=32 AF_ROLL_TAG=0000 AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES={g} {PY} -u exe/train_deform.py "
            f"--data {R}/d_ref --hold_traj 0000 --out {pre}.run --tag OVS {OVS_FLAGS} "
            f"--pool_fill {R}/pool/fill --pool_cfg {R}/pool/cfg --pool_combos {S}_jelly "
            f"--eval_len {F} --eval_dt_sub {N} && "
            f"{PY} -u exe/ovs_stab.py conv --scene {o} --seq {pre}.dump.pt --out {pre}.npz --substeps {N}")


def sim_cmd(u, g):
    """스캔·반복·기준 단위의 명령."""
    if u.m == "gf":
        return gf_cmd(u.S, u.N, u.pre, g)
    if u.m == "ovs":
        return ovs_cmd(u.S, u.N, u.pre, g)
    dump = ""
    if u.m == "ipg" and "/robust/r1/" in u.pre:      # 반복 확정 r1 = 시각 지표용 전체 입자 덤프
        dump = u.pre + "_dump"
    return mpm_cmd(u.S, u.m, u.N, u.pre + ".npz", g, dump=dump, ckpt=5 if u.m == "ipg" or u.N >= 1024 else 50)


# ------------------------------------------------------------------------------------------------ 판정
_VCACHE = {}


def verdict(d, m, N, S):
    """1 안정 / 0 발산 / None (아직). 끝 표식이 있어야 판정한다."""
    pre = f"{d}/{m}_N{N}"
    if not os.path.exists(pre + ".done"):
        return None
    k = (pre, os.path.getmtime(pre + ".done"))
    if k in _VCACHE:
        return _VCACHE[k]
    r = subprocess.run([PY, f"{WT}/exe/stab_is_stable.py", d, m, str(N), O(S)], capture_output=True, text=True, cwd=WT)
    v = (r.stdout.strip().splitlines() or ["-"])[-1]
    v = {"1": 1, "0": 0}.get(v)
    if v is None:
        log("[판정 실패]", pre, r.stdout[-200:], r.stderr[-300:])
        v = 0 if os.path.exists(pre + ".done") and not os.path.exists(pre + ".npz") else None
    if v is not None:
        _VCACHE[k] = v
    return v


def ref_done(S):
    return os.path.exists(f"{O(S)}/res/pg_N1024.done")


# ------------------------------------------------------------------------------------------------ 사슬 (형상 × 방법)
def chain(S, m, want):
    """현재 파일 상태에서 이 사슬이 필요로 하는 단위를 want 에 넣고, 확정 N (또는 '>128') / None 을 돌려준다."""
    if not ref_done(S):
        return None
    if m in ("ovs", "ipg") and not selftest_ok(m):
        return None
    sc = f"{O(S)}/scan"
    heavy = m == "ovs" and S == "pillow2sofa"
    pr = {"ipg": 1, "ovs": 2, "pg": 4, "gf": 4}[m]
    mk = lambda d, N, k="scan": Unit(f"{S}/{k}/{m}/{N}", f"{d}/{m}_N{N}", "sim", S, m, N, heavy, pr)
    lo, hi = 0, None
    for N in POW:
        v = verdict(sc, m, N, S)
        if v is None:
            want.append(mk(sc, N)); return None
        if v == 1:
            hi = N; break
        lo = N
    if hi is None:
        return ">128"
    nmin, pend = None, 0
    for N in range(lo + 1, hi):
        v = verdict(sc, m, N, S)
        if v == 1:
            if pend == 0:
                nmin = N
            break
        if v is None:
            want.append(mk(sc, N)); pend += 1
            if pend >= SPEC:
                break
    if pend:
        return None
    if nmin is None:
        nmin = hi
    N = nmin
    while N <= POW[-1]:
        vs = verdict(sc, m, N, S)
        reps = (1, 2) if vs == 1 else (1, 2, 3)
        if vs == 0:
            N += 1; continue
        rv = [verdict(f"{O(S)}/robust/r{k}", m, N, S) for k in reps]
        if 0 in rv:
            N += 1; continue
        miss = [k for k, v in zip(reps, rv) if v is None]
        if miss:
            for k in miss:
                os.makedirs(f"{O(S)}/robust/r{k}", exist_ok=True)
                want.append(mk(f"{O(S)}/robust/r{k}", N, f"robust/r{k}"))
            return None
        return N
    return ">128"


# ------------------------------------------------------------------------------------------------ 이어 돌리기 자가 시험
STD = f"{A5}/selftest"


def selftest_ok(m):
    return os.path.exists(f"{STD}/{m}.pass")


def selftest_units(want):
    """우리: 6 프레임 롤아웃을 프레임 4 조각이 생기면 실제로 끊고(kill) 이어 돌려, 끊지 않은 실행과 비교.
    i-PG: 8 프레임을 체크포인트(2 프레임 간격)가 프레임 4 를 넘으면 끊고 이어 돌려 비교. plane, N1."""
    os.makedirs(STD, exist_ok=True)
    for m in ("ovs", "ipg"):
        if selftest_ok(m) or os.path.exists(f"{STD}/{m}.fail"):
            continue
        for tag in ("ref", "cut"):
            pre = f"{STD}/{m}_{tag}"
            if not os.path.exists(pre + ".done"):
                want.append(Unit(f"selftest/{m}/{tag}", pre, f"self_{m}_{tag}", "plane", m, 1, False, 0))


def selftest_cmd(u, g):
    if u.m == "ovs":   # 끊지 않는 쪽은 조각도 안 쓴다 (순수한 한 번 실행)
        return ovs_cmd("plane", 1, u.pre, g, frames=8, part_every=2, parts=u.kind.endswith("cut"))
    return mpm_cmd("plane", "ipg", 1, u.pre + ".npz", g, ckpt=2 if u.kind.endswith("cut") else 0, frames=8)


def selftest_cut_ready(u):
    """끊을 때가 됐나: 우리 = 조각 c00004 가 생김, i-PG = 체크포인트가 프레임 4 이상."""
    if u.m == "ovs":
        return os.path.exists(f"{u.pre}.parts/c00004.pt")
    import re
    try:   # tqdm 이 5/8 이상이면 프레임 4 체크포인트가 이미 저장됐다
        hits = re.findall(r"\| (\d+)/8 \[", open(u.log, errors="ignore").read())
        return bool(hits) and int(hits[-1]) >= 5 and os.path.exists(u.pre + ".npz.ckpt.pt")
    except Exception:                                     # noqa: BLE001
        return False


def selftest_judge():
    for m in ("ovs", "ipg"):
        a_, b_ = f"{STD}/{m}_ref", f"{STD}/{m}_cut"
        if selftest_ok(m) or os.path.exists(f"{STD}/{m}.fail") or not (os.path.exists(a_ + ".done") and os.path.exists(b_ + ".done")):
            continue
        try:
            if m == "ovs":
                import torch
                A = torch.load(a_ + ".dump.pt", map_location="cpu", weights_only=False)["pred"].float().numpy()
                B = torch.load(b_ + ".dump.pt", map_location="cpu", weights_only=False)["pred"].float().numpy()
                sm = f"{ST}/plane/smoke/ovs_N1.dump.pt"     # 780da52 의 3 프레임 시험 (코드 판 사이 대조)
                C = torch.load(sm, map_location="cpu", weights_only=False)["pred"].float().numpy() if os.path.exists(sm) else None
            else:
                A, B, C = np.load(a_ + ".npz")["x"], np.load(b_ + ".npz")["x"], None
            L = float(np.linalg.norm(A[0].max(0) - A[0].min(0)))
            T = min(len(A), len(B))
            d = [float(np.linalg.norm(A[t] - B[t], axis=-1).mean() / L) for t in range(T)]
            rec = dict(frames=T, mean_dist_over_L=d, max=max(d), resumed=open(b_ + ".log", errors="ignore").read().count("이어"))
            if C is not None:
                rec["vs_780da52_3f"] = [float(np.linalg.norm(A[t] - C[t], axis=-1).mean() / L) for t in range(min(3, len(C)))]
            ok = T == (8 if m == "ovs" else 9) and max(d) < 1e-3 and rec["resumed"] > 0
            json.dump(rec, open(f"{STD}/{m}.json", "w"), indent=1)
            open(f"{STD}/{m}.{'pass' if ok else 'fail'}", "w").write(json.dumps(rec) + "\n")
            log(f"[자가 시험] {m} {'통과' if ok else '실패'}", json.dumps(rec)[:400])
        except Exception as e:                            # noqa: BLE001
            open(f"{STD}/{m}.fail", "w").write(repr(e)); log("[자가 시험] 비교 오류", m, repr(e))


# ------------------------------------------------------------------------------------------------ 시각 지표
def vis_units(S, nconf, want):
    """확정 N 이 모두 나온 형상: 렌더용 덤프 -> rep_ip 렌더 -> stab_vis. 끝나면 stab/<S>/vis/vis_a5.json."""
    V = f"{O(S)}/vis"; os.makedirs(V, exist_ok=True)
    items = [("pg", 1024)] + [(m, nconf[m]) for m in METHODS if isinstance(nconf[m], int)]
    pngs, ready = [], True
    for m, N in items:
        if m in ("pg", "gf"):
            pre = f"{V}/{m}_N{N}"
            if not os.path.exists(pre + ".done"):
                want.append(Unit(f"{S}/visdump/{m}/{N}", pre, "visdump", S, m, N, False, 3)); ready = False; continue
            src = pre + "_dump" if m == "pg" else pre
        elif m == "ipg":
            src = f"{O(S)}/robust/r1/ipg_N{N}_dump"
        else:   # ovs: 확정 N 의 스캔 실행 덤프 (없으면 r1)
            p = f"{O(S)}/scan/ovs_N{N}"
            src = (p if os.path.exists(p + ".dump.pt") else f"{O(S)}/robust/r1/ovs_N{N}") + ".dump.pt"
        rpre = f"{V}/{m}_N{N}_render"
        if not os.path.exists(rpre + ".done"):
            u = Unit(f"{S}/render/{m}/{N}", rpre, "render", S, m, N, False, 3); u.src = src
            want.append(u); ready = False; continue
        pngs.append((m, N, f"{V}/png_{m}_N{N}"))
    if not ready:
        return False
    if os.path.exists(f"{V}/vis_a5.done"):
        return True
    u = Unit(f"{S}/vismetric", f"{V}/vis_a5", "vismetric", S, None, None, False, 3); u.pngs = pngs
    want.append(u)
    return False


def vis_cmd(u, g):
    S, o, V, F = u.S, O(u.S), f"{O(u.S)}/vis", SCENES[u.S]
    if u.kind == "visdump":
        if u.m == "gf":
            return gf_cmd(S, u.N, u.pre, g, vis=True)
        return mpm_cmd(S, "pg", u.N, u.pre + ".npz", g, dump=u.pre + "_dump", ckpt=20)
    if u.kind == "render":
        rp = f"{V}/{u.m}_N{u.N}"
        if u.m == "ovs":
            conv = (f"{PY} - <<'PYEOF'\nimport torch, numpy as np\nq = torch.load('{u.src}', map_location='cpu', weights_only=False)\n"
                    f"X = q['pred'].float().numpy(); d = dict(traj_full=X)\n"
                    f"if q.get('F') is not None: d['F_full'] = q['F'].float().numpy()[-X.shape[0]:].astype(np.float32)\n"
                    f"np.savez('{rp}_full.npz', **d)\nPYEOF\n")
            return (f"cd {WT} && {conv} CUDA_VISIBLE_DEVICES={g} {PY} -u exe/rep_ip.py --sim {o}/init --cfg {o}/config.json --shape {S} "
                    f"--method ours --frames {F} --no_ref --render_npz {rp}_full.npz --video {rp}.mp4 --png_dir {V}/png_{u.m}_N{u.N} --out {rp}.r.npz")
        return (f"cd {WT} && CUDA_VISIBLE_DEVICES={g} {PY} -u exe/rep_ip.py --sim {u.src} --cfg {o}/config.json --shape {S} --method ours "
                f"--frames {F} --render_ref --video {rp}.mp4 --png_dir {V}/png_{u.m}_N{u.N} --out {rp}.r.npz")
    name = {"pg": "PG", "gf": "GF", "ipg": "iPG", "ovs": "ours"}
    runs = " ".join(f"{name[m]}_N{N}={p}" for m, N, p in u.pngs if not (m == "pg" and N == 1024))
    return f"cd {WT} && CUDA_VISIBLE_DEVICES={g} {PY} -u exe/stab_vis.py --ref {V}/png_pg_N1024 --runs {runs} --out {V}/vis_a5.json"


# ------------------------------------------------------------------------------------------------ 카드·실행
RUN = {}          # key -> dict(unit, proc, gpu, t0, cut)
R2_LAST = [0.0]


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return {int(a): int(b) for a, b in (ln.split(",") for ln in out.strip().splitlines())}
    except Exception as e:                                # noqa: BLE001
        log("[nvidia-smi 실패]", repr(e)); return {}


def pick_gpu(u, mem):
    mine = {}
    for r in RUN.values():
        mine.setdefault(r["gpu"], []).append(r["unit"])
    for g in GPU_ORDER:                                   # 빈 카드 먼저
        if g not in mine and mem.get(g, 99999) < FREE_MIB:
            return g
    if u.heavy:
        return None
    for g in GPU_ORDER:                                   # 내 카드에 가벼운 것 둘까지
        us = mine.get(g, [])
        if 0 < len(us) < 2 and not any(q.heavy for q in us):
            return g
    return None


def launch(u, g):
    if u.kind == "sim":
        cmd = sim_cmd(u, g)
    elif u.kind.startswith("self_"):
        cmd = selftest_cmd(u, g)
    else:
        cmd = vis_cmd(u, g)
    os.makedirs(os.path.dirname(u.pre), exist_ok=True)
    lg = u.log
    if os.path.exists(lg):                                # 지난 시도의 로그는 지우지 않고 옆으로 (판정이 옛 오류를 읽지 않게)
        k = 1
        while os.path.exists(f"{lg}.try{k}"):
            k += 1
        os.replace(lg, f"{lg}.try{k}")
    env = dict(os.environ, **ENV)
    f = open(lg, "w")
    f.write(f"[A5 드라이버] {time.strftime('%F %T')} GPU {g} 커밋 {os.path.basename(WT)}\n$ {cmd}\n"); f.flush()
    p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                         start_new_session=True)
    RUN[u.key] = dict(unit=u, proc=p, gpu=g, t0=time.time(), cut=False, f=f)
    open(u.pre + ".pid", "w").write(f"{os.uname().nodename} {p.pid} {os.environ.get('PBS_JOBID', '')}\n")
    log(f"[띄움] {u.key} GPU{g} pid {p.pid}")


def finish(key, rc):
    r = RUN.pop(key); u = r["unit"]; r["f"].close()
    txt = open(u.log, errors="ignore").read()[-20000:] if os.path.exists(u.log) else ""
    if u.kind.startswith("self_") and u.kind.endswith("cut") and r["cut"]:
        log(f"[자가 시험] {u.key} 끊음 (rc {rc}) -> 이어 돌리기"); return       # 표식 없이 -> 다음 틱에 같은 명령으로 이어 돈다
    oom = "out of memory" in txt.lower() or "CUDA_ERROR_OUT_OF_MEMORY" in txt
    if rc != 0 and (rc < 0 or rc in (129, 130, 137, 143) or oom):
        log(f"[중단] {u.key} rc {rc}{' 메모리 부족' if oom else ''} -> 다음에 이어 돈다"); return
    mark = u.pre + ".done"
    json.dump(dict(rc=rc, t=time.strftime("%F %T"), gpu=r["gpu"], commit=os.path.basename(WT)), open(mark, "w"))
    log(f"[끝] {u.key} rc {rc}")


def poll():
    for key in list(RUN):
        r = RUN[key]; u = r["unit"]; rc = r["proc"].poll()
        if rc is not None:
            finish(key, rc); continue
        if u.kind == "self_ovs_cut" or u.kind == "self_ipg_cut":
            # 이미 한 번 이어 돌렸으면(조각/체크포인트에서 시작) 끊지 않는다
            if not os.path.exists(u.pre + ".cut") and selftest_cut_ready(u):
                os.killpg(r["proc"].pid, signal.SIGKILL); r["cut"] = True
                open(u.pre + ".cut", "w").write(time.strftime("%F %T"))
                log(f"[자가 시험] {key} 실제로 끊음")
        elif os.path.exists(u.log) and time.time() - os.path.getmtime(u.log) > STALL_S:
            os.killpg(r["proc"].pid, signal.SIGKILL)
            open(u.log, "a").write(f"\nError: 로그가 {STALL_S // 3600} 시간 동안 멈춰 끊음 (발산으로 처리)\n")
            r["proc"].wait(); r["f"].close(); RUN.pop(key)
            json.dump(dict(rc=-999, t=time.strftime("%F %T"), stall=True), open(u.pre + ".done", "w"))
            log(f"[멈춤] {key} -> 발산 처리")


ORPHAN = {}       # 접두사 -> pid (드라이버 기동 전부터 살아 있던 내 실행)


def adopt():
    """드라이버만 죽고 실행은 살아 있는 경우: 같은 단위를 두 번 띄우지 않도록 살아 있는 pid 를 찾아 기다린다 (기동 때 한 번)."""
    busy = {}
    pfs = glob.glob(f"{A5}/**/*.pid", recursive=True)
    for S in SCENES:
        pfs += glob.glob(f"{O(S)}/**/*.pid", recursive=True)
    for pf in pfs:
        try:
            host, pid, _ = (open(pf).read().split() + ["", "", ""])[:3]
            if host == os.uname().nodename and os.path.exists(f"/proc/{int(pid)}"):
                busy[pf[:-4]] = int(pid)
        except Exception:                                 # noqa: BLE001
            pass
    return busy


R2P = [None]


def r2_sync(force=False):
    """R2 에 통째로 (제외 없음): stab/<형상> 전체와 stab/a5. 앞 전송이 안 끝났으면 건너뛴다 (마지막은 기다린다)."""
    if R2P[0] is not None and R2P[0].poll() is None:
        if not force:
            return
        R2P[0].wait()
    if not force and time.time() - R2_LAST[0] < 1800:
        return
    R2_LAST[0] = time.time()
    cmd = " ; ".join([f"~/bin/rclone copy {O(S)} r2:storage/result/anchorflow/stab_a5/{S} --transfers 8" for S in SCENES]
                     + [f"~/bin/rclone copy {A5} r2:storage/result/anchorflow/stab_a5/a5"])
    R2P[0] = subprocess.Popen(f"( {cmd} ) >> {A5}/rclone.log 2>&1", shell=True, start_new_session=True)
    if force:
        R2P[0].wait()


def status(lines):
    tmp = f"{A5}/status.txt.tmp"
    open(tmp, "w").write("\n".join(lines) + "\n"); os.replace(tmp, f"{A5}/status.txt")


def main():
    log(f"[시작] 워크트리 {WT}, 잡 {os.environ.get('PBS_JOBID', '?')}, 호스트 {os.uname().nodename}")
    ORPHAN.update(adopt())
    if ORPHAN:
        log("[기다림] 기동 전부터 살아 있는 실행", list(ORPHAN))
    while True:
        poll()
        for k_ in [k_ for k_, p_ in ORPHAN.items() if not os.path.exists(f"/proc/{p_}")]:
            ORPHAN.pop(k_)
        busy_ext = set(ORPHAN)
        selftest_judge()
        want = []
        for S in SCENES:
            if not ref_done(S):
                want.append(Unit(f"{S}/ref", f"{O(S)}/res/pg_N1024", "sim", S, "pg", 1024, False, 0))
        selftest_units(want)
        nconf, lines = {}, []
        for S in SCENES:
            nconf[S] = {m: chain(S, m, want) for m in METHODS}
            lines.append(f"{S}: 기준 {'끝' if ref_done(S) else '진행'}  " + "  ".join(f"{m}={nconf[S][m]}" for m in METHODS))
        json.dump(nconf, open(f"{A5}/nconf.json.tmp", "w")); os.replace(f"{A5}/nconf.json.tmp", f"{A5}/nconf.json")
        all_vis = True
        for S in SCENES:
            if all(nconf[S][m] is not None for m in METHODS):
                all_vis &= vis_units(S, nconf[S], want)
            else:
                all_vis = False
        if all_vis and not os.path.exists(f"{A5}/report.done"):
            want.append(Unit("report", f"{A5}/report", "report", "plane", None, None, False, 0))
        done_all = os.path.exists(f"{A5}/report.done")
        # 중복 제거·우선순위
        seen, todo = set(), []
        for u in sorted(want, key=lambda q: (q.prio, q.key)):
            if u.key in seen or u.key in RUN or u.done or u.pre in busy_ext:
                continue
            seen.add(u.key); todo.append(u)
        mem = gpu_mem() if todo else {}
        for u in todo:
            if u.kind == "report":
                if not RUN:                               # CPU 일: 다른 실행 다 끝나고
                    cmd = (f"cd {WT} && {PY} -u exe/stab_a5_report.py --root {ST} --out {A5}")
                    f = open(u.log, "w")
                    rc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **ENV)).returncode
                    f.close()
                    if rc == 0:
                        json.dump(dict(rc=0, t=time.strftime("%F %T")), open(u.pre + ".done", "w")); log("[보고] 끝")
                    else:
                        log(f"[보고] 실패 rc {rc} (로그 {u.log})"); open(f"{A5}/REPORT_FAIL", "w").write(str(rc))
                continue
            g = pick_gpu(u, mem)
            if g is None:
                continue
            launch(u, g)
            mem[g] = mem.get(g, 0) + 1000                 # 이번 틱에 다시 비었다고 보지 않게
        lines.append(f"실행 중 {len(RUN)}: " + ", ".join(f"{k}@{r['gpu']}" for k, r in RUN.items()))
        lines.append(f"대기 {len([u for u in todo if u.key not in RUN])}: " + ", ".join(u.key for u in todo if u.key not in RUN)[:1500])
        lines.append(f"자가 시험: ovs {'통과' if selftest_ok('ovs') else ('실패' if os.path.exists(STD + '/ovs.fail') else '대기')}, "
                     f"ipg {'통과' if selftest_ok('ipg') else ('실패' if os.path.exists(STD + '/ipg.fail') else '대기')}")
        status(lines)
        open(f"{A5}/heartbeat", "w").write(f"{int(time.time())} {os.environ.get('PBS_JOBID', '?')} {os.uname().nodename} {os.getpid()}\n")
        r2_sync()
        if done_all and not RUN:
            r2_sync(force=True)
            open(f"{A5}/STAB_A5_DONE", "w").write(time.strftime("%F %T") + "\n")
            log("STAB_A5_DONE"); status(lines + ["STAB_A5_DONE"])
            break
        for _ in range(12):                               # 60 초 대기 (자가 시험 끊기만 5 초마다 본다)
            time.sleep(5)
            if any(r["unit"].kind.endswith("_cut") for r in RUN.values()):
                poll()


if __name__ == "__main__":
    try:
        main()
    finally:
        for r in RUN.values():                            # 드라이버가 죽을 때 내 실행만 같이 정리 (다음 기동 때 이어 돈다)
            try:
                os.killpg(r["proc"].pid, signal.SIGTERM)
            except Exception:                             # noqa: BLE001
                pass
