"""Spring-Gaus 공식 동역학 학습 (train.py train_dynamic -> train_step) 의 정확한 이어 돌리기 (exe/stab_sg.py --mode train 이 부름).

공식 코드는 고치지 않고 모듈 전역만 감쌈 (결과 불변):
  - 저장: train_step 의 반복 루프 (etqdm) 를 감싸 매 EVERY 반복이 끝난 직후 (다음 반복 시작 전) <FIT>/sg_resume/cur/checkpoint/ 에 원자적으로 씀
      공식 dy_reload 형식 (Spring_Mass.pth.tar = 모델 state_dict, gaussians.ply (_xyz = init_xyz_all, 공식 저장과 같음), anchors.pt, dy_n_step.json)
      + state.pt (끝난 반복 번호, n_step, 시뮬레이터 옵티마이저·스케줄러, 가우시안 옵티마이저 (Adam 모멘트·step), 난수 python·numpy·torch·CUDA 전부,
                  시뮬레이터의 state_dict 밖 텐서·정수 속성 (af_nstep 등), dump_cfg.yaml 원문)
    SG_RESUME_R2_MAP="<로컬 앞부분>=<R2 앞부분>" 이면 저장 직후 rclone 으로 R2 에 바로 올림 (뒤에서).
  - 재개: 같은 cfg (dump_cfg.yaml 원문 같음) 의 가장 늦은 저장점을 찾아 공식 dy_reload 경로로 시뮬레이터·가우시안을 만들고 (공식 규약: 정제 단계 건너뜀),
    옵티마이저·난수·반복 번호를 넣은 뒤 그 다음 반복부터 공식 train_step 을 그대로 돌림. n_step 증가 일정 (N_STEP_ITER) 은 반복 번호로 정해지므로 그대로 이어짐.
  - SG_RESUME_LEGACY=1 이면 옛 체크포인트 (공식 SAVE_INTERVAL 100: exp/*/checkpoints_dynamic/checkpoint/) 에서도 재개 (기본 끔) -- 단 공식 저장에는 가우시안 옵티마이저 상태가 없어
    그 Adam 모멘트만 새로 시작 (근사; 로그에 '옛 체크포인트 근사 재개' 표시). 시뮬레이터 옵티마이저·난수·n_step·반복 번호는 정확.
  환경: SG_NO_EMPTY_CACHE=1 (동역학 반복 안 empty_cache 끔, 결과 불변 가속), SG_RESUME=0 끔, SG_RESUME_EVERY (기본 5), SG_RESUME_STOP_AT (검증용: 이 반복 저장 뒤 프로세스 종료)."""
import glob
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

EVERY = int(os.environ.get("SG_RESUME_EVERY", "5"))
STOP_AT = int(os.environ.get("SG_RESUME_STOP_AT", "0"))


def _log(s):
    print(f"[sg_resume {time.strftime('%F %T')}] {s}", flush=True)


def _rng_get():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def _rng_set(r):
    random.setstate(r["py"]); np.random.set_state(r["np"]); torch.set_rng_state(r["torch"]); torch.cuda.set_rng_state_all(r["cuda"])


def _extra(sim):
    """state_dict 밖의 텐서·스칼라 속성 (공식 + stab_sg 덧붙임 af_*). 목록·None 은 구성 때 정해지는 값이라 뺌."""
    pk = set(dict(sim.named_parameters()).keys()) | set(dict(sim.named_buffers()).keys())
    out = {}
    for k, v in vars(sim).items():
        if k.startswith("_") or k in pk:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, (int, float, bool)) and not isinstance(v, np.generic):
            out[k] = v
    return out


def _cands(FIT, dump_cfg, n_iter):
    c = []
    for n in ("cur", "old"):
        f = f"{FIT}/sg_resume/{n}/checkpoint/state.pt"
        if os.path.exists(f):
            try:
                s = torch.load(f, map_location="cpu", weights_only=False)
                if s.get("dump_cfg") == dump_cfg:
                    c.append((s["iter"], 1, os.path.dirname(f), s))
                else:
                    _log(f"{f}: cfg 다름 -> 안 씀")
            except Exception as ex:                                                  # noqa: BLE001
                _log(f"{f}: 읽기 실패 {ex!r}")
    for d in (glob.glob(f"{FIT}/exp/*/checkpoints_dynamic/checkpoint/") if os.environ.get("SG_RESUME_LEGACY", "0") == "1" else []):   # 옛 체크포인트 (근사) 는 켤 때만
        need = ("train_param.pth.tar", "Spring_Mass.pth.tar", "random_state.pkl", "gaussians.ply", "anchors.pt", "dy_n_step.json")
        if not all(os.path.exists(d + q) for q in need):
            continue
        dc = os.path.join(d, "../../dump_cfg.yaml")
        if not (os.path.exists(dc) and open(dc).read() == dump_cfg):
            _log(f"옛 체크포인트 {d}: cfg 다름 -> 안 씀"); continue
        try:
            ep = int(torch.load(d + "train_param.pth.tar", map_location="cpu", weights_only=False)["epoch"])
        except Exception as ex:                                                      # noqa: BLE001
            _log(f"{d}: train_param 읽기 실패 {ex!r}"); continue
        c.append((ep, 0, d.rstrip("/"), None))
    c = [q for q in c if 0 < q[0]]
    return sorted(c, key=lambda q: (q[0], q[1]))[-1] if c else None


def install(T, FIT, arg, cfg, exp_dir):
    if os.environ.get("SG_RESUME", "1") == "0":
        _log("꺼짐 (SG_RESUME=0)"); return
    dump_cfg = open(f"{exp_dir}/dump_cfg.yaml").read()
    n_iter = int(cfg.DYNAMIC.ITERATIONS)
    best = _cands(FIT, dump_cfg, n_iter)
    ctx = dict(start=1, load=None)
    if best is not None:
        it, new, ck, st = best
        arg.dy_reload = f"{ck}/Spring_Mass.pth.tar"                                   # 공식 dy_reload 경로: 가우시안 ply·앵커·가중치·n_step 적재, 정제 단계 건너뜀
        fz = cfg.is_frozen() if hasattr(cfg, "is_frozen") else False
        if fz:
            cfg.defrost()
        cfg.DYNAMIC.PRETRAINED = arg.dy_reload
        if fz:
            cfg.freeze()
        ctx.update(start=it + 1, load=(new, ck, st))
        _log(f"재개: 반복 {it} 끝난 저장점 {ck} ({'새 형식 정확' if new else '옛 체크포인트 근사 재개 (가우시안 Adam 모멘트만 새로)'}) -> 반복 {it + 1} 부터")
    else:
        _log("저장점 없음 -> 처음부터")
    m = os.environ.get("SG_RESUME_R2_MAP", "")
    r2 = None
    if "=" in m:
        lp, rp = m.split("=", 1)
        if os.path.realpath(FIT).startswith(os.path.realpath(lp)) or FIT.startswith(lp):
            rel = os.path.relpath(os.path.realpath(FIT), os.path.realpath(lp))
            r2 = f"{rp.rstrip('/')}/{rel}/sg_resume/cur"
    ctx["r2"], ctx["sync"] = r2, None

    get_sim0, train_step0, etqdm0 = T.get_simulator, T.train_step, T.etqdm

    def get_simulator(*a, **k):
        sim, g = get_sim0(*a, **k)
        ctx["sim"], ctx["g"] = sim, g
        return sim, g

    def save(it):
        sim, g = ctx["sim"], ctx["g"]
        R = f"{FIT}/sg_resume"; tmp = f"{R}/tmp/checkpoint"
        shutil.rmtree(f"{R}/tmp", ignore_errors=True); os.makedirs(tmp)
        torch.save(sim.state_dict(), f"{tmp}/Spring_Mass.pth.tar")
        x0 = g._xyz
        g._xyz = sim.init_xyz_all.detach().clone()                                    # 공식 저장과 같음
        g.save_ply(f"{tmp}/gaussians.ply")
        g._xyz = x0
        torch.save(sim.init_xyz.detach().clone().cpu(), f"{tmp}/anchors.pt")
        json.dump(sim.n_step, open(f"{tmp}/dy_n_step.json", "w"))
        torch.save(dict(iter=it, n_step=sim.n_step, sim_opt=sim.optimizer.state_dict(),
                        sim_sched=(sim.scheduler.state_dict() if sim.scheduler is not None else None),
                        g_opt=(g.optimizer.state_dict() if g.optimizer is not None else None),
                        rng=_rng_get(), extra=_extra(sim), dump_cfg=dump_cfg, t=time.time()), f"{tmp}/state.pt")
        if os.path.isdir(f"{R}/cur"):
            shutil.rmtree(f"{R}/old", ignore_errors=True); os.rename(f"{R}/cur", f"{R}/old")
        os.rename(f"{R}/tmp", f"{R}/cur")
        shutil.rmtree(f"{R}/old", ignore_errors=True)
        _log(f"저장 반복 {it} n_step {sim.n_step}")
        if r2 and (ctx["sync"] is None or ctx["sync"].poll() is not None):
            ctx["sync"] = subprocess.Popen(["rclone", "sync", f"{R}/cur", r2, "-q"], stdin=subprocess.DEVNULL)
        if STOP_AT and it >= STOP_AT:
            if ctx["sync"] is not None:
                ctx["sync"].wait()
            _log(f"SG_RESUME_STOP_AT {STOP_AT}: 종료 (검증용)")
            sys.stdout.flush(); os._exit(17)

    class Bar:
        def __init__(self, it_, **k):
            self.bar = etqdm0([i for i in it_ if i >= ctx["start"]], **k)

        def __iter__(self):
            first = True
            for i in self.bar:
                if first and ctx.get("rng") is not None:
                    _rng_set(ctx.pop("rng"))
                if not first and (i - 1) % EVERY == 0:
                    save(i - 1)
                first = False
                yield i

        def __getattr__(self, n):
            return getattr(self.bar, n)

    def train_step(*a, **k):
        names = ("args", "cfg", "cfg_stage", "stage", "max_frame", "simulator", "scene", "gaussians")
        kw = dict(zip(names, a), **k)
        if kw["stage"] != "dynamic":
            return train_step0(*a, **k)
        simulator, gaussians = kw["simulator"], kw["gaussians"]
        ctx["sim"], ctx["g"] = simulator, gaussians
        if ctx["load"] is not None:
            new, ck, st = ctx["load"]
            if new:
                simulator.optimizer.load_state_dict(st["sim_opt"])
                if st["sim_sched"] is not None and simulator.scheduler is not None:
                    simulator.scheduler.load_state_dict(st["sim_sched"])
                if st["g_opt"] is not None and gaussians.optimizer is not None:
                    gaussians.optimizer.load_state_dict(st["g_opt"])
                for k, v in st["extra"].items():
                    cur = getattr(simulator, k, None)
                    if isinstance(v, torch.Tensor):
                        if isinstance(cur, torch.Tensor) and cur.shape == v.shape and not torch.equal(cur.detach().cpu(), v):
                            _log(f"속성 {k} 가 저장값과 다름 (최대 차 {float((cur.detach().cpu().double() - v.double()).abs().max()):.3g}) -> 저장값으로")
                        setattr(simulator, k, v.to(cur.device if isinstance(cur, torch.Tensor) else "cuda").requires_grad_(bool(cur.requires_grad) if isinstance(cur, torch.Tensor) else False))
                    else:
                        setattr(simulator, k, v)
                simulator.n_step = st["n_step"]
                ctx["rng"] = st["rng"]
            else:
                tp = torch.load(f"{ck}/train_param.pth.tar", map_location="cpu", weights_only=False)
                simulator.optimizer.load_state_dict(tp["optimizer"])
                rs = pickle.load(open(f"{ck}/random_state.pkl", "rb"))
                ctx["rng"] = dict(py=rs.random_rng_state, np=rs.numpy_rng_state, torch=rs.torch_rng_state, cuda=rs.torch_cuda_rng_state_all)
            _log(f"적재 끝: 반복 {ctx['start']} 부터, n_step {simulator.n_step}")
        T.etqdm = Bar
        ec0 = torch.cuda.empty_cache
        if os.environ.get("SG_NO_EMPTY_CACHE", "0") == "1":                         # 가속: 프레임마다 부르는 캐시 비우기만 끔 (할당기 동작만; 계산·난수 불변 -- exe/sgr_test 로 확인)
            torch.cuda.empty_cache = lambda: None
            _log("torch.cuda.empty_cache 끔 (SG_NO_EMPTY_CACHE=1)")
        try:
            return train_step0(*a, **k)
        finally:
            T.etqdm = etqdm0
            torch.cuda.empty_cache = ec0
            if ctx["sync"] is not None:
                ctx["sync"].wait()

    T.get_simulator = get_simulator
    T.train_step = train_step
