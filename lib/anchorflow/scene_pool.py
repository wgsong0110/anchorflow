"""교사 궤적 없이 도는 **상태 풀**.

미리 만든 궤적을 전혀 읽지 않는다. 씬의 정지 상태(채우기 입자 집합)에서 출발해
손잡이 계획을 직접 뽑고, 학생이 한 스텝씩 굴린 상태를 풀에 담아 둔다. 상태마다
누적 물리잔차를 들고 있다가 문턱을 넘으면 버리고 새 초기 상태로 채운다.

풀 원소: (씬 번호, 제어 입자 색인, 목표점, 경과 프레임, x, v, F, 앵커, 누적잔차)
"""
import glob
import json
import math
import os

import numpy as np
import torch

__all__ = ["load_scenes", "HandlePlan", "StatePool", "target_grid"]


def target_grid(domain, margin, n_side, cfg=None, dev="cpu", clear=0.15):
    """목표점 후보 [P,3]. **유한하고 고정**이다.

    매번 연속 분포에서 뽑으면 같은 목표를 두 번 볼 일이 없어, 학생이 같은 과제를
    반복해서 배울 기회가 없다. 마진 상자 안의 격자로 후보를 고정하고 바닥면
    안쪽으로 들어가는 점은 버린다 (도달할 수 없는 목표다).
    """
    t = torch.linspace(margin, domain - margin, n_side)
    P = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    for bc in ((cfg or {}).get("boundary_conditions") or []):
        if bc.get("type") != "surface_collider":
            continue
        pt = torch.as_tensor(bc["point"], dtype=P.dtype)
        nr = torch.as_tensor(bc["normal"], dtype=P.dtype)
        nr = nr / nr.norm().clamp_min(1e-12)
        # 면에서 띄우는 여유는 **손잡이 반경**이면 된다. margin(목표 상자 여유)을
        # 쓰면 바닥 쪽 후보가 통째로 날아가 손잡이가 늘 위로만 끌게 된다.
        P = P[((P - pt) * nr).sum(-1) > clear]
    return P.to(dev)


def load_scenes(fill_dir, cfg_dir, combos, n_pts, dev, seed=0):
    """(태그, 씬) 목록. 씬은 정지 상태와 물성만 들고 있다.

    fill_dir 의 `pgfill_{형상}.npy` 가 PG 채우기로 만든 입자 집합이고,
    cfg_dir 의 `{형상}_{물성}_t.json` 이 물성이다. 궤적 파일은 쓰지 않는다.
    """
    out = []
    g = torch.Generator().manual_seed(seed)
    for tag in combos:
        shape, mat = tag.split("_", 1)
        f = os.path.join(fill_dir, f"pgfill_{shape}.npy")
        if not os.path.exists(f):
            alt = sorted(glob.glob(os.path.join(fill_dir, f"fill_{shape}.npy")))
            if not alt:
                print(f"[풀] {tag}: 채우기 파일이 없다 -- 건너뛴다", flush=True)
                continue
            f = alt[0]
        xf = torch.from_numpy(np.load(f)).float()
        n_full = xf.shape[0]
        sel = torch.randperm(n_full, generator=g)[:n_pts].sort().values
        x0 = xf[sel].to(dev)
        cj = os.path.join(cfg_dir, f"{shape}_{mat}_t.json")
        if not os.path.exists(cj):
            cj = os.path.join(cfg_dir, f"{shape}_{mat}.json")
        cfg = json.load(open(cj))
        ng = int(cfg.get("n_grid", 100))
        gl = float(cfg.get("grid_lim", 2.0))
        dx = gl / ng
        vi = (x0 / dx).long().clamp(0, ng - 1)
        fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
        cnt = torch.zeros(ng ** 3, device=dev).index_add_(
            0, fl, torch.ones(x0.shape[0], device=dev))
        # 부분표본이므로 셀당 개수를 원본 비율로 되돌린다
        mass = ((dx ** 3) / cnt[fl].clamp(min=1.0)) * float(cfg["density"]) \
            * (float(n_full) / x0.shape[0])
        ext = float((x0.max(0).values - x0.min(0).values).norm())
        out.append((tag, dict(x0=x0, mass=mass, cfg=cfg, ext=ext,
                              n_full=n_full, sel=sel.to(dev), tag=tag)))
        print(f"[풀] {tag}: 입자 {x0.shape[0]} (원본 {n_full}), 물체 {ext:.4f}, "
              f"E={cfg['E']:g} nu={cfg['nu']:g}", flush=True)
    return out


class HandlePlan:
    """손잡이 계획. 제어 입자와 목표점만 자유 정보이고 나머지는 규칙이다.

    속도는 **가속도와 최고속도를 고정**하고 짠다 -- 목표까지 걸리는 시간을
    고정하면 먼 목표는 빠르게, 가까운 목표는 느리게 끌려가 같은 물성이 전혀
    다른 속도 영역을 보게 된다. 지령 속도는

        s = min(vmax, a·t, sqrt(2 a d))

    로, 출발에서 a 로 가속해 vmax 로 순항하고 남은 거리 d 가 감속 거리에
    들어오면 a 로 감속해 목표에서 속도 0 으로 멈춘다.
    """

    def __init__(self, n_ctrl, idx, target, radius, frame_dt=1.0 / 60.0,
                 frames=60, acc=2.4, vmax=0.6, tol=5e-3):
        self.k = n_ctrl
        self.idx = idx                      # [K] 제어 입자 (부분표본 색인)
        self.target = target                # [K,3]
        self.radius = radius
        self.frames = frames                # 한 계획의 **상한** (도달하면 더 짧다)
        self.dt = frame_dt
        self.acc = acc
        self.vmax = vmax
        self.tol = tol

    @staticmethod
    def sample(x0, n_ctrl, radius, gen, dev, cand, frames=60,
               acc=2.4, vmax=0.6, tol=5e-3):
        """제어 입자와 목표점을 뽑는다. 손잡이끼리 2R 안에 겹치지 않게 한다."""
        n = x0.shape[0]
        idx = []
        for _ in range(200):
            c = int(torch.randint(n, (1,), generator=gen, device=dev))
            if all(float((x0[c] - x0[j]).norm()) >= 2.0 * radius for j in idx):
                idx.append(c)
            if len(idx) == n_ctrl:
                break
        while len(idx) < n_ctrl:            # 좁은 물체면 겹침을 허용한다
            idx.append(int(torch.randint(n, (1,), generator=gen, device=dev)))
        idx = torch.tensor(idx, device=dev, dtype=torch.long)
        pick = torch.randint(cand.shape[0], (n_ctrl,), generator=gen, device=dev)
        return HandlePlan(n_ctrl, idx, cand[pick].clone(), radius,
                          frames=frames, acc=acc, vmax=vmax, tol=tol)

    def velocity(self, x_now, elapsed):
        """[K,3] 이번 프레임의 명령 속도."""
        vec = self.target - x_now[self.idx]
        dist = vec.norm(dim=-1)
        dirv = vec / dist.clamp_min(1e-9).unsqueeze(-1)
        t = float(elapsed) * self.dt
        s_ramp = self.acc * max(t, self.dt)                 # 출발에서 가속
        s_stop = (2.0 * self.acc * dist.clamp_min(0.0)).sqrt()   # 멈출 수 있는 속도
        s = torch.clamp(s_stop, max=min(self.vmax, s_ramp))
        s = s * (dist > self.tol).to(s.dtype)               # 도달하면 정지
        return dirv * s.unsqueeze(-1)

    def arrived(self, x_now):
        return bool(((self.target - x_now[self.idx]).norm(dim=-1)
                     <= self.tol).all())

    def weights(self, x_now):
        """[N,K] 감쇠 가중치 (1-q^2)^2. 교사와 같은 규약."""
        c = x_now[self.idx]
        q = ((x_now.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1)
             / max(self.radius, 1e-6)).clamp(0, 1)
        return (1.0 - q * q) ** 2

    def pack(self):
        """체크포인트용. 제어 입자와 목표점만 담으면 나머지는 규칙으로 복원된다."""
        return dict(k=self.k, idx=self.idx.cpu(), target=self.target.cpu(),
                    radius=self.radius, frames=self.frames, dt=self.dt,
                    acc=self.acc, vmax=self.vmax, tol=self.tol)

    @staticmethod
    def unpack(d, dev):
        return HandlePlan(int(d["k"]), d["idx"].to(dev), d["target"].to(dev),
                          float(d["radius"]), frame_dt=float(d["dt"]),
                          frames=int(d["frames"]), acc=float(d.get("acc", 2.4)),
                          vmax=float(d.get("vmax", 0.6)),
                          tol=float(d.get("tol", 5e-3)))


class StatePool:
    """살아 있는 상태들의 집합. 누적 잔차가 문턱을 넘으면 버린다."""

    def __init__(self, scenes, size, n_ctrl, radius, dev, gen,
                 frames=60, thresh=0.05, window=30,
                 domain=2.0, margin=0.15, start_mid=False, keep_prob=0.5,
                 acc=2.4, vmax=0.6, n_side=4):
        self.scenes = scenes
        self.size = size
        self.n_ctrl = n_ctrl
        self.radius = radius
        self.dev = dev
        self.gen = gen
        self.frames = frames
        # 문턱은 **고정**이다. 판정에 쓰는 양이 정류 잔차를 길이로 환산해
        # 물체 크기로 나눈 무차원 수라, 물성·형상이 달라도 같은 자로 잰다.
        self.thresh = thresh
        # 누적은 **최근 window 프레임**만 본다. 전체 평균으로 두면 초반 잔차를
        # 계속 끌고 다녀, 물리적으로 멀쩡한데도 오래 살았다는 이유로 버려진다.
        self.window = window
        # 새 상태를 계획의 **중간 지점**에서 시작할지. 정지 상태에서만 출발하면
        # 관성항이 "가만히 있어라" 를 원하고 탄성항이 0 이라 아무것도 안 하는 것이
        # 거의 최적이 되어 기울기가 사라진다.
        self.start_mid = start_mid
        # 문턱을 넘어도 이 확률로는 버리지 않고 **그대로 둔다** (전진도 취소).
        # 매번 버리면 풀이 신규로만 차서 학생이 같은 상태를 이어 볼 기회가 없다.
        self.keep_prob = keep_prob
        self.n_keep = 0
        self.domain = domain            # 시뮬 영역 [0, domain]^3
        self.margin = margin            # 목표점은 이만큼 안쪽에서 뽑는다
        # 손잡이 운동은 **가속도와 최고속도**로 정한다 (도달 시간은 거리에 따라 다름)
        self.acc = acc
        self.vmax = vmax
        # 목표점 후보는 유한 고정 집합이다. 씬마다 바닥면이 같으므로 한 번만 짠다.
        self.cand = target_grid(domain, margin, n_side,
                                scenes[0][1]["cfg"] if scenes else None,
                                dev=dev, clear=radius)
        self.fresh_res = []                 # 신규 상태 잔차 (중앙값 기준용)
        # **비운 채로 시작한다.** 배치에 섞이는 신규 상태가 살아남아야 풀이
        # 차오르고, 버린 자리도 즉시 메우지 않는다 -- 풀의 크기 자체가
        # "지금 정책이 몇 스텝을 버티는가" 를 말해 주는 신호가 된다.
        self.items = []
        self.n_drop = 0
        self.n_replan = 0

    def state_dict(self):
        """풀 전체와 누적 집계. 재개할 때 빈 풀에서 0 부터 다시 세면 TB 의
        누적 곡선이 끊기고, 무엇보다 학생이 쌓아 둔 상태들을 잃는다."""
        items = []
        for q in self.items:
            if q is None:
                continue
            items.append(dict(
                si=int(q["si"]), x=q["x"].detach().cpu(),
                v=q["v"].detach().cpu(), F=q["F"].detach().cpu(),
                p=(q["p"].detach().cpu() if torch.is_tensor(q["p"]) else None),
                plan=q["plan"].pack(), elapsed=int(q["elapsed"]),
                hist=list(q["hist"]), age=int(q["age"])))
        return dict(items=items, n_drop=self.n_drop, n_keep=self.n_keep,
                    n_replan=self.n_replan, fresh_res=list(self.fresh_res),
                    tags=[t for t, _ in self.scenes])

    def load_state_dict(self, d):
        """씬 구성이 달라졌으면 그 상태만 버리고 나머지는 그대로 이어받는다."""
        tags = [t for t, _ in self.scenes]
        old = list(d.get("tags") or tags)
        self.items = []
        n_skip = 0
        for q in d.get("items", []):
            si = int(q["si"])
            tag = old[si] if si < len(old) else None
            if tag not in tags:
                n_skip += 1
                continue
            self.items.append(dict(
                si=tags.index(tag), x=q["x"].to(self.dev),
                v=q["v"].to(self.dev), F=q["F"].to(self.dev),
                p=(q["p"].to(self.dev) if torch.is_tensor(q["p"]) else None),
                plan=HandlePlan.unpack(q["plan"], self.dev),
                elapsed=int(q["elapsed"]), hist=list(q["hist"]),
                age=int(q["age"])))
        self.n_drop = int(d.get("n_drop", 0))
        self.n_keep = int(d.get("n_keep", 0))
        self.n_replan = int(d.get("n_replan", 0))
        self.fresh_res = list(d.get("fresh_res") or [])
        return len(self.items), n_skip

    def fresh(self, si=None):
        """새 초기 상태: 정지 자세, v=0, F=I, 새 손잡이 계획."""
        if si is None:
            si = int(torch.randint(len(self.scenes), (1,), generator=self.gen,
                                   device=self.dev))
        tag, sc = self.scenes[si]
        x = sc["x0"].clone()
        plan = HandlePlan.sample(x, self.n_ctrl, self.radius, self.gen,
                                 self.dev, self.cand, self.frames,
                                 acc=self.acc, vmax=self.vmax)
        el = 0
        if self.start_mid:
            el = int(torch.randint(self.frames, (1,), generator=self.gen,
                                   device=self.dev))
        return dict(si=si, x=x, v=torch.zeros_like(x),
                    F=torch.eye(3, device=self.dev).expand(
                        x.shape[0], 3, 3).contiguous(),
                    p=None, plan=plan, elapsed=el, hist=[], age=0)

    def threshold(self):
        return self.thresh

    def sample(self, n_pool, n_fresh):
        """배치 구성: 풀에서 n_pool 개, 새 상태 n_fresh 개."""
        self.items = [q for q in self.items if q is not None]   # 빈자리 정리
        picks = []
        if self.items and n_pool > 0:
            sel = torch.randint(len(self.items), (n_pool,), generator=self.gen,
                                device=self.dev).tolist()
            picks += [("pool", i) for i in sel]
        picks += [("fresh", None)] * n_fresh
        return picks

    def put_back(self, slot, st, res, prev=None):
        """한 스텝 진행한 상태를 되돌려 놓는다. 문턱을 넘으면 버린다.

        prev 가 있으면 (전진 전 상태) 문턱을 넘었을 때 keep_prob 확률로 그
        상태로 되돌려 풀에 그대로 남긴다 -- 이번 스텝을 없던 일로 한다.
        """
        st["hist"].append(float(res))
        if len(st["hist"]) > self.window:
            st["hist"] = st["hist"][-self.window:]
        st["age"] += 1
        mean_res = float(np.mean(st["hist"]))
        if st["age"] == 1:
            self.fresh_res.append(float(res))
        st["elapsed"] += 1
        # **목표에 닿았을 때** 새 제어 입자·목표점을 뽑는다. 고정 시간이 지나서
        # 뽑는 것이 아니다 -- 도달 시간은 거리에 따라 다르다. frames 는 끌어도
        # 도달하지 못하는 경우(물체가 딸려오지 않거나 발산) 빠져나오는 상한이다.
        if (st["elapsed"] >= self.frames
                or st["plan"].arrived(st["x"])):
            st["plan"] = HandlePlan.sample(st["x"], self.n_ctrl, self.radius,
                                           self.gen, self.dev, self.cand,
                                           self.frames, acc=self.acc,
                                           vmax=self.vmax)
            st["elapsed"] = 0
            self.n_replan += 1
        bad = (not bool(torch.isfinite(st["x"]).all())) or \
            (not np.isfinite(res)) or mean_res > self.thresh
        if bad:
            keep = (prev is not None and self.keep_prob > 0
                    and float(torch.rand(1, generator=self.gen,
                                         device=self.dev)) < self.keep_prob
                    and bool(torch.isfinite(prev["x"]).all()))
            if keep:
                self.n_keep += 1
                if slot is not None:
                    self.items[slot] = prev    # 전진을 취소하고 그대로 둔다
                elif len(self.items) < self.size:
                    self.items.append(prev)    # 신규였으면 그대로 풀에 넣는다
                return True
            self.n_drop += 1
            if slot is not None:
                self.items[slot] = None        # 자리를 비워 둔다 (메우지 않는다)
            return False
        if slot is not None:
            self.items[slot] = st
        elif len(self.items) < self.size:
            self.items.append(st)
        return True
