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

__all__ = ["load_scenes", "HandlePlan", "StatePool"]


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

    속도 프로파일은 생성기와 같다 -- 전체 1 초(60 프레임)를 가속 0.25 s,
    등속 0.5 s, 감속 0.25 s 로 나누고 최고속도는 거리/0.75 로 잡는다.
    """

    def __init__(self, n_ctrl, idx, target, radius, frame_dt=1.0 / 60.0,
                 frames=60):
        self.k = n_ctrl
        self.idx = idx                      # [K] 제어 입자 (부분표본 색인)
        self.target = target                # [K,3]
        self.radius = radius
        self.frames = frames
        self.dt = frame_dt
        self.t_acc = 0.25
        self.t_tot = frames * frame_dt

    @staticmethod
    def sample(x0, n_ctrl, radius, gen, dev, ext, frames=60,
               dist_lo=0.30, dist_hi=0.70, domain=2.0, margin=0.15):
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
        # 목표점은 **시뮬레이션 영역에 마진을 준 상자 안에서** 뽑는다. 방향과
        # 거리를 뽑아 경계로 눌러 버리면(clamp) 목표점이 벽에 쏠린다.
        lo_b, hi_b = margin, domain - margin
        tgt = torch.empty(n_ctrl, 3, device=dev)
        for k in range(n_ctrl):
            for _ in range(64):
                d = torch.randn(3, generator=gen, device=dev)
                d = d / d.norm().clamp_min(1e-9)
                dist = (dist_lo + (dist_hi - dist_lo)
                        * float(torch.rand(1, generator=gen, device=dev))) * ext
                cand = x0[idx[k]] + d * dist
                if bool(((cand >= lo_b) & (cand <= hi_b)).all()):
                    tgt[k] = cand
                    break
            else:
                tgt[k] = (lo_b + (hi_b - lo_b)
                          * torch.rand(3, generator=gen, device=dev))
        return HandlePlan(n_ctrl, idx, tgt, radius, frames=frames)

    def velocity(self, x_now, elapsed):
        """[K,3] 이번 프레임의 명령 속도."""
        vec = self.target - x_now[self.idx]
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dirv = vec / dist
        t = float(elapsed) * self.dt
        cruise = self.t_tot - 2.0 * self.t_acc
        vpeak = dist.squeeze(-1) / max(cruise + self.t_acc, 1e-6)
        if t < self.t_acc:
            s = vpeak * (t / self.t_acc)
        elif t < self.t_acc + cruise:
            s = vpeak
        else:
            r = max(0.0, (self.t_tot - t) / self.t_acc)
            s = vpeak * min(1.0, r)
        return dirv * s.unsqueeze(-1)

    def weights(self, x_now):
        """[N,K] 감쇠 가중치 (1-q^2)^2. 교사와 같은 규약."""
        c = x_now[self.idx]
        q = ((x_now.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1)
             / max(self.radius, 1e-6)).clamp(0, 1)
        return (1.0 - q * q) ** 2


class StatePool:
    """살아 있는 상태들의 집합. 누적 잔차가 문턱을 넘으면 버린다."""

    def __init__(self, scenes, size, n_ctrl, radius, dev, gen,
                 frames=60, thresh=0.05, window=30,
                 domain=2.0, margin=0.15):
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
        self.domain = domain            # 시뮬 영역 [0, domain]^3
        self.margin = margin            # 목표점은 이만큼 안쪽에서 뽑는다
        self.fresh_res = []                 # 신규 상태 잔차 (중앙값 기준용)
        self.items = [self.fresh() for _ in range(size)]
        self.n_drop = 0
        self.n_replan = 0

    def fresh(self, si=None):
        """새 초기 상태: 정지 자세, v=0, F=I, 새 손잡이 계획."""
        if si is None:
            si = int(torch.randint(len(self.scenes), (1,), generator=self.gen,
                                   device=self.dev))
        tag, sc = self.scenes[si]
        x = sc["x0"].clone()
        plan = HandlePlan.sample(x, self.n_ctrl, self.radius, self.gen,
                                 self.dev, sc["ext"], self.frames,
                                 domain=self.domain, margin=self.margin)
        return dict(si=si, x=x, v=torch.zeros_like(x),
                    F=torch.eye(3, device=self.dev).expand(
                        x.shape[0], 3, 3).contiguous(),
                    p=None, plan=plan, elapsed=0, hist=[], age=0)

    def threshold(self):
        return self.thresh

    def sample(self, n_pool, n_fresh):
        """배치 구성: 풀에서 n_pool 개, 새 상태 n_fresh 개."""
        picks = []
        if self.items and n_pool > 0:
            sel = torch.randint(len(self.items), (n_pool,), generator=self.gen,
                                device=self.dev).tolist()
            picks += [("pool", i) for i in sel]
        picks += [("fresh", None)] * n_fresh
        return picks

    def put_back(self, slot, st, res):
        """한 스텝 진행한 상태를 되돌려 놓는다. 문턱을 넘으면 버린다."""
        st["hist"].append(float(res))
        if len(st["hist"]) > self.window:
            st["hist"] = st["hist"][-self.window:]
        st["age"] += 1
        mean_res = float(np.mean(st["hist"]))
        if st["age"] == 1:
            self.fresh_res.append(float(res))
        st["elapsed"] += 1
        if st["elapsed"] >= self.frames:
            # 계획을 다 썼는데 살아 있으면 새 제어 입자·목표점으로 이어 간다
            _, sc = self.scenes[st["si"]]
            st["plan"] = HandlePlan.sample(st["x"], self.n_ctrl, self.radius,
                                           self.gen, self.dev, sc["ext"],
                                           self.frames, domain=self.domain,
                                           margin=self.margin)
            st["elapsed"] = 0
            self.n_replan += 1
        bad = (not bool(torch.isfinite(st["x"]).all())) or \
            (not np.isfinite(res)) or mean_res > self.thresh
        if bad:
            self.n_drop += 1
            if slot is not None:
                self.items[slot] = self.fresh()
            return False
        if slot is not None:
            self.items[slot] = st
        elif len(self.items) < self.size:
            self.items.append(st)
        return True
