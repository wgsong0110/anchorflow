"""제어점 궤적 -- 웨이포인트를 Catmull-Rom 으로 잇고 속도 상한을 지키게 시간 배분.

교사 MPM 에서 물체 표면 입자 몇 개를 **제어점**으로 잡아 매 스텝 위치·속도를
강제(Dirichlet)하는 데 쓴다. 궤적은 이렇게 만든다.

  * n 스텝마다 새 표적을 뽑는다
  * 확률 p  : **지금 입자 위치 중 하나**를 골라, 그 자리의 바깥 법선 반대쪽으로
              침투 깊이만큼 들어간 점  -> 물체를 찌르거나 누르는 조작
  * 확률 1-p: 물체 바깥의 임의 점                    -> 잡아 빼거나 스쳐 지나가는 조작
  * 웨이포인트를 Catmull-Rom 으로 잇고, **구간 길이 / 속도 상한** 으로 스텝 수를
    정해 어디서도 v_max 를 넘지 않게 한다 (긴 구간에는 스텝을 더 준다)

모든 손잡이(n, p, 깊이, v_max, 제어점 수)는 config 로 열려 있고 시드로 재현된다.

법선은 t0 구름에서 이웃 PCA 로 잡고 중심 반대쪽을 바깥으로 본다. "안쪽" 을
중심 방향으로만 잡으면 오목한 자리에서 엉뚱한 데를 찌른다.
"""
import numpy as np


def _knn(x, k):
    """격자 해시로 이웃 k 개. scipy/sklearn 없이 돌아야 한다 (로컬에 없다)."""
    n = len(x)
    lo = x.min(0)
    # 칸 하나에 평균 몇 개가 들어가게 잡는다
    cell = float(np.cbrt(np.prod(x.max(0) - lo + 1e-12) * max(k, 8) / max(n, 1)))
    c = np.floor((x - lo) / cell).astype(np.int64)
    key = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791)
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    table = {int(kk): order[s:e] for kk, s, e in zip(ks, starts, ends)}
    out = np.zeros((n, k), np.int64)
    off = np.array([(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1)
                    for l in (-1, 0, 1)])
    for i in range(n):
        cand = []
        for o in off:
            cc = c[i] + o
            kk = int((cc[0] * 73856093) ^ (cc[1] * 19349663) ^ (cc[2] * 83492791))
            v = table.get(kk)
            if v is not None:
                cand.append(v)
        cand = np.concatenate(cand) if cand else np.arange(n)
        if len(cand) < k + 1:
            cand = np.arange(n)
        d = np.linalg.norm(x[cand] - x[i], axis=1)
        out[i] = cand[np.argsort(d)[1:k + 1]]
    return out


def estimate_normals(x, k=16):
    """이웃 PCA 의 최소 고유벡터. 중심에서 멀어지는 쪽을 바깥으로 맞춘다."""
    nb = _knn(x, min(k, max(len(x) - 1, 1)))
    d = x[nb] - x[:, None, :]
    cov = np.einsum("nki,nkj->nij", d, d) / d.shape[1]
    # 비유한 값이 하나라도 섞이면 eigh 가 통째로 터진다 (겪었다)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(3) * 1e-12
    try:
        w, v = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        v = np.tile(np.eye(3), (cov.shape[0], 1, 1))
    nrm = v[:, :, 0]
    out = x - x.mean(0)
    flip = (nrm * out).sum(1) < 0
    nrm[flip] *= -1.0
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)


def surface_mask(x, dx=None):
    """격자 칸으로 채워 보고 **빈 면이웃이 있는** 칸의 입자를 표면으로 본다.

    칸이 입자 간격보다 작으면 속이 꽉 찬 물체도 전부 표면으로 나온다 (겪었다).
    dx 를 안 주면 입자 간격의 두 배로 잡는다.
    """
    if dx is None:
        sub = x[::max(1, len(x) // 4000)]
        nb = _knn(sub, 2)
        dx = 2.0 * float(np.median(np.linalg.norm(sub[nb[:, 0]] - sub, axis=1)))
    c = np.floor(x / dx).astype(np.int64)
    c -= c.min(0)
    shape = c.max(0) + 3
    occ = np.zeros(tuple(shape), bool)
    occ[c[:, 0] + 1, c[:, 1] + 1, c[:, 2] + 1] = True
    nb = np.ones_like(occ)
    for ax in range(3):
        for s in (-1, 1):
            nb &= np.roll(occ, s, axis=ax)
    inner = nb[c[:, 0] + 1, c[:, 1] + 1, c[:, 2] + 1]
    return ~inner


def _catmull_rom(p0, p1, p2, p3, u):
    """u in [0,1] 한 구간. 표준 Catmull-Rom (tension 0.5)."""
    u = u[:, None]
    return 0.5 * ((2 * p1)
                  + (-p0 + p2) * u
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


class ControlTraj:
    """제어점 K 개의 스텝별 위치·속도.

    Parameters
    ----------
    x0 : (N,3)  t0 입자 위치
    ctrl : (K,) 제어점으로 쓸 입자 색인
    dt : 한 스텝의 시간
    steps : 만들 스텝 수
    every_n : 새 표적을 뽑는 주기 (스텝)
    p_touch : 물체 쪽 표적을 고를 확률
    depth : 침투 깊이 (물체 지름 대비 비율)
    v_max : 속도 상한
    seed : 재현용
    """

    def __init__(self, x0, ctrl, dt, steps, every_n=20, p_touch=0.6,
                 depth=0.08, v_max=0.5, seed=0, normals=None, bounds=None):
        self.x0 = np.asarray(x0, np.float64)
        self.ctrl = np.asarray(ctrl, np.int64)
        self.K = len(self.ctrl)
        self.dt = float(dt)
        self.steps = int(steps)
        self.every_n = int(every_n)
        self.p_touch = float(p_touch)
        self.v_max = float(v_max)
        self.rng = np.random.default_rng(seed)
        self.ext = float(np.linalg.norm(self.x0.max(0) - self.x0.min(0)))
        self.depth = float(depth) * self.ext
        self.center = 0.5 * (self.x0.max(0) + self.x0.min(0))
        self.radius = 0.5 * self.ext
        self.normals = (estimate_normals(self.x0) if normals is None
                        else np.asarray(normals, np.float64))
        # 격자 밖으로 나가면 warp 가 범위 밖 주소를 건드려 죽는다. 끌고 가는
        # 입자까지 생각해 **안쪽으로 묶는다** (실제로 겪었다: 반경 12% 로 5681 개를
        # 끌고 z>2 로 나가 CUDA illegal memory access).
        self.bounds = (None if bounds is None
                       else (np.asarray(bounds[0], np.float64),
                             np.asarray(bounds[1], np.float64)))
        self._build()

    # ---------------------------------------------------------------- 표적
    def _clip(self, q):
        if self.bounds is None:
            return q
        return np.clip(q, self.bounds[0], self.bounds[1])

    def _target(self, cur):
        if self.rng.random() < self.p_touch:
            j = int(self.rng.integers(len(self.x0)))
            return self._clip(cur[j] - self.normals[j] * self.depth)
        d = self.rng.normal(size=3)
        d /= np.linalg.norm(d) + 1e-12
        r = self.radius * self.rng.uniform(1.15, 1.7)
        return self._clip(self.center + d * r)

    # ------------------------------------------------------------- 궤적 생성
    def _build(self):
        # 웨이포인트: 제어점마다 따로. 표적은 t0 구름에서 뽑는다 (교사 실행과
        # 무관하게 미리 정해져야 시드로 재현된다).
        n_way = int(np.ceil(self.steps / self.every_n)) + 3
        W = np.zeros((self.K, n_way, 3))
        W[:, 0] = self.x0[self.ctrl]
        for i in range(1, n_way):
            for k in range(self.K):
                W[k, i] = self._target(self.x0)
        self.W = W

        # 구간 i 는 **W[i] 에서 W[i+1] 까지**이고 양옆 점으로 접선을 잡는다.
        # (i, i+1, i+2, i+3) 으로 잡으면 W[0] 을 건너뛰어 첫 스텝에 순간이동한다.
        def quad(k, i):
            m = n_way - 1
            return (W[k, max(i - 1, 0)], W[k, i], W[k, min(i + 1, m)],
                    W[k, min(i + 2, m)])

        self.seg_steps = []
        for i in range(n_way - 1):
            need = self.every_n
            for k in range(self.K):
                u = np.linspace(0, 1, 64)
                pts = _catmull_rom(*quad(k, i), u)
                L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                # 표본으로 잰 길이는 실제보다 짧게 나오므로 조금 여유를 둔다
                need = max(need, int(np.ceil(1.01 * L
                                             / max(self.v_max * self.dt, 1e-12))))
            self.seg_steps.append(need)
            if sum(self.seg_steps) >= self.steps:
                break

        # Catmull-Rom 은 호길이 매개변수가 아니라, u 를 고르게 띄우면 구간 안에서
        # 속도가 들쭉날쭉해 상한을 1.4 배까지 넘는다. **호길이로 다시 매개화**한다.
        def arc_sample(k, i, ns):
            uu = np.linspace(0.0, 1.0, 256)
            pts = _catmull_rom(*quad(k, i), uu)
            seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            if cum[-1] < 1e-12:
                return np.repeat(pts[-1:], ns, 0)
            want = cum[-1] * (np.arange(1, ns + 1) / ns)
            u_eq = np.interp(want, cum, uu)
            return _catmull_rom(*quad(k, i), u_eq)

        pos = []
        for i, ns in enumerate(self.seg_steps):
            seg = np.stack([arc_sample(k, i, ns) for k in range(self.K)], 1)
            pos.append(seg)
        P = np.concatenate(pos, 0)
        if len(P) < self.steps:                              # 모자라면 마지막을 잡고 있는다
            P = np.concatenate([P, np.repeat(P[-1:], self.steps - len(P), 0)], 0)
        self.P = self._clip(P[:self.steps])
        V = np.zeros_like(self.P)
        V[1:] = (self.P[1:] - self.P[:-1]) / self.dt
        V[0] = (self.P[0] - self.x0[self.ctrl]) / self.dt
        self.V = V

    # ------------------------------------------------------------------ 조회
    def pos(self, step):
        return self.P[min(step, self.steps - 1)]

    def vel(self, step):
        return self.V[min(step, self.steps - 1)]

    def max_speed(self):
        return float(np.linalg.norm(self.V, axis=-1).max())


def pick_control_points(x, dx=None, k=4, seed=0):
    """표면 입자 중 **서로 떨어진** k 개를 고른다 (한 자리에 몰리면 조작이 안 된다)."""
    m = surface_mask(x, dx)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    first = int(rng.choice(idx))
    chosen = [first]
    d = np.linalg.norm(x[idx] - x[first], axis=1)
    for _ in range(k - 1):
        j = int(idx[np.argmax(d)])
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(x[idx] - x[j], axis=1))
    return np.array(chosen, np.int64), m
