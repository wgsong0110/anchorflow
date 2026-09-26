"""GS-Verse 포팅. 공개 구현(C#/Unity)의 물리와 결합을 그대로 옮긴다.

공개된 저장소(Anastasiya999/GS-Verse)의 물리는 논문이 말하는 XPBD 가 아니라
`SplatDeformate.VertexSpringJob` 의 **정점별 복원 스프링 + 감쇠** 명시적 적분이다:

    v <- v - (x - x_rest) * springForce * dt
    v <- v * (1 - damping * dt)
    x <- x + v * dt

정점끼리 결합이 없고 부피항도 중력도 없다 (기본값 springForce=20, damping=5).
가우시안·입자는 GaMeS 방식으로 삼각형에 붙는다 -- 위치는 세 정점의 무게중심
조합이고 회전·크기는 삼각형 기하에서 나온다. 여기서는 표면 밖/안의 입자도
담아야 하므로 무게중심 좌표에 **법선 방향 오프셋**을 하나 더 둔다:

    x_p = a0 v0 + a1 v1 + a2 v2 + b * n(삼각형)

삼각형이 변형되면 그 국소 틀 [e1, e2, n] 의 변화가 그대로 입자를 옮기고, 같은
틀의 변화로 변형구배 F 도 만든다 (그쪽이 회전·크기를 만드는 길과 같다).
"""
import numpy as np
import torch

__all__ = ["mesh_from_points", "bind_points", "GSVerseSim"]


def mesh_from_points(x, n_grid=100, grid_lim=2.0, iso=0.5, smooth=1):
    """채우기 입자에서 표면 메시를 뽑는다 (PG 채우기가 쓰는 marching cubes).

    반환: (정점 [V,3] 시뮬 좌표, 삼각형 [T,3])
    """
    import mcubes
    dx = grid_lim / n_grid
    idx = (x / dx).long().clamp(0, n_grid - 1)
    occ = torch.zeros(n_grid, n_grid, n_grid, device=x.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # 격자를 살짝 뭉개 표면을 매끄럽게 한다 (구멍 난 등위면을 피한다)
    for _ in range(max(smooth, 0)):
        occ = torch.nn.functional.avg_pool3d(
            occ[None, None], 3, stride=1, padding=1)[0, 0]
        occ = occ / occ.max().clamp_min(1e-12)
    v, f = mcubes.marching_cubes(occ.detach().cpu().numpy().astype(np.float32),
                                 float(iso))
    v = torch.from_numpy(np.ascontiguousarray(v)).float().to(x.device) * dx
    f = torch.from_numpy(np.ascontiguousarray(f.astype(np.int64))).to(x.device)
    # marching cubes 는 바늘 같은 삼각형을 낸다. 그 국소 틀 [e1,e2,n] 이 거의
    # 특이해서 무게중심 좌표를 풀면 정밀도가 날아간다 (결합 오차 5e-2 를 봤다).
    ar = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]],
                     dim=-1).norm(dim=-1) * 0.5
    keep = ar > 1e-3 * float(ar.median())
    return v, f[keep]


def _tri_frame(v, f):
    """삼각형의 국소 틀. 반환 (원점 v0 [T,3], e1, e2, n)."""
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    n = torch.cross(e1, e2, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v0, e1, e2, n


def bind_points(x, v, f, chunk=4096):
    """입자를 가장 가까운 삼각형에 묶는다. 반환 (삼각형 색인, a1, a2, b).

    x = v0 + a1 e1 + a2 e2 + b n 이 되도록 국소 틀에서 좌표를 푼다 (정확히
    복원되므로 t=0 에서 위치가 한 비트도 달라지지 않는다).
    """
    v0, e1, e2, n = _tri_frame(v, f)
    ctr = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    A = torch.stack([e1, e2, n], -1).double()             # [T,3,3]
    ti = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        e = min(s + chunk, x.shape[0])
        ti[s:e] = torch.cdist(x[s:e], ctr).argmin(1)
    # 배수가 아니라 **풀어서** 얻는다. 기저가 정칙이면 정확히 복원된다.
    loc = torch.linalg.solve(A[ti], (x - v0[ti]).double().unsqueeze(-1))
    loc = loc.squeeze(-1).to(x.dtype)
    return ti, loc[:, 0], loc[:, 1], loc[:, 2]


class GSVerseSim:
    """정점별 복원 스프링 적분기 + 삼각형 결합 복원."""

    def __init__(self, x_rest, n_grid=100, grid_lim=2.0, spring=20.0,
                 damping=5.0, radius=0.15, substeps=1):
        self.dev = x_rest.device
        self.vr, self.f = mesh_from_points(x_rest, n_grid, grid_lim)
        self.v = self.vr.clone()
        self.vel = torch.zeros_like(self.v)
        self.spring, self.damping = float(spring), float(damping)
        self.radius = float(radius)
        self.substeps = max(int(substeps), 1)
        self.ti, self.a1, self.a2, self.b = bind_points(x_rest, self.vr, self.f)
        self.A0 = torch.stack(_tri_frame(self.vr, self.f)[1:], -1)   # [T,3,3]
        self.A0i = torch.linalg.pinv(self.A0)
        fit = float((self.positions() - x_rest).norm(dim=-1).max())
        print(f"[GS-Verse] 정점 {self.v.shape[0]} 삼각형 {self.f.shape[0]}, "
              f"결합 오차 최대 {fit:.2e}", flush=True)

    def positions(self):
        v0, e1, e2, n = _tri_frame(self.v, self.f)
        t = self.ti
        return (v0[t] + self.a1.unsqueeze(-1) * e1[t]
                + self.a2.unsqueeze(-1) * e2[t] + self.b.unsqueeze(-1) * n[t])

    def F(self):
        """입자별 변형구배. 국소 틀의 변화 A A0^{-1} 를 쓴다 (그쪽 회전·크기와 같은 길)."""
        A = torch.stack(_tri_frame(self.v, self.f)[1:], -1)
        return (A @ self.A0i)[self.ti]

    def step(self, dt, handle_pos=None, handle_vel=None):
        """한 프레임. 손잡이는 PG 규약 (1-q^2)^2 가중으로 정점 속도에 섞는다."""
        h = dt / self.substeps
        for _ in range(self.substeps):
            if handle_pos is not None:
                for k in range(handle_pos.shape[0]):
                    q = ((self.v - handle_pos[k]).norm(dim=-1)
                         / max(self.radius, 1e-9)).clamp(0, 1)
                    w = ((1.0 - q * q) ** 2).unsqueeze(-1)
                    self.vel = (1.0 - w) * self.vel + w * handle_vel[k]
            self.vel = self.vel - (self.v - self.vr) * self.spring * h
            self.vel = self.vel * (1.0 - self.damping * h)
            self.v = self.v + self.vel * h
        return self.positions()
