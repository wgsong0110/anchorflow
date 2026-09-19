"""파괴·유동·접합 씬을 Taichi MLS-MPM 으로 직접 만든다.

기성 솔버 둘은 우리가 원하는 세 가지를 다 못 준다.

  PhysGaussian  소성 연화뿐이라 **끊어지지 않는다**
  GaussianFluent CD-MPM  Cam-Clay 라 흙처럼 전체가 흘러내린다
  둘 다          눌러 붙여도 **접합이 안 된다** (응집 모델이 없다)

그래서 구성식을 직접 넣는다. 재질 셋을 담는다.

  0 elastic_damage  탄성으로 버티다 최대 신장이 문턱을 넘으면 손상이 쌓이고
                    응력이 (1-d) 로 줄어 그 자리에서 끊어진다. 손상은 비가역이다
  1 fluid           약압축성. F 를 부피비 J 하나로 줄이고 압력만 낸다
  2 cohesive        **비대칭 항복**: 압축에서는 무제한으로 흘러(접촉이 아물고)
                    인장에서는 유한한 항복을 버틴다. 찰흙이 실제로 하는 일이고,
                    이래야 눌러 붙인 두 덩어리가 한 몸이 된다

형상은 절차적으로 찍는다 (3DGS 자산이 필요 없다). 출력은 프레임마다 h5 로,
키 이름을 기성 러너와 맞춰 두어 기존 렌더러·측정 도구가 그대로 붙는다.
"""
# `from __future__ import annotations` 를 쓰면 안 된다 -- 그러면 모든 주석이
# 문자열이 되어 타이치가 커널 인자 타입을 못 읽는다 ("Invalid type annotation").
import argparse
import os

import h5py
import numpy as np
import taichi as ti

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True,
                choices=("fracture", "flow", "merge", "tear", "impact",
                         "collide", "sand"))
ap.add_argument("--out", required=True)
ap.add_argument("--n_grid", type=int, default=64)
ap.add_argument("--frames", type=int, default=240)
ap.add_argument("--substeps", type=int, default=24)
ap.add_argument("--dt", type=float, default=1e-4)
ap.add_argument("--spacing", type=float, default=0.008,
                help="입자 간격 (월드 단위, 영역은 [0,1]^3)")
ap.add_argument("--E", type=float, default=5e3)
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--rho", type=float, default=1000.0)
ap.add_argument("--dmg_thresh", type=float, default=1.06,
                help="최대 특이값이 이 값을 넘으면 손상이 쌓인다")
ap.add_argument("--dmg_rate", type=float, default=40.0)
ap.add_argument("--yield_t", type=float, default=0.02,
                help="cohesive 의 **인장 체적** 항복 (로그 변형 단위)")
ap.add_argument("--yield_s", type=float, default=0.004,
                help="cohesive 의 **전단** 항복. 낮을수록 모양을 쉽게 잊는다")
ap.add_argument("--friction", type=float, default=35.0,
                help="모래의 마찰각(도)")
ap.add_argument("--pull", type=float, default=0.35, help="구동기 속도")
ap.add_argument("--v0", type=float, default=0.0,
                help="초기 속도 크기 (impact/collide 에서 쓴다)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=a.seed)

dx = 1.0 / a.n_grid
inv_dx = float(a.n_grid)
p_vol = (a.spacing) ** 3
p_mass = p_vol * a.rho
mu0 = a.E / (2 * (1 + a.nu))
lam0 = a.E * a.nu / ((1 + a.nu) * (1 - 2 * a.nu))
K_fluid = a.E
_sf = np.sin(np.radians(a.friction))
ALPHA = float(np.sqrt(2.0 / 3.0) * 2.0 * _sf / (3.0 - _sf))


# ----------------------------------------------------------- 형상 만들기
def box(lo, hi, sp):
    g = np.stack(np.meshgrid(*[np.arange(lo[i] + sp / 2, hi[i], sp)
                               for i in range(3)], indexing="ij"), -1)
    return g.reshape(-1, 3)


def ball(c, r, sp):
    q = box([c[i] - r for i in range(3)], [c[i] + r for i in range(3)], sp)
    return q[np.linalg.norm(q - np.array(c), axis=1) <= r]


rng = np.random.RandomState(a.seed)
if a.scene == "fracture":
    pts = box([0.28, 0.44, 0.44], [0.72, 0.56, 0.56], a.spacing)
    mat = np.zeros(len(pts), np.int32)
elif a.scene == "flow":
    pts = box([0.10, 0.10, 0.10], [0.34, 0.60, 0.60], a.spacing)
    mat = np.ones(len(pts), np.int32)
elif a.scene == "tear":
    # 얇은 판을 x 로 당긴다. 한쪽 가장자리에 홈을 파 두면 응력이 그 끝에 몰려
    # 균열이 홈에서 출발해 z 방향으로 달린다 -- 이것이 인열이다.
    pts = box([0.24, 0.47, 0.32], [0.76, 0.53, 0.68], a.spacing)
    notch = (np.abs(pts[:, 0] - 0.5) < 1.2 * a.spacing) & (pts[:, 2] < 0.44)
    pts = pts[~notch]
    mat = np.zeros(len(pts), np.int32)
elif a.scene == "sand":
    # 모래 기둥이 제 무게로 무너진다 -- 소성 유동
    pts = box([0.34, 0.34, 0.06], [0.58, 0.58, 0.62], a.spacing)
    mat = np.full(len(pts), 3, np.int32)
elif a.scene == "impact":
    # 취성 구가 바닥에 떨어져 깨진다. 당기는 구동기가 없고 충돌이 파괴를 만든다.
    pts = ball([0.5, 0.5, 0.62], 0.13, a.spacing)
    mat = np.zeros(len(pts), np.int32)
elif a.scene == "collide":
    # 두 취성 구가 정면으로 부딪혀 깨진다.
    p1 = ball([0.30, 0.5, 0.5], 0.11, a.spacing)
    p2 = ball([0.70, 0.5, 0.5], 0.11, a.spacing)
    pts = np.concatenate([p1, p2])
    mat = np.zeros(len(pts), np.int32)
else:                                        # merge
    p1 = ball([0.38, 0.5, 0.5], 0.10, a.spacing)
    p2 = ball([0.62, 0.5, 0.5], 0.10, a.spacing)
    pts = np.concatenate([p1, p2])
    mat = np.full(len(pts), 2, np.int32)
pts = pts + rng.uniform(-a.spacing / 4, a.spacing / 4, pts.shape)
N = len(pts)
print(f"[씬] {a.scene}, 입자 {N}, 격자 {a.n_grid}, dx {dx:.4f}, "
      f"입자간격 {a.spacing}", flush=True)

x = ti.Vector.field(3, float, N)
v = ti.Vector.field(3, float, N)
C = ti.Matrix.field(3, 3, float, N)
F = ti.Matrix.field(3, 3, float, N)
Jf = ti.field(float, N)              # 유체의 부피비
dmg = ti.field(float, N)
mt = ti.field(ti.i32, N)
gv = ti.Vector.field(3, float, (a.n_grid,) * 3)
gm = ti.field(float, (a.n_grid,) * 3)

x.from_numpy(pts.astype(np.float32))
mt.from_numpy(mat)


V0 = np.zeros((N, 3), np.float32)
if a.scene == "impact":
    V0[:, 2] = -a.v0
elif a.scene == "collide":
    V0[:, 0] = np.where(pts[:, 0] < 0.5, a.v0, -a.v0)
v0f = ti.Vector.field(3, float, N)
v0f.from_numpy(V0)


@ti.kernel
def init():
    for p in x:
        v[p] = v0f[p]
        C[p] = ti.Matrix.zero(float, 3, 3)
        F[p] = ti.Matrix.identity(float, 3)
        Jf[p] = 1.0
        dmg[p] = 0.0


@ti.kernel
def substep(t: ti.f32, grav: ti.f32, pull: ti.f32):
    for I in ti.grouped(gm):
        gm[I] = 0.0
        gv[I] = ti.Vector([0.0, 0.0, 0.0])

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 3) + a.dt * C[p]) @ F[p]
        stress = ti.Matrix.zero(float, 3, 3)

        if mt[p] == 1:                                   # 유체
            Jf[p] *= 1.0 + a.dt * C[p].trace()
            press = K_fluid * (Jf[p] - 1.0)
            stress = -press * ti.Matrix.identity(float, 3)
            F[p] = ti.Matrix.identity(float, 3)
        else:
            U, sig, V = ti.svd(F[p])
            s0 = ti.max(sig[0, 0], ti.max(sig[1, 1], sig[2, 2]))
            if mt[p] == 2:
                # 찰흙: **전단만** 자유롭게 흐르고(모양을 잊어 접촉이 아문다)
                # **체적은 탄성**으로 남겨 압력이 살아 있어야 한다. 앞판은 체적까지
                # 풀어버려서 압력이 0 이 되는 바람에 두 공이 서로 통과했다.
                e0 = ti.log(ti.max(sig[0, 0], 1e-4))
                e1 = ti.log(ti.max(sig[1, 1], 1e-4))
                e2 = ti.log(ti.max(sig[2, 2], 1e-4))
                tr = e0 + e1 + e2
                d0, d1, d2 = e0 - tr / 3.0, e1 - tr / 3.0, e2 - tr / 3.0
                nrm = ti.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-12
                if nrm > a.yield_s:                      # 전단 항복 (아주 낮다)
                    k = a.yield_s / nrm
                    d0, d1, d2 = d0 * k, d1 * k, d2 * k
                tr = ti.min(tr, 3.0 * a.yield_t)         # 인장 체적은 여기까지만
                sig[0, 0] = ti.exp(d0 + tr / 3.0)
                sig[1, 1] = ti.exp(d1 + tr / 3.0)
                sig[2, 2] = ti.exp(d2 + tr / 3.0)
                F[p] = U @ sig @ V.transpose()
            elif mt[p] == 3:
                # 모래: Drucker-Prager (Klar et al. 2016). 인장은 못 버티고,
                # 전단은 마찰각이 정하는 원뿔 위로 되돌린다 -- 소성 유동이다.
                e0 = ti.log(ti.max(sig[0, 0], 1e-4))
                e1 = ti.log(ti.max(sig[1, 1], 1e-4))
                e2 = ti.log(ti.max(sig[2, 2], 1e-4))
                tr = e0 + e1 + e2
                d0, d1, d2 = e0 - tr / 3.0, e1 - tr / 3.0, e2 - tr / 3.0
                nrm = ti.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-12
                if tr > 0.0:                             # 인장이면 응집이 없다
                    sig[0, 0] = 1.0; sig[1, 1] = 1.0; sig[2, 2] = 1.0
                else:
                    dg = nrm + (3.0 * lam0 + 2.0 * mu0) / (2.0 * mu0) * tr * ALPHA
                    if dg > 0.0:
                        k = dg / nrm
                        sig[0, 0] = ti.exp(e0 - k * d0)
                        sig[1, 1] = ti.exp(e1 - k * d1)
                        sig[2, 2] = ti.exp(e2 - k * d2)
                F[p] = U @ sig @ V.transpose()
            J = sig[0, 0] * sig[1, 1] * sig[2, 2]
            R = U @ V.transpose()
            stress = (2.0 * mu0 * (F[p] - R) @ F[p].transpose()
                      + ti.Matrix.identity(float, 3) * lam0 * J * (J - 1.0))
            if mt[p] == 0:
                # 최대 신장이 문턱을 넘은 만큼 손상이 쌓인다. 줄지 않는다.
                if s0 > a.dmg_thresh:
                    dmg[p] = ti.min(1.0, dmg[p] + a.dmg_rate * a.dt
                                    * (s0 - a.dmg_thresh))
                stress = (1.0 - dmg[p]) * stress

        stress = (-a.dt * p_vol * 4.0 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            off = ti.Vector([i, j, k])
            dpos = (off.cast(float) - fx) * dx
            wt = w[i][0] * w[j][1] * w[k][2]
            ti.atomic_add(gv[base + off], wt * (p_mass * v[p] + affine @ dpos))
            ti.atomic_add(gm[base + off], wt * p_mass)

    # 격자 갱신과 구동기를 **한 루프 안에서** 한다. 타이치는 최상위 for 만
    # 병렬 루프로 인정해서, if 안에 for 를 넣으면 struct_for 중첩으로 거부한다.
    for I in ti.grouped(gm):
        if gm[I] > 0:
            gv[I] = gv[I] / gm[I]
            gv[I][2] += a.dt * grav
            for d in ti.static(range(3)):
                if I[d] < 3 and gv[I][d] < 0:
                    gv[I][d] = 0.0
                if I[d] > a.n_grid - 3 and gv[I][d] > 0:
                    gv[I][d] = 0.0
            # 구동기: 양끝을 x 방향으로 강제한다
            if pull != 0.0:
                px = float(I[0]) * dx
                if px < 0.32:
                    gv[I] = ti.Vector([-pull, 0.0, 0.0])
                elif px > 0.68:
                    gv[I] = ti.Vector([pull, 0.0, 0.0])

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        nv = ti.Vector.zero(float, 3)
        nC = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            off = ti.Vector([i, j, k])
            dpos = off.cast(float) - fx
            wt = w[i][0] * w[j][1] * w[k][2]
            g = gv[base + off]
            nv += wt * g
            nC += 4.0 * inv_dx * wt * g.outer_product(dpos)
        v[p], C[p] = nv, nC
        x[p] += a.dt * nv
        # 유체는 압력만으로 버티므로 한 입자가 영역을 벗어나면 다음 P2G 의 격자
        # 색인이 범위를 넘어 커널이 죽는다. 영역 안으로 집어넣고 속도를 눕힌다.
        for d in ti.static(range(3)):
            if x[p][d] < 2.0 * dx:
                x[p][d] = 2.0 * dx
                if v[p][d] < 0:
                    v[p][d] = 0.0
            if x[p][d] > 1.0 - 2.0 * dx:
                x[p][d] = 1.0 - 2.0 * dx
                if v[p][d] > 0:
                    v[p][d] = 0.0


import time
init()
os.makedirs(a.out, exist_ok=True)
t_sim = t_io = 0.0
t_all = time.time()
GRAV = {"fracture": 0.0, "flow": -9.8, "merge": 0.0, "tear": 0.0,
        "impact": -9.8, "collide": 0.0, "sand": -9.8}[a.scene]
for f in range(a.frames + 1):
    _t = time.time()
    xs = x.to_numpy(); vs = v.to_numpy(); Fs = F.to_numpy().reshape(-1, 9)
    with h5py.File(os.path.join(a.out, "sim_%010d.h5" % f), "w") as h:
        h.create_dataset("x", data=xs.T)
        h.create_dataset("v", data=vs.T)
        h.create_dataset("f_tensor", data=Fs.T)
        h.create_dataset("damage", data=dmg.to_numpy())
        h.create_dataset("time", data=np.array([[f * a.substeps * a.dt]]))
    t_io += time.time() - _t
    if f == a.frames:
        break
    _t = time.time()
    # merge 는 먼저 누르고(안쪽), 머물다가, 당긴다(바깥쪽)
    pull = 0.0
    if a.scene in ("fracture", "tear"):
        pull = a.pull
    elif a.scene == "merge":
        fr = f / a.frames
        pull = (-a.pull * 0.6 if fr < 0.25 else
                (0.0 if fr < 0.45 else a.pull))
    for _ in range(a.substeps):
        substep(f * a.substeps * a.dt, GRAV, pull)
    ti.sync()
    t_sim += time.time() - _t
    if f % 20 == 0:
        fin = np.isfinite(xs).all(1)
        print(f"  f{f:4d}  손상 중앙 {np.median(dmg.to_numpy()):.3f} "
              f"비유한 {int((~fin).sum())}  x[{xs[fin].min():.3f},"
              f"{xs[fin].max():.3f}]", flush=True)
print(f"[저장] {a.out}  {a.frames + 1} 프레임", flush=True)
print(f"[시간] 전체 {time.time() - t_all:.1f}s = 시뮬 {t_sim:.1f}s "
      f"({t_sim / max(a.frames, 1) * 1e3:.1f} ms/프레임, 서브스텝 {a.substeps}) "
      f"+ h5 쓰기 {t_io:.1f}s", flush=True)
print("SYNTH_OK", flush=True)
