"""GaussianFluent `particle_impulse` 경계 (config boundary_conditions type="particle_impulse") 를 우리 판 입자 속도 충격으로 거는 감싸개.

사용: python -u exe/impulse_wrap.py --impulse_bc 1 [<다음 감싸개.py> <그 인자...>] exe/fem_an_rt_sim.py <실행기 인자 ...>
--impulse_bc 가 없거나 0 이면 아무것도 안 함 (결과 불변). config 에 particle_impulse 가 없어도 할 일 없음.

GF 정의 (decode_param.py + mpm_solver_warp.add_impulse_on_particles):
  force (3), num_dt (기본 1), start_time (기본 0), point (기본 [1,1,1]), size (기본 [1,1,1]).
  대상 = 처음 위치 |x_p - point| < size (축별, 한 번만 골라 고정). 서브스텝 시각 time (= k * substep_dt) 이 [start, start + substep_dt * num_dt) 이면
  P2G 직전에 매 서브스텝 v_p += force / m_p * substep_dt.  => 총 속도 변화 = force/m_p * substep_dt * (창 안 서브스텝 수).
우리 구현: 서브스텝 개수를 프레임 [fi*h, (fi+1)*h) 와 겹치는 것만 세어, 그 프레임 S.step 직전에 입자 속도 s.v 에 합친 dv 를 더한다 (운동량 같음; 입자 질량 s.m 사용).
  s.v 는 그래프 밖 장치 버퍼라 그래프 재생 전에 바꾸면 그 프레임 P2G 에 쓰임 (S.step 이 되돌림용 백업을 뜨기 전이라 재시도에도 유지; 같은 t 로 다시 불러도 한 번만).
  --pbc (TREL/PVB) 는 구속 입자 속도를 정해진 값으로 두는 장치라 재사용하지 않음 -- 충격은 구속이 아니라 속도 더하기이므로 PVB 에 넣으면 의미가 다름.
못 옮기는 것: GF 는 서브스텝마다 충격이 들어가 그 사이 격자 상호작용이 있으나 우리는 한 프레임 한 번에 합쳐 준다 (num_dt 가 작은 충격이면 같은 운동량).
  GF 의 APIC C 는 우리 판에 없음 (영향 없음으로 둠).
"""
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))


def frame_counts(start, num_dt, dt, h, fi):
    """프레임 fi 에 속한 GF 서브스텝 (k*dt, k 정수) 중 [start, start+dt*num_dt) 안의 개수."""
    eps = 1e-6
    k0 = math.ceil(start / dt - eps)
    k1 = math.ceil(start / dt + num_dt - eps)                                          # time < start + dt*num_dt  <=>  k < start/dt + num_dt
    f0 = round(fi * h / dt)
    f1 = round((fi + 1) * h / dt)
    return max(0, min(k1, f1) - max(k0, f0))


def select_mask(x0, point, size):
    """GF selection_add_impulse_on_particles: |x - point| < size 축별 전부."""
    import torch
    return ((x0 - torch.tensor(point, dtype=x0.dtype, device=x0.device)).abs() < torch.tensor(size, dtype=x0.dtype, device=x0.device)).all(-1)


def install():
    import torch
    from anchorflow import fem_an_rt as ART
    orig = ART.RTSolver.step
    st = dict(init=False, imps=[], done=set())

    def step(s, t):
        if not st["init"]:
            st["init"] = True
            f = sys._getframe(1)
            while f is not None and not ("cfg" in f.f_globals and "d" in f.f_globals and f.f_globals.get("__name__") == "__main__"):
                f = f.f_back
            assert f is not None, "[impulse] 실행기 전역 (cfg, d) 을 못 찾음"
            g = f.f_globals; cfg = g["cfg"]
            x0 = g["d"]["x"][0].to(s.v.device).float()                                 # GF 좌표 (띄우기 전) 처음 위치
            for b in (cfg.get("boundary_conditions") or []):
                if b.get("type") != "particle_impulse":
                    continue
                m_o = select_mask(x0, [float(q) for q in b.get("point", [1, 1, 1])], [float(q) for q in b.get("size", [1, 1, 1])])
                m_s = m_o[s.perm.to(m_o.device)]                                       # 입자 정렬 순서로
                st["imps"].append(dict(force=[float(q) for q in b["force"]], num_dt=int(b.get("num_dt", 1)), start=float(b.get("start_time", 0.0)), mask=m_s))
                print(f"[impulse] force {b['force']} num_dt {b.get('num_dt', 1)} start {b.get('start_time', 0.0)} 대상 입자 {int(m_o.sum())}/{m_o.numel()}", flush=True)
            st["dt"] = float(cfg["substep_dt"]); st["h"] = float(s.h)
        key = round(t / st["h"])
        if st["imps"] and key not in st["done"]:
            st["done"].add(key)
            for im in st["imps"]:
                n = frame_counts(im["start"], im["num_dt"], st["dt"], st["h"], key)
                if n > 0:
                    fo = torch.tensor(im["force"], device=s.v.device, dtype=torch.float32)
                    mk = im["mask"].to(s.v.device)
                    s.v[mk] += fo[None, :] / s.m[mk][:, None] * (st["dt"] * n)
                    print(f"[impulse] 프레임 {key}: 서브스텝 {n} 개 분량 속도 충격 (대상 {int(mk.sum())})", flush=True)
        return orig(s, t)

    ART.RTSolver.step = step


if __name__ == "__main__":
    argv = sys.argv[1:]
    ON = 0
    while argv and argv[0] == "--impulse_bc":
        ON = int(float(argv[1])); del argv[:2]
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
            _done[0] = True
            install()
        return _orig(path, *a, **kw)

    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name="__main__")
