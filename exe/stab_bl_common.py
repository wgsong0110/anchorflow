"""안정 최소 서브스텝 표 (doc/pareto.md 절 S) 의 공개 코드 베이스라인 (PhysTwin · Simplicits · Spring-Gaus) 공통 부분.

각 실행기 (exe/stab_phystwin.py · stab_simplicits.py · stab_sg.py) 가 서로 다른 환경 (af · simp) 에서 돌아서
anchorflow 라이브러리를 import 하지 않고 여기 것만 쓴다. 정의는 우리 실행기와 같음:
  - 씬: --data 의 .pt (d["x"] [T+1,P,3] 정답 입자 궤적, d["v"] 속도, d["cfg"] PG config) -- pareto_vrgs.py 와 같은 파일.
  - EXT = 0 프레임 바운딩박스 대각선 (deform_model.extent 와 같은 식), 프레임 오차 = 평균 |x−x_gt| / EXT (전 입자).
  - 발산 = 프레임 위치 비유한 또는 프레임 오차 > 2 (절 BD 정의) -> 그 프레임에서 멈춤.
  - 입자 질량 = PG 규약 (격자 칸 부피 / 칸 안 입자 수 x 밀도; deform_model.traj_mass 와 같은 식), --region_density 면 영역 밀도 (region_rho 와 같은 식).
  - 출력 summary.json = bnv·우리 실행기와 같은 열쇠 (diverged = 발산 프레임 또는 None, err_mean, err_mean_done, ms_mean, ms_max) -> exe/stab.py parse 가 그대로 읽음.
  - 덤프 pred.pt = {pred [L,P,3], F [L,P,3,3]} (프레임 1..L, 전 입자) -> exe/lb_roll_to_h5.py -> exe/lb_render.py (stab.py visual).
"""
import json
import math
import os
import time

import numpy as np
import torch


def add_scene_args(ap):
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--floor", choices=["vel", "none"], default="vel")
    ap.add_argument("--cfg_bc", action="store_true", help="config 의 cuboid 고정 경계 (ficus 화분)")
    ap.add_argument("--region_density", action="store_true")
    ap.add_argument("--pg_vdamp", action="store_true", help="(받기만 함 -- 베이스라인은 자기 감쇠 규약을 씀)")
    ap.add_argument("--warm", type=int, default=2, help="시간 평균에서 뺄 첫 프레임 수")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--dump_F", action="store_true")


def load_scene(a, dev="cpu"):
    d = torch.load(a.data, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    T = int(d["x"].shape[0]) - 1
    L = min(a.frames, T)
    x0 = d["x"][0].double()
    # 질량 (deform_model.traj_mass 와 같은 식)
    ng = int(cfg.get("n_grid", 100))
    dx = float(cfg.get("grid_lim", 2.0)) / ng
    vi = (x0 / dx).long().clamp(0, ng - 1)
    fl = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
    cn = torch.zeros(ng ** 3, dtype=torch.float64).index_add_(0, fl, torch.ones(x0.shape[0], dtype=torch.float64))
    rho = torch.full((x0.shape[0],), float(cfg["density"]), dtype=torch.float64)
    if a.region_density:
        for r in (cfg.get("additional_material_params") or []):
            if float(r.get("E", cfg["E"])) != float(cfg["E"]) or float(r.get("nu", cfg["nu"])) != float(cfg["nu"]):
                raise SystemExit(f"--region_density: E/nu 가 다른 영역은 없음 ({r})")
            q = torch.as_tensor(r["point"], dtype=torch.float64)
            sz = torch.as_tensor(r["size"], dtype=torch.float64)
            rho[((x0 > q - sz) & (x0 < q + sz)).all(-1)] = float(r["density"])
    vol = (dx ** 3) / cn[fl]
    FZ = None
    if a.floor == "vel":
        pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
        assert abs(float(pl["normal"][2]) - 1.0) < 1e-6, "바닥 법선이 z 가 아님"
        FZ = float(pl["point"][2])
    BOX, BOXW, MOV = parse_bcs(cfg, L * float(cfg["frame_dt"])) if a.cfg_bc else ([], [], [])
    ROT = []                                                                             # paper_remeasure2: PG 회전 구동기 -> [(점, 단위 법선, 반 높이, 반지름, [(회전 몫, 이동 몫, 시작, 끝), ..]), ..]
    if a.cfg_bc:
        _grp = {}
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "enforce_particle_velocity_rotation":
                nv_ = torch.as_tensor(bc["normal"], dtype=torch.float64); nv_ = (nv_ / nv_.norm()).tolist()
                k_ = (tuple(float(q) for q in bc["point"]), tuple(nv_), float(bc["half_height_and_radius"][0]), float(bc["half_height_and_radius"][1]))
                _grp.setdefault(k_, []).append((float(bc["rotation_scale"]), float(bc["translation_scale"]), float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))))
        ROT = [(k_[0], list(k_[1]), k_[2], k_[3], w_) for k_, w_ in _grp.items()]
    g = [float(q) for q in cfg["g"]]
    assert abs(g[0]) < 1e-12 and abs(g[1]) < 1e-12 and g[2] <= 0, f"중력이 -z 또는 0 이 아님 {g}"   # ficus 공식 config = 0
    S = dict(d=d, cfg=cfg, L=L, x0=x0, X=d["x"][:L + 1], v0=d["v"][0].double(), vol=vol, rho=rho, mass=vol * rho,
             EXT=float((x0.max(0).values - x0.min(0).values).norm()), FZ=FZ, BOX=BOX, BOXW=BOXW, MOV=MOV, ROT=ROT, g=g, frame_dt=float(cfg["frame_dt"]),
             E=float(cfg["E"]), nu=float(cfg["nu"]))
    return S


def parse_bcs(cfg, horizon):
    """config 고정·구동 경계 -> (BOX 상시 고정 [(점, 크기)], BOXW 구간 고정 [(점, 크기, 시작, 끝)], MOV 구간 등속 구동 [(점, 크기, 시작, 끝, 속도)]).
    상자 안 판정은 처음 위치 (PG selection_enforce_particle_velocity_translation 은 경계 추가 때 = 0 프레임 입자 위치로 마스크).
      cuboid (격자 속도 0): 예전 그대로 (구간이 롤아웃 전체면 BOX, 아니면 BOXW).
      enforce_particle_translation: 속도 0 이고 구간이 롤아웃 전체면 BOX (예전 그대로), 속도 0 구간 -> BOXW,
        [blx] 속도 ≠ 0 -> MOV (bread: 위·아래 띠를 ±0.2 m/s 로 당김, 아래 띠는 2.2 s 까지).
      [blx] release_particles_sequentially (wolf): PG mpm_solver_warp.release_particles_sequentially 와 같은 50 개 속도 0 상자
        (법선 축: 중심 end_position, 반폭 |start−end|/50 x (50−i), 다른 축: 중심 1 반폭 1; 구간 [start_time, end_time/50 x (i+1))) -> BOXW 50 개."""
    BOX, BOXW, MOV = [], [], []
    for bc in (cfg.get("boundary_conditions") or []):
        ty = bc.get("type")
        if ty == "cuboid":
            assert max(abs(float(q)) for q in bc["velocity"]) == 0.0
            w_ = cuboid_window(bc, horizon)
            if w_ is None:
                BOX.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]]))
            else:                                                                        # final_pa_bl: 시간 구간 cuboid (pillow2sofa 방석 받침)
                BOXW.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]], w_[0], w_[1]))
        elif ty == "enforce_particle_translation":
            pt, sz = [float(q) for q in bc["point"]], [float(q) for q in bc["size"]]
            vel = [float(q) for q in bc["velocity"]]
            t0, t1 = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
            if max(abs(q) for q in vel) == 0.0:
                if t0 <= 0.0 and t1 > horizon:
                    BOX.append((pt, sz))                                                 # paper_remeasure2: 정지 영역 (vasedeck 윗부분·pillow2sofa 소파)
                else:
                    BOXW.append((pt, sz, t0, t1))
            else:
                MOV.append((pt, sz, t0, t1, vel))
        elif ty == "release_particles_sequentially":
            nl = 50                                                                      # PG 가 인자를 무시하고 50 으로 둠
            nv = [float(q) for q in bc["normal"]]
            pt, sz, ax = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1
            for k in range(3):
                if nv[k] == 0:
                    pt[k], sz[k] = 1.0, 1.0
                else:
                    ax, pt[k] = k, float(bc["end_position"])
            hl = abs(float(bc["start_position"]) - float(bc["end_position"])) / nl
            etp = float(bc["end_time"]) / nl
            for i in range(nl):
                s2 = list(sz); s2[ax] = hl * (nl - i)
                BOXW.append((list(pt), s2, float(bc["start_time"]), etp * (i + 1)))
    return BOX, BOXW, MOV


def plastic_params(cfg, E=None):
    """[blx] 재질 확장 (원 논문은 탄성 전용): PG config -> 소성 매개변수. E 는 각 방법이 논문 절차로 맞춘 값이 있으면 그것 (반환 사상의 mu·lam), 없으면 config.
    PG mpm_solver_warp 기본값: softening 0.1, hardening 0, xi 0, friction_angle 25 (sin 은 3.14159265 로)."""
    mat = cfg.get("material", "jelly")
    E = float(cfg["E"]) if E is None else float(E)
    nu = float(cfg["nu"])
    mu = E / (2.0 * (1.0 + nu)); lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    phi = float(cfg.get("friction_angle", 25.0))
    sp = math.sin(phi / 180.0 * 3.14159265)
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * sp / (3.0 - sp)
    ys = float(cfg.get("yield_stress", 0.0))
    return dict(material=mat, E=E, nu=nu, mu=mu, lam=lam, friction_angle=phi, alpha=alpha,
                cdp=(3.0 * lam + 2.0 * mu) / (2.0 * mu) * alpha,                         # PG sand: δγ = |ε̂| + (3λ+2μ)/(2μ) α tr ε
                yield_stress=ys, softening=float(cfg.get("softening", 0.1)), hardening=int(cfg.get("hardening", 0)),
                eps_y=(math.sqrt(1.5) * ys / E if E > 0 else 0.0))                       # 스프링 항복 로그 변형률: 단축 응력에서 PG von Mises 항복 (|dev τ| = σy) 이 되는 축 변형률


def spring_plastic_update(cur, rest, pp, tr=None):
    """[blx] 스프링 소성 휴지 길이 (표준 방식: 변형률이 항복을 넘으면 휴지 길이가 따라감). cur·rest 같은 모양 텐서, 로그 변형률 ε = log(cur/rest).
      plasticine (PG von Mises): |ε| > ε_y 이면 rest = cur exp(−sign(ε) ε_y)  (ε_y = sqrt(3/2) σy / E; 단축 응력 환산).
      sand (PG Drucker–Prager, 스프링 근사 -- 정확히 옮길 수 없음): tr = 스프링 양 끝 점의 체적 변형률 추정 (점마다 이어진 스프링 ε 평균 x 3) 의 평균,
        축 편차 e = ε − tr/3, |ε̂| ≈ sqrt(3/2)|e| (축대칭 상태), δγ = sqrt(3/2)|e| + cdp tr.
        δγ ≤ 0: 그대로 / δγ > 0 · tr > 0 (인장): rest = cur (응력 0, PG F_e = U Vᵀ) / δγ > 0 · tr ≤ 0: e 를 δγ 만큼 줄임 (PG H = ε − ε̂ δγ/|ε̂|) -> rest = cur exp(−(tr/3 + e'))."""
    eps = torch.log(cur.clamp(min=1e-12) / rest.clamp(min=1e-12))
    if pp["material"] == "plasticine":
        ey = pp["eps_y"]
        if ey <= 0:
            return rest
        over = eps.abs() > ey
        return torch.where(over, cur * torch.exp(-torch.sign(eps) * ey), rest)
    if pp["material"] == "sand":
        e = eps - tr / 3.0
        en = math.sqrt(1.5) * e.abs()
        dg = en + pp["cdp"] * tr
        e2 = e * (1.0 - dg / en.clamp(min=1e-30))
        r_c = cur * torch.exp(-(tr / 3.0 + e2))
        r_t = cur
        return torch.where(dg <= 0, rest, torch.where(tr > 0, r_t, r_c))
    return rest


def cuboid_window(bc, horizon):
    """[final_pa_bl] PG cuboid 의 시간 구간 (PhysGaussian mpm_utils set_velocity_on_cuboid: start_time <= t < end_time 동안만 상자 안 격자 속도 = velocity,
    그 뒤 자유). 구간이 롤아웃 [0, horizon] 전체를 덮으면 None (= 예전처럼 상시 고정 -- 결과 불변), 아니면 (start, end).
    reset=1 (end 뒤 15 dt 동안 전체 격자 속도 0) 은 end 가 롤아웃 밖일 때만 허용."""
    t0, t1 = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
    if t0 <= 0.0 and t1 > horizon:
        return None
    assert int(bc.get("reset", 0)) == 0, f"구간 안에서 끝나는 reset cuboid 는 미지원 {bc}"
    return (t0, t1)


def box_active(t0, t1, t, eps=1e-6):
    """PG 조건 start <= t < end (t 는 프레임 시작 시각; 부동소수 오차 eps)."""
    return t0 <= t + eps and t + eps < t1


def time_scale(Kl, g):
    """닮음 변환 시간 배율 (절 S.3 공식 밖 (1)): 길이 Kl 배일 때 중력이 9.8 이 되게 c = sqrt(Kl |g| / 9.8).
    중력 0 (ficus 공식 config) 이면 맞출 중력이 없어 c = 1 (길이만 바꿈; 사용자 결정 11-05)."""
    gz = abs(float(g[2]))
    return (Kl * gz / 9.8) ** 0.5 if gz > 0 else 1.0


def rot_membership(x, rot):
    """회전 구동 원기둥 (처음 위치 기준) 안 입자 마스크 [(점, 법선, 반 높이, 반지름, 구간들)] -> 그룹마다 bool [N]."""
    out = []
    for (pt, nv, hh, rr, _w) in rot:
        n_ = torch.as_tensor(nv, device=x.device, dtype=x.dtype); off = x - torch.as_tensor(pt, device=x.device, dtype=x.dtype)
        ax = off @ n_; rp = off - ax[:, None] * n_
        out.append((ax.abs() < hh) & (rp.norm(dim=-1) < rr))
    return out


def rot_traj(x0, rot, masks, L, h):
    """구동 입자의 프레임별 위치 [L+1, N, 3] (PG 정의: v = rs (r_perp x n) + ts n = 각속도 −rs n 의 강체 회전 + n 방향 이동, 프레임 [(f−1)h, f h) 와 겹친 몫 f 만큼 곱해 합침).
    프레임 f 위치는 f−1 위치에서 한 프레임 회전 (로드리게스) -- 우리 판 노드 구속·VR-GS 구동과 같은 프레임 평균 속도."""
    X = x0.unsqueeze(0).repeat(L + 1, 1, 1).clone()
    for (pt, nv, hh, rr, wins), msk in zip(rot, masks):
        n_ = torch.as_tensor(nv, dtype=x0.dtype); p_ = torch.as_tensor(pt, dtype=x0.dtype)
        for f in range(1, L + 1):
            t = (f - 1) * h
            rs = ts = 0.0
            for (r_, t_, t0, t1) in wins:
                fr = max(0.0, min(t + h, t1) - max(t, t0)) / h
                rs += r_ * fr; ts += t_ * fr
            th = -rs * h                                            # 각속도 -rs n
            r = X[f - 1][msk] - p_
            ax = (r @ n_)[:, None] * n_; rp = r - ax
            rot_ = ax + rp * torch.cos(torch.tensor(th, dtype=x0.dtype)) + torch.cross(n_.expand_as(rp), rp, dim=-1) * torch.sin(torch.tensor(th, dtype=x0.dtype))
            X[f][msk] = p_ + rot_ + ts * h * n_
    return X


def in_boxes(x, boxes):
    """|x − point| < size (모든 축) -- vrgs_xpbd 고정 상자와 같은 판정."""
    m = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for b_ in boxes:
        pt, sz = b_[0], b_[1]
        m |= ((x - torch.as_tensor(pt, device=x.device, dtype=x.dtype)).abs() < torch.as_tensor(sz, device=x.device, dtype=x.dtype)).all(-1)
    return m


class Roll:
    """프레임마다 예측 전 입자 위치 (와 F) 를 받아 오차·발산·시간·덤프를 모은다."""

    def __init__(s, a, S, N, extra=None):
        s.a, s.S, s.N = a, S, N
        s.errs, s.ms, s.div = [], [], None
        s.extra = dict(extra or {})
        s.P = torch.empty(S["L"], S["x0"].shape[0], 3) if a.dump else None
        s.PF = torch.empty(S["L"], S["x0"].shape[0], 3, 3) if (a.dump and a.dump_F) else None
        os.makedirs(a.out, exist_ok=True)

    def frame(s, f, x, F=None, ms=None):
        """f = 1..L, x [P,3] (예측 전 입자, 시뮬 좌표). 반환: 계속하면 True."""
        x = x.detach().to("cpu", torch.float64)
        gt = s.S["X"][f].double()
        e = float((x - gt).norm(dim=-1).mean()) / s.S["EXT"] if torch.isfinite(x).all() else float("nan")
        if not math.isfinite(e) or e > 2.0:
            s.div = f
            s.errs.append(float("nan"))
            return False
        s.errs.append(e)
        if ms is not None:
            s.ms.append(ms)
        if s.P is not None:
            s.P[f - 1] = x.float()
        if s.PF is not None:
            s.PF[f - 1] = (torch.eye(3).expand(x.shape[0], 3, 3) if F is None else F.detach().to("cpu", torch.float32))
        if f % 10 == 0 or f == s.S["L"]:
            print(f"  [프레임 {f}/{s.S['L']}] 오차 {100 * e:.3f}%" + (f" {ms:.2f} ms" if ms is not None else ""), flush=True)
        return True

    def fail(s, f, why):
        s.div = f
        s.errs.append(float("nan"))
        s.extra["fail"] = why
        print(f"  [실패] 프레임 {f}: {why}", flush=True)

    def finish(s):
        a = s.a
        sc = s.ms[a.warm:] if len(s.ms) > a.warm else s.ms
        fin = [q for q in s.errs if math.isfinite(q)]
        r = dict(tag=a.tag, N=s.N, frames=len(s.errs), diverged=s.div, err_mean=(float(np.mean(s.errs)) if s.div is None else None),
                 err_mean_done=(float(np.mean(fin)) if fin else None), ms_mean=(float(np.mean(sc)) if sc else None),
                 ms_max=(float(np.max(sc)) if sc else None), errs=s.errs, **s.extra)
        json.dump(r, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
        if s.P is not None and s.div is None:
            torch.save({"pred": s.P, "F": s.PF}, os.path.join(a.out, "pred.pt"))
        print(f"[끝] {a.tag} N {s.N}: " + (f"발산 프레임 {s.div}" if s.div else f"오차 {100 * r['err_mean']:.3f}%")
              + (f", 평균 {r['ms_mean']:.2f} ms (최대 {r['ms_max']:.2f})" if r["ms_mean"] is not None else ""), flush=True)
        return r


def sync_ms(t0):
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0)
