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
    BOX = []
    if a.cfg_bc:
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") == "cuboid":
                assert max(abs(float(q)) for q in bc["velocity"]) == 0.0
                BOX.append(([float(q) for q in bc["point"]], [float(q) for q in bc["size"]]))
    g = [float(q) for q in cfg["g"]]
    assert abs(g[0]) < 1e-12 and abs(g[1]) < 1e-12 and g[2] <= 0, f"중력이 -z 또는 0 이 아님 {g}"   # ficus 공식 config = 0
    S = dict(d=d, cfg=cfg, L=L, x0=x0, X=d["x"][:L + 1], v0=d["v"][0].double(), vol=vol, rho=rho, mass=vol * rho,
             EXT=float((x0.max(0).values - x0.min(0).values).norm()), FZ=FZ, BOX=BOX, g=g, frame_dt=float(cfg["frame_dt"]),
             E=float(cfg["E"]), nu=float(cfg["nu"]))
    return S


def time_scale(Kl, g):
    """닮음 변환 시간 배율 (절 S.3 공식 밖 (1)): 길이 Kl 배일 때 중력이 9.8 이 되게 c = sqrt(Kl |g| / 9.8).
    중력 0 (ficus 공식 config) 이면 맞출 중력이 없어 c = 1 (길이만 바꿈; 사용자 결정 11-05)."""
    gz = abs(float(g[2]))
    return (Kl * gz / 9.8) ** 0.5 if gz > 0 else 1.0


def in_boxes(x, boxes):
    """|x − point| < size (모든 축) -- vrgs_xpbd 고정 상자와 같은 판정."""
    m = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for (pt, sz) in boxes:
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
