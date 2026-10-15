"""서브스텝 안정성: 기본 실행 조합(train_deform 출력만 최적화 --ov_seq)으로 우리 방법을 돌리기 위한 앞뒤 처리.

init : stab/<형상>/init 의 h5 에서 float32 초기 상태(.pt, F=I)와 풀 채우기·설정·데이터 디렉토리를 만든다
state: 롤아웃 덤프 -> train_deform 궤적 .pt (x·v·F, 0 프레임 = 초기 상태; 문턱 검증의 정답 궤적·초기 상태로 쓴다)
conv : --ov_roll 롤아웃 덤프 (프레임 경계 x·v·F) -> rep_ip --stab_out 과 같은 형식의 npz (같은 부분표본·같은 stabstat)
"""
import argparse, glob, json, os, sys
import numpy as np, torch, h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from anchorflow import stabstat as ss

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["init", "conv", "state"])
ap.add_argument("--scene", required=True, help="stab/<형상> 디렉토리 (init/ 와 config.json)")
ap.add_argument("--out", required=True, help="init: 출력 디렉토리 / conv: npz 경로")
ap.add_argument("--seq", default="", help="conv: AF_ROLL_DUMP 롤아웃 덤프 (.pt: pred, v, F)")
ap.add_argument("--substeps", type=int, default=1)
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.scene, "config.json")))
h5 = sorted(glob.glob(os.path.join(a.scene, "init", "*.h5")))[0]
with h5py.File(h5, "r") as f:
    X0 = np.array(f["x"])
X0 = (X0.T if X0.shape[0] == 3 and X0.shape[-1] != 3 else X0).astype(np.float32)
N = X0.shape[0]
V0 = np.tile(np.asarray(cfg.get("init_velocity", [0, 0, 0]), np.float32), (N, 1))
# 질량·부피: rep_ip·시뮬레이터와 같은 정의 (셀마다 세고 dx³/개수)
ng, gl = int(cfg.get("n_grid", 100)), float(cfg.get("grid_lim", 2.0))
dx = gl / ng
cell = np.floor(X0 / dx).astype(np.int64)
key = (cell[:, 0] * (ng + 8) + cell[:, 1]) * (ng + 8) + cell[:, 2]
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
VOL = dx ** 3 / cnt[inv].astype(np.float64)
MASS = float(cfg["density"]) * VOL
FL = [b for b in cfg.get("boundary_conditions", []) if b["type"] == "surface_collider"]
ZF = float(FL[0]["point"][2]) if FL else 0.0
EK = dict(vol=VOL, E=float(cfg["E"]), nu=float(cfg["nu"]))
g = cfg.get("g", [0.0, 0.0, -9.8])

if a.mode == "init":
    os.makedirs(os.path.join(a.out, "pool", "fill"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "pool", "cfg"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "d_ref"), exist_ok=True)
    shape = os.path.basename(os.path.normpath(a.scene))
    x = torch.from_numpy(X0)
    T = int(cfg.get("frame_num", 60)) + 1    # 롤아웃 길이 = 데이터 프레임 수 - 1 (정답 자리는 초기 상태 반복, 측정에 안 쓴다)
    st = {"x": x.expand(T, N, 3).clone(), "v": torch.from_numpy(V0).expand(T, N, 3).clone(),
          "F": torch.eye(3).expand(T, N, 3, 3).clone(), "sel": torch.arange(N), "cfg": cfg, "n_full": N,
          "meta": {"source": "h5 초기 상태 (float32)", "sim": h5}, "gaussians_first": N,
          "F_is_true": True}                    # 정지 초기 상태라 F = I 가 실제 값 (위치에서 되살리지 않는다)
    p = os.path.join(a.out, "init_state.pt")
    torch.save(st, p)
    np.save(os.path.join(a.out, "pool", "fill", f"pgfill_{shape}.npy"), X0)
    json.dump(cfg, open(os.path.join(a.out, "pool", "cfg", f"{shape}_jelly.json"), "w"), indent=1)
    d = os.path.join(a.out, "d_ref", "0000.pt")
    if os.path.lexists(d):
        os.remove(d)
    os.symlink(p, d)
    print(f"[초기 상태] {p}  입자 {N}  바닥 {ZF}", flush=True)
elif a.mode == "state":
    q = torch.load(a.seq, map_location="cpu", weights_only=False)
    x0 = torch.from_numpy(X0)[None]
    st = {"x": torch.cat([x0, q["pred"].float()]), "v": torch.cat([torch.from_numpy(V0)[None], q["v"].float()]),
          "F": torch.cat([torch.eye(3).expand(1, N, 3, 3), q["F"].float()]), "sel": torch.arange(N), "cfg": cfg, "n_full": N,
          "meta": {"source": "기본 실행 조합 출력만 최적화 롤아웃", "dump": a.seq, "substeps": a.substeps},
          "gaussians_first": N, "F_is_true": True}
    torch.save(st, a.out)
    print(f"[상태 저장] {a.out}  {st['x'].shape[0]} 프레임", flush=True)
else:
    q = torch.load(a.seq, map_location="cpu", weights_only=False)
    xs, vs, Fs = q["pred"].float().numpy(), q["v"].float().numpy(), q["F"].float().numpy()
    sub = ss.subset(N)
    rows = [ss.frame_stats(X0, V0, np.tile(np.eye(3), (N, 1, 1)), MASS, g, ZF, **EK)]
    X = [X0[sub]]
    for t in range(xs.shape[0]):
        rows.append(ss.frame_stats(xs[t], vs[t], Fs[t], MASS, g, ZF, **EK))
        X.append(xs[t][sub])
    K = list(ss.KEYS)
    np.savez_compressed(a.out, sub=sub, x=np.stack(X).astype(np.float32), t=np.zeros(len(X) - 1), keys=np.array(K),
                        substeps=a.substeps, dt=float(cfg["frame_dt"]) / a.substeps,
                        stats=np.array([[r.get(k, np.nan) for k in K] for r in rows]))
    print(f"[안정성 저장] {a.out}  {len(X)} 프레임", flush=True)
