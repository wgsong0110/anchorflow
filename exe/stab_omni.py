"""[stab1115] OmniPhysGS (공식 저장소 wgsxm/OmniPhysGS 의 MPMModel p2g2p, 수정 없음) 의 서브스텝 안정성 실행기. stab_mpm.py 와 같은 입출력 npz.
공식 코드는 영역 [0,1]^3 (num_grids 격자, 입자 부피 = prod(size)/N 균일) 이라 우리 장면 (영역 [0,2], 격자 100) 을 길이 s=0.5 배로 줄여 넣는다:
 x'=s x, v'=s v, g'=s g, E'=s^2 E (같은 밀도·시간) -> 궤적이 정확히 s 배 (num_grids=100 = 우리 격자 dx 0.02 의 s 배), 입자 부피 = size 의 곱 / N 을 우리 평균 입자 부피 x s^3 로.
재질: 학습 단계 (신경 구성모델 선택) 는 쓰지 않고 정답과 같은 고정 공회전 탄성 (공식 CorotatedElasticity, E 1e6 nu 0.3) + 항등 소성으로 고정, MPM 솔버 부분만 쓴다.
바닥 = 공식 surface_collider (sticky, 격자 노드), 상자 = 공식 clip_bound.
  python exe/stab_omni.py --repo <OmniPhysGS> --h5 init.h5 --config config.json --substeps N --out r.npz [--dump DIR] [--frames F]
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "lib"))
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/home/dkta/work/OmniPhysGS"); ap.add_argument("--h5", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--substeps", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--dump", default=""); ap.add_argument("--ckpt_every", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.repo); os.chdir(a.repo)
import h5py, numpy as np, torch
import warp as wp
wp.init()
from tqdm import tqdm
from anchorflow import stabstat as ss
from src.mpm_core import MPMModel, set_boundary_conditions
from src.constitutive_models.physical_constitutive_models import CorotatedElasticity
cfg = json.load(open(a.config)); S = 0.5
GL = float(cfg["grid_lim"]); assert abs(GL * S - 1.0) < 1e-9
n_grid = int(cfg["n_grid"]); dx = GL / n_grid
NF = a.frames or int(cfg["frame_num"]); frame_dt = float(cfg["frame_dt"]); dt = frame_dt / a.substeps
G = [float(q) for q in cfg["g"]]
fl = [b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]; FZ = float(fl["point"][2])
with h5py.File(a.h5) as h:
    x = np.array(h["x"]); x = np.ascontiguousarray((x.T if x.shape[0] == 3 else x).astype(np.float32))
N = x.shape[0]
X = torch.as_tensor(x).cuda()
cell = (X / dx).floor().long(); key = (cell[:, 0] * n_grid + cell[:, 1]) * n_grid + cell[:, 2]
_, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
VOL = (dx ** 3 / cnt[inv].float()).contiguous(); MASS = VOL.double().cpu().numpy() * float(cfg["density"])
SUB = ss.subset(N)
E, nu = float(cfg["E"]), float(cfg["nu"])
vol_p = float(VOL.mean()) * S ** 3
sim = {"num_grids": n_grid, "dt": dt, "gravity": [g * S for g in G], "clip_bound": 0.5, "damping": 1.0}
mat = {"center": [0.5] * 3, "size": [vol_p * N, 1.0, 1.0], "rho": float(cfg["density"])}
model = MPMModel(sim, mat, init_pos=X * S, enable_train=False, device=torch.device("cuda"))
set_boundary_conditions(model, [{"type": "surface_collider", "point": [p * S for p in fl["point"]], "normal": fl["normal"], "surface": fl["surface"],
                                  "friction": float(fl.get("friction", 0.0)), "start_time": 0.0, "end_time": 1000.0}])
elas = CorotatedElasticity().cuda()
elas.log_E = torch.tensor([E * S * S], device="cuda").log(); elas.nu = torch.tensor([nu], device="cuda")
xs = X * S; v = torch.zeros_like(xs); C = torch.zeros(N, 3, 3, device="cuda"); F = torch.eye(3, device="cuda").repeat(N, 1, 1)
_EK = dict(vol=VOL.double().cpu().numpy(), E=E, nu=nu)
rows = [ss.frame_stats(x, np.zeros_like(x), np.tile(np.eye(3), (N, 1, 1)), MASS, G, FZ, **_EK)]; XS = [x[SUB]]; T = []
if a.dump:
    os.makedirs(a.dump, exist_ok=True)
    with h5py.File(f"{a.dump}/sim_{0:010d}.h5", "w") as hh:
        hh.create_dataset("x", data=x); hh.create_dataset("F", data=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (N, 1)))
print(f"[omni] 입자 {N}, N {a.substeps} (dt {dt:.3e}), {NF} 프레임, 입자 부피 {vol_p:.3e}", flush=True)
for f in tqdm(range(1, NF + 1), desc=f"omni N{a.substeps}"):
    t0 = time.time(); bad = False
    with torch.no_grad():
        try:
            for s in range(a.substeps):
                stress = elas(F)
                xs, v, C, F = model(xs, v, C, F, stress)
        except Exception as e:                                   # 공식 구성모델의 유한성 단언 -> 발산
            print("  발산 (예외)", repr(e)[:120], flush=True); bad = True
    torch.cuda.synchronize(); T.append(time.time() - t0)
    xt = (xs / S).cpu().numpy(); vt = (v / S).cpu().numpy(); Ft = F.cpu().numpy()
    if bad:
        xt = np.full_like(xt, np.nan)
    rows.append(ss.frame_stats(xt, vt, Ft, MASS, G, FZ, **_EK))
    rows[-1]["safe_n"] = rows[-1]["safe_split"] = rows[-1]["safe_inert"] = 0.0
    if a.dump and not bad:
        with h5py.File(f"{a.dump}/sim_{f:010d}.h5", "w") as hh:
            hh.create_dataset("x", data=xt.astype(np.float32)); hh.create_dataset("F", data=Ft.reshape(N, 9).astype(np.float32))
    XS.append(xt[SUB])
    if rows[-1]["nan"] > 0.5:
        print(f"  발산 프레임 {f}", flush=True); break
np.savez_compressed(a.out, sub=SUB, x=np.stack(XS).astype(np.float32), t=np.array(T),
                    stats=np.array([[r.get(k, np.nan) for k in ss.KEYS + ss.SAFE_KEYS] for r in rows]),
                    keys=np.array(ss.KEYS + ss.SAFE_KEYS), substeps=a.substeps, dt=dt, safe=0)
print(f"[저장] {a.out}  프레임당 {np.mean(T):.3f}s", flush=True)
