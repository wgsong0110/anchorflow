"""NGFF 베이스라인 평가 (lbdata test 20 궤적).

각 평가 궤적의 0 프레임(키포인트 2048 FPS, 초기 강체 속도 v0)에서 79 스텝 롤아웃하고, NGFF videogen/eval.py 와
똑같이 upsample_gaussian_splatting(k=8: 0 프레임 기준 최근접 키포인트 8 개 위치 평균)으로 전 가우시안에 올린다.
오차 = 프레임별 (가우시안 평균 L2 위치 오차) / L, 프레임 1..79 평균.  L = 그 궤적 0 프레임 가우시안 bbox 대각선 (고정 상수).
"still" = 0 프레임에 그대로 서 있는 예측. ratio = ngff / still.
GT 는 lbdata 원본 h5 (float32) 를 쓴다.

  cd /home/dkta/work/lb/NGFF_lb && python /home/dkta/work/anchorflow/exe/ngff_lb_eval.py --exp_dir EXP --out OUT
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--exp_dir", required=True)
ap.add_argument("--ckpt", default="ngff_best.pth")
ap.add_argument("--ngff", default="/home/dkta/work/lb/NGFF_lb")
ap.add_argument("--data_path", default="/home/dkta/work/lb/ngff_data/mpm")
ap.add_argument("--group", default="lb_test")
ap.add_argument("--cache_dir", default="/home/dkta/work/lb/ngff_data/cache")
ap.add_argument("--src", default="/home/dkta/work/lbdata/test")
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--t0", type=int, default=0,
                help="이 프레임의 상태(키포인트·질량중심·각도·질량중심 속도)에서 시작해 steps-1 프레임까지 롤아웃. "
                     "오차는 프레임 t0+1..steps-1 평균, L 은 그대로 0 프레임 대각선, 정지 = t0 프레임에 그대로 (우리 t0=12 규약과 같다)")
ap.add_argument("--save_ids", default="0000,0005,0015")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.ngff)
os.chdir(a.ngff)
import h5py                                                      # noqa: E402
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402
from accelerate.utils import set_seed                           # noqa: E402
from dynamic_models import NGFFobj                               # noqa: E402
from dynamic_models.DGSDataset import DGSDataset                 # noqa: E402
from utils.transformation_utils import euler_xyz_to_matrix, upsample_gaussian_splatting   # noqa: E402

torch.set_grad_enabled(False)
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
ar = json.load(open(f"{a.exp_dir}/args.json"))
set_seed(ar.get("seed", 0))
model = NGFFobj(input_dim=ar["output_dim"], hidden_dim=ar["hidden_dim"], output_dim=ar["output_dim"], num_layers=ar["num_layers"],
                mass=ar["mass"], dt=ar["dt"], ode_method=ar["ode_method"], r=0.1, step_size=ar["step_size"],
                threshold=ar["threshold"], rtol=ar["rtol"], atol=ar["atol"],
                gravity=ar.get("gravity", -5.0), boundary=ar.get("boundary"))
sd = torch.load(f"{a.exp_dir}/{a.ckpt}", map_location="cpu")
model.load_state_dict(sd)
model.to(dev).eval()
print(f"[모델] {a.exp_dir}/{a.ckpt} gravity={float(model.gravity):.2f} boundary={model.boundary.tolist()} dt={model.dt} step={model.step_size}", flush=True)

gdir = os.path.join(a.data_path, a.group)
scenes = [os.path.join(gdir, s) for s in sorted(os.listdir(gdir)) if not s.startswith(".")]
ds = DGSDataset(scenes, num_frames=a.steps, num_keypoints=ar["num_keypoints"], chunk=ar["chunk"], dtype=torch.float32,
                cache_dir=a.cache_dir)
save_ids = set(a.save_ids.split(",")) if a.save_ids else set()
res = {}
for idx in tqdm(range(len(ds)), desc="eval"):
    d = ds[idx]
    name = os.path.basename(d["path"].rstrip("/"))
    st = json.load(open(os.path.join(d["path"], "init_state.json")))
    nid = os.path.basename(st["src"].rstrip("/"))
    data = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    t0 = time.time()
    T0 = a.t0
    point_seq, com_seq, angle_seq = model(data["points"][:, T0], data["com"][:, T0], data["com_vel"][:, T0], data["angle"][:, T0],
                                          data["angle_vel"][:, T0], data["padding"], data["knn"], pred_len=a.steps - 1 - T0,
                                          external_forces=None)
    R_seq = euler_xyz_to_matrix(angle_seq)
    fut = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)        # (B,T,O,N,3)
    torch.cuda.synchronize(); t_roll = time.time() - t0
    B, T, O, N, C = fut.shape
    mask = data["padding"].unsqueeze(1).reshape(B, 1, O * N, 1).repeat(1, T, 1, C).bool()
    fut = fut.reshape(B, T, -1, C)[mask].reshape(B, T, -1, C)
    # GT (float32 원본)
    NG = int(st["NG"])
    gt = []
    for t in range(T0, a.steps):
        with h5py.File(os.path.join(st["src"], f"sim_{t:010d}.h5")) as h:
            gt.append(torch.as_tensor(np.array(h["x"][:NG])))
    gt = torch.stack(gt).to(dev).float()                                                    # (T,NG,3)
    up_in = dict(data)
    up_in["pos"] = fut
    with h5py.File(os.path.join(st["src"], "sim_0000000000.h5")) as h:                       # 이웃은 0 프레임(정지) 기준 그대로
        g0 = torch.as_tensor(np.array(h["x"][:NG])).to(dev).float()
    up = upsample_gaussian_splatting(up_in, g0[None], torch.zeros(1, NG, 1, dtype=torch.long, device=dev), k=8)
    pred = up["pos"][0]                                                                     # (T,NG,3)
    L = float((g0.max(0)[0] - g0.min(0)[0]).norm())
    e = (pred - gt).norm(dim=-1).mean(1) / L                                                # (T,)
    es = (gt[0:1] - gt).norm(dim=-1).mean(1) / L
    ek = (fut[0] - gt[:, d["keypoint_indices"].to(dev)]).norm(dim=-1).mean(1) / L         # 키포인트만 (업샘플 전)
    r = {"name": name, "src": st["src"], "v0": st["v0"], "L": L,
         "ngff": float(e[1:].mean()), "still": float(es[1:].mean()), "ratio": float(e[1:].mean() / es[1:].mean()),
         "ngff_keypoints": float(ek[1:].mean()), "frame0_upsample_err": float(e[0]),
         "ngff_last": float(e[-1]), "still_last": float(es[-1]), "rollout_sec": t_roll,
         "ngff_per_frame": e.tolist(), "still_per_frame": es.tolist()}
    res[nid] = r
    print(f"{nid} L={L:.4f} ngff={r['ngff']*100:.2f}% still={r['still']*100:.2f}% ratio={r['ratio']:.3f} "
          f"kp={r['ngff_keypoints']*100:.2f}% f0={r['frame0_upsample_err']*100:.3f}%", flush=True)
    if nid in save_ids:
        torch.save(pred.half().cpu(), os.path.join(a.out, f"pred_test{nid}.pt"))
    del gt, pred, up, fut


def summ(keys):
    rs = [res[k] for k in keys]
    if not rs:
        return None
    return {"n": len(rs), "ngff": float(np.mean([r["ngff"] for r in rs])), "still": float(np.mean([r["still"] for r in rs])),
            "ratio_of_means": float(np.mean([r["ngff"] for r in rs]) / np.mean([r["still"] for r in rs])),
            "mean_ratio": float(np.mean([r["ratio"] for r in rs])),
            "ngff_keypoints": float(np.mean([r["ngff_keypoints"] for r in rs]))}


ks = sorted(res)
out = {"exp_dir": a.exp_dir, "ckpt": a.ckpt, "steps": a.steps, "t0": a.t0,
       "metric": "mean over frames 1..T-1 of (mean L2 over all Gaussians) / L, L = frame-0 Gaussian bbox diagonal",
       "upsample": "NGFF videogen/eval.py upsample_gaussian_splatting, k=8",
       "summary_all": summ(ks), "summary_0_14": summ([k for k in ks if int(k) < 15]),
       "summary_15_19": summ([k for k in ks if int(k) >= 15]), "per_traj": res}
json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
for k in ("summary_all", "summary_0_14", "summary_15_19"):
    print(k, json.dumps(out[k]), flush=True)
print(f"[완료] {a.out}/metrics.json", flush=True)
