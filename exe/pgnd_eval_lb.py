"""PGND(RSS 2025) 학습 모델을 lbdata 평가 20 궤적에서 굴리고, 우리 지표와 PGND 공식 지표를 낸다.

시작 조건 (사용자 결정 2026-10-11): PGND 공식 3 프레임 정답 히스토리 = 물리 프레임 0, 6, 12 (0.1 s 스텝).
  프레임 12 의 공식 FPS 1000 입자 (eval.py: sim.uniform=True, gripper_forcing=False, num_steps=1000) 에서
  11 스텝 자기회귀 -> 물리 프레임 18, 24, .., 78.
가우시안 상향 표본: PGND 자체 방식 (experiments/train/gs.py rollout_and_render + real_world/utils/render_utils.py
  interpolate_motions): 뼈 = 직전 예측 입자, 관계 knn 16, 가중치 = 가우시안별 knn 16 역거리, 뼈마다 SVD 강체 회전.
  시작 가우시안 = 프레임 12 정답 위치 (인덱스 일치 297,877 개). 가우시안이 많아 덩어리로 나눠 부를 뿐 수식은 같다.
우리 지표: 프레임별 (가우시안 평균 L2 위치 오차) / L, 물리 프레임 18..78 (11 개) 평균. L = 0 프레임 가우시안 bbox 대각선.
  still = 프레임 12 가우시안에 그대로 멈춘 예측. ratio = pgnd / still.
PGND 공식 지표 (metric_eval.do_metric 와 같은 계산): MDE (같은 1000 인덱스), CD·EMD (예측 1000 대 정답 10k 부분집합 전체),
  inverse_preprocess 로 되돌린 좌표, 예측 스텝 1..11.

  python exe/pgnd_eval_lb.py --pgnd /home/dkta/work/lb/pgnd --train_name lego/train --iteration -1 --out DIR
"""
import argparse
import glob
import json
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--pgnd", default="/home/dkta/work/lb/pgnd")
ap.add_argument("--train_name", default="lego/train")
ap.add_argument("--iteration", type=int, default=-1, help="-1 = ckpt/ 에서 가장 큰 번호")
ap.add_argument("--episodes", default="100-119")
ap.add_argument("--out", required=True)
ap.add_argument("--no_official", action="store_true", help="PGND 공식 MDE/CD/EMD 생략")
ap.add_argument("--chunk", type=int, default=40000)
ap.add_argument("--n_gauss", type=int, default=297877)
ap.add_argument("--save_quat", action="store_true", help="가우시안별 누적 회전(PGND 쿼터니언 wxyz, 우리 좌표, 프레임 12 기준)도 저장 -> 렌더용")
ap.add_argument("--oracle", action="store_true", help="모델 대신 정답 입자(같은 FPS 1000)를 상향 표본 -> 상향 표본 자체 오차의 하한")
a = ap.parse_args()

sys.path[:0] = [a.pgnd, os.path.join(a.pgnd, "experiments")]
os.chdir(a.pgnd)
import h5py                                          # noqa: E402
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import warp as wp                                    # noqa: E402
import yaml                                          # noqa: E402
from omegaconf import OmegaConf                      # noqa: E402
from sklearn.neighbors import NearestNeighbors       # noqa: E402
from torch.utils.data import DataLoader              # noqa: E402
from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch  # noqa: E402
from pgnd.material import PGNDModel                  # noqa: E402
from pgnd.data import RealTeleopBatchDataset         # noqa: E402
from real_world.utils.render_utils import interpolate_motions          # noqa: E402
from train.metric import mean_dist, chamfer_dist, em_distance, inverse_preprocess  # noqa: E402

FLIP = torch.tensor([1.0, -1.0, -1.0])               # lb_to_pgnd.py 파일 좌표 <-> 우리 좌표 (자기 역)
log_root = os.path.join(a.pgnd, "experiments", "log")
exp = os.path.join(log_root, a.train_name)
cfg = OmegaConf.create(yaml.load(open(f"{exp}/hydra.yaml"), Loader=yaml.CLoader))
it = a.iteration
if it < 0:
    it = max(int(os.path.basename(p)[:-3]) for p in glob.glob(f"{exp}/ckpt/*.pt") if os.path.basename(p)[:-3].isdigit())
# eval.py __main__ 과 같게
cfg.sim.num_steps = 1000
cfg.sim.gripper_forcing = False
cfg.sim.uniform = True
dev = torch.device("cuda:0")
wp.init(); wp.ScopedTimer.enabled = False; wp.set_module_options({"fast_math": False})
wpdev = wp.get_device("cuda:0")
ck = torch.load(f"{exp}/ckpt/{it:06d}.pt", map_location=dev)
material = PGNDModel(cfg).to(dev); material.load_state_dict(ck["material"]); material.requires_grad_(False); material.eval()
friction = torch.tensor(cfg.model.friction.value, device=dev).reshape(-1, 1)
src_root = os.path.join(log_root, str(cfg.train.source_dataset_name))
md = json.load(open(f"{src_root}/metadata.json"))
skip = int(cfg.train.dataset_load_skip_frame) * int(cfg.train.dataset_skip_frame)
nh = int(cfg.sim.n_history)
f0 = nh * skip                                       # 시작 물리 프레임 (12)
os.makedirs(a.out, exist_ok=True)
print(f"[모델] {exp} iter {it}  skip {skip}  start frame {f0}", flush=True)


def knn_relations(bones, k=16):                      # gs.py GSRenderer.knn_relations (k_rel=16)
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(bones.detach().cpu().numpy())
    _, idx = knn.kneighbors(bones.detach().cpu().numpy())
    return idx[:, 1:]


def knn_weights(bones, pts, k=16):                   # gs.py GSRenderer.knn_weights (k_wgt=16)
    knn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(bones.detach().cpu().numpy())
    _, idx = knn.kneighbors(pts.detach().cpu().numpy())
    bs = bones[idx]
    dist = torch.norm(bs - pts[:, None], dim=-1)
    w = 1 / (dist + 1e-6)
    w = w / w.sum(dim=-1, keepdim=True)
    wa = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)
    wa[torch.arange(pts.shape[0])[:, None], idx] = w
    return wa


def h5x(path, n=None):
    with h5py.File(path) as h:
        x = np.asarray(h["x"], dtype=np.float32)
    return torch.from_numpy(x if n is None else x[:n])


def rollout(ep):
    random.seed(cfg.seed + ep); np.random.seed(cfg.seed + ep); torch.manual_seed(cfg.seed + ep)
    ds = RealTeleopBatchDataset(cfg, dataset_root=os.path.join(log_root, str(cfg.train.dataset_name) + "_evalscript", "state"),
                                source_data_root=src_root, device=dev, num_steps=cfg.sim.num_steps,
                                eval_episode_name=f"episode_{ep:04d}")
    init_state, actions, gt_states, ds_idx = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
    x, v, x_his, v_his, clip_bound, enabled, _ = init_state
    x, v, x_his, v_his = x.to(dev), v.to(dev), x_his.to(dev), v_his.to(dev)
    actions = actions.to(dev)
    T = gt_states[0].shape[1]; B, Np = 1, x.shape[1]
    sim = CacheDiffSimWithFrictionBatch(cfg, T, B, wpdev, requires_grad=True)
    statics = StaticsBatch(); statics.init(shape=(B, Np), device=wpdev)
    statics.update_clip_bound(clip_bound); statics.update_enabled(enabled)
    colliders = CollidersBatch(); ng = cfg.sim.num_grippers
    colliders.init(shape=(B, ng), device=wpdev)
    if ng > 0:
        colliders.initialize_grippers(actions[:, 0])
    enabled = enabled.to(dev)
    xs = [x[0].clone()]
    with torch.no_grad():
        for step in range(T):
            if ng > 0:
                colliders.update_grippers(actions[:, step])
            x_in, v_in = x.clone(), v.clone()
            pred = material(x, v, x_his, v_his, enabled)
            if not torch.isfinite(pred).all():
                raise RuntimeError(f"episode {ep} step {step}: pred not finite")
            x, v = sim(statics, colliders, step, x, v, friction, pred)
            x_his = torch.cat([x_his.reshape(B, Np, -1, 3)[:, :, 1:], x_in[:, :, None]], dim=2).reshape(B, Np, -1)
            v_his = torch.cat([v_his.reshape(B, Np, -1, 3)[:, :, 1:], v_in[:, :, None]], dim=2).reshape(B, Np, -1)
            xs.append(x[0].clone())
    return torch.stack(xs).cpu(), ds_idx[0].cpu()        # (T+1, 1000, 3) PGND 좌표, 10k 부분집합 안 인덱스


res = []
e0, e1 = [int(s) for s in a.episodes.split("-")]
for ep in range(e0, e1 + 1):
    t0 = time.time()
    traj_p, di = rollout(ep)
    t_roll = time.time() - t0
    if a.oracle:                                     # 같은 1000 인덱스의 정답 궤적 (PGND 좌표) 으로 바꿔 끼운다
        _o = torch.from_numpy(np.load(f"{src_root}/episode_{ep:04d}/traj.npz")["xyz"]).float()
        _, _of = inverse_preprocess(cfg, traj_p.clone(), _o.clone())
        traj_p = None
        traj_or = _of[f0::skip][:, di]
    ed = f"{src_root}/episode_{ep:04d}"
    xyz_orig = torch.from_numpy(np.load(f"{ed}/traj.npz")["xyz"]).float()      # (81, 10k, 3) 파일 좌표
    sel = torch.from_numpy(np.load(f"{ed}/sel.npy"))
    if a.oracle:
        traj_f, xyz_orig_f = traj_or, inverse_preprocess(cfg, torch.zeros(1, 1, 3), xyz_orig.clone())[1]
    else:
        traj_f, xyz_orig_f = inverse_preprocess(cfg, traj_p.clone(), xyz_orig.clone())   # 파일 좌표로
    gt_traj = xyz_orig_f[f0::skip]
    assert len(gt_traj) == len(traj_f), (len(gt_traj), len(traj_f))
    frames = list(range(f0, f0 + skip * len(traj_f), skip))                          # 12, 18, .., 78
    off = {}
    if not a.no_official:
        t1 = time.time(); m = []
        for i in range(1, len(traj_f)):
            xp, xg = traj_f[i].cuda(), gt_traj[i].cuda()
            m.append([mean_dist(xp, xg[di.cuda()]), chamfer_dist(xp, xg), em_distance(xp, xg)])
        m = np.array(m)
        off = dict(mde=float(m[:, 0].mean()), chamfer=float(m[:, 1].mean()), emd=float(m[:, 2].mean()),
                   mde_last=float(m[-1, 0]), chamfer_last=float(m[-1, 1]), emd_last=float(m[-1, 2]),
                   per_step=m.tolist(), sec=time.time() - t1, units="file coords (= sim units, rigid flip of ours)")
    # 가우시안 상향 표본 (우리 좌표)
    pts = (traj_f * FLIP).cuda()                                                     # (12, 1000, 3)
    src = md[ep]["path"]
    G0 = h5x(f"{src}/sim_{0:010d}.h5", a.n_gauss)
    L = float((G0.max(0)[0] - G0.min(0)[0]).norm())
    xg = h5x(f"{src}/sim_{f0:010d}.h5", a.n_gauss).cuda()
    still = xg.clone()
    q = torch.zeros(xg.shape[0], 4, device="cuda"); q[:, 0] = 1
    preds, preds_q, e, es = [], [], [], []
    t2 = time.time()
    for i in range(1, len(pts)):
        prev, cur = pts[i - 1], pts[i]
        rel = knn_relations(prev)
        nx, nq = [], []
        for c in range(0, xg.shape[0], a.chunk):
            w = knn_weights(prev, xg[c:c + a.chunk])
            xx, qq, _ = interpolate_motions(bones=prev, motions=cur - prev, relations=rel, weights=w,
                                            xyz=xg[c:c + a.chunk], quat=q[c:c + a.chunk], step=f"{i-1}->{i}")
            nx.append(xx); nq.append(qq)
        xg = torch.cat(nx); q = torch.nn.functional.normalize(torch.cat(nq), dim=-1)
        gt = h5x(f"{src}/sim_{frames[i]:010d}.h5", a.n_gauss).cuda()
        e.append(float((xg - gt).norm(dim=1).mean()) / L); es.append(float((still - gt).norm(dim=1).mean()) / L)
        preds.append(xg.half().cpu())
        if a.save_quat:
            preds_q.append(q.half().cpu())
    t_up = time.time() - t2
    torch.save({"x": torch.stack(preds), "frames": frames[1:], "start_frame": f0,
                "particles": (traj_f * FLIP).float(), "particle_idx": sel[di], "iteration": it,
                "coords": "ours (lbdata sim coords)", "src": src,
                "q": (torch.stack(preds_q) if a.save_quat else None)}, f"{a.out}/pred_{ep:04d}.pt")
    e, es = np.array(e), np.array(es)
    r = dict(episode=ep, split=md[ep]["split"], name=md[ep]["name"], L=L, pgnd=float(e.mean()), still=float(es.mean()),
             ratio=float(e.mean() / es.mean()), pgnd_last=float(e[-1]), still_last=float(es[-1]),
             pgnd_per_frame=e.tolist(), still_per_frame=es.tolist(), frames=frames[1:],
             speed=json.load(open(f"{src}/meta.json")).get("speed"), official=off,
             rollout_sec=t_roll, upsample_sec=t_up)
    res.append(r)
    print(f"[{ep:04d}] {md[ep]['split']}/{md[ep]['name']} L={L:.3f} pgnd={r['pgnd']*100:.2f}% still={r['still']*100:.2f}% "
          f"ratio={r['ratio']:.3f}" + (f" | MDE {off['mde']:.4f} CD {off['chamfer']:.4f} EMD {off['emd']:.4f} ({off['sec']:.0f}s)" if off else "")
          + f" | roll {t_roll:.1f}s up {t_up:.1f}s", flush=True)


def summ(rs):
    if not rs:
        return {}
    out = {"n": len(rs), "pgnd": float(np.mean([r["pgnd"] for r in rs])), "still": float(np.mean([r["still"] for r in rs])),
           "mean_ratio": float(np.mean([r["ratio"] for r in rs]))}
    out["ratio_of_means"] = out["pgnd"] / out["still"]
    if rs[0]["official"]:
        for k in ("mde", "chamfer", "emd"):
            out["official_" + k] = float(np.mean([r["official"][k] for r in rs]))
    return out


inr = [r for r in res if (r["speed"] or 0) <= 1.5]
oor = [r for r in res if (r["speed"] or 0) > 1.5]
out = {"iteration": it, "train_name": a.train_name, "start_frame": f0, "frames": res[0]["frames"] if res else [],
       "metric": "mean over physical frames 18..78 of (mean L2 over 297,877 Gaussians) / L, L = frame-0 Gaussian bbox diagonal; "
                 "still = Gaussians frozen at frame 12",
       "all": summ(res), "in_range": summ(inr), "out_of_range": summ(oor), "per_traj": res}
json.dump(out, open(f"{a.out}/metrics.json", "w"), indent=1)
for k in ("all", "in_range", "out_of_range"):
    print(k, json.dumps(out[k]), flush=True)
print(f"[완료] {a.out}/metrics.json", flush=True)
