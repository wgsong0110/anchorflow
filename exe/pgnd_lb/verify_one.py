"""변환된 episode_0000 을 PGND 자체 dataset 전처리(RealTeleopBatchDataset.preprocess)로 읽어 확인한다."""
import os, sys, shutil, yaml
P = "/home/dkta/work/lb/pgnd"
sys.path[:0] = [P, P + "/experiments"]; os.chdir(P)
import numpy as np, torch
from hydra import initialize_config_dir, compose
from pgnd.data import RealTeleopBatchDataset
ov = ["train.source_dataset_name=data/lego_lb/sub_episodes_v", "sim.preprocess_scale=0.45", "train.dataset_load_skip_frame=6",
      "train.training_start_episode=0", "train.training_end_episode=1"]
with initialize_config_dir(config_dir=P + "/experiments/cfg", version_base="1.2"):
    cfg = compose(config_name="default", overrides=ov)
root = "/home/dkta/work/lb/pgnd_verify/state"; shutil.rmtree(root, ignore_errors=True)
ds = RealTeleopBatchDataset(cfg, dataset_root=root, source_data_root=P + "/experiments/log/data/lego_lb/sub_episodes_v",
                            device="cpu", num_steps=cfg.sim.num_steps_train, train=True, dataset_non_overwrite=True)
info = yaml.safe_load(open(root + "/episode_0000/info.yaml"))
pts = sorted(os.listdir(root + "/episode_0000")); print("info", info, "| .pt files", sum(p.endswith(".pt") for p in pts))
xs = ds.episode_xs[0]; print("loaded frames x", tuple(xs.shape), "(skip 6 -> 0,6,..,78)")
print("min", xs.amin((0, 1)).tolist(), "max", xs.amax((0, 1)).tolist())
com = xs.mean(1)
print("COM y per loaded frame", [round(float(c), 4) for c in com[:, 1]])
print("COM x,z drift", (com[-1] - com[0])[[0, 2]].tolist())
raw = np.load(P + "/experiments/log/data/lego_lb/sub_episodes_v/episode_0000/traj.npz")["xyz"]
print("raw file frames", raw.shape, "ours z(=-file z) COM per 6 frames", [round(float(-raw[f, :, 2].mean()), 4) for f in range(0, 81, 6)])
acc = (com[2, 1] - 2 * com[1, 1] + com[0, 1]) / 0.1 ** 2
print(f"COM y second difference over frames 0,6,12: {float(acc):.3f} (free fall would be -9.8*0.45 = {-9.8*0.45:.3f})")
print("len(ds) windows", len(ds))
init_state, actions, gt = ds[0]
x, v, xh, vh, cb, en, ev = init_state
print("sample x", tuple(x.shape), "v", tuple(v.shape), "x_his", tuple(xh.shape), "actions", tuple(actions.shape), "gt_x", tuple(gt[0].shape),
      "grip open", actions[0, 0, 14].item(), "grip pos", actions[0, 0, :3].tolist())
