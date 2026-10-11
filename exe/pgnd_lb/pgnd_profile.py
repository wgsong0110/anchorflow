"""PGND 학습 반복 시간 분해 (공식 설정, lbdata). 결과에 영향 없는 측정만 한다."""
import os, sys, time
P = "/home/dkta/work/lb/pgnd"
sys.path[:0] = [P, P + "/experiments"]; os.chdir(P)
import torch
from hydra import initialize_config_dir, compose
ov = ["gpus=[0]", "overwrite=False", "resume=True", "debug=True", "train.source_dataset_name=data/lego_lb/sub_episodes_v",
      "sim.preprocess_scale=0.45", "train.dataset_load_skip_frame=6", "train.num_workers=8", "train.batch_size=32",
      "train.training_start_episode=0", "train.training_end_episode=100", "train.eval_start_episode=100", "train.eval_end_episode=120",
      "train.name=lego_profile/train", "train.dataset_name=lego_smoke/dataset", "train.num_iterations=100000"]
with initialize_config_dir(config_dir=P + "/experiments/cfg", version_base="1.2"):
    cfg = compose(config_name="default", overrides=ov)
from train import train_eval as te
tr = te.Trainer(cfg); tr.load_train_dataset(); tr.init_train()
from pgnd.material.network import utils as nu
orig = nu.get_grid_locations
G = []
def wrap(x, n, dx):
    xg, gi = orig(x, n, dx); G.append(xg.shape[1]); return xg, gi
import pgnd.material.pgnd as pm; pm.get_grid_locations = wrap
def run(n, tag):
    torch.cuda.synchronize(); t = time.time(); tr.train(0, n, save=False); torch.cuda.synchronize()
    print(f"[{tag}] {n} it: {(time.time()-t)/n:.3f} s/it, active grid nodes mean {sum(G)/len(G):.0f} max {max(G)}, "
          f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True); G.clear()
run(5, "warmup anomaly=on")
run(20, "official anomaly=on")
torch.autograd.set_detect_anomaly(False)
run(20, "anomaly=off")
# 데이터 로딩만
t = time.time()
for _ in range(20): next(tr.dataloader)
print(f"[data] {(time.time()-t)/20:.3f} s/batch", flush=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    tr.train(0, 3, save=False)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
