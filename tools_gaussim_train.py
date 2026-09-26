"""GausSim 학습 진입점. 공개된 `mmgs.apis.train_model` 을 부르는 것뿐이다.

저장소에 `tools/train.py` 가 빠져 있어 (README 의 "Full code" 미체크) 이것만
보태면 공개된 학습 경로가 그대로 돈다. 모델·데이터셋·스케줄은 손대지 않는다.

  python tools_gaussim_train.py <config> [--work-dir DIR] [--gpus N]
"""
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist

from mmgs import __version__
from mmgs.apis import set_random_seed, train_model
from mmgs.datasets import build_dataset
from mmgs.models import build_simulator
from mmgs.utils import collect_env, get_root_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--launcher", default="none",
                    choices=["none", "pytorch", "slurm", "mpi"])
    ap.add_argument("--local_rank", type=int, default=0)
    a = ap.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(a.local_rank)

    cfg = Config.fromfile(a.config)
    cfg.work_dir = (a.work_dir or osp.join("./work_dirs",
                                           osp.splitext(osp.basename(a.config))[0]))
    if a.resume_from:
        cfg.resume_from = a.resume_from
    distributed = a.launcher != "none"
    if distributed:
        init_dist(a.launcher, **cfg.get("dist_params", {}))
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        cfg.gpu_ids = [0]

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger = get_root_logger(log_file=osp.join(cfg.work_dir, f"{ts}.log"),
                            log_level=cfg.log_level)
    meta = dict(env_info="\n".join(f"{k}: {v}" for k, v in collect_env().items()),
                seed=a.seed, exp_name=osp.basename(a.config),
                mmgs_version=__version__, config=cfg.pretty_text)
    set_random_seed(a.seed, deterministic=a.deterministic)
    cfg.seed = a.seed

    model = build_simulator(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    if cfg.get("workflow", [("train", 1)]) and len(cfg.workflow) == 2:
        val = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val))
    train_model(model, datasets, cfg, distributed=distributed, validate=True,
                timestamp=ts, meta=meta)


if __name__ == "__main__":
    main()
