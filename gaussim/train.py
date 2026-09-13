"""GausSim 학습 엔트리.

공개 저장소에 `tools/train.py` 가 빠져 있다(README 의 "Full code" 가 TODO).
mmgs.apis.train_model 과 configs/gssim/iccv/pudding.py 는 그대로 있으므로,
mmcls/mmseg 계열의 표준 엔트리 형태로 그 둘을 잇기만 한다. 학습 로직은
저장소 코드를 그대로 쓰고 여기서는 아무것도 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmgs import __version__
from mmgs.apis import set_random_seed, train_model
from mmgs.datasets import build_dataset
from mmgs.models import build_simulator
from mmgs.utils import collect_env, get_root_logger


def parse_args():
    p = argparse.ArgumentParser(description="Train a GausSim simulator")
    p.add_argument("config", help="학습 config 경로")
    p.add_argument("--work-dir", help="로그와 체크포인트를 쓸 디렉토리")
    p.add_argument("--resume-from", help="이어서 학습할 체크포인트")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cfg-options", nargs="+", action=DictAction, default={})
    p.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"],
                   default="none")
    p.add_argument("--local_rank", type=int, default=0)
    a = p.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(a.local_rank)
    return a


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs",
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpu_ids = range(args.gpus)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get("dist_params", {}))
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info = "\n".join([f"{k}: {v}" for k, v in collect_env().items()])
    logger.info("환경:\n" + env_info)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    logger.info(f"분산 학습: {distributed}")

    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed

    model = build_simulator(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmgs_version=__version__)

    train_model(model, datasets, cfg, distributed=distributed,
                validate=(not args.no_validate), timestamp=timestamp, meta=meta)


if __name__ == "__main__":
    main()
