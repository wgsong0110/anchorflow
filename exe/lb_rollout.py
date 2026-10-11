"""학습된 변형 모델의 **추론 전용** 롤아웃 + 평가 + 영상 (lb 데이터셋).

train_deform.py --infer 대신 쓴다: 학습 스크립트를 통째로 띄우지 않고 (풀·통계
재계산·덤프 없음) 체크포인트 하나로 모델을 세워 프레임마다 망을 한 번씩 통과시킨다.
한 스텝·롤아웃·지표는 학습과 같은 라이브러리(lib/anchorflow/deform_model.py) 에서
오므로 train_deform 의 평가와 같은 수치가 나온다.

  - 모델 구성·플래그는 체크포인트에 저장된 args 가 정한다 (--net_inertial 등)
  - 초기 상태: x[t0], v = 궤적에 저장된 v (--v0_traj 로 학습한 모델) 또는 차분, F[t0]
  - 지표: 프레임별 입자 평균 위치 오차 / 물체 크기, 정지 기준, 비 (train_deform 과 같다)
  - 영상: 프레임마다 메모리에서 바로 3DGS 렌더 (F_render = F_cum · R, F_cum 은 망 변형장
    야코비안의 누적 -- train_deform 의 AF_ROLL_DUMP_F 경로와 같다). GT 영상과 나란히.

  python exe/lb_rollout.py --ckpt /home/dkta/work/lbours/main2/MAIN_best.pt \
      --tags 0000,0015,0005 --t0 0 --len 80 --video_dir OUT --json OUT/ev.json \
      --tb /home/dkta/work/lbours/tb --tb_tag main2_best_rollout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from anchorflow import deform_model as DM                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="train_deform 체크포인트 (MAIN_best.pt 등)")
ap.add_argument("--traj_dir", default="/home/dkta/work/lbpt/test", help="lb_to_pt.py 가 만든 .pt 들")
ap.add_argument("--tags", default="all", help="쉼표 목록 (예 0000,0015) 또는 all")
ap.add_argument("--t0", type=int, default=0)
ap.add_argument("--len", type=int, default=80)
ap.add_argument("--video_dir", default="", help="주면 궤적마다 <tag>.mp4 를 쓴다")
ap.add_argument("--gt_dir", default="/home/dkta/work/lbdata/test",
                help="GT 영상 <gt_dir>/<tag>/video.mp4 (있으면 왼쪽에 나란히)")
ap.add_argument("--no_gt", action="store_true", help="GT 영상을 붙이지 않는다")
ap.add_argument("--shape", default="lego")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--json", default="", help="궤적별·요약 지표 JSON")
ap.add_argument("--tb", default="", help="TensorBoard 루트")
ap.add_argument("--tb_tag", default="", help="TB 실행 이름 (기본: <체크포인트 폴더>_<파일>)")
ap.add_argument("--fast", action="store_true",
                help="빠른 경로 (lib/anchorflow/fast_roll: 융합 커널 + CUDA 그래프, 같은 식)")
ap.add_argument("--prof", action="store_true", help="모델 한 스텝의 구간별 시간 (동기화가 들어가 조금 느려진다)")
a = ap.parse_args()
dev = "cuda"

t_all = time.time()
M = DM.EvalModel(a.ckpt, dev, fast=a.fast)
ma = M.a
print(f"[모델] {a.ckpt} 단계 {M.step_ck} | 층 {ma.gnn_layers} hidden {ma.hidden} "
      f"net_inertial {ma.net_inertial} v_from_dt {ma.v_from_dt} v0_traj {ma.v0_traj} "
      f"서브스텝 {M.sub} | 기준 크기 {M.ext0:.5f} ({M.ref})", flush=True)

tags = (sorted(os.path.splitext(f)[0] for f in os.listdir(a.traj_dir) if f.endswith(".pt"))
        if a.tags in ("", "all") else [t for t in a.tags.split(",") if t])
REND = None
if a.video_dir:
    os.makedirs(a.video_dir, exist_ok=True)
    import imageio.v2 as imageio
    from anchorflow.lb_gs_render import LegoRenderer


def sync():
    torch.cuda.synchronize()


PROF = {}


class _Sec:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        sync(); self.t0 = time.time()

    def __exit__(self, *_):
        sync(); PROF[self.name] = PROF.get(self.name, 0.0) + time.time() - self.t0


SEC = _Sec if a.prof else None


rows, timing = {}, {}
for tag in tags:
    t_load = time.time()
    d = torch.load(os.path.join(a.traj_dir, f"{tag}.pt"), map_location="cpu", weights_only=False)
    for k in ("x", "v", "F"):
        if k in d and torch.is_tensor(d[k]):
            d[k] = d[k].to(dev)
    T = d["x"].shape[0]
    L = min(a.len, T - a.t0 - 1)
    if L < 1:
        print(f"[건너뜀] {tag}: 프레임 {T} 로 t0={a.t0} 롤아웃 불가", flush=True)
        continue
    gsel = torch.arange(d["x"].shape[1], device=dev)
    with torch.no_grad():
        x, v = M.init(d, a.t0)
        ext = float(d["_ext"])
        Fcum = None
        if a.video_dir:
            Fcum = DM.take(DM.traj_F(d, M.frame_dt, dev)[a.t0], gsel, dev).float().clone()
            Rm = torch.as_tensor(np.array(d["meta"]["R"]), device=dev, dtype=torch.float32)
            if REND is None:
                REND = LegoRenderer(d["cfg"], shape=a.shape, dev=dev)
            gt_rd = None
            gp = os.path.join(a.gt_dir, tag, "video.mp4")
            if not a.no_gt and os.path.exists(gp):
                gt_rd = imageio.get_reader(gp)
            vw = imageio.get_writer(os.path.join(a.video_dir, f"{tag}.mp4"), fps=a.fps,
                                    codec="libx264", quality=8)
    sync()
    t_load = time.time() - t_load
    t_rend = [0.0]
    t_step = [0.0]

    def put(xr, Fr, k):
        """프레임 k (0 = 초기 상태) 를 렌더해 쓴다. GT 는 원 궤적의 프레임 t0+k."""
        t_ = time.time()
        img = REND.image(xr, Fr @ Rm)
        if gt_rd is not None:
            try:
                g = gt_rd.get_data(a.t0 + k)
                if g.shape[0] == img.shape[0]:
                    img = np.concatenate([g[..., :3], img], 1)
            except IndexError:
                pass
        vw.append_data(img)
        t_rend[0] += time.time() - t_

    def step_fn(i, x, v):
        global Fcum
        sync(); t_ = time.time()
        x2, v2, J = M.frame(d, x, v, sec=SEC)
        sync(); t_step[0] += time.time() - t_
        if Fcum is not None:
            Fcum = J.float() @ Fcum
        return x2, v2, J

    def on_frame(i, x2, gt, _J, _xs):
        if a.video_dir:
            put(x2, Fcum, i + 1)

    with torch.no_grad():
        t_roll = time.time()
        if a.video_dir:
            put(x, Fcum, 0)
        errs, stills = DM.rollout_eval(step_fn, x, v,
                                       lambda i: DM.take(d["x"][a.t0 + i + 1], gsel, dev),
                                       L, ext, on_frame=on_frame)
        sync()
        t_roll = time.time() - t_roll
    if a.video_dir:
        vw.close()
        if gt_rd is not None:
            gt_rd.close()
    w = DM.window_row(L, errs, stills)
    rows[tag] = dict(windows={a.t0: w})
    timing[tag] = dict(load_s=t_load, rollout_render_s=t_roll, model_s=t_step[0],
                       render_s=t_rend[0], frames=L)
    print(f"[롤아웃] {tag} t0={a.t0:3d}: {L} 프레임, 평균 {100*w['err_mean']:.3f}% "
          f"(정지 {100*w['still_mean']:.3f}%, 비 {w['ratio']:.2f})  "
          f"[{t_roll:.2f} s = 모델 {t_step[0]:.2f} + 렌더 {t_rend[0]:.2f}, 적재 {t_load:.1f} s]",
          flush=True)
    if PROF:
        print("    [구간] " + " ".join(f"{k} {v:.2f}s" for k, v in sorted(PROF.items(), key=lambda z: -z[1])), flush=True)
        PROF.clear()
    del d
    torch.cuda.empty_cache()

S = DM.summarize(rows)
if S is None:
    raise SystemExit("평가한 궤적이 없다")
print(f"\n[요약] 전체 창 평균 {100*S['err_mean']:.3f}% (정지 {100*S['still_mean']:.3f}%, "
      f"비 {S['ratio']:.2f}) -- {S['n']} 궤적, 총 {time.time() - t_all:.1f} s", flush=True)

if a.json:
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump(dict(ckpt=os.path.abspath(a.ckpt), step=M.step_ck, t0=a.t0, len=a.len,
                   traj_dir=a.traj_dir, model_args=vars(ma), net_ext0=M.ext0,
                   summary=S, rollout=rows, timing=timing,
                   video_dir=(os.path.abspath(a.video_dir) if a.video_dir else None)),
              open(a.json, "w"), indent=1, ensure_ascii=False)
    print(f"[JSON] {a.json}", flush=True)
if a.tb:
    from torch.utils.tensorboard import SummaryWriter
    _ck = os.path.abspath(a.ckpt)
    name = a.tb_tag or (f"{os.path.basename(os.path.dirname(_ck))}_"
                        f"{os.path.splitext(os.path.basename(_ck))[0]}")
    tbw = SummaryWriter(os.path.join(a.tb, name))
    DM.tb_write_eval(tbw, rows, M.step_ck, prefix="eval")
    tbw.close()
    print(f"[TB] {os.path.join(a.tb, name)} (단계 {M.step_ck})", flush=True)
print("LB_ROLLOUT_OK")
