"""앵커 변형 모델을 GaussianFluent 파괴 궤적 위에서 학습한다.

한 프레임이 한 스텝이다 -- 서브스텝은 없다. 모델은 앵커 상태와 앵커별로 집계한
가우시안 상태를 받아 앵커마다 (변위, 반경, 온도) 를 내고, 그것으로 만든 변형 사상이
다음 프레임의 가우시안 위치와 모양을 정한다 (lib/anchorflow/deform.py).

앵커 초기화가 이 설계의 이점 하나를 그대로 쓴다: 앵커는 FPS 로 고른 **가우시안**
이므로, 어느 프레임에서 시작하든 그 프레임의 가우시안 위치가 곧 앵커의 초기
위치다. 그래서 임의의 시점에서 창을 열어 몇 프레임 펼치는 학습이 자연스럽다.

손실은 둘이다.

  위치   다음 프레임 가우시안 위치. 물체 크기로 나눈다.
  모양   변형 사상의 야코비안 J 를 GT 변형구배의 한 프레임 증분 F_{t+1} F_t^-1 에
         맞춘다. 위치만 채점하면 모양은 자유롭게 틀려도 되는데, 모양은 렌더링에
         곧바로 들어간다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import math
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="gen_gf_trajs.py 가 만든 .pt 들의 디렉토리")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="deform")
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--batch", type=int, default=1,
                help="한 스텝에 평균 낼 창의 수. 창마다 손실이 수십 배 다르므로 "
                     "하나만 쓰면 기울기가 그 차이에 휘둘린다")
ap.add_argument("--unroll", type=int, default=1, help="한 창에서 펼칠 프레임 수")
ap.add_argument("--unroll_final", type=int, default=4,
                help="(미사용) 예전의 중간 증가 일정. 지금은 unroll 고정")
ap.add_argument("--unroll_at", type=float, default=0.4,
                help="이 비율을 지나면 unroll 을 unroll_final 로 늘린다")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lambda_J", type=float, default=0.1)
ap.add_argument("--shape_loss", default="frob", choices=("frob", "bures", "none"),
                help="모양을 어떻게 채점할지. frob 은 J 와 GT 증분의 Frobenius, "
                     "bures 는 위치까지 포함한 입자별 Bures-Wasserstein (길이^2 "
                     "단위라 lambda_J 가 필요 없다), none 은 위치만")
ap.add_argument("--lambda_anchor", type=float, default=0.0,
                help="앵커 변위를 GT 에 직접 맞추는 보조 손실의 가중. 앵커는 FPS 로 "
                     "고른 **가우시안**이라 그 정답 변위를 정확히 안다 -- 스키닝을 "
                     "거치는 간접 경로보다 훨씬 쉬운 문제이고, 진단으로도 쓴다")
ap.add_argument("--sigma0", type=float, default=0.5,
                help="정준 가우시안의 등방 표준편차를 입자 간격의 몇 배로 볼지. "
                     "이 씬의 ply 는 sigma ~ 1e-9 로 사실상 점이라, 모양 항을 쓰려면 "
                     "부피에서 온 크기를 줘야 한다")
ap.add_argument("--n_pts", type=int, default=20000, help="한 스텝에 쓰는 가우시안 수")
ap.add_argument("--eval_t0", type=int, nargs="+", default=[5, 40, 80],
                help="롤아웃을 시작할 프레임들. 이 궤적은 충돌 직후와 안정된 뒤의 "
                     "프레임당 변위가 수십 배 달라서, 한 구간만 보면 오해한다")
ap.add_argument("--eval_len", type=int, default=15)
ap.add_argument("--voxel", action="store_true",
                help="앵커를 매 프레임 복셀 다운샘플링으로 새로 뽑는다. 앵커 선정과 "
                     "소속과 집계가 한 패스로 접히고, FPS 가 사라진다")
ap.add_argument("--vox_ens", type=int, default=0,
                help="오프셋이 다른 격자 몇 개를 앙상블할지 (0 이면 격자 하나)")
ap.add_argument("--refps", action="store_true",
                help="매 프레임 현재 배치에서 앵커를 FPS 로 다시 뽑는다. 기본은 "
                     "t=0 에 한 번 뽑고 모델이 낸 변위로만 옮기는 것인데, 그러면 "
                     "앵커가 재질에서 떨어져 나가도 되돌아올 길이 없다")
ap.add_argument("--arch", default="conv",
                choices=("attn", "conv", "unet", "conv_sep", "conv_par",
                         "unet_sep", "unet_par"),
                help="격자 위 신경망. 집계는 셀, 출력은 격자점(c2g) 이다")
ap.add_argument("--vox_res", type=int, default=16,
                help="conv/unet 일 때 격자 한 변의 칸 수 (앵커 = 칸 전체)")
ap.add_argument("--no_mat", action="store_true",
                help="물성이 한 종류뿐이면 물성 특징은 상수라 뺀다")
ap.add_argument("--control", action="store_true",
                help="제어점 궤적을 학생에게도 준다. 앵커 특징에 **제어점으로부터의 "
                     "상대 위치**와 **이번 스텝 강제 변위**를 더하고, 강제되는 "
                     "입자의 다음 위치는 네트워크 출력 대신 궤적 값으로 덮어쓴다")
ap.add_argument("--n_ctrl", type=int, default=4, help="제어점 수 (특징 차원을 정한다)")
ap.add_argument("--grip", action="store_true",
                help="물리 집게 정보를 조건 입력으로 준다 (덮어쓰기는 없다)")
ap.add_argument("--n_arms", type=int, default=2, help="집게 팔 수")
ap.add_argument("--vox_knn_feat", action="store_true",
                help="복셀 앵커의 특징을 칸 하드 할당이 아니라 **스키닝과 같은 kNN "
                     "이웃**으로 집계한다. 지금은 앵커가 본 가우시안과 옮기는 "
                     "가우시안이 달라 특징->변위 대응이 어긋난다 -- FPS 경로는 둘이 "
                     "같은 idx 를 쓴다")
ap.add_argument("--damage", action="store_true",
                help="앵커마다 **손상률**을 하나 더 내게 하고, 그것을 자기 앵커들로 "
                     "섞어 **가우시안마다** 손상을 쌓는다. 손상이 온도를 깎아 "
                     "배정이 딱딱해지므로, 망가진 가우시안은 연속체에서 빠져 "
                     "가장 가까운 앵커만 따라간다")
ap.add_argument("--lambda_dmg", type=float, default=1.0,
                help="손상 지도 가중치. 정답은 그 결합이 GT 에서 실제로 늘어난 배율")
ap.add_argument("--dmg_thresh", type=float, default=2.0,
                help="GT 결합 길이가 이 배가 되면 손상 1 로 본다")
ap.add_argument("--shape_pts", type=int, default=0,
                help="모양 손실을 이 개수의 입자로만 잰다 (0 이면 전부). 손실이 "
                     "입자 평균이라 부분표본도 불편추정이고, svdvals 가 2 만 개 "
                     "3x3 에서 13 ms 라 여기가 한 스텝의 3 분의 1 이다")
ap.add_argument("--gpu_data", type=int, default=1,
                help="궤적을 GPU 에 상주시킨다. CPU 색인 + 전송이 한 스텝의 "
                     "5 분의 1 이라 그냥 올리는 쪽이 빠르다")
ap.add_argument("--gpu_data_mb", type=float, default=8000.0,
                help="이 용량을 넘으면 GPU 상주를 건너뛴다")
ap.add_argument("--motion_frac", type=float, default=0.0,
                help="창을 뽑을 때 GT 변위가 큰 프레임을 이 비율만큼 우선한다. "
                     "이 궤적은 100 프레임 중 ~30 만 움직이고 나머지는 완전히 "
                     "정지라, 균등하게 뽑으면 학습의 70%가 '아무 일도 안 일어남'이다")
ap.add_argument("--hold_last", type=int, default=20,
                help="각 궤적의 마지막 몇 프레임을 평가용으로 뗀다")
ap.add_argument("--hold_traj", default=None,
                help="통째로 홀드아웃할 궤적 태그 (쉼표로 구분)")
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--resume", default=None)
ap.add_argument("--small_out", action="store_true",
                help="출력층을 0 이 아니라 기본 초기화의 1/100 로 시작")
ap.add_argument("--skin_nbr", default="knn",
                choices=("knn", "corners", "n20"),
                help="스키닝 이웃: kNN16(거리 topk) / 꼭짓점 8 / 산술 20")
ap.add_argument("--skin_corners", action="store_true",
                help="스키닝 이웃을 kNN 대신 **자기 칸의 8 꼭짓점**으로 -- 탐색이\n                     공짜다. 가중치는 여전히 학습 반경으로 정해진다")
ap.add_argument("--ens", type=int, default=1,
                help="원점을 어긋나게 둔 격자를 몇 개 앙상블할지 (변위 평균)")
ap.add_argument("--metrics", action="store_true",
                help="롤아웃에서 CD/EMD 까지 잰다 (Spring-Gaus 정의)")
ap.add_argument("--cd_pts", type=int, default=2048,
                help="CD/EMD 표본 크기 (EMD 가 O(n^3) 이라 필요하다)")
ap.add_argument("--transfer", default="skin", choices=("skin", "tri"),
                help="격자점 변위를 가우시안으로 옮기는 법. skin 은 kNN 위 "
                     "학습 반경 소프트맥스, tri 는 고정 trilinear")
ap.add_argument("--no_gn", action="store_true",
                help="conv 블록의 GroupNorm 제거 (크기 정보 보존)")
ap.add_argument("--stat_occ", action="store_true",
                help="입력 표준화 통계를 **찬 셀만**으로 잡는다")
ap.add_argument("--r2", default=None, help="체크포인트를 올릴 R2 경로")
ap.add_argument("--seed", type=int, default=0)
# --- Phase 2: 물리 잔차(backward Euler 증분 포텐셜) 학습 ---------------------
ap.add_argument("--phase2", action="store_true",
                help="교사 위치 대신 **증분 포텐셜**을 목적함수로 미세조정한다. "
                     "교사 프레임이 필요 없으므로 상태를 어디서 뽑아도 된다")
ap.add_argument("--phys_w", type=float, default=1.0, help="물리 항 가중")
ap.add_argument("--phys_sup", type=float, default=0.0,
                help="교사 위치 손실을 함께 쓸 가중 (0 이면 물리 단독)")
ap.add_argument("--phys_K", type=int, default=1,
                help="한 표본에서 펼칠 스텝 수. 뒤 스텝의 상태는 자기 출력이라 "
                     "그대로 on-policy 표본이 된다")
ap.add_argument("--phys_K_warm", type=int, default=0,
                help="K 를 1 에서 --phys_K 로 올리는 데 쓰는 스텝 수")
ap.add_argument("--phys_noise", type=float, default=0.0,
                help="상태 교란 크기 (물체 크기 대비). 매끄러운 저주파 장을 더하고 "
                     "F 도 (I+grad u)F 로 함께 흔든다")
ap.add_argument("--phys_probe", type=int, default=0,
                help="교사 프레임에서 에너지·잔차만 이만큼 재고 끝낸다 (정상성 검사)")
ap.add_argument("--val_every", type=int, default=0,
                help="이 간격마다 홀드아웃 롤아웃으로 재고 best 체크포인트를 남긴다")
ap.add_argument("--val_n", type=int, default=4, help="검증에 쓸 궤적 수")
ap.add_argument("--val_len", type=int, default=10, help="검증 롤아웃 길이")
ap.add_argument("--noise", type=float, default=0.0,
                help="Phase 1 에서 창의 **시작 상태**를 매끄러운 저주파 장으로 흔든다 "
                     "(물체 크기 대비 최대 비율). 정답은 그대로 두므로 모델이 "
                     "벗어난 곳에서 돌아오는 보정을 배운다")
ap.add_argument("--wd", type=float, default=0.0,
                help="가중치 감쇠 (0 보다 크면 Adam 대신 AdamW 를 쓴다). 한 스텝 "
                     "교사 오차만 내려가고 홀드아웃이 안 따라오는 과적합을 친다")
ap.add_argument("--drop", type=float, default=0.0,
                help="conv 블록 사이 채널 드롭아웃 비율 (Dropout3d). 검증·롤아웃 "
                     "에서는 자동으로 꺼진다")
ap.add_argument("--det_reg", type=float, default=0.0,
                help="한 스텝 변형장의 야코비안 행렬식이 뒤집히는 것(det<=0)을 "
                     "벌한다. relu(margin - det)^2 의 입자 평균에 이 가중치를 곱한다")
ap.add_argument("--det_margin", type=float, default=0.1,
                help="det 가 이 값 아래로 내려가면 벌점이 붙는다 (0 이면 뒤집힘만)")
ap.add_argument("--loss_last", action="store_true",
                help="언롤 창에서 **마지막 프레임만** 손실로 쓴다. 중간 프레임의 "
                     "정답 읽기와 손실 계산이 빠지지만, 역전파는 여전히 창 전체를 "
                     "거슬러 올라간다 (마지막 상태가 앞 스텝에 의존하므로)")
ap.add_argument("--tb", default=None, help="TensorBoard 이벤트를 쓸 디렉토리")
ap.add_argument("--fe_state", action="store_true",
                help="탄성 변형구배 F_e 를 **입자 상태로** 들고 다닌다. 교사의 F 로 "
                     "시작해 매 스텝 J F_e 로 밀고 항복면에 사영하며, 그 주응력 "
                     "로그(Hencky)를 셀 입력에 더한다. 총 변형만으로는 같은 모양이라도 "
                     "소성으로 얼마나 흘렀는지 구분할 수 없다")
ap.add_argument("--mat_film", action="store_true",
                help="물성을 셀 특징에 붙이지 않고 **FiLM** 으로 넣는다. 씬 안에서 "
                     "물성이 상수라 붙이면 채널 하나를 상수로 채우는 셈이다")
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)

from anchorflow import deform                                  # noqa: E402
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import voxel                                    # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, anchor_knn,  # noqa: E402
                               bc_features, bond_stretch, bures_w2_sq,
                               fps, gauss_stretch, grid_knn,
                               jacobian_of, skin, skin_with_jacobian)

# ---------------------------------------------------------------- 데이터
files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
if not files:
    raise SystemExit(f"궤적이 없다: {a.data}")
hold = set((a.hold_traj or "").split(",")) - {""}
TR, held = [], []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    # 궤적은 half 로 저장해 두었다 (디스크 절약). 학습·통계는 float32 로 올린다.
    for _k in ("x", "v", "F"):
        if _k in d and torch.is_tensor(d[_k]) and d[_k].dtype == torch.float16:
            d[_k] = d[_k].float()
    tag = os.path.splitext(os.path.basename(f))[0]
    (held if tag in hold else TR).append((tag, d))
if not TR:
    raise SystemExit("학습할 궤적이 없다")
_MB = sum(sum(v.numel() * v.element_size() for v in d.values()
               if torch.is_tensor(v)) for _t, d in TR + held) / 1e6
if a.gpu_data and _MB < a.gpu_data_mb:
    for _t, d in TR + held:
        for k in ("x", "v", "F"):
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dev)
    print(f"[데이터] 궤적 {_MB:.0f} MB 를 GPU 에 올렸다", flush=True)
elif a.gpu_data:
    print(f"[데이터] 궤적이 {_MB:.0f} MB 라 GPU 상주를 건너뛴다 "
          f"(--gpu_data_mb {a.gpu_data_mb:.0f})", flush=True)
print(f"[데이터] 학습 {len(TR)} 궤적, 홀드아웃 {len(held)} 궤적", flush=True)
for tag, d in TR + held:
    c = d["cfg"]
    print(f"  {tag}: {d['x'].shape[0]} 프레임 x {d['x'].shape[1]} 입자, "
          f"E={c['E']:g} nu={c['nu']:g} xi={c.get('xi', 0):g} g={c['g']}", flush=True)

cfg0 = TR[0][1]["cfg"]
FRAME_DT = float(cfg0["frame_dt"])
X0 = TR[0][1]["x"][0]
EXT = float((X0.max(0).values - X0.min(0).values).norm())
N_FULL = X0.shape[0]

# 앵커는 t=0 배치에서 FPS 로 고른 **가우시안**이다. 인덱스로 들고 있으면 어느
# 프레임에서든 그 프레임의 가우시안 위치로 앵커를 초기화할 수 있다.



def _san(t):
    """특징은 O(1) 이 정상이다. 얇은 구름에서 국소 맞춤이 튀면 여기서 잘라 준다."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_FCAP, _FCAP)


if os.environ.get("AF_ANOMALY"):
    torch.autograd.set_detect_anomaly(True)             # NaN 이 어느 연산에서 나오는지
_FCAP = float(os.environ.get("AF_FEAT_CAP", "50"))   # 특징 상한 (발산 방지)
X0d = X0.to(dev)
AIDX = fps(X0d, a.n_anchors, a.seed)
H = float(torch.cdist(X0d[AIDX], X0d[AIDX]).topk(
    2, largest=False).values[:, 1].median())          # 앵커 간격
# 잎처럼 촘촘한 구름에서는 간격이 물체 크기의 2% 아래로 내려가는데, 그러면 1/H
# 로 나누는 집계 특징이 1e6 까지 커져 첫 스텝에 발산한다 (겪었다). 하한을 둔다.
_h_min = float(os.environ.get("AF_H_MIN", "0.05")) * EXT
if H < _h_min:
    print(f"[앵커] 간격 {H:.5f} 이 너무 좁아 {_h_min:.5f} 로 올린다", flush=True)
    H = _h_min
print(f"[앵커] {a.n_anchors} 개 (FPS), 간격 {H:.5f}, 물체 {EXT:.4f}", flush=True)

# 질량: 격자 점유로 부피를 재고 config 의 밀도를 곱한다 (MPM 이 하는 것과 같다)
ng = int(cfg0.get("n_grid", 100))
dx = float(cfg0.get("grid_lim", 2.0)) / ng
vi = (X0d / dx).long().clamp(0, ng - 1)
flat = (vi[:, 0] * ng + vi[:, 1]) * ng + vi[:, 2]
cnt = torch.zeros(ng ** 3, device=dev).index_add_(
    0, flat, torch.ones(N_FULL, device=dev))
MASS = ((dx ** 3) / cnt[flat]) * float(cfg0["density"])

# 물성은 앵커 특징에 들어간다. 궤적마다 다르므로 궤적별로 만든다.
def mat_feat(cfg):
    # 중력은 방향이라 평행이동 등변성을 깨지 않는다. 궤적마다 다르므로 넣는다.
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0
    # np.log 는 float64 를 돌려주고, 목록에 섞이면 텐서가 통째로 double 이 된다
    # 항복응력과 구성모델 종류도 넣는다 -- 형상·손잡이가 같아도 물성이 다르면
    # 궤적이 다르므로, 이걸 안 주면 학생은 세 물성의 평균만 배운다.
    if a.no_mat:
        return torch.zeros(0, device=dev, dtype=torch.float32)
    _mats = ("jelly", "metal", "foam", "sand")
    _oh = [1.0 if cfg.get("material", "jelly") == m else 0.0 for m in _mats]
    return torch.cat([torch.tensor(
        [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
         np.log(float(cfg["density"])),
         np.log1p(float(cfg.get("yield_stress", 0.0))),
         float(cfg.get("friction_angle", 0.0)) / 45.0] + _oh,
        device=dev, dtype=torch.float32), g])


# 프레임별 평균 GT 변위. 어디가 실제로 움직이는 구간인지 여기서 정해진다.
for _t, _d in TR + held:
    _xx = _d["x"]
    _ss = torch.randperm(_xx.shape[1])[:2000]
    mv = (_xx[1:, _ss] - _xx[:-1, _ss]).norm(dim=-1).mean(1)      # [T-1]
    _d["motion"] = mv
    print(f"  {_t}: 변위 중앙 {float(mv.median())/EXT*100:.4f}% "
          f"상위10% {float(mv.quantile(0.9))/EXT*100:.4f}% "
          f"움직이는 프레임(중앙의 3배 초과) {int((mv > 3*mv.median()).sum())}/{len(mv)}",
          flush=True)

VEL_SCALE = EXT / FRAME_DT
# 정준 공분산. ply 의 것을 쓸 수 없어(사실상 0) 입자 간격에서 만든다 -- MPM 이
# 부피를 쓰는 것과 같은 근거다. 등방이므로 L0 = sigma0 * I.
SIG0 = a.sigma0 * float(dx)
N_MAT = 0 if a.no_mat else 13   # logE, nu, xi, logρ, log1p(항복), φ, 종류4, g3
N_FILM = N_MAT if (a.mat_film and not a.no_mat) else 0
n_bc = bc_features(X0d[:2], cfg0).shape[-1]
n_feat_probe = None

# ---------------------------------------------------------------- 모델
net = None
opt = None
step0 = 0


def build(n_feat):
    global net, opt
    if a.arch.startswith(("conv", "unet")):
        from anchorflow.conv_stepper import ConvStepper
        net = ConvStepper(n_feat=n_feat, hidden=a.hidden, depth=a.depth,
                          h=H, scale=0.02 * EXT,
                          arch=a.arch.replace("conv", "plain"),
                          skin_out=(a.transfer == "skin"),
                          damage=a.damage, n_mat=N_FILM,
                          drop=a.drop).to(dev)
    else:
        net = DeformNet(n_feat=n_feat, hidden=a.hidden, depth=a.depth,
                        heads=a.heads, scale=0.02 * EXT, h=H, ext=EXT,
                        seed=a.seed, damage=a.damage).to(dev)
    opt = (torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
           if a.wd > 0 else torch.optim.Adam(net.parameters(), lr=a.lr))
    n = sum(p.numel() for p in net.parameters())
    print(f"[모델] 입력 {n_feat}, 파라미터 {n/1e6:.2f}M", flush=True)


def take(t, idx_gpu):
    """궤적에서 부분표본을 떼어 GPU 로.

    궤적이 GPU 에 올라가 있으면 그냥 색인한다 (0.013 ms). CPU 에 있으면 색인을
    CPU 에서 해야 하고 (장치가 섞이면 torch 가 거부한다) 그것이 위치 1.61 ms,
    F 4.08 ms 로 한 스텝의 5 분의 1 을 먹는다 -- 궤적 하나가 98 MB 라 다 올려도
    7 개에 700 MB 다. 못 올릴 이유가 없었다."""
    if t.is_cuda:
        return t[idx_gpu]
    return t[idx_gpu.cpu()].to(dev, non_blocking=True)


VOX_OFFS = (np.array([np.random.RandomState(i).rand(3)
                     for i in range(a.vox_ens)]) if a.vox_ens else None)
VOX_CELL = None      # 첫 호출에서 앵커 간격으로 정한다
VOX_LO = None
VOX_DIMS = None      # 격자 치수는 궤적 전체로 한 번만 잡는다 (동기화 제거)


def vox_feats(d, gsel, x, v):
    """복셀 앵커와 그 특징. -> (p, feat, idx)"""
    cfg = d["cfg"]
    X = take(d["x"][0], gsel)
    cell = VOX_CELL * (max(a.vox_ens, 1) ** (1.0 / 3.0))
    vb = voxel.build(x, X, v / VEL_SCALE, MASS[gsel], cell, lo=VOX_LO,
                     offsets=VOX_OFFS, dims=VOX_DIMS)
    idx, _ = voxel.neighbors(x, vb, a.k)
    if a.vox_knn_feat:
        # 앵커 위치만 복셀로 정하고, 특징은 FPS 경로와 똑같이 그 앵커가 실제로
        # 옮길 가우시안들(같은 idx)로 집계한다.
        # voxel.neighbors 는 3x3x3 안에 앵커가 k 개보다 적으면 빈 자리를 -1 로
        # 채운다. 그대로 집계에 넘기면 커널이 음수 색인으로 메모리를 벗어난다 --
        # 가장 가까운 앵커로 메운다 (그 가우시안에서 한 번 더 세는 것뿐이다).
        idxf = torch.where(idx < 0, idx[:, :1].expand_as(idx), idx).clamp(min=0)
        feat, _ = aggregate(x, v / VEL_SCALE, X, MASS[gsel], idxf, vb.M, cell,
                            pa=vb.pos)
        feat = _san(feat)
        return vb.pos, feat, idx
    W, cx, cX, cv, cnt, g2 = vb.moments
    M = vb.M
    S = g2[:, :9].reshape(M, 3, 3) / W.reshape(M, 1, 1)
    iu = torch.triu_indices(3, 3, device=dev)
    S6 = S[:, iu[0], iu[1]] / (cell * cell)
    tr = S.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(M, 1, 1)
    I3 = torch.eye(3, device=dev)
    om = torch.linalg.solve(
        (tr * I3 - S) * W.reshape(M, 1, 1) + 1e-8 * I3,
        g2[:, 9:12].unsqueeze(-1)).squeeze(-1)
    Fa = g2[:, 12:21].reshape(M, 3, 3) @ torch.linalg.inv(
        g2[:, 21:30].reshape(M, 3, 3) + (1e-6 * cell * cell) * I3)
    det = torch.linalg.det(Fa).reshape(M, 1)
    feat = torch.cat([
        torch.log(W).reshape(M, 1), torch.log1p(cnt), (cx - cX) / cell,
        torch.zeros_like(cx),            # 앵커 위치 = 질량중심이라 상대값이 0 이다
        cv, S6, om, Fa.reshape(M, 9),
        torch.sign(det) * torch.log(det.abs().clamp(min=1e-6))], -1)
    return vb.pos, feat, idx


def ctrl_feat_pts(d, t, q, cell):
    """**가우시안마다** 손잡이 특징 -> [N, 7K].

    셀 중심 하나로 뭉개면 같은 칸 안에서 손잡이에 가까운 입자와 먼 입자가
    구분되지 않는다. 길이는 셀 크기로 정규화한다.
    """
    K = a.n_ctrl
    N = q.shape[0]
    if "ctrl_pos" not in d:
        return torch.zeros(N, 7 * K, device=q.device, dtype=q.dtype)
    P = d["ctrl_pos"].to(q.device, q.dtype)
    tt = min(t, P.shape[0] - 1)
    c = P[tt]
    dc = P[min(tt + 1, P.shape[0] - 1)] - c
    k = c.shape[0]
    if k < K:
        z = torch.zeros(K - k, 3, device=q.device, dtype=q.dtype)
        c, dc = torch.cat([c, z]), torch.cat([dc, z])
    c, dc = c[:K], dc[:K]
    rel = (q.unsqueeze(1) - c.unsqueeze(0)) / cell
    frc = dc.unsqueeze(0).expand(N, K, 3) / cell
    if "ctrl_R" in d:
        R = d["ctrl_R"].to(q.device, q.dtype)
        Rt = R[min(t, R.numel() - 1)].clamp(min=1e-6)
        qq = ((q.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / Rt).clamp(0, 1)
        w = (1.0 - qq * qq) ** 2
    else:
        w = torch.zeros(N, K, device=q.device, dtype=q.dtype)
    return torch.cat([rel.reshape(N, -1), frc.reshape(N, -1), w], -1)


def ctrl_feat(d, t, p):
    """앵커마다 (제어점 상대위치, 이번 스텝 강제 변위) -> [A, 6K].

    상대 위치는 **어디를 누르고 있는지**, 강제 변위는 **어느 쪽으로 미는지**를
    말해 준다. 둘 다 앵커 간격 H 로 나눠 크기를 맞춘다.
    """
    K = a.n_ctrl
    A = p.shape[0]
    if "ctrl_pos" not in d:
        return torch.zeros(A, 6 * K, device=p.device, dtype=p.dtype)
    P = d["ctrl_pos"].to(p.device, p.dtype)              # [T, k, 3]
    tt = min(t, P.shape[0] - 1)
    c = P[tt]
    dc = P[min(tt + 1, P.shape[0] - 1)] - c
    k = c.shape[0]
    if k < K:                                            # 모자라면 0 으로 채운다
        c = torch.cat([c, torch.zeros(K - k, 3, device=p.device, dtype=p.dtype)])
        dc = torch.cat([dc, torch.zeros(K - k, 3, device=p.device, dtype=p.dtype)])
    c, dc = c[:K], dc[:K]
    rel = (p.unsqueeze(1) - c.unsqueeze(0)) / H          # [A, K, 3]
    frc = dc.unsqueeze(0).expand(A, K, 3) / H
    # 손잡이 **소속 가중치**. 반경 R 안의 입자는 계획 속도로 끌려가므로,
    # "이 앵커가 얼마나 끌려가는가" 를 알려 주지 않으면 위치만으로는 알 수 없다.
    if "ctrl_R" in d:
        R = d["ctrl_R"].to(p.device, p.dtype)
        Rt = R[min(t, R.numel() - 1)].clamp(min=1e-6)
        q = ((p.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / Rt).clamp(0, 1)
        w = (1.0 - q * q) ** 2                            # [A, K]
    else:
        w = torch.zeros(A, K, device=p.device, dtype=p.dtype)
    return torch.cat([rel.reshape(A, -1), frc.reshape(A, -1),
                      w.reshape(A, -1)], -1)


def grip_feat(d, t, p):
    """앵커마다 집게 정보 -> [A, 13*arms].

    (상대위치 3, 판 법선 3, 이번 스텝 판 이동 3, 판 접선 3, 간격 1) x 팔 수.
    잡은 자리·무는 방향·움직이는 방향을 다 담아야 학생이 조작을 따라갈 수 있다.
    길이는 앵커 간격 H 로 나눠 크기를 맞춘다.
    """
    A, M = p.shape[0], a.n_arms
    if "grip" not in d:
        return torch.zeros(A, 13 * M, device=p.device, dtype=p.dtype)
    G = d["grip"].to(p.device, p.dtype)                  # [T, arms, 13]
    tt = min(t, G.shape[0] - 1)
    g0 = G[tt]
    g1 = G[min(tt + 1, G.shape[0] - 1)]
    k = g0.shape[0]
    if k < M:
        pad = torch.zeros(M - k, 13, device=p.device, dtype=p.dtype)
        g0, g1 = torch.cat([g0, pad]), torch.cat([g1, pad])
    g0, g1 = g0[:M], g1[:M]
    c = g0[:, :3]                                        # 중심
    R = g0[:, 3:12].reshape(M, 3, 3)                     # 행이 축 (법선, 접선1, 접선2)
    gap = g0[:, 12:13]
    rel = (p.unsqueeze(1) - c.unsqueeze(0)) / H          # [A, M, 3]
    mov = ((g1[:, :3] - c) / H).unsqueeze(0).expand(A, M, 3)
    nrm = R[:, 0].unsqueeze(0).expand(A, M, 3)
    tan = R[:, 1].unsqueeze(0).expand(A, M, 3)
    gg = gap.reshape(1, M, 1).expand(A, M, 1)
    return torch.cat([rel, nrm, mov, tan, gg], -1).reshape(A, -1)


def ctrl_local(d, gsel):
    """제어점 명단을 **부분표본 좌표계**로 옮긴다.

    `ctrl_mem` 은 궤적 전체(4 만 입자) 기준인데 학습은 `gsel` 로 솎은 것만 본다.
    그대로 색인하면 범위를 벗어나 CUDA 가 죽는다 (겪었다). 전체 크기의 불린
    마스크를 만들어 `gsel` 로 다시 읽으면 좌표계가 맞는다.
    """
    key = (int(gsel.numel()), int(gsel[0]), int(gsel[-1]), id(d))
    if d.get("_cl_key") == key:
        return d["_cl"]
    Nf = d["x"].shape[1]
    out = []
    for k, mm in enumerate(d.get("ctrl_mem", [])):
        full = torch.zeros(Nf, dtype=torch.bool)
        full[mm] = True
        loc = torch.nonzero(full[gsel.cpu()]).squeeze(-1).to(gsel.device)
        # 대응하는 offset (제어점 기준 상대 위치) 도 같은 순서로
        g = gsel[loc].cpu()
        off = d["x"][0][g] - d["x"][0][d["ctrl"][k]]
        out.append((loc, off))
    d["_cl"] = out
    d["_cl_key"] = key
    return out


def free_mask(d, n, device, gsel):
    """강제되지 **않은** 입자만 True. 손실은 여기서만 잰다.

    강제된 입자는 궤적 값으로 덮어쓰므로 오차가 정확히 0 이다. 그대로 평균에
    넣으면 손실이 희석돼 모델이 좋아 보인다 (그리고 기울기도 안 준다).
    """
    m = torch.ones(n, dtype=torch.bool, device=device)
    for loc, _ in ctrl_local(d, gsel):
        m[loc] = False
    return m


def apply_control(d, t, gsel, x2):
    """강제되는 입자의 다음 위치를 **궤적 값으로 덮어쓴다**.

    교사가 그 입자들을 Dirichlet 으로 박았으므로, 학생이 거기를 예측하게 두면
    맞출 수 없는 것을 맞추라고 시키는 셈이다.
    """
    if "ctrl_mem" not in d or "ctrl_pos" not in d:
        return x2
    P = d["ctrl_pos"].to(x2.device, x2.dtype)
    t1 = min(t + 1, P.shape[0] - 1)
    x2 = x2.clone()
    for k, (loc, off) in enumerate(ctrl_local(d, gsel)):
        if k >= P.shape[1] or loc.numel() == 0:
            continue
        # 무리는 제어점과 **같은 offset 으로** 움직인다 (교사가 그렇게 박는다)
        x2[loc] = P[t1, k] + off.to(x2.device, x2.dtype)
    return x2


_ENS_SHIFT = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5),
              (0.0, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
              (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)]

from anchorflow import trilinear as TRI          # noqa: E402
from anchorflow import vox_anchor                # noqa: E402


def fe_invariants(fe):
    """F_e 의 주응력 로그 [N,3]. 회전에 불변이라 그대로 특징으로 쓸 수 있다."""
    C = fe.transpose(-1, -2) @ fe
    sig = torch.linalg.eigvalsh(C.double()).clamp_min(1e-12).sqrt()
    return sig.clamp_min(0.01).log().to(fe.dtype)


def cell_feats(d, t, gsel, x, v, shift=None, fe=None):
    """학습·통계·추론이 **모두 같은** 입력을 쓰도록 한 군데서 만든다.

    -> (_in [셀, F], p 셀중심, grid_shape (격자점, 셀), tri (평탄idx, 가중치),
        (lo, h, n) 격자, crow 셀 색인)
    """
    cfg = d["cfg"]
    X = take(d["x"][0], gsel)
    lo, hh, nn3 = vox_anchor.grid_for(x, a.vox_res ** 3)
    if shift is not None:          # 앙상블: 격자 원점을 반 칸씩 어긋나게 둔다
        lo = lo - torch.tensor(shift, device=lo.device, dtype=lo.dtype) * hh
        nn3 = nn3 + 1              # 어긋난 만큼 한 칸 더 덮는다
    # 집계는 **셀 기준** (가우시안당 한 번), 출력은 격자점 기준 -> c2g 가 옮긴다
    crow, cw, ncell = TRI.cell_index(x, lo, hh, nn3)
    M_cell = ncell[0] * ncell[1] * ncell[2]
    ccen = (torch.stack(torch.meshgrid(
        *[torch.arange(ncell[dd], device=dev, dtype=x.dtype) for dd in range(3)],
        indexing="ij"), -1).reshape(-1, 3) + 0.5) * hh + lo
    feat = _san(TRI.tri_feats(x, v / VEL_SCALE, X, MASS[gsel], crow, cw,
                              M_cell, ccen, hh))
    # 손잡이는 **가우시안마다** 만들어 같은 셀 집계에 싣는다 (질량가중 평균)
    cond = None
    if a.control:
        cf = ctrl_feat_pts(d, t, x, hh)
        wm = cw * MASS[gsel].unsqueeze(1)
        num = torch.zeros(M_cell, cf.shape[-1], device=dev, dtype=cf.dtype)
        num.index_add_(0, crow.reshape(-1),
                       (wm.unsqueeze(-1) * cf.unsqueeze(1)).reshape(-1, cf.shape[-1]))
        den = torch.zeros(M_cell, 1, device=dev, dtype=cf.dtype)
        den.index_add_(0, crow.reshape(-1), wm.reshape(-1, 1))
        cond = _san(num / den.clamp(min=1e-12))
    p = ccen                       # 조건·경계 특징은 셀 중심에서 읽는다
    flat_c, w_c = TRI.corners(x, lo, hh, nn3)      # 출력(격자점) -> 가우시안
    grid_pts = tuple(int(t) for t in nn3)
    grid_shape = (grid_pts, tuple(ncell))
    tri = (flat_c, w_c)
    if a.fe_state:
        # 입자별 불변량을 셀로 질량가중 평균한다 (집계 가중치는 cell_feats 와 같다)
        _ei = (fe_invariants(fe) if fe is not None
               else torch.zeros(x.shape[0], 3, device=dev, dtype=x.dtype))
        _wm = cw * MASS[gsel].unsqueeze(1)
        _num = torch.zeros(M_cell, 3, device=dev, dtype=x.dtype)
        _num.index_add_(0, crow.reshape(-1),
                        (_wm.unsqueeze(-1) * _ei.unsqueeze(1)).reshape(-1, 3))
        _den = torch.zeros(M_cell, 1, device=dev, dtype=x.dtype)
        _den.index_add_(0, crow.reshape(-1), _wm.reshape(-1, 1))
        _fecell = _num / _den.clamp(min=1e-12)
    if a.mat_film:
        extra = bc_features(p, cfg) / hh      # 물성은 FiLM 으로 따로 들어간다
    else:
        extra = torch.cat([mat_feat(cfg).reshape(1, N_MAT).expand(p.shape[0],
                                                                  N_MAT),
                           bc_features(p, cfg) / hh], -1)
    if a.fe_state:
        extra = torch.cat([extra, _fecell], -1)
    if cond is not None:
        extra = torch.cat([extra, cond], -1)
    if a.grip:
        extra = torch.cat([extra, grip_feat(d, t, p)], -1)
    # 국소 변형구배 맞춤이 잎처럼 얇은 구름에서 특이해지면 특징이 1e7 까지 튀어
    # 첫 스텝에 발산한다 (겪었다). 입력은 O(1) 이 정상이므로 잘라서 넣는다.
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_FCAP, _FCAP)
    extra = torch.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_FCAP, _FCAP)
    _in = torch.cat([feat, extra], -1)
    return _in, p, grid_shape, tri, (lo, hh, nn3), crow


# ---------------------------------------------------------------- 구간 시간
# AF_PROF=1 이면 한 스텝의 구간별 시간을 모은다. CUDA 는 비동기라 구간을 재려면
# 매번 동기화해야 하므로, 켜면 전체가 느려진다 -- 어디가 비싼지 가릴 때만 쓴다.
_PROF = {}
_PROF_ON = bool(os.environ.get("AF_PROF"))


class _pt:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _PROF_ON:
            torch.cuda.synchronize()
            self.t0 = time.time()
        return self

    def __exit__(self, *_):
        if _PROF_ON:
            torch.cuda.synchronize()
            _PROF[self.name] = _PROF.get(self.name, 0.0) + (time.time() - self.t0)


def step_once(d, t, gsel, p, x, v, need_J=True, dmg=None, idx_prev=None,
              x0=None, p0=None, fe=None):
    """한 프레임. -> (x_next, p_next, v_next, J, dp, aidx_next)

    aidx_next 는 --refps 일 때만 뜻이 있다: 다음 프레임의 앵커가 **현재 부분표본의
    몇 번째 가우시안인지**. 매 스텝 다시 뽑으면 앵커의 정체가 바뀌므로, 앵커 손실이
    비교할 정답도 그때그때 그 가우시안들의 GT 변위로 바뀐다.
    """
    # 앙상블: 원점을 어긋나게 둔 격자 여러 개의 변위를 평균한다. 같은 가중치를
    # 쓰므로 파라미터는 늘지 않고, 격자 위치 때문에 생기는 편향만 씻긴다.
    if a.arch == "attn":
        # 예전 어텐션 스테퍼 경로를 그대로 되살린 것이다: FPS 앵커에 가우시안을
        # kNN 으로 모아 특징을 만들고, 한 층에서 모든 앵커가 서로를 본 뒤,
        # 학습된 반경 스키닝으로 가우시안을 옮긴다. 앵커는 자기 변위만큼 간다.
        cfg = d["cfg"]
        X = take(d["x"][0], gsel)
        idx, _ = anchor_knn(x, p, a.k)
        feat, _ = aggregate(x, v / VEL_SCALE, X, MASS[gsel], idx, p.shape[0], H,
                            pa=p)
        feat = _san(feat)
        extra = torch.cat([
            mat_feat(cfg).reshape(1, N_MAT).expand(p.shape[0], N_MAT),
            bc_features(p, cfg) / H], -1)
        if a.control:
            extra = torch.cat([extra, ctrl_feat(d, t, p)], -1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0,
                                neginf=0.0).clamp(-_FCAP, _FCAP)
        extra = torch.nan_to_num(extra, nan=0.0, posinf=0.0,
                                 neginf=0.0).clamp(-_FCAP, _FCAP)
        out = net(p, torch.cat([feat, extra], -1), FRAME_DT)
        dp, log_r, log_t = out[0], out[1], out[2]
        dp = torch.nan_to_num(dp, nan=0.0, posinf=0.0,
                              neginf=0.0).clamp(-0.5 * H, 0.5 * H)
        log_r = torch.nan_to_num(log_r, nan=0.0, posinf=0.0,
                                 neginf=0.0).clamp(-4.0, 4.0)
        log_t = torch.nan_to_num(log_t, nan=0.0, posinf=0.0,
                                 neginf=0.0).clamp(-4.0, 4.0)
        x2 = skin(x, p, dp, log_r, log_t, idx, H)[0]
        if a.control:
            x2 = apply_control(d, t, gsel, x2)
        J = (jacobian_of(lambda q: skin(q, p, dp, log_r, log_t, idx, H)[0], x)
             if need_J else None)
        return x2, p + dp, (x2 - x) / FRAME_DT, J, dp, None, dmg, idx, fe

    shifts = _ENS_SHIFT[:a.ens] if a.ens > 1 else [None]
    acc, warps, dp = 0.0, [], None
    for _sh in shifts:
        with _pt("셀집계"):
            _in, p, grid_shape, tri, (lo, hh, nn3), crow = cell_feats(
                d, t, gsel, x, v, shift=_sh, fe=fe)
        _mv = (mat_feat(d["cfg"]).reshape(1, N_MAT) if N_FILM else None)
        with _pt("신경망"):
            out = net(p, _in, FRAME_DT, grid_shape[0], cells=grid_shape[1],
                      mat=_mv)
        dp = out[0]
        # 얇은 잎 같은 구름에서는 한 번의 큰 출력이 다음 스텝의 kNN 을 망가뜨려
        # (NaN 거리 -> 엉뚱한 색인) CUDA assert 로 죽는다. 물리적으로 말이 되는
        # 범위로 잘라 둔다: 한 스텝에 앵커 간격의 절반을 넘게 움직이지 않는다.
        dp = torch.nan_to_num(dp, nan=0.0, posinf=0.0,
                              neginf=0.0).clamp(-0.5 * H, 0.5 * H)
        if a.transfer == "skin":
            # 격자점을 앵커로 두고 kNN + 학습 반경 소프트맥스로 옮긴다. 격자가
            # 규칙적이라 kNN 은 탐색 없이 구한다 (격자점 = 원점을 반 칸 당긴
            # 격자의 칸 중심).
            if a.skin_corners or a.skin_nbr == "corners":
                sidx = tri[0]          # cell_feats 가 이미 만든 꼭짓점 색인
            elif a.skin_nbr == "n20":
                sidx = vox_anchor.nbr20(x, lo, hh, nn3)
            else:
                lo_g = lo - 0.5 * hh
                sidx = vox_anchor.knn(x, lo_g, hh, nn3, a.k)
            gpos = (torch.stack(torch.meshgrid(
                *[torch.arange(int(nn3[dd]), device=dev, dtype=x.dtype)
                  for dd in range(3)], indexing="ij"), -1).reshape(-1, 3)
                ) * hh + lo
            log_r = (out[1] if len(out) > 2
                     else torch.full((dp.shape[0],), math.log(hh), device=dev))
            log_t = out[2] if len(out) > 2 else torch.zeros_like(log_r)
            with _pt("스키닝"):
                if a.fe_state or a.det_reg > 0:
                    # 해석적 야코비안을 그대로 쓴다 (자동미분 세 번보다 싸다)
                    xe, _w8, _Jf = skin_with_jacobian(x, gpos, dp, log_r,
                                                      log_t, sidx, float(hh))
                else:
                    xe = skin(x, gpos, dp, log_r, log_t, sidx, float(hh))[0]
                    _Jf = None
            warps.append(lambda q, _g=gpos, _d=dp, _lr=log_r, _lt=log_t,
                         _i=sidx, _h=float(hh): skin(q, _g, _d, _lr, _lt,
                                                     _i, _h)[0])
        else:
            xe = x + TRI.g2p(tri[0], tri[1], dp)
            warps.append(lambda q, _lo=lo, _h=hh, _n=nn3, _d=dp:
                         q + TRI.g2p(*TRI.corners(q, _lo, _h, _n), _d))
        acc = acc + (xe - x)
    x2 = x + acc / len(shifts)
    dmg_out = None
    if a.control:
        x2 = apply_control(d, t, gsel, x2)
    # 격자는 다음 프레임에 x2 로부터 다시 잡는다. 앵커를 옮길 필요가 없다.
    ai, p_next = None, p
    J = None
    if need_J:
        def _warp(q):
            return q + sum(w(q) - q for w in warps) / len(warps)

        J = jacobian_of(_warp, x)
    fe_next = fe
    if a.fe_state and fe is not None and _Jf is not None:
        # 교사와 같은 절차: 시험 변형구배를 밀고 항복면으로 사영한다
        with _pt("F_e갱신"):
            _ftr = _Jf @ fe
            with torch.no_grad():
                _ps, _dlg = phys_resid.psi_of(_ftr, d["cfg"], FRAME_DT)
                fe_next = phys_resid.plastic_step(_ftr, _dlg).detach()
    # det 벌점은 **해석적** 야코비안으로 잰다. 자동미분 야코비안은 need_J 일 때만
    # 있고 세 배 비싸다. 둘 다 없으면 벌점을 못 매기므로 None 으로 돌려준다.
    return (x2, p_next, (x2 - x) / FRAME_DT, J, dp, ai, dmg, crow, fe_next,
            (_Jf if _Jf is not None else J))


_F_MSG = []


def traj_F(d):
    """탄성 변형구배 [T,N,3,3]. 궤적에 든 것은 전부 항등이라 위치에서 되살린다."""
    if d.get("_Fok") is None:
        _Fin = d["F"]
        _id = float((_Fin[:3].float().to(dev)
                     - torch.eye(3, device=dev)).abs().max()) < 1e-6
        if _id:
            t_ = time.time()
            d["F"] = phys_resid.rebuild_F(d["x"].to(dev).float(), d["cfg"],
                                          FRAME_DT).cpu() if not a.gpu_data \
                else phys_resid.rebuild_F(d["x"].to(dev).float(), d["cfg"],
                                          FRAME_DT)
            if not _F_MSG:
                print(f"[Phase2] 궤적의 F 가 항등이라 위치에서 복원한다 "
                      f"(궤적당 {time.time()-t_:.1f}초)", flush=True)
                _F_MSG.append(1)
        d["_Fok"] = True
    return d["F"]


def traj_mass(d):
    """궤적 자기 배치·자기 밀도로 잰 입자 질량 [N_full]. 전조합 학습에서는 씬마다
    밀도도 형상도 다르므로 cfg0 의 것을 쓰면 안 된다."""
    if "_mass" in d:
        return d["_mass"]
    c = d["cfg"]
    x0 = d["x"][0].to(dev).float()
    ng_ = int(c.get("n_grid", 100))
    dx_ = float(c.get("grid_lim", 2.0)) / ng_
    vi_ = (x0 / dx_).long().clamp(0, ng_ - 1)
    fl_ = (vi_[:, 0] * ng_ + vi_[:, 1]) * ng_ + vi_[:, 2]
    cn_ = torch.zeros(ng_ ** 3, device=dev).index_add_(
        0, fl_, torch.ones(x0.shape[0], device=dev))
    d["_mass"] = ((dx_ ** 3) / cn_[fl_]) * float(c["density"])
    d["_ext"] = float((x0.max(0).values - x0.min(0).values).norm())
    return d["_mass"]


def phys_window(d, t0, K, gsel, sigma, gen):
    """Phase 2 의 한 표본. 교사 프레임에서 상태를 뽑아 **노이즈를 섞고** K 스텝
    펼치며 매 스텝의 증분 포텐셜을 더한다.

    교사 다음 프레임이 필요 없다 -- 목적함수가 상태만으로 정의되므로 교란된
    상태에서도 정답(그 상태에서 출발한 backward Euler 해)이 있다. 지도학습이었다면
    교란할 때마다 교사를 다시 돌려야 했다.
    """
    mass_full = traj_mass(d)
    mass = mass_full[gsel] * (float(N_FULL) / gsel.numel())   # 부분표본 보정
    ext = d.get("_ext", EXT)
    cfg = d["cfg"]
    vol = mass / float(cfg["density"])
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    h = FRAME_DT
    norm = float(mass.sum()) * (ext ** 2) / (h * h)

    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / h
    F = take(traj_F(d)[t0], gsel).float()
    if sigma > 0:
        u, gu = phys_resid.smooth_noise(x, sigma * ext, ext, gen)
        x = x + u
        F = (torch.eye(3, device=dev) + gu) @ F
        # 변위 교란을 한 스텝에 걸친 것으로 보면 속도도 그만큼 달라져 있다
        v = v + float(torch.rand(1, generator=gen, device=dev)) * u / h
    p = take(d["x"][t0], AIDX)
    e_tot, r_free, r_ring, parts = 0.0, 0.0, 0.0, None
    for i in range(K):
        xtil = x + h * v
        x2, p, v, J, _dp, _ai, _dmg, _cr, _fe, _Jd = step_once(
            d, t0 + i, gsel, p, x, v, need_J=True)
        F_tr = J @ F
        fm = free_mask(d, x2.shape[0], dev, gsel) if a.control else None
        E, dlog, parts = phys_resid.ip_energy(
            x2, xtil, F_tr, mass, vol, cfg, h, free=fm, g=g, norm=norm)
        e_tot = e_tot + E
        if i == 0:
            with torch.no_grad():
                # 잔차 대용: 자유낙하 예측에서 얼마나 벗어났나. 구속 입자는 따로
                # 본다 -- 거기는 반력이 실어 나르는 곳이라 값이 큰 게 정상이다.
                rr = (x2 - xtil).norm(dim=-1) / ext
                if fm is None:
                    r_free, r_ring = float(rr.mean()), 0.0
                else:
                    r_free = float(rr[fm].mean()) if int(fm.sum()) else 0.0
                    nfm = ~fm
                    r_ring = float(rr[nfm].mean()) if int(nfm.sum()) else 0.0
        F = phys_resid.plastic_step(F_tr, dlog).detach() if K > 1 else F_tr
        x = x2
    return e_tot / K, r_free, r_ring, parts


def window(d, t0, L, gsel):
    """궤적 d 의 t0 에서 L 프레임. 앵커는 그 프레임의 GT 가우시안으로 초기화.

    같은 창에서 **아무것도 안 했을 때**의 오차도 함께 낸다. 이 궤적은 충돌 직후와
    안정된 뒤의 프레임당 변위가 수십 배 차이라, 손실의 절대값만 보면 어려운 창을
    뽑았는지 모델이 나빠졌는지 구별할 수 없다. 정지 기준선과의 비를 봐야 한다
    (기준선은 모델과 무관한 양이므로 자체 변위 정규화가 아니다).
    """
    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / FRAME_DT
    if a.noise > 0:
        # 시작 상태만 흔들고 **정답은 그대로 둔다**. 그러면 모델이 "벗어난 곳에서
        # 제자리로 돌아오는" 보정을 배운다 -- 롤아웃에서 실제로 필요한 능력이고,
        # 교사 궤적을 다시 돌릴 필요가 없다. 앵커도 같은 장으로 옮겨야 배치와
        # 앵커가 어긋나지 않는다.
        _sg = a.noise * (10.0 ** (-2.0 * (1.0 - float(
            torch.rand(1, generator=gen, device=dev)))))
        _u, _gu = phys_resid.smooth_noise(x, _sg * EXT, EXT, gen)
        x = x + _u
        v = v + float(torch.rand(1, generator=gen, device=dev)) * _u / FRAME_DT
    # 앵커는 그 프레임의 GT 가우시안이다. --refps 면 현재 부분표본에서 매 스텝
    # 다시 뽑으므로 시작도 부분표본 안에서 잡는다.
    ai = fps(x, a.n_anchors, a.seed) if a.refps else None
    p = x[ai] if a.refps else take(d["x"][t0], AIDX)
    loss_x = loss_J = loss_a = loss_d = loss_det = 0.0
    n_used = 0
    still = a_rel = d_rel = 0.0
    x_still = x.clone()
    x0w, p0w = x.clone(), p.clone()          # 손상의 기준 배치
    dmg, idx_prev = None, None
    fe = (take(traj_F(d)[t0], gsel).float() if a.fe_state else None)
    for i in range(L):
        ai_now = ai
        x2, p, v, J, dp, ai, dmg, idx_prev, fe, Jdet = step_once(
            d, t0 + i, gsel, p, x, v, need_J=a.lambda_J > 0,
            dmg=dmg, idx_prev=idx_prev, x0=x0w, p0=p0w, fe=fe)
        if a.damage:
            # 정답: 그 가우시안이 자기 앵커들에 대해 GT 에서 실제로 늘어난 배율.
            # 앵커도 가우시안이라 양끝의 GT 위치를 그대로 집을 수 있다.
            gx = take(d["x"][t0 + i + 1], gsel)
            gp = (gx[ai_now] if a.refps else take(d["x"][t0 + i + 1], AIDX))
            refw = (x0w.unsqueeze(1) - p0w[idx_prev]).norm(dim=-1)
            sg = (((gx.unsqueeze(1) - gp[idx_prev]).norm(dim=-1)
                   / refw.clamp_min(1e-9) - 1.0).clamp_min(0.0)).mean(1)
            tgt = (sg / max(a.dmg_thresh - 1.0, 1e-6)).clamp(0.0, 1.0)
            ld = ((dmg - tgt) ** 2).mean()
            loss_d = loss_d + ld
            d_rel = d_rel + float(dmg.mean())
        # 앵커의 정답 변위: 앵커가 가우시안이므로 그 가우시안의 GT 변위 그대로다
        if a.voxel:
            # 복셀 앵커는 특정 가우시안이 아니라 그 칸의 질량중심이라, 정답 변위도
            # 그 칸 구성원들의 평균 변위로 잡는다.
            dp_gt = None
        elif a.arch.startswith(("conv", "unet")):
            # 격자점 변위는 특정 가우시안에 대응하지 않는다 -- 앵커 손실 없음
            dp_gt = None
        elif a.refps:
            gt_now = take(d["x"][t0 + i + 1], gsel)
            dp_gt = gt_now[ai_now] - x[ai_now]
        else:
            dp_gt = (take(d["x"][t0 + i + 1], AIDX)
                     - take(d["x"][t0 + i], AIDX))
        la = (torch.zeros((), device=dev) if dp_gt is None
              else ((dp - dp_gt) ** 2).sum(-1).mean() / (EXT ** 2))
        loss_a = loss_a + la
        a_rel = a_rel + (0.0 if dp_gt is None else float(la) ** 0.5 / max(
            float((dp_gt ** 2).sum(-1).mean()) ** 0.5 / EXT, 1e-20))
        # --loss_last 면 중간 프레임의 정답은 아예 읽지 않는다
        gt = (take(d["x"][t0 + i + 1], gsel)
              if ((not a.loss_last) or i == L - 1) else None)
        if gt is not None and os.environ.get("AF_DIAG2"):
            _u = gt - x                              # 정답 변위
            _pd = x2 - x                             # 망이 낸 변위
            _c = float((_pd * _u).sum() / (_pd.norm() * _u.norm()).clamp(min=1e-20))
            _cv = float((v * FRAME_DT * _u).sum()
                        / ((v * FRAME_DT).norm() * _u.norm()).clamp(min=1e-20))
            print(f"  [출력] |dp|/|u| {float(_pd.norm()/_u.norm().clamp(min=1e-20)):.4f}  "
                  f"cos(dp,u) {_c:+.4f}  cos(v*dt,u) {_cv:+.4f}  "
                  f"|u| {float(_u.norm()):.4e}", flush=True)
        fm = free_mask(d, x2.shape[0], x2.device, gsel) if a.control else None
        _use = (not a.loss_last) or (i == L - 1)
        if _use and fm is not None:  # 강제된 입자는 오차 0 이라 평균을 희석시킨다
            loss_x = loss_x + ((x2[fm] - gt[fm]) ** 2).sum(-1).mean() / (EXT ** 2)
            still = still + float(((x_still[fm] - gt[fm]) ** 2).sum(-1).mean()) / (EXT ** 2)
            n_used += 1
        elif _use:
            loss_x = loss_x + ((x2 - gt) ** 2).sum(-1).mean() / (EXT ** 2)
            still = still + float(((x_still - gt) ** 2).sum(-1).mean()) / (EXT ** 2)
            n_used += 1
        if a.det_reg > 0 and Jdet is not None:
            # 뒤집힌 요소(det<=0)는 물리적으로 불가능하고, 롤아웃이 터지는 자리는
            # 대개 여기다. 여유 margin 을 두어 0 에 닿기 전에 밀어낸다.
            _det = torch.linalg.det(Jdet.float())
            _pen = torch.relu(a.det_margin - _det) ** 2
            loss_det = loss_det + (_pen[fm].mean() if fm is not None
                                   else _pen.mean())
        if a.shape_loss != "none" and a.lambda_J > 0:
            # 복원한 F 는 디스크·메모리를 아끼려 half 로 들고 있다
            F0 = take(traj_F(d)[t0 + i], gsel).float()
            F1 = take(traj_F(d)[t0 + i + 1], gsel).float()
            Jgt = F1 @ torch.linalg.inv(F0 + 1e-4 * torch.eye(3, device=dev))
            if a.shape_loss == "frob":
                _d = ((J - Jgt) ** 2).sum((-1, -2))
                loss_J = loss_J + (_d[fm].mean() if fm is not None else _d.mean())
            else:
                # 현재 프레임 가우시안의 인수 L_t = sigma0 * F_t. 예측/정답 공분산은
                # 각각 (J L_t)(J L_t)^T, (Jgt L_t)(Jgt L_t)^T 이므로 인수만 넘기면 된다.
                Lt = SIG0 * F0
                _b = bures_w2_sq(x2, gt, J @ Lt, Jgt @ Lt)
                loss_J = loss_J + ((_b[fm].mean() if fm is not None else _b.mean())
                                   / (EXT ** 2))
        x = x2
    _nu = max(n_used, 1)
    return (loss_x / _nu,
            (loss_J / L if a.lambda_J > 0 else torch.zeros((), device=dev)),
            still / _nu, loss_a / L, a_rel / L,
            (loss_d / L if a.damage else torch.zeros((), device=dev)),
            d_rel / L,
            (loss_det / L if a.det_reg > 0 else torch.zeros((), device=dev)))


if a.small_out:
    from anchorflow import conv_stepper as _CS
    _CS._ZERO_OUT = False
    print('[구조] 출력층 가중치 = 기본 초기화 x 0.01', flush=True)
if a.no_gn:
    from anchorflow import conv_stepper as _CS
    _CS._NORM = False
    print('[구조] conv 블록 GroupNorm 제거', flush=True)

# 특징 차원을 한 번 재서 모델을 세운다 -- 반드시 **학습과 같은 경로**로 잰다
with torch.no_grad():
    _d = TR[0][1]
    _g = torch.arange(min(a.n_pts, N_FULL), device=dev)
    _x = take(_d["x"][1], _g)
    _v = (_x - take(_d["x"][0], _g)) / FRAME_DT
    if a.arch == "attn":
        # 어텐션 경로는 셀이 아니라 **앵커** 기준이라 폭이 다르다. 학습과 같은
        # 조립으로 한 번 재서 맞춘다 (예전에 여기만 옛 경로로 남겨 두어 폭이
        # 우연히 같아 들키지 않은 적이 있다).
        _p0 = take(_d["x"][1], AIDX)
        _i0, _ = anchor_knn(_x, _p0, a.k)
        _f0, _ = aggregate(_x, _v / VEL_SCALE, take(_d["x"][0], _g),
                           MASS[_g], _i0, _p0.shape[0], H, pa=_p0)
        n_feat = (_f0.shape[-1] + N_MAT + n_bc
                  + (7 * a.n_ctrl if a.control else 0))
    else:
        _fe0 = (take(traj_F(_d)[1], _g).float() if a.fe_state else None)
        n_feat = cell_feats(_d, 1, _g, _x, _v, fe=_fe0)[0].shape[-1]
if a.voxel:
    VOX_CELL = H
    # 격자는 모든 궤적을 덮도록 공간에 고정한다 (프레임마다 새로 잡으면 물체가
    # 떨어지는 것만으로 모든 복셀 키가 바뀐다)
    VOX_LO = (torch.stack([dd["x"].reshape(-1, 3).min(0).values
                           for _t, dd in TR + held]).min(0).values
              - 4 * H).to(dev)
    _hi = torch.stack([dd["x"].reshape(-1, 3).max(0).values
                       for _t, dd in TR + held]).max(0).values.to(dev)
    _mx = (((_hi - VOX_LO) / (H * (max(a.vox_ens, 1) ** (1.0 / 3.0))))
           .floor().long() + 4).tolist()
    VOX_DIMS = (_mx[1] + 3, _mx[2] + 3,
                (_mx[0] + 3) * (_mx[1] + 3) * (_mx[2] + 3))
    with torch.no_grad():
        _g = torch.arange(min(a.n_pts, N_FULL), device=dev)
        _x = take(TR[0][1]["x"][0], _g)
        _p, _f, _i = vox_feats(TR[0][1], _g, _x, torch.zeros_like(_x))
        n_feat = (_f.shape[-1] + N_MAT + n_bc + (7 * a.n_ctrl if a.control else 0)
              + (13 * a.n_arms if a.grip else 0))
    print(f"[복셀] 한 변 {VOX_CELL:.5f}, 앵커 {_p.shape[0]} 개, 입력 {n_feat}",
          flush=True)

build(n_feat)

# 입력 표준화 통계는 실제로 뽑는 것과 같은 분포에서 모은다. 채널 스케일이 네 자릿수
# 넘게 벌어져 있어서(질량은 log 라 -16, 다음 변위를 결정하는 속도는 7e-4) 그냥 넣으면
# 정작 중요한 채널이 첫 Linear 에서 묻힌다.
with torch.no_grad():
    # 학습이 실제로 넣는 것과 **같은 경로**로 통계를 잡는다. 예전에는 여기만
    # 어텐션 경로(grid_knn+aggregate+ctrl_feat)로 남아 있어, 폭이 우연히 같은
    # 탓에 에러 없이 전혀 다른 양의 평균/표준편차가 들어갔다 (학습이 안 된 원인).
    samp = []
    gstat = torch.Generator(device=dev).manual_seed(a.seed + 7)
    for _ in range(32):
        _tag, _dd = TR[int(torch.randint(len(TR), (1,), generator=gstat, device=dev))]
        _t = int(torch.randint(1, _dd["x"].shape[0] - 2, (1,), generator=gstat,
                               device=dev))
        _gs = torch.randperm(N_FULL, generator=gstat,
                             device=dev)[:min(a.n_pts, N_FULL)].sort().values
        _x = take(_dd["x"][_t], _gs)
        _v = (_x - take(_dd["x"][_t - 1], _gs)) / FRAME_DT
        if a.arch == "attn":
            _pp = take(_dd["x"][_t], AIDX)
            _ii, _ = anchor_knn(_x, _pp, a.k)
            _ff, _ = aggregate(_x, _v / VEL_SCALE, take(_dd["x"][0], _gs),
                               MASS[_gs], _ii, _pp.shape[0], H, pa=_pp)
            _ex = torch.cat([
                mat_feat(_dd["cfg"]).reshape(1, N_MAT).expand(_pp.shape[0],
                                                              N_MAT),
                bc_features(_pp, _dd["cfg"]) / H], -1)
            if a.control:
                _ex = torch.cat([_ex, ctrl_feat(_dd, _t, _pp)], -1)
            _s = torch.cat([_san(_ff), _san(_ex)], -1)
            _cr = None
        else:
            _fes = (take(traj_F(_dd)[_t], _gs).float() if a.fe_state else None)
            _s, _, _, _, _, _cr = cell_feats(_dd, _t, _gs, _x, _v, fe=_fes)
        if a.stat_occ and _cr is not None:  # 빈 칸이 96% 라 통계를 장악한다
            _m = torch.zeros(_s.shape[0], dtype=torch.bool, device=_s.device)
            _m[_cr.reshape(-1)] = True
            _s = _s[_m]
        samp.append(_s)
    samp = torch.cat(samp, 0)
    net.set_input_stats(samp)
    _sn = (samp - net.in_mu) / net.in_sd
    print(f"[표준화 확인] 표본 {tuple(samp.shape)}  정규화 후 평균 "
          f"{float(_sn.mean()):+.4f} 표준편차 {float(_sn.std()):.4f} "
          f"최대절대값 {float(_sn.abs().max()):.1f}  "
          f"sd=1 로 남은 채널 {int((net.in_sd - 1).abs().lt(1e-9).sum())}"
          f"/{samp.shape[1]}", flush=True)
    print(f"[표준화] 표본 {samp.shape[0]} x {samp.shape[1]}, 채널 표준편차 "
          f"최소 {float(net.in_sd.min()):.2e} 최대 {float(net.in_sd.max()):.2e}, "
          f"평균 절대값 최대 {float(net.in_mu.abs().max()):.2e}", flush=True)

if a.resume and os.path.exists(a.resume):
    ck = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    step0 = int(ck["step"])
    _rng_ck = ck
    print(f"[재개] {a.resume} step {step0}", flush=True)

os.makedirs(a.out, exist_ok=True)
gen = torch.Generator(device=dev).manual_seed(a.seed)
_rng_ck = globals().get("_rng_ck")
if _rng_ck is not None:
    # 상태 복원은 **실패해도 학습을 막지 않는다**. 장치나 토치 판이 다르면
    # 형식이 맞지 않는데, 그것 때문에 이어달리기 자체가 죽으면 곤란하다.
    _ok = []
    for _nm, _fn, _key in (
            ("표본", lambda v: gen.set_state(v), "rng_gen"),
            ("cpu", torch.set_rng_state, "rng_cpu"),
            ("cuda", (torch.cuda.set_rng_state_all
                      if torch.cuda.is_available() else None), "rng_cuda"),
            ("numpy", np.random.set_state, "rng_np"),
            ("python", random.setstate, "rng_py")):
        _v = _rng_ck.get(_key)
        if _v is None or _fn is None:
            continue
        try:
            _fn(_v)
            _ok.append(_nm)
        except Exception as _e:
            print(f"[재개] {_nm} 난수 상태 복원 실패 ({_e}) -- 그대로 간다",
                  flush=True)
    if _ok:
        print(f"[재개] 난수 상태 복원: {', '.join(_ok)}", flush=True)
hist = []
t_start = time.time()
_VAL = (held if held else TR)[:a.val_n]
_best = float(globals().get("_rng_ck", {}).get("best", float("inf"))
              if globals().get("_rng_ck") else float("inf"))


def quick_val():
    """홀드아웃 지표 두 가지를 함께 낸다.

    (1) 짧은 롤아웃의 정지기준 대비 비 -- best 체크포인트의 기준
    (2) **학습과 똑같은 목적함수**를 홀드아웃에서 잰 값 -- 과적합을 읽으려면
        train/test 가 같은 자여야 한다
    """
    net.eval()
    tot = ref = 0.0
    obj = 0.0
    with torch.enable_grad():
        for _tag, d in _VAL:
            gs = torch.arange(0, N_FULL, max(1, N_FULL // a.n_pts),
                              device=dev)[:a.n_pts]
            for _t0 in (3, 10, 20):
                if _t0 + a.unroll + 1 >= d["x"].shape[0]:
                    continue
                obj += float(window(d, _t0, a.unroll, gs)[0])
    obj /= max(len(_VAL) * 3, 1)
    for _tag, d in _VAL:
        gsel = torch.arange(0, N_FULL, max(1, N_FULL // a.n_pts),
                            device=dev)[:a.n_pts]
        t0 = 3
        x = take(d["x"][t0], gsel)
        v = (x - take(d["x"][t0 - 1], gsel)) / FRAME_DT
        p = take(d["x"][t0], AIDX)
        x_still = x.clone()
        for i in range(a.val_len):
            with torch.enable_grad():
                x2, p, v, _, _, _, _, _, _, _ = step_once(
                    d, t0 + i, gsel, p, x, v, need_J=False)
            x2, p, v = x2.detach(), p.detach(), v.detach()
            gt = take(d["x"][t0 + i + 1], gsel)
            fm = (free_mask(d, x2.shape[0], dev, gsel) if a.control
                  else slice(None))
            tot += float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
            ref += float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
            x = x2
    net.train()
    return tot / max(ref, 1e-20), obj


def save_ck(name, step):
    # 난수 상태를 함께 남긴다. 이게 없으면 재개한 뒤 창 표본이 다른 흐름을 타서
    # 곡선이 이어지지 않는다 (TensorBoard 에서 바로 보인다).
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                "step": step, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                "n_feat": n_feat, "args": vars(a),
                "best": _best,
                "rng_gen": gen.get_state(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "rng_np": np.random.get_state(),
                "rng_py": random.getstate()},
               os.path.join(a.out, f"{a.tag}_{name}.pt"))
    if a.r2:
        os.system(f"rclone copy {a.out} {a.r2} --include '*.pt' "
                  f"--include '*.json' >/dev/null 2>&1 &")


if a.phys_probe:
    # 정상성 검사: **교사 프레임 자체**의 증분 포텐셜과 잔차. 이것이 도달 가능한
    # 바닥이고, 여기서 값이 터지면 구성모델이나 질량을 잘못 꽂은 것이다.
    print("[Phase2 검사] 교사 프레임의 에너지·잔차", flush=True)
    for _i in range(a.phys_probe):
        tag, d = TR[int(torch.randint(len(TR), (1,), generator=gen, device=dev))]
        T = d["x"].shape[0]
        t0 = int(torch.randint(1, T - 2, (1,), generator=gen, device=dev))
        gsel = torch.randperm(N_FULL, generator=gen,
                              device=dev)[:a.n_pts].sort().values
        mass = traj_mass(d)[gsel] * (float(N_FULL) / gsel.numel())
        ext = d.get("_ext", EXT)
        cfg = d["cfg"]
        vol = mass / float(cfg["density"])
        gvec = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
        h = FRAME_DT
        norm = float(mass.sum()) * (ext ** 2) / (h * h)
        x0_ = take(d["x"][t0], gsel)
        v0_ = (x0_ - take(d["x"][t0 - 1], gsel)) / h
        x1_ = take(d["x"][t0 + 1], gsel).clone().requires_grad_(True)
        F0_ = take(traj_F(d)[t0], gsel).float()
        F1_ = take(traj_F(d)[t0 + 1], gsel).float()
        fm = free_mask(d, x1_.shape[0], dev, gsel) if a.control else None
        E, _dl, parts = phys_resid.ip_energy(
            x1_, x0_ + h * v0_, F1_, mass, vol, cfg, h, free=fm, g=gvec,
            norm=norm)
        rr = phys_resid.residual(E * norm, x1_, mass, ext) * h * h
        _fr = float(rr[fm].mean()) if fm is not None else float(rr.mean())
        print(f"  {tag} t0={int(t0):3d} [{phys_resid.mat_name(cfg)}] "
              f"E={float(E):.4e} (관성 {parts[0]:.3e} 탄성 {parts[1]:.3e} "
              f"중력 {parts[2]:.3e})  잔차 {100*_fr:.4f}%", flush=True)
    raise SystemExit(0)

TBW = None
if a.tb:
    from torch.utils.tensorboard import SummaryWriter
    # 재개하면 같은 폴더에 새 이벤트 파일이 생기는데, 이전 파일에 그 지점 이후
    # 기록이 남아 있으면 TensorBoard 가 두 곡선을 겹쳐 그려 스텝이 꼬인다.
    # purge_step 으로 재개 지점 이후를 버리게 한다.
    TBW = SummaryWriter(os.path.join(a.tb, a.tag),
                        purge_step=step0 if step0 else None)
    print(f"[TB] {os.path.join(a.tb, a.tag)}", flush=True)

pbar = tqdm(range(step0, a.iters), desc="학습", ncols=90)
for it in pbar:
    L = a.unroll            # 학습 중 펼치기 길이는 바꾸지 않는다
    opt.zero_grad(set_to_none=True)
    lx = lJ = la = ldm = ldet = 0.0
    still = arel = dmean = 0.0
    for _ in range(a.batch):
        tag, d = TR[int(torch.randint(len(TR), (1,), generator=gen, device=dev))]
        if os.environ.get("AF_FIXWIN"):           # 한 창만 반복 -- 과적합 진단용
            tag, d = TR[0]
        T = d["x"].shape[0] - a.hold_last
        hi = max(T - L - 1, 2)
        if a.motion_frac > 0 and float(torch.rand(1, generator=gen,
                                                  device=dev)) < a.motion_frac:
            w_ = d["motion"][1:hi].clamp(min=1e-12)
            t0 = 1 + int(torch.multinomial(w_.to(dev), 1, generator=gen))
        else:
            t0 = int(torch.randint(1, hi, (1,), generator=gen, device=dev))
        gsel = torch.randperm(N_FULL, generator=gen,
                              device=dev)[:a.n_pts].sort().values
        if os.environ.get("AF_FIXWIN"):
            t0 = 5
            gsel = torch.arange(0, N_FULL, max(1, N_FULL // a.n_pts),
                                device=dev)[:a.n_pts]
        if a.phase2:
            # K 램프: 1 -> phys_K. 드리프트는 뒤 스텝에서 생기므로 결국 늘린다.
            Kp = a.phys_K
            if a.phys_K_warm > 0:
                Kp = 1 + int((a.phys_K - 1) * min(1.0, it / a.phys_K_warm))
            # 교란 세기는 배치마다 로그균등 -- 여러 세기를 동시에 본다
            sg = 0.0
            if a.phys_noise > 0:
                _u = float(torch.rand(1, generator=gen, device=dev))
                sg = a.phys_noise * (10.0 ** (-2.0 * (1.0 - _u)))
            wE, r_free, r_ring, _pt = phys_window(d, t0, Kp, gsel, sg, gen)
            loss_b = a.phys_w * wE
            if a.phys_sup > 0:
                wx, wJ, wst, wa, wrel, wd, wdm, wdet = window(d, t0, L, gsel)
                loss_b = loss_b + a.phys_sup * wx + a.det_reg * wdet
            else:
                wx = wE.detach(); wJ = torch.zeros((), device=dev)
                wst = r_ring; wa = torch.zeros((), device=dev)
                wrel = r_free; wd = torch.zeros((), device=dev); wdm = 0.0
            (loss_b / a.batch).backward()
            lx = lx + float(wE) / a.batch
            still = still + r_ring / a.batch
            arel = arel + r_free / a.batch
            continue
        wx, wJ, wst, wa, wrel, wd, wdm, wdet = window(d, t0, L, gsel)
        # 창마다 바로 역전파해 누적한다 -- 창 여러 개의 그래프를 동시에 들고 있으면
        # 야코비안까지 붙어 메모리가 배치 수만큼 늘어난다
        with _pt("역전파"):
            ((wx + a.lambda_J * wJ + a.lambda_anchor * wa
              + a.lambda_dmg * wd + a.det_reg * wdet) / a.batch).backward()
        lx = lx + float(wx) / a.batch
        lJ = lJ + float(wJ) / a.batch
        ldet = ldet + float(wdet) / a.batch
        still = still + wst / a.batch
        la = la + float(wa) / a.batch
        arel = arel + wrel / a.batch
        ldm = ldm + float(wd) / a.batch
        dmean = dmean + wdm / a.batch
    if _PROF_ON and it > 0 and it % 10 == 0:
        _tot = sum(_PROF.values())
        _msg = "  ".join(f"{k} {v/it*1000:.1f}ms({100*v/max(_tot,1e-9):.0f}%)"
                         for k, v in sorted(_PROF.items(), key=lambda z: -z[1]))
        print(f"[구간 {it}] 합 {_tot/it*1000:.0f}ms/it  {_msg}", flush=True)
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if os.environ.get("AF_DIAG") and it < int(os.environ["AF_DIAG"]):
        _prev = {n: q.detach().clone() for n, q in net.named_parameters()}
        _g = [(n, float(q.grad.norm())) for n, q in net.named_parameters()
              if q.grad is not None]
        _ng = sum(1 for n, q in net.named_parameters() if q.grad is None)
        print(f"  [진단 {it}] 기울기 있는 파라미터 {len(_g)}, 없는 것 {_ng}, "
              f"|g| 합 {sum(v for _, v in _g):.3e}", flush=True)
        for n, v in sorted(_g, key=lambda z: -z[1])[:6]:
            print(f"    {n:34} |g| {v:.3e}", flush=True)
    opt.step()
    if os.environ.get("AF_DIAG") and it < int(os.environ["AF_DIAG"]):
        _d = [(n, float((q.detach() - _prev[n]).norm())) for n, q in
              net.named_parameters()]
        print(f"    변화 합 {sum(v for _, v in _d):.3e}  "
              f"상위 {[(n.split('.')[-2:], round(v, 8)) for n, v in sorted(_d, key=lambda z: -z[1])[:3]]}",
              flush=True)
    hist.append((lx, lJ, still, la, arel, ldm, dmean))
    if TBW is not None and it % 20 == 0:
        if a.phase2:
            TBW.add_scalar("phase2/E", lx, it)
            TBW.add_scalar("phase2/자유잔차", arel, it)
            TBW.add_scalar("phase2/구속잔차", still, it)
        else:
            TBW.add_scalar("목적함수/학습", lx, it)
            TBW.add_scalar("학습/위치오차%", 100 * lx ** 0.5, it)
            TBW.add_scalar("학습/정지기준%", 100 * still ** 0.5, it)
            TBW.add_scalar("학습/비", (lx / max(still, 1e-20)) ** 0.5, it)
        TBW.add_scalar("학습/기울기노름", float(gn), it)
        if a.det_reg > 0:
            TBW.add_scalar("학습/det벌점", ldet, it)
    if it % 20 == 0:
        if a.phase2:
            pbar.set_postfix(E=f"{lx:.3e}", 자유잔차=f"{100*arel:.3f}%",
                             구속잔차=f"{100*still:.3f}%", K=Kp,
                             gn=f"{float(gn):.1e}")
        else:
            pbar.set_postfix(x=f"{100*lx**0.5:.3f}%",
                             정지=f"{100*still**0.5:.3f}%",
                             비=f"{(lx/max(still,1e-20))**0.5:.2f}",
                             앵커비=f"{arel:.2f}", J=f"{lJ:.1e}", L=L,
                             **({"det": f"{ldet:.1e}"} if a.det_reg > 0 else {}),
                             **({"손상": f"{dmean:.3f}", "d손실": f"{ldm:.1e}"}
                                if a.damage else {}),
                             gn=f"{float(gn):.1e}")
    if a.val_every and ((it + 1) % a.val_every == 0 or it == a.iters - 1):
        _v, _vo = quick_val()
        if TBW is not None:
            TBW.add_scalar("검증/비", _v, it)
            TBW.add_scalar("목적함수/검증", _vo, it)
        if _v < _best:
            _best = _v
            save_ck("best", it + 1)
            print(f"  [검증 {it+1}] 비 {_v:.4f} 목적 {_vo:.3e} -- best 갱신"
                  + (f"  (관성 {_pt[0]:.3e} 탄성 {_pt[1]:.3e} 중력 {_pt[2]:.3e})"
                     if a.phase2 and _pt else ""), flush=True)
        else:
            print(f"  [검증 {it+1}] 비 {_v:.4f} 목적 {_vo:.3e} "
                  f"(best {_best:.4f})", flush=True)
    if (it + 1) % a.save_every == 0 or it == a.iters - 1:
        save_ck("last", it + 1)

# ---------------------------------------------------------------- 평가
_ROLLDUMP = None


@torch.no_grad()
def rollout(d, t0, L, gsel):
    # 드롭아웃이 켜져 있으면 롤아웃이 확률적이 된다 -- 평가는 항상 eval 로.
    net.eval()
    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / FRAME_DT
    p = x[fps(x, a.n_anchors, a.seed)] if a.refps else take(d["x"][t0], AIDX)
    errs, stills = [], []
    x_still = x.clone()
    x0e, p0e = x.clone(), p.clone()
    dmg_e, idx_e = None, None
    fe_r = (take(traj_F(d)[t0], gsel).float() if a.fe_state else None)
    cds, ems, cds_s, ems_s = [], [], [], []
    for i in range(L):
        with torch.enable_grad():
            x2, p, v, _, _, _, dmg_e, idx_e, fe_r, _ = step_once(
                d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
        x2 = x2.detach(); p = p.detach(); v = v.detach()
        gt = take(d["x"][t0 + i + 1], gsel)
        fm = free_mask(d, x2.shape[0], x2.device, gsel) if a.control else slice(None)
        errs.append(float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / EXT)
        stills.append(float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / EXT)
        if _ROLLDUMP is not None:
            _ROLLDUMP.append((x2.detach().cpu(), gt.detach().cpu()))
        if a.metrics:
            cds.append(chamfer(x2, gt) / (EXT ** 2))
            ems.append(emd(x2, gt, t0 * 1000 + i) / EXT)
            cds_s.append(chamfer(x_still, gt) / (EXT ** 2))
            ems_s.append(emd(x_still, gt, t0 * 1000 + i) / EXT)
        x = x2
    net.train()
    return errs, stills, cds, ems, cds_s, ems_s


def chamfer(p_, q_, chunk=4096):
    """양방향 평균 최근접 **제곱** 거리의 합 (Spring-Gaus 관례)."""
    def one(u, w_):
        s_, n_ = 0.0, u.shape[0]
        for i_ in range(0, n_, chunk):
            dd = torch.cdist(u[i_:i_ + chunk], w_)
            s_ += float((dd.min(1).values ** 2).sum())
        return s_ / n_
    return one(p_, q_) + one(q_, p_)


def emd(p_, q_, seed):
    """최적 일대일 대응의 평균 이동량. 두 구름에서 **같은 인덱스**를 뽑는다."""
    from scipy.optimize import linear_sum_assignment
    g_ = torch.Generator().manual_seed(int(seed))
    i_ = torch.randperm(p_.shape[0], generator=g_)[:a.cd_pts].to(p_.device)
    dd = torch.cdist(p_[i_], q_[i_]).double().cpu().numpy()
    r_, c_ = linear_sum_assignment(dd)
    return float(dd[r_, c_].mean())


gsel = torch.arange(min(a.n_pts, N_FULL), device=dev)
rows = {}
for tag, d in TR + held:
    T = d["x"].shape[0]
    rows[tag] = dict(held=(tag in hold), windows={})
    for t0 in a.eval_t0:
        L = min(a.eval_len, T - t0 - 1)
        if L < 2:
            continue
        _rd = os.environ.get("AF_ROLL_DUMP")
        _want = (_rd and tag.endswith(os.environ.get("AF_ROLL_TAG", "")) 
                 and t0 == int(os.environ.get("AF_ROLL_T0", "3")))
        _ROLLDUMP = [] if _want else None
        e, st, cd, em, cds_, ems_ = rollout(d, t0, L, gsel)
        if _want and _ROLLDUMP:
            torch.save({"pred": torch.stack([a_ for a_, _ in _ROLLDUMP]),
                        "gt": torch.stack([b_ for _, b_ in _ROLLDUMP]),
                        "x0": take(d["x"][t0], gsel).cpu(),
                        "ctrl_pos": d.get("ctrl_pos"), "t0": t0, "tag": tag,
                        "EXT": EXT}, _rd)
            print(f"[롤아웃 덤프] {_rd}  {tag} t0={t0}", flush=True)
        _ROLLDUMP = None
        rows[tag]["windows"][t0] = dict(L=L, err=e, still=st,
                                        err_mean=float(np.mean(e)),
                                        still_mean=float(np.mean(st)),
                                        **(dict(cd=float(np.mean(cd)),
                                                emd=float(np.mean(em)),
                                                cd_still=float(np.mean(cds_)),
                                                emd_still=float(np.mean(ems_)))
                                           if a.metrics else {}))
        print(f"[롤아웃] {tag}{' (홀드아웃)' if tag in hold else ''} t0={t0:3d}: "
              f"{L} 프레임, 평균 {100*np.mean(e):.3f}% "
              f"(정지 {100*np.mean(st):.3f}%, 비 "
              f"{np.mean(e)/max(np.mean(st),1e-12):.2f})"
              + (f"  CD {np.mean(cd):.3e} (정지 {np.mean(cds_):.3e})"
                 f"  EMD {100*np.mean(em):.3f}% (정지 {100*np.mean(ems_):.3f}%)"
                 if a.metrics else ""), flush=True)
r_all = [(w["err_mean"], w["still_mean"]) for r in rows.values()
         for w in r["windows"].values()]
print(f"\n[요약] 전체 창 평균 {100*np.mean([x for x,_ in r_all]):.3f}% "
      f"(정지 {100*np.mean([y for _,y in r_all]):.3f}%, 비 "
      f"{np.mean([x/max(y,1e-12) for x,y in r_all]):.2f}) "
      f"-- 비가 1 보다 작아야 도움이 된 것이다", flush=True)
if a.metrics:
    _c = [w["cd"] for r in rows.values() for w in r["windows"].values()]
    _cs = [w["cd_still"] for r in rows.values() for w in r["windows"].values()]
    _e = [w["emd"] for r in rows.values() for w in r["windows"].values()]
    _es = [w["emd_still"] for r in rows.values() for w in r["windows"].values()]
    print(f"[지표] CD {np.mean(_c):.4e} (정지 {np.mean(_cs):.4e}, 비 "
          f"{np.mean(_c)/max(np.mean(_cs),1e-30):.3f})   "
          f"EMD {100*np.mean(_e):.4f}% (정지 {100*np.mean(_es):.4f}%, 비 "
          f"{np.mean(_e)/max(np.mean(_es),1e-30):.3f})", flush=True)

json.dump(dict(tag=a.tag, args=vars(a), extent=EXT, h=H, n_feat=n_feat,
               minutes=(time.time() - t_start) / 60,
               loss_hist=hist[::20], rollout=rows),
          open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1,
          ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
print("DEFORM_OK")
