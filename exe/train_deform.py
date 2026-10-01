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

import contextlib
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
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--batch", type=int, default=1,
                help="한 스텝에 평균 낼 창의 수. 창마다 손실이 수십 배 다르므로 "
                     "하나만 쓰면 기울기가 그 차이에 휘둘린다")
ap.add_argument("--unroll", type=int, default=1, help="한 창에서 펼칠 프레임 수")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--sigma0", type=float, default=0.5,
                help="정준 가우시안의 등방 표준편차를 입자 간격의 몇 배로 볼지. "
                     "이 씬의 ply 는 sigma ~ 1e-9 로 사실상 점이라, 모양 항을 쓰려면 "
                     "부피에서 온 크기를 줘야 한다")
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
ap.add_argument("--arch", default="sgnn",
                choices=("sgnn", "attn", "conv", "unet", "conv_sep",
                         "conv_par", "unet_sep", "unet_par"),
                help="sgnn(기본): **사면체 복합체** 위 메시지 패싱. 점유 사면체의 "
                     "꼭짓점만 노드이고 간선은 그 사면체의 변(방향까지 구분한 "
                     "26 종). conv/unet 은 격자 위 컨볼루션 (옛 경로)")
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
ap.add_argument("--gpu_data", type=int, default=1,
                help="궤적을 GPU 에 상주시킨다. CPU 색인 + 전송이 한 스텝의 "
                     "5 분의 1 이라 그냥 올리는 쪽이 빠르다")
ap.add_argument("--gpu_data_mb", type=float, default=8000.0,
                help="이 용량을 넘으면 GPU 상주를 건너뛴다")
ap.add_argument("--hold_traj", default=None,
                help="통째로 홀드아웃할 궤적 태그 (쉼표로 구분)")
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--resume", default=None)
ap.add_argument("--small_out", action="store_true",
                help="출력층을 0 이 아니라 기본 초기화의 1/100 로 시작")
ap.add_argument("--metrics", action="store_true",
                help="롤아웃에서 CD/EMD 까지 잰다 (Spring-Gaus 정의)")
ap.add_argument("--cd_pts", type=int, default=2048,
                help="CD/EMD 표본 크기 (EMD 가 O(n^3) 이라 필요하다)")
# 변형은 로컬·글로벌 두 단계가 **순차·독립**이라 따로 고른다. 격자-입자
# 전달은 사면체 복합체의 barycentric 하나다.
ap.add_argument("--jac", default="analytic", choices=("analytic", "auto"),
                help="변형장 야코비안 계산법. analytic 은 닫힌 형식 (RQS "
                     "기울기 대각 x 커널 ∇u 연쇄) -- 역전파 3회짜리 auto "
                     "(jacobian_of) 보다 싸고 no_grad 롤아웃에서도 돈다")
ap.add_argument("--n_nodes", type=int, default=32,
                help="--arch sgnn 의 **노드 간격**을 정한다 (물체를 몇 노드로 "
                     "덮을지). 사면체 변 길이 = 물체/n_nodes 이고, 발판 격자는 "
                     "그 2 배 간격으로 내부에서 잡힌다")
ap.add_argument("--gnn_no_norm", action="store_true",
                help="메시지 패싱 갱신의 LayerNorm 제거 (격자 conv 에서 "
                     "GroupNorm 이 크기 정보를 지운 전례가 있다)")
ap.add_argument("--gnn_sum", action="store_true",
                help="메시지 집계를 평균 대신 합으로 (경계 노드 희석 방지)")
ap.add_argument("--gnn_layers", type=int, default=8,
                help="--arch sgnn 의 메시지 패싱 층 수. 수용영역이 층 수로만 "
                     "늘어난다 (체대각 간선이 있어 한 홉이 소큐브를 가로지른다)")
ap.add_argument("--det_eps", type=float, default=0.1,
                help="사면체 det(grad Phi) 의 하한. 이보다 작은 사면체는 제대로 "
                     "된 셀이 아니라고 보고 **탄성항에서 빼고**, det 를 이 위로 "
                     "되돌리는 복구 손실을 준다")
ap.add_argument("--lr_cos", action="store_true",
                help="학습률을 코사인으로 0 까지 감쇠한다. PT/KB/SH/SL/SW 전 "
                     "실행이 초반(250~1750 스텝) 최고 뒤 악화했고, 후반 큰 "
                     "보폭이 그 원인 후보다")
ap.add_argument("--det_every", type=int, default=50,
                help="det 통계(무효 비율·최소·중앙) 를 몇 스텝마다 로그에 "
                     "남길지. 0 이면 끔")
ap.add_argument("--det_w", type=float, default=100.0,
                help="det 복구 손실의 가중치. 물리를 맞추는 것보다 셀이 유효한 "
                     "것이 우선이라 크게 둔다")
ap.add_argument("--dt_cond", action="store_true",
                help="dt 를 log10 푸리에로 인코딩해 층마다 FiLM 으로 넣는다 "
                     "(어텐션 경로의 DtFiLM 과 같은 것). 0 초기화라 켜는 "
                     "순간에는 항등이고, 옛 체크포인트에 얹어도 값이 안 변한다")
ap.add_argument("--dt_scale", action="store_true",
                help="망의 변위 출력 크기를 dt 에 비례시킨다 (변위~v*dt 의 1 차 "
                     "관계를 구조로 박는다). --dt_cond 와 함께 쓰는 것이 기본")
ap.add_argument("--dt_sub", type=int, default=1,
                help="한 스텝의 dt = frame_dt / dt_sub. 1 이면 지금까지와 같다")
ap.add_argument("--dt_sub_set", default="",
                help="쉼표 목록 (예 1,2,4,8). 물리·풀 학습에서 창마다 여기서 "
                     "하나 뽑아 dt 를 바꾼다 -- dt 를 조건 변수로 학습시키는 "
                     "핵심 스위치다. 비우면 --dt_sub 고정")
ap.add_argument("--v_from_dt", action="store_true",
                help="입자 속도를 (x2-x)/h 차분이 아니라 **변형장을 t 로 미분**해 "
                     "얻는다: v = dPhi_t(x)/dt |_{t=h}. t 는 스칼라라 순방향 "
                     "AD 한 번으로 [N,3] 이 통째로 나온다. --dt_cond 필요")
ap.add_argument("--obj", default="pts", choices=("pts", "grid"),
                help="증분 포텐셜을 어디서 재는가. pts(기본) 는 **입자에서 바로** "
                     "잰다 -- MPM 격자를 전혀 거치지 않고, 탄성항의 F 를 변형장 "
                     "야코비안으로 민다. grid 는 예전 경로(P2G 로 노드 증분을 "
                     "모으고 G2P 미분으로 ∇Δu 를 되받는다) 로 비교용으로만 남긴다")
ap.add_argument("--f_from_jac", action="store_true",
                help="변형구배를 격자 B-스플라인 공간미분(g2p_grad)이 아니라 "
                     "**변형장 자신의 야코비안**으로 민다: F <- grad_x Phi * F. "
                     "g2p_grad 는 셀 내부 재배열·스키닝 가중치를 무시해 실제로 "
                     "입자를 옮긴 사상과 다른 F 를 만든다")
ap.add_argument("--eval_dt_sub", type=int, default=0,
                help="평가·롤아웃에서 한 프레임을 몇 서브스텝으로 나눌지 "
                     "(0 이면 --dt_sub). 프레임 경계는 그대로라 교사와 계속 "
                     "같은 자리에서 비교된다")
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
ap.add_argument("--resume_fresh", action="store_true",
                help="가중치만 이어받고 스텝·옵티마이저·난수는 새로 시작한다. "
                     "다른 단계로 넘어갈 때 쓴다 (예: 증류 -> RL)")
ap.add_argument("--no_pool", dest="pool", action="store_false",
                help="**상태 풀이 기본이고 유일한 학습 경로다.** 정지 상태에서 "
                     "출발해 손잡이 계획을 직접 뽑고, 모델이 상태를 한 스텝씩 "
                     "갱신하며 그 상태를 풀에 담는다. 교사 궤적은 학습에 전혀 "
                     "쓰지 않는다 (평가·검증에서만 기준으로 쓴다). 이 플래그를 "
                     "주면 학습 경로가 없어 즉시 종료한다")
ap.add_argument("--pool_size", type=int, default=128,
                help="풀 크기. 크면 한 상태가 다시 뽑히기까지 오래 걸려 낡은 "
                     "정책이 만든 상태만 쌓인다 (1024 면 약 85 반복에 한 번). "
                     "128 이면 약 10 반복마다 전진한다")
ap.add_argument("--pool_grid", default="domain", choices=("domain", "fit"),
                help="domain 이면 시뮬 영역 전체를 vox_res^3 으로 고정해 모든 "
                     "상태가 같은 격자를 쓴다 (영역을 벗어난 상태는 버린다). "
                     "fit 이면 예전처럼 매 스텝 물체에 맞춰 새로 잡는다")
ap.add_argument("--pool_fresh", type=float, default=0.25,
                help="배치에서 새 초기 상태로 채우는 비율")
ap.add_argument("--pool_thresh", type=float, default=0.10,
                help="폐기 문턱 (고정). 정류 잔차를 길이로 환산해 물체 크기로 "
                     "나눈 무차원 값이라 물성·형상이 달라도 같은 자다. 실측으로 "
                     "학습 초기 누적잔차 중앙이 0.024~0.033 인데 문턱을 그 "
                     "근처(0.01~0.02)로 두면 거의 전부 즉시 폐기돼 학생이 같은 "
                     "상태를 이어 볼 기회가 없다 -- 0.05 이상에서 고른다")
ap.add_argument("--pool_frames", type=int, default=240,
                help="계획 한 회의 **상한** 프레임. 목표에 닿으면 그보다 일찍 "
                     "새 계획으로 넘어간다 (도달 시간은 거리에 따라 다르다)")
ap.add_argument("--pool_fill", default="",
                help="풀이 읽을 채움 디렉토리(pgfill_<형상>.npy). 비우면 AF_WORK. "
                     "교사와 같은 입자 집합을 쓰려면 그쪽 캐시를 가리킬 것")
ap.add_argument("--pool_combos", default="all",
                help="쉼표로 구분한 형상_물성. all 이면 12 조합 전부 (기본)")
ap.add_argument("--ctrl_zbias", type=float, default=0.0,
                help="손잡이 목표를 물체 크기의 이 배수만큼 **위로** 올린다. "
                     "교사 시나리오(s400706)는 들어올리는 구동이라 명령 z성분 "
                     "평균이 +0.379 인데, 목표 격자가 z 로 대칭이면 그 구동을 "
                     "재현할 수 없다 -- 그러면 학생이 평가에서 만나는 입력을 "
                     "학습 중에 본 적이 없게 된다")
ap.add_argument("--ctrl_acc", type=float, default=2.4,
                help="손잡이 가속도. 목표까지 걸리는 시간을 고정하는 대신 "
                     "가속도와 최고속도를 고정한다 -- 시간을 고정하면 먼 목표는 "
                     "빠르게 가까운 목표는 느리게 끌려 속도 영역이 뒤섞인다")
ap.add_argument("--ctrl_vmax", type=float, default=0.6,
                help="손잡이 최고속도")
ap.add_argument("--bc_soft", action="store_true",
                help="바닥·경계를 옛 벌점(bc_energy) 으로 되돌린다. 기본은 하드 "
                     "사영이다 -- 벌점은 관성항과 겨루어 새고(실측 관통 0.97%) "
                     "PG/i-PG 가 격자 속도를 박는 것과 조건이 다르다")
ap.add_argument("--ctrl_soft", action="store_true",
                help="손잡이를 옛 감쇠 가중 (1-q^2)^2 로 되돌린다. 기본은 하드 "
                     "Dirichlet(반경 안 1, 밖 0) 이고 i-PG 교사와 같은 조건이다")
ap.add_argument("--lambda_bc", type=float, default=1.0,
                help="구속 일치 항 가중치. 구속 입자(손잡이 + 바닥·경계 활성)에서 "
                     "**망 출력 변위**가 하드 사영값과 같아지게 한다. 이게 없으면 "
                     "망은 그 자리에서 무엇을 내든 덮어쓰기가 대신 처리해 주므로 "
                     "변형장이 경계조건과 어긋난 채 남고 그 불일치가 주변으로 샌다")
ap.add_argument("--out_var", action="store_true",
                help="**학습 루프를 그대로 쓰고** 망 대신 프레임별 출력 변수를 "
                     "최적화한다. 임의 프레임 샘플링·배치·손실 모두 학습과 같고, "
                     "갱신 대상만 망 파라미터에서 **그 상태의 노드 출력**으로 "
                     "바뀐다. 상태는 전진하지 않으므로 각 상태의 E 가 이 "
                     "매개화의 한 스텝 하한까지 내려간다")
ap.add_argument("--out_var_lr", type=float, default=3e-3)
ap.add_argument("--ov_init",
                choices=["zero", "randn", "net", "target", "predict"],
                default="predict",
                help="--ov_roll 자유변수의 초기값. zero=항등 변형장(기본), "
                     "randn=--ov_init_std 규모의 난수, net=망 출력, "
                     "target=모든 노드를 h·v̄ + h²g 로 (평행이동 진단), "
                     "predict=노드별 관성 예측자 h·v_n + h²g (기본). 암시적 "
                     "솔버가 뉴턴을 시작하는 그 지점이다 -- 0 에서 출발하면 "
                     "탄성 강성에 묻혀 자유낙하조차 못 찾는다")
ap.add_argument("--ov_init_std", type=float, default=1e-3)
ap.add_argument("--fbar", choices=["none", "nodal"], default="none",
                help="변형구배를 어떻게 잴지. none=셀 하나로(기본), "
                     "nodal=**노드 평균 변형구배** (Bonet 의 averaged nodal "
                     "deformation gradient): 셀별 F 를 꼭짓점에 모아 평균하고 "
                     "입자가 그 노드값을 barycentric 으로 받는다. 입자의 F 가 "
                     "1-ring 의 모든 셀에 의존하게 되어 이웃 셀끼리 직접 묶인다")
ap.add_argument("--ov_opt", choices=["adam", "lbfgs"], default="lbfgs",
                help="--ov_roll 의 최적화기. 탄성 강성(E=2e6)이 관성항보다 "
                     "원시 값으로 700 배 커서 문제가 심하게 비등방이다 -- "
                     "Adam 은 0 에서 출발하면 자유낙하조차 못 찾는다(실측: "
                     "필요 변위의 6%%). 선탐색이 있는 L-BFGS 가 기본이다")
ap.add_argument("--ov_roll", type=int, default=0,
                help="롤아웃에서 **프레임마다** 노드 출력을 직접 최적화한다 "
                     "(반복 수). 망은 초기화에만 쓰이고 상태는 정상 전진한다 "
                     "-- 이 매개화의 롤아웃 하한을 잰다. --out_var 의 한 스텝 "
                     "측정과 달리 평가 창 전체에서 성립하는 수치다")
ap.add_argument("--oracle_roll", action="store_true",
                help="망 출력 자리에 자유 변수를 넣고 매 프레임 물리손실을 "
                     "최소화하는 오라클 롤아웃. 학습·평가 코드를 그대로 쓴다")
ap.add_argument("--oracle_steps", type=int, default=300)
ap.add_argument("--oracle_lr", type=float, default=1e-3)
ap.add_argument("--oracle_lr_shape", type=float, default=1e-2,
                help="반경·두께(log) 변수의 학습률. 변위와 단위가 달라 따로 둔다")
ap.add_argument("--oracle_out", default="")
ap.add_argument("--oracle_sub", type=int, default=1,
                help="프레임을 이만큼 서브스텝으로 나눠 **각 서브스텝마다** 출력을 "
                     "최적화하고 상태를 전진시킨다. 증분 포텐셜은 dt 가 클수록 "
                     "PG 와 벌어지므로 dt 를 i-PG 수준까지 내려 확인하는 데 쓴다")
ap.add_argument("--oracle_curve", action="store_true",
                help="스텝마다 목적함수와 교사오차를 기록해 러닝 커브를 낸다")
ap.add_argument("--oracle_snap", default="",
                help="쉼표 목록. 그 프레임 수에 도달하면 그때까지의 롤아웃을 "
                     "따로 저장한다 (진행 중 비교용)")
ap.add_argument("--roll_scen", default="",
                help="벤치 시나리오 npz 로 학생을 굴려 프레임별 상태를 덤프한다. "
                     "--pool 과 --pool_combos <조합 하나> 를 함께 준다")
ap.add_argument("--roll_out", default="", help="--roll_scen 덤프 경로")
ap.add_argument("--pool_targets", type=int, default=5,
                help="목표점 후보 격자의 한 변. 후보를 **유한 고정** 집합으로 "
                     "두어 같은 목표를 여러 번 보게 한다 (5 면 최대 125 개)")
ap.add_argument("--pool_window", type=int, default=30,
                help="폐기 판정에 쓰는 잔차 평균의 창 길이. 1 로 두면 누적 "
                     "없이 **이번 스텝 잔차만** 보고 판정한다")
ap.add_argument("--pool_keep", type=float, default=0.5,
                help="누적 잔차가 문턱을 넘어도 이 확률로는 버리지 않고 "
                     "전진을 취소해 상태를 그대로 풀에 남긴다 (기본 0.5). "
                     "매번 버리면 풀이 신규로만 차서 같은 상태를 이어 볼 수 없다")
ap.add_argument("--pool_whiten", action="store_true",
                help="손실을 이동 RMS 로 나눠 스케일을 고정한다 (RL 의 보상 "
                     "표준화와 같은 취지). 최소점은 그대로이고 기울기 크기만 "
                     "일정해져, 물성마다 1e-7~1e-5 로 널뛰는 문제를 없앤다")
ap.add_argument("--pool_start_mid", action="store_true",
                help="새 상태를 손잡이 계획의 무작위 지점에서 시작한다. 정지 "
                     "상태에서만 출발하면 '아무것도 안 하기' 가 거의 최적이라 "
                     "기울기가 사라진다")
ap.add_argument("--pool_loss", default="energy",
                choices=("energy", "residual"),
                help="손실을 i-PG 목적함수 값으로 둘지(energy, 기본) 그 기울기인 "
                     "정류 잔차로 둘지. 잔차는 2 계 미분을 타서 더 비싸다")
ap.add_argument("--rl", action="store_true",
                help="액터-크리틱으로 학습한다. 보상은 -(i-PG 손실), 미래는 가치함수 "
                     "V 가 대신 보므로 롤아웃을 거슬러 미분하지 않는다 (BPTT 길이 1). "
                     "상태는 에피소드 안에서 학생 자신의 출력으로 이어진다")
ap.add_argument("--rl_reward", default="residual",
                choices=("residual", "energy"),
                help="보상. residual 은 -(정류 잔차)^2 로 **0 에 유계**하고 정답에서 "
                     "정확히 0 이다. energy 는 증분 포텐셜 자체인데 중력항 때문에 "
                     "아래로 유계가 아니라 에피소드로 누적하면 자유낙하가 최적이 "
                     "된다 (실제로 30 스텝 만에 발산했다)")
ap.add_argument("--rl_gamma", type=float, default=0.95, help="할인율")
ap.add_argument("--rl_term", type=float, default=3.0,
                help="앵커 판본에서 가우시안-앵커 거리가 처음의 이 배를 넘으면 "
                     "에피소드를 끝내고 큰 음수 보상을 준다")
ap.add_argument("--rl_term_pen", type=float, default=10.0, help="종료 벌점")
ap.add_argument("--phys_probe", type=int, default=0,
                help="교사 프레임에서 에너지·잔차만 이만큼 재고 끝낸다 (정상성 검사)")
ap.add_argument("--val_every", type=int, default=0,
                help="이 간격마다 홀드아웃 롤아웃으로 재고 best 체크포인트를 남긴다")
ap.add_argument("--val_n", type=int, default=4, help="검증에 쓸 궤적 수")
ap.add_argument("--val_len", type=int, default=10, help="검증 롤아웃 길이")
ap.add_argument("--wd", type=float, default=0.0,
                help="가중치 감쇠 (0 보다 크면 Adam 대신 AdamW 를 쓴다). 한 스텝 "
                     "교사 오차만 내려가고 홀드아웃이 안 따라오는 과적합을 친다")
ap.add_argument("--det_reg", type=float, default=0.0,
                help="한 스텝 변형장의 야코비안 행렬식이 뒤집히는 것(det<=0)을 "
                     "벌한다. relu(margin - det)^2 의 입자 평균에 이 가중치를 곱한다")
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

import torch.autograd.forward_ad as _fwAD                       # noqa: E402
from anchorflow import deform                                  # noqa: E402
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import simplex as SX                            # noqa: E402
from anchorflow import simplex_gnn as _SGM                      # noqa: E402
from anchorflow.simplex_gnn import (SimplexGNN, node_moments,    # noqa: E402
                                    scatter_to_nodes)
from anchorflow import voxel                                    # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, anchor_knn,  # noqa: E402
                               bc_features, bond_stretch, bures_w2_sq,
                               fps, gauss_stretch, grid_knn,
                               jacobian_of)

# ---------------------------------------------------------------- 데이터
files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
if not files:
    raise SystemExit(f"궤적이 없다: {a.data}")
hold = set((a.hold_traj or "").split(",")) - {""}
TR, held = [], []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    # 궤적은 half 로 저장돼 있다. 예전에는 여기서 float32 로 올렸는데, 그러면
    # 궤적당 메모리가 두 배가 되어 전 조합(960 개)이 57 GB 라 GPU 에 못 올라가고
    # 매 스텝 CPU 에서 부분표본을 gather 하느라 계산(0.4 s)보다 접근(5 s)이
    # 비싸졌다. half 로 두면 28 GB 라 GPU 에 상주하고, take 가 쓸 때 캐스팅한다
    # -- 저장이 이미 half 이므로 수치는 한 비트도 달라지지 않는다.
    # 궤적의 v 는 아무도 읽지 않는다 (속도는 위치 차이로 만든다). 전 조합이면
    # 이것만으로 7 GB 를 차지하므로 적재에서 아예 뺀다.
    d.pop("v", None)
    tag = os.path.splitext(os.path.basename(f))[0]
    (held if tag in hold else TR).append((tag, d))
if not TR and a.pool:
    # 풀 모드는 학습 궤적을 아예 읽지 않는다 (초기 상태는 씬의 정지 자세에서
    # 만들고 손잡이 계획도 직접 뽑는다). 궤적이 홀드아웃뿐이면 구조 정보
    # -- cfg, 초기 자세, 경계 상자 -- 만 그쪽에서 본다.
    TR = list(held)
    print("[풀] 학습 궤적 없음 -- 구조 정보만 홀드아웃에서 읽는다", flush=True)
if not TR:
    raise SystemExit("학습할 궤적이 없다")
_MB = sum(sum(v.numel() * v.element_size() for v in d.values()
               if torch.is_tensor(v)) for _t, d in TR + held) / 1e6
if a.gpu_data and _MB < a.gpu_data_mb:
    for _t, d in TR + held:
        for k in ("x", "F"):
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

# --- 지금 스텝의 dt -----------------------------------------------------------
# step_once 와 손잡이 명령, 속도 환산이 **모두** 이 하나를 본다. FRAME_DT 는
# 궤적의 프레임 간격(교사 상태를 읽고 속도를 만드는 기준)으로 남고, _DT[0] 은
# 학생이 실제로 밟는 스텝이다. 둘을 갈라 놓아야 dt 를 흔들어도 초기 속도나
# 교사 프레임 색인 같은 **물리량**이 같이 흔들리지 않는다.
_DT = [FRAME_DT]


@contextlib.contextmanager
def dt_scope(dt):
    _o = _DT[0]
    _DT[0] = float(dt)
    try:
        yield
    finally:
        _DT[0] = _o


# K 언롤 동안 노드 격자를 고정하는 자리. None 이면 스텝마다 새로 잡는다.
_GRID = [None]


@contextlib.contextmanager
def grid_pin(x):
    """K 서브스텝이 **같은 이산화**를 보게 노드 격자를 한 번만 잡는다.

    conv 경로는 격자가 cfg(n_grid, grid_lim) 로 고정이라 서브스텝마다 같은
    격자를 본다. sgnn 은 격자를 x 의 바운딩박스에서 잡으므로 그대로 두면 K
    스텝이 서로 다른 격자 위에서 합성된다 -- 물체가 부풀면 간격 h 까지 커지고,
    망 출력은 간격 단위라 그만큼 함께 커져 되먹임이 된다. K=1 에서는 격자를
    한 번만 잡으므로 이 고정이 아무것도 바꾸지 않는다.
    """
    _o = _GRID[0]
    _GRID[0] = SX.grid_for_nodes(x, a.n_nodes)
    try:
        yield
    finally:
        _GRID[0] = _o


if not a.bc_soft:
    # 하드 사영을 쓰면 같은 경계를 벌점으로 또 세지 않는다.
    os.environ["AF_NO_BC"] = "1"

_DT_SUBS = [float(q) for q in a.dt_sub_set.split(",") if q.strip()]
EVAL_SUB = int(a.eval_dt_sub or a.dt_sub)


def sample_sub(gen=None):
    """이번 창의 서브스텝 배수. --dt_sub_set 이 있으면 거기서 뽑는다."""
    if not _DT_SUBS:
        return float(a.dt_sub)
    i = int(torch.randint(len(_DT_SUBS), (1,), generator=gen, device=dev))
    return _DT_SUBS[i]


if _DT_SUBS or a.dt_sub != 1 or a.dt_cond or a.dt_scale:
    print(f"[dt] frame_dt {FRAME_DT:.5g}, 학습 배수 "
          f"{_DT_SUBS if _DT_SUBS else a.dt_sub}, 평가 배수 {EVAL_SUB}, "
          f"조건화 {'on' if a.dt_cond else 'off'}, "
          f"출력 비례 {'on' if a.dt_scale else 'off'}", flush=True)
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
    # 궤적은 half 로 상주한다 -- 통계는 float 로 올려서 낸다 (quantile 이 half 를
    # 받지 않고, 차이가 작아 half 로 재면 자릿수도 모자란다)
    mv = (_xx[1:, _ss].float() - _xx[:-1, _ss].float()).norm(dim=-1).mean(1)
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


if a.gnn_no_norm:
    _SGM._MP_NORM = False
    print("[구조] 메시지 패싱 LayerNorm 제거", flush=True)
if a.gnn_sum:
    _SGM._MP_MEAN = False
    print("[구조] 메시지 집계를 합으로", flush=True)


def build(n_feat):
    global net, opt
    if a.arch != "sgnn":
        raise SystemExit("사면체 복합체 경로만 남았다 -- --arch sgnn 을 쓸 것 "
                         "(격자 conv/unet 은 태그 grid-rqs-final 에 있다)")
    net = SimplexGNN(n_feat=n_feat, hidden=a.hidden, layers=a.gnn_layers,
                     scale=0.02 * EXT, dt_cond=a.dt_cond, dt_ref=FRAME_DT,
                     dt_scale=a.dt_scale, n_mat=N_FILM,
                     ).to(dev)
    opt = (torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
           if a.wd > 0 else torch.optim.Adam(net.parameters(), lr=a.lr))
    global SCHED
    SCHED = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
             if a.lr_cos and a.iters > 0 else None)


def take(t, idx_gpu):
    """궤적에서 부분표본을 떼어 GPU 로.

    궤적이 GPU 에 올라가 있으면 그냥 색인한다 (0.013 ms). CPU 에 있으면 색인을
    CPU 에서 해야 하고 (장치가 섞이면 torch 가 거부한다) 그것이 위치 1.61 ms,
    F 4.08 ms 로 한 스텝의 5 분의 1 을 먹는다 -- 궤적 하나가 98 MB 라 다 올려도
    7 개에 700 MB 다. 못 올릴 이유가 없었다."""
    if t.is_cuda:
        r = t[idx_gpu]
    else:
        r = t[idx_gpu.cpu()].to(dev, non_blocking=True)
    return r.float() if r.dtype == torch.float16 else r


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


def ctrl_anchor(d, gsel):
    """손잡이가 **붙들고 있는 입자**를 현재 부분표본 좌표계로 옮긴다 -> [T,K].

    색인이 두 겹이다: `ctrl_id` 는 원본 전체(25 만) 기준, 궤적에는 `sel` 로 솎은
    2 만 개가 들어 있고, 학습은 거기서 다시 `gsel` 로 솎는다.

    전체 -> 2 만 은 궤적마다 한 번만 하면 된다. 2 만 -> 현재 표본 은 `gsel` 이
    스텝마다 바뀌므로 매번 하는데, **GPU 에서 벡터로** 처리한다 (파이썬 루프로
    최근접을 돌았더니 한 스텝이 계산보다 비쌌다 -- 0.6 초가 4 초가 됐다).
    """
    if "_ca20" not in d:
        cid = d["ctrl_id"].cpu()
        sel = d["sel"].cpu()
        inv = torch.full((int(d["n_full"]),), -1, dtype=torch.long)
        inv[sel] = torch.arange(sel.numel())
        l20 = inv[cid.reshape(-1).clamp(0, inv.numel() - 1)].reshape(cid.shape)
        miss = l20 < 0
        if bool(miss.any()):
            X0 = d["x"][0].float().cpu()
            P = d["ctrl_pos"].float().cpu()
            idx = torch.nonzero(miss)
            tt = idx[:, 0].clamp(max=P.shape[0] - 1)
            c = P[tt, idx[:, 1]]                          # [m,3]
            l20[miss] = torch.cdist(c, X0).argmin(1)
        d["_ca20"] = l20.to(gsel.device if gsel is not None else dev)
    l20 = d["_ca20"]
    if gsel is None:
        return l20                      # 2 만 좌표계 색인 그대로 (표본 구성용)
    n20 = d["x"].shape[1]
    # 색인은 반드시 범위 안에 있어야 한다 -- 넘으면 CUDA 가 비동기 assert 로
    # 죽어서 원인 지점을 못 찾는다 (전 조합 학습에서 겪었다).
    l20 = l20.clamp(0, n20 - 1)
    gs = gsel.clamp(0, n20 - 1)
    invg = torch.full((n20,), -1, dtype=torch.long, device=gsel.device)
    invg[gs] = torch.arange(gs.numel(), device=gsel.device)
    loc = invg[l20.reshape(-1)].reshape(l20.shape)
    miss = loc < 0
    if bool(miss.any()):
        # 표본에 없는 제어 입자는 프레임 0 에서 가장 가까운 표본 입자로 대신한다
        x0all = d["x"][0].float().to(gsel.device)
        x0g = x0all[gs]                                    # [N,3]
        tgt = x0all[l20.reshape(-1)[miss.reshape(-1)]]     # [m,3]
        loc[miss] = torch.cdist(tgt, x0g).argmin(1)
    return loc


def ctrl_center(d, gsel, t, x):
    """손잡이 중심 [K,3]. **지목된 손잡이 입자의 현재 위치**다 (i-PG 와 같다).

    두 가지를 고쳐 둔다.

    1) 대체 입자를 **프레임 0 기준으로 한 번만** 고른다. 예전에는 `ctrl_anchor`
       가 프레임마다 "그 프레임 교사 중심에 가장 가까운 저장 입자" 를 골랐다.
       그러면 중심 입자의 정체가 매 프레임 바뀌고, 그 입자는 그 전까지 공 안에
       없어 구동을 안 받았으니 뒤처져 있다 -- 중심이 밀리고 오차가 누적된다
       (실측: 프레임 43 에 교사 중심과 0.308 = 반경의 2 배).
    2) 손잡이 입자가 저장 표본에 **있으면** 그것을 그대로 쓴다. 전체를 저장하는
       궤적(새 i-PG 교사)에서는 대체가 아예 일어나지 않는다.
    """
    if "_cc_fix" not in d:
        cid = d["ctrl_id"]
        cid = (cid[0] if cid.dim() == 2 else cid).reshape(-1).cpu()
        sel = d["sel"].cpu()
        inv = torch.full((int(d["n_full"]),), -1, dtype=torch.long)
        inv[sel] = torch.arange(sel.numel())
        loc = inv[cid.clamp(0, inv.numel() - 1)]
        miss = loc < 0
        if bool(miss.any()):
            # 프레임 0 의 손잡이 중심에 가장 가까운 저장 입자로 한 번만 정한다
            X0 = d["x"][0].float().cpu()
            P = d["ctrl_pos"].float().cpu()[0]
            loc[miss] = torch.cdist(P[miss], X0).argmin(1)
        d["_cc_fix"] = loc.to(dev)
    loc = d["_cc_fix"]
    if gsel is not None and gsel.numel() != d["x"].shape[1]:
        raise SystemExit("ctrl_center 는 전체 표본(gsel=arange)만 받는다")
    return x[loc]


def ctrl_weights(d, t, x, loc_t):
    """학생 상태에서 손잡이 가중치 [N,K].

    기본은 **하드 Dirichlet** 이다: 반경 안이면 1, 밖이면 0. i-PG 가 격자
    속도를 명령값으로 박는 것과 같은 조건이고, 교사도 그렇게 다시 만들었다.
    `--ctrl_soft` 를 켜면 옛 감쇠 가중 (1-q^2)^2 로 돌아간다 (옛 교사 궤적으로
    돌린 결과를 재현할 때만 쓴다).
    """
    R = d["ctrl_R"].to(x.device, x.dtype)
    Rt = R[min(t, R.numel() - 1)].clamp(min=1e-6)
    c = x[loc_t]                                          # [K,3] 학생 상태
    q = ((x.unsqueeze(1) - c.unsqueeze(0)).norm(dim=-1) / Rt).clamp(0, 1)
    if a.ctrl_soft:
        return (1.0 - q * q) ** 2, c
    return (q < 1.0).to(x.dtype), c


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
        off = (d["x"][0][g] - d["x"][0][d["ctrl"][k]]).float()
        out.append((loc, off))
    d["_cl"] = out
    d["_cl_key"] = key
    return out


def free_mask(d, n, device, gsel, x=None, t=0):
    """강제되지 **않은** 입자만 True. 손실은 여기서만 잰다.

    강제된 입자는 명령으로 밀리므로 학생이 정할 양이 아니다. 그대로 평균에
    넣으면 손실이 희석되고(그리고 기울기도 가짜다) 물리손실에서는 그 입자들의
    관성·중력 잔차까지 최소화하려 들어 운동을 자유낙하로 오해하게 된다.

    예전에는 `ctrl_mem`(궤적에 없는 키)을 읽어 **항상 전부 True** 였다.
    지금은 학생 상태에서 손잡이 가중치를 재어 w > 0.5 인 입자를 제외한다.
    """
    m = torch.ones(n, dtype=torch.bool, device=device)
    if x is not None and "ctrl_id" in d and "ctrl_R" in d:
        loc = ctrl_anchor(d, gsel)
        tt = min(t, loc.shape[0] - 1)
        w, _ = ctrl_weights(d, tt, x, loc[tt])
        m &= ~(w.max(1).values > 0.5)
        return m
    for loc, _ in ctrl_local(d, gsel):
        m[loc] = False
    return m


def apply_control(d, t, gsel, x2, x=None, dt=None):
    """강제되는 입자의 다음 위치를 **궤적 값으로 덮어쓴다**.

    교사가 그 입자들을 Dirichlet 으로 박았으므로, 학생이 거기를 예측하게 두면
    맞출 수 없는 것을 맞추라고 시키는 셈이다.
    """
    if "ctrl_id" not in d or "ctrl_vel" not in d or x is None:
        return x2
    # 교사(PG)가 하는 것과 같은 규약: 손잡이가 붙든 **입자를 중심으로** 반경 안
    # 입자의 속도를 감쇠 가중치로 섞는다. 중심은 교사가 측정한 위치가 아니라
    # **학생 자신의 상태에서** 그 입자가 가 있는 곳이고, 섞는 값은 주어진 명령
    # 속도다 -- 그래야 롤아웃에 교사의 시뮬 결과가 새지 않는다.
    loc = ctrl_anchor(d, gsel)
    tt = min(t, loc.shape[0] - 1)
    w, _c = ctrl_weights(d, tt, x, loc[tt])               # [N,K]
    vc = d["ctrl_vel"].to(x.device, x.dtype)[min(tt, d["ctrl_vel"].shape[0] - 1)]
    # 여러 손잡이가 겹치면 가장 센 것을 따른다
    wm, ki = w.max(1)
    _h = _DT[0] if dt is None else dt
    d_cmd = _h * _CTRL_SCALE[0] * vc[ki]                  # [N,3] 명령 변위
    wm = wm.unsqueeze(-1)
    return x + (1.0 - wm) * (x2 - x) + wm * d_cmd


_ENS_SHIFT = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5),
              (0.0, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
              (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)]

from anchorflow import vox_anchor                # noqa: E402


def fe_invariants(fe):
    """F_e 의 주응력 로그 [N,3]. 회전에 불변이라 그대로 특징으로 쓸 수 있다."""
    C = fe.transpose(-1, -2) @ fe
    sig = torch.linalg.eigvalsh(C.double()).clamp_min(1e-12).sqrt()
    return sig.clamp_min(0.01).log().to(fe.dtype)


def det_take():
    """직전 스텝 det 로 (탄성 마스크, 복구 손실) 을 만들고 비운다.

    d = det(grad Phi) 가 eps 아래인 사면체는 유효한 셀이 아니라고 보고 탄성
    에서 빼고, d >= eps 로 되돌리는 손실을 크게 준다.

    손실은 **유효한 사면체에서 정확히 0** 이어야 한다. 예전 softplus 판은
    d=1 에서도 1.2e-5 를 남겼고, det_w=100 을 곱하면 물리 목적함수(1.8e-5)
    의 67 배가 되어 학습이 물리 대신 "det 를 키우는 일" 을 했다 (실측: det
    중앙 1.06 으로 체적이 계속 부풀고, 구조를 뭘 바꿔도 E 가 9e-4 로 동일).
    그래서 경첩(hinge) 으로 바꾼다 -- d >= eps 면 값도 기울기도 0 이다:

        pen = relu(eps-d)^2/eps^2 + beta2*relu(-d)^2 + beta3*relu(-d)^3

    음수에서도 정의되고 큰 음수일수록 3 차항이 더 세게 민다.
    """
    dt_ = _DET_LAST[0]
    _DET_LAST[0] = None
    if dt_ is None:
        return None, 0.0
    eps = float(a.det_eps)
    bad = dt_ < eps
    neg = torch.relu(-dt_)
    pen = (torch.relu(eps - dt_) / max(eps, 1e-6)) ** 2 \
        + 10.0 * neg ** 2 + 10.0 * neg ** 3
    _DET_BAD.append(float(bad.to(torch.float32).mean()))
    _DET_MIN.append(float(dt_.min()))
    _DET_MED.append(float(dt_.median()))
    for _q in (_DET_BAD, _DET_MIN, _DET_MED):
        if len(_q) > 200:
            _q.pop(0)
    return (~bad), a.det_w * pen.mean()


def det_report():
    """무효 사면체 비율·det 통계 한 줄. 진행바는 폭에 잘려 못 믿는다."""
    if not _DET_BAD:
        return ""
    n = len(_DET_BAD)
    # 창 최소(최근 200 표본) 와 **현재 스텝** 을 함께 보인다 -- 창만 보면
    # 오래된 위반이 남아 회복을 못 읽는다
    return (f"무효 {100*sum(_DET_BAD)/n:.3f}% (지금 {100*_DET_BAD[-1]:.3f}%)  "
            f"det 최소 {min(_DET_MIN):.4f} (지금 {_DET_MIN[-1]:.4f}) "
            f"중앙 {sum(_DET_MED)/n:.4f}")


def ip_of(x, du, vel, F, mass, vol, cfg, h, ng, gl, g=None, norm=None,
          free=None, jac=None, elastic_mask=None):
    """증분 포텐셜. --obj 에 따라 입자에서 바로(기본) 또는 격자에서 잰다.

    반환 규약은 둘이 같다: (E, dlog, F_trial, parts).
    """
    if _OBJ_PTS:
        if jac is None:
            raise RuntimeError(
                "--obj pts 는 변형장 야코비안이 필요하다 (step_once 가 "
                "None 을 돌려줬다 -- 전달·앙상블 설정을 확인할 것)")
        return phys_resid.pts_ip_energy(
            x, du, vel, F, jac, mass, vol, cfg, h, ng, gl,
            g=g, norm=norm, free=free, elastic_mask=elastic_mask)
    return phys_resid.grid_ip_energy(
        x, du, vel, F, mass, vol, cfg, h, ng, gl,
        g=g, norm=norm, free=free, jac=(jac if a.f_from_jac else None))



# --- 삭제 과정에서 함께 날아간 전역들 (복구) -----------------------
_PROF_ON = bool(os.environ.get("AF_PROF"))
_PROF = {}
_CUR_T = [0]              # step_once 가 남기는 현재 프레임
_CTRL_SCALE = [1.0]       # 손잡이 명령 변위 배수 (서브스텝이면 1/K)
_DP_HOOK = [None]
# 망을 아예 부르지 않고 바로 출력을 내는 경로. None 을 돌려주면 망을 쓴다.
_DP_FAST = [None]
# "want" 를 넣으면 다음 셀집계 결과를 붙잡아 이후 호출에서 재사용한다
_NF_HOLD = [None]
_OV_TGT = [None]           # --ov_init target 이 쓸 자유낙하 평행이동 [3]
# AF_OV_CURVE=경로 를 주면 프레임별 (반복, E) 곡선을 그 경로에 남긴다
_CURVE = [] if os.environ.get("AF_OV_CURVE") else None
# 롤아웃 덤프에 격자점을 함께 담을지. (노드 위치, 노드 변위, 유효 여부)
_NOD_M = [None]            # 손잡이로 덮는 노드 (최적화에서 뺀다)
_NODE_CAP = [False]
_NODE_LAST = [None]
_BC_LAST = [None]          # (구속 보정량 [N,3], 활성집합 [N,1]) -- 최근 step_once
_BC_DIAG = []              # (활성비, 보정최대/ext, 바닥아래/ext, 손잡이오차/ext)
# [오차/ext, 구속 입자 수, 명령 변위 평균] -- 학습·평가 공통으로 갱신된다
_CTRL_ERR = [0.0, 0, 0.0]
_OV_LOG = []               # --ov_roll: 프레임별 (최적화 전 E, 후 E)


def bc_report():
    """하드 구속이 실제로 걸렸나 한 줄. 진행바는 폭에 잘려 못 믿는다.

    보정 최대가 0 에 가까워야 망이 경계조건을 스스로 내고 있다는 뜻이고,
    바닥 아래 깊이가 0 이어야 사영이 실제로 막고 있다는 뜻이다.
    """
    if not _BC_DIAG:
        return ""
    n = len(_BC_DIAG)
    ar = sum(q[0] for q in _BC_DIAG) / n
    cm = max(q[1] for q in _BC_DIAG)
    pd = max(q[2] for q in _BC_DIAG)
    ce = max(q[3] for q in _BC_DIAG)
    return (f"구속 활성 {100 * ar:.2f}%  보정최대 {cm:.2e}  "
            f"바닥아래 {pd:.2e} (지금 {_BC_DIAG[-1][2]:.2e})  "
            f"손잡이오차 {ce:.2e}")
_F_MSG = []
_PL_DROP = []
_PL_MSG = []
_PL_RMS = [0.0]
_RL_MSG = []
_DET_LAST = [None]       # 직전 스텝의 사면체 det -- 마스킹·복구에 쓴다
_DET_BAD = []            # 최근 스텝의 무효 사면체 비율 (보고용)
_DET_MIN = []            # 최근 스텝의 det 최소값
_DET_MED = []            # 최근 스텝의 det 중앙값


class _tsec:
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


if os.environ.get("AF_ANOMALY"):     # in-place/NaN 범인을 짚을 때
    torch.autograd.set_detect_anomaly(True)
    print("[디버그] autograd anomaly detection on", flush=True)

_VDT_MSG = []            # --v_from_dt 를 못 쓸 때의 경고를 한 번만
_OBJ_PTS = (a.obj == "pts")   # 목적함수를 입자에서 바로 재는가 (격자 미사용)
_RQS_PEN = [None]        # step_once 가 쌓는 RQS 연속성 벌점, 손실 지점이 소비


def node_feats(d, t, gsel, x, v, fe=None):
    """사면체 복합체의 **노드** 특징. 입자 물리량을 자기 사면체의 barycentric
    가중으로 4 꼭짓점에 뿌려 모은다 (P2G 와 같은 구조, 전달 가중치 재사용).

    -> (_in [M,F], npos [M,3], (lo,lat,nn), rows [N,4], lam [N,4], uniq [M],
        간선 (src,dst,cls))
    """
    cfg = d["cfg"]
    X = take(d["x"][0], gsel)
    lo, lat, nn = (_GRID[0] if _GRID[0] is not None
                   else SX.grid_for_nodes(x, a.n_nodes))
    idx, lam, _aux = SX.locate(x, lo, lat, nn)   # tau 에 무관 -- 한 번만
    rows, uniq = SX.active_nodes(idx)          # **점유 사면체의 꼭짓점만**
    Mn = int(uniq.numel())
    npos = SX.node_pos(lo, lat, nn, uniq)
    hn = lat.s                                 # 길이 단위 = 면내 간격 h
    mw = MASS[gsel]
    # 국소 통계(2 차 모멘트·각속도·국소 F) -- 탄성항이 보는 것을 담으려면
    # 변위·속도 평균만으로는 모자란다 (격자 시절 tri_feats 와 같은 항목)
    feat = _san(node_moments(x, v / VEL_SCALE, X, mw, rows, lam, Mn, npos, hn))
    parts = []
    if a.fe_state:
        parts.append(fe_invariants(fe) if fe is not None
                     else torch.zeros(x.shape[0], 3, device=dev, dtype=x.dtype))
    if a.control:
        parts.append(ctrl_feat_pts(d, t, x, hn))
    if parts:
        feat = torch.cat([feat, _san(scatter_to_nodes(
            _san(torch.cat(parts, -1)), rows, lam, Mn, mass=mw))], -1)
    extra = [bc_features(npos, cfg) / hn]
    if N_MAT and not N_FILM:
        extra.insert(0, mat_feat(cfg).reshape(1, N_MAT).expand(Mn, N_MAT))
    _in = torch.cat([_san(feat), _san(torch.cat(extra, -1))], -1)
    _in = torch.nan_to_num(_in, nan=0.0, posinf=0.0,
                           neginf=0.0).clamp(-_FCAP, _FCAP)
    src, dst, cls = SX.edges_of(rows, uniq, nn)
    return (_in, npos, (lo, lat, nn), rows, lam, uniq, (src, dst, cls), _aux)


def step_once(d, t, gsel, p, x, v, need_J=True, dmg=None, idx_prev=None,
              x0=None, p0=None, fe=None):
    """한 프레임. -> (x_next, p_next, v_next, J, dp, aidx_next)

    aidx_next 는 --refps 일 때만 뜻이 있다: 다음 프레임의 앵커가 **현재 부분표본의
    몇 번째 가우시안인지**. 매 스텝 다시 뽑으면 앵커의 정체가 바뀌므로, 앵커 손실이
    비교할 정답도 그때그때 그 가우시안들의 GT 변위로 바뀐다.
    """
    _CUR_T[0] = int(t)          # 프레임별 출력 변수를 찾는 데 쓴다
    # 앙상블: 원점을 어긋나게 둔 격자 여러 개의 변위를 평균한다. 같은 가중치를
    # 쓰므로 파라미터는 늘지 않고, 격자 위치 때문에 생기는 편향만 씻긴다.
    if a.arch == "sgnn":
        # --- 사면체 복합체 경로 -------------------------------------------
        # 변형장은 "GNN + barycentric + 손잡이 혼합" 뿐이다. 노드 특징·간선은
        # tau 에 무관하므로 밖에서 만들고, tau 에 탄젠트를 얹어 한 번 통과
        # 시키면 dPhi/dt (각 지점의 속도) 가 나온다.
        with _tsec("셀집계"):
            # 한 프레임 안에서 x·v 가 고정이면 격자·locate·간선·노드특징이 전부
            # 그대로다 (tau 에도 무관하다). --ov_roll 은 같은 프레임을 수백 번
            # 돌므로 한 번만 만들어 재사용한다.
            if _NF_HOLD[0] is not None and _NF_HOLD[0] != "want":
                _nf = _NF_HOLD[0]
            else:
                _nf = node_feats(d, t, gsel, x, v, fe=fe)
                if _NF_HOLD[0] == "want":
                    _NF_HOLD[0] = _nf
            (_in, npos, (lo, lat, nn), rows, lam, uniq,
             (esrc, edst, ecls), _lax) = _nf
        _mv = (mat_feat(d["cfg"]).reshape(1, N_MAT) if N_FILM else None)
        # 손잡이는 **노드 변위를 명령값으로 덮는다** (i-PG 가 격자 속도를 박는
        # 것과 같다). 예전에는 입자 단계에서 덮었는데, 그러면 (a) 변형장 자체는
        # 경계조건을 모르고 (b) 중심을 입자에서 재-유도하느라 뒤처짐이 누적됐다
        # (실측: 프레임 43 에 교사 중심과 0.308 = 반경의 2 배, 원 안 입자가
        # 3115 -> 577 개). 노드에 박으면 둘 다 사라지고 L_bc 도 필요 없다.
        _nod_m = None            # 명령으로 덮을 노드 (불리언 [M])
        _nod_v = None            # 그 노드의 명령 속도 [M,3]
        if a.control and "ctrl_id" in d and "ctrl_vel" in d:
            _tt = min(t, d["ctrl_vel"].shape[0] - 1)
            _cvc = d["ctrl_vel"].to(x.device, x.dtype)[_tt]        # [K,3]
            _cen = ctrl_center(d, gsel, _tt, x)                    # [K,3]
            _Rc = d["ctrl_R"].to(x.device, x.dtype)
            _Rc = _Rc[min(_tt, _Rc.numel() - 1)].clamp_min(1e-6)
            # 노드-손잡이 거리로 **지시함수** (하드). 가장 가까운 손잡이를 따른다
            _dn = (npos.unsqueeze(1) - _cen.unsqueeze(0)).norm(dim=-1)   # [M,K]
            _dmin, _kmin = _dn.min(1)
            _nod_m = _dmin < _Rc
            _nod_v = _cvc[_kmin]                                   # [M,3]

        _NOD_M[0] = _nod_m

        if _OV_TGT[0] == "want":
            # **노드별 관성 예측자** h·v_n + h²g. v_n 은 barycentric 가중 질량
            # 평균이라 P2G 와 같은 사상이다. 자유변수의 출발점으로 쓴다.
            _Mn = npos.shape[0]
            _mw = (MASS[gsel] if MASS.numel() != x.shape[0] else MASS
                   ).unsqueeze(-1).to(x.dtype)
            _ws = torch.zeros(_Mn, 1, device=dev, dtype=x.dtype)
            _vs = torch.zeros(_Mn, 3, device=dev, dtype=x.dtype)
            for _j in range(4):
                _w = lam[:, _j:_j + 1] * _mw
                _ws.index_add_(0, rows[:, _j], _w)
                _vs.index_add_(0, rows[:, _j], _w * v)
            _vn = _vs / _ws.clamp_min(1e-30)
            _gq = torch.as_tensor(d["cfg"]["g"], device=dev, dtype=x.dtype)
            _hq = float(_DT[0])
            _OV_TGT[0] = _hq * _vn + (_hq ** 2) * _gq

        def _fieldS(tau):
            # 출력을 자유변수로 갈아끼우는 경로(--ov_roll)에서는 변수가 생긴
            # 뒤의 순전파가 통째로 버려진다. 프레임당 수백 번이므로 건너뛴다.
            out = (_DP_FAST[0](npos.shape[0], x.dtype, x.device)
                   if _DP_FAST[0] is not None else None)
            if out is None:
                with _tsec("신경망"):
                    out = net(_in, esrc, edst, ecls, tau, mat=_mv)
                if _DP_HOOK[0] is not None:
                    out = _DP_HOOK[0](out)
            dpn = torch.nan_to_num(out[0], nan=0.0, posinf=0.0, neginf=0.0)
            # 셀(사면체) 내부는 **항등**이다. 로컬 변환은 쓰지 않는다.
            # locate 결과(rows, lam)를 재사용한다 -- tau 에 무관하므로 jvp 안에서
            # 다시 돌 이유가 없고, Mtot 크기 zeros+index_copy 도 필요 없다.
            if _nod_m is not None:
                # **노드 변위를 명령으로 덮는다.** detach 로 기울기를 끊어 그
                # 노드는 최적화 대상에서 빠진다 (상수 경계조건이다).
                dpn = torch.where(_nod_m.unsqueeze(-1),
                                  (tau * _CTRL_SCALE[0] * _nod_v).detach(),
                                  dpn)
                with torch.no_grad():
                    _CTRL_ERR[1] = int(_nod_m.sum())
                    _CTRL_ERR[2] = float(
                        (tau * _nod_v)[_nod_m].norm(dim=-1).mean()
                        if bool(_nod_m.any()) else 0.0)
            q = x + SX.g2p_pre(rows, lam, dpn)
            _qraw = q
            _act = torch.zeros_like(q[:, :1])
            if _nod_m is not None:
                # 구속 노드 넷에 모두 둘러싸인 입자는 명령만큼 간다 -- 그 입자를
                # 증분 포텐셜에서 뺀다 (반력이 실어 나르는 자리다).
                _act = _act + _nod_m[rows].all(1).to(q.dtype).unsqueeze(-1)
            if not a.bc_soft:
                _duP, _fa = phys_resid.bc_project(
                    x, q - x, d["cfg"], tau,
                    float(d["cfg"].get("grid_lim", 2.0)),
                    int(d["cfg"]["n_grid"]))
                q = x + _duP
                _act = _act + _fa.to(q.dtype).unsqueeze(-1)
            # corr 은 구속이 원 출력을 얼마나 고쳤나다. 이걸 줄이면 망이 경계조건을
            # **스스로** 내게 된다 (자유 입자는 정확히 0 이라 기여가 없다).
            _corr = q - _qraw
            _ex = (dpn, _corr, _act)
            return q, _ex

        _use_dtS = a.v_from_dt and _DP_HOOK[0] is None and a.dt_cond
        # (아래 jvp/야코비안 구간도 계측한다)
        _tau0 = torch.as_tensor(float(_DT[0]), device=dev, dtype=x.dtype)
        if _use_dtS:
            with _tsec("jvp(속도)"):
                (x2, _outs), (v_next, _) = torch.func.jvp(
                    _fieldS, (_tau0,), (torch.ones_like(_tau0),))
        else:
            x2, _outs = _fieldS(_tau0)
            v_next = (x2 - x) / _DT[0]
        dpn = _outs[0]
        # 셀 내부가 항등이라 변형장은 사면체별 아핀이다 -> 야코비안은 닫힌 형식
        # 하나로 나온다 (재배열이 있던 시절에는 합성이라 역전파 3 회가 필요했다).
        with _tsec("야코비안"):
            _u, Jf = SX.g2p_jac_pre(rows, lam, _lax, lat, dpn)
            Jf = torch.eye(3, device=dev, dtype=x.dtype) + Jf
            if a.fbar == "nodal":
                # **노드 평균 변형구배.** 셀 하나로 재면 그 셀의 일그러짐이
                # 이웃에게 아무 비용도 물리지 않아, 손잡이 구속이 셀을 한 겹씩
                # 건너며만 전해진다 (실측: 공이 손잡이에서 떨어져 나갔다).
                # 이 격자는 모든 사면체의 부피가 같으므로 부피 가중은 평균과
                # 같다. 평균에 드는 것은 **입자가 있는 셀**뿐이다.
                _tid = SX.tet_id(lo, lat, nn, _lax)
                _tu, _tinv = torch.unique(_tid, return_inverse=True)
                _T = int(_tu.numel())
                _Jt = torch.zeros(_T, 3, 3, device=dev, dtype=Jf.dtype)
                _Jt = _Jt.index_copy(0, _tinv, Jf)          # 셀별 F
                _rt = torch.zeros(_T, 4, device=dev, dtype=rows.dtype)
                _rt = _rt.index_copy(0, _tinv, rows)
                _Fn = torch.zeros(npos.shape[0], 3, 3, device=dev,
                                  dtype=Jf.dtype)
                _cn = torch.zeros(npos.shape[0], device=dev, dtype=Jf.dtype)
                _one = torch.ones(_T, device=dev, dtype=Jf.dtype)
                for _j in range(4):
                    _Fn = _Fn.index_add(0, _rt[:, _j], _Jt)
                    _cn = _cn.index_add(0, _rt[:, _j], _one)
                _Fn = _Fn / _cn.clamp_min(1.0).reshape(-1, 1, 1)
                Jf = (lam.reshape(-1, 4, 1, 1) * _Fn[rows]).sum(1)
        if _NODE_CAP[0]:
            # **유효 격자점** = 입자가 든 사면체의 꼭짓점. 나머지는 변위가 어떤
            # 값이든 입자를 하나도 옮기지 않으므로 뜻이 없다.
            _sup = torch.zeros(npos.shape[0], dtype=torch.bool, device=dev)
            _sup[rows.reshape(-1)] = True
            # 입자별 셀 구분: 0 = 꼭짓점 넷 다 손잡이 밖, 1 = **섞인 셀**
            # (일부만 손잡이 안), 2 = 넷 다 손잡이 안. 1 이 손잡이가 몸통으로
            # 힘을 넘기는 유일한 통로다.
            if _nod_m is None:
                _pc = torch.zeros(x.shape[0], dtype=torch.uint8)
            else:
                _hit = _nod_m[rows]
                _pc = (_hit.any(1).to(torch.uint8)
                       + _hit.all(1).to(torch.uint8)).cpu()
            _NODE_LAST[0] = (npos.detach().cpu(), dpn.detach().cpu(),
                             _sup.cpu(), _pc)
        _DET_LAST[0] = torch.linalg.det(Jf)
        # 구속 보정량·활성집합을 창 쪽으로 넘긴다 (L_bc 와 free 마스크에 쓴다)
        _BC_LAST[0] = (_outs[-2], _outs[-1])
        return (x2, p, v_next, None, dpn, None, dmg, None, fe, Jf)
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


_OV_CUR = [None]           # --out_var 에서 지금 최적화 중인 상태 사전

@contextlib.contextmanager
def traj_scope(d):
    """평가·검증 동안 전역 MASS/EXT/N_FULL 을 **그 궤적의** 값으로 맞춘다.

    풀 루프가 `MASS, EXT, N_FULL = sc["mass"], sc["ext"], n_p` 로 전역을 씬 값
    으로 덮어쓴다. 그 뒤 교사 궤적으로 평가하면 node_feats 의 `MASS[gsel]` 이
    **풀 씬의 앞쪽 질량**을 집어 (입자 집합이 다르다) 질량 가중 노드 특징이
    통째로 어긋난다. EXT 도 달라 절대 오차 퍼센트가 틀린다 (비는 분자·분모가
    같은 EXT 라 상쇄되지만 절대값은 틀린다).
    """
    global MASS, EXT, N_FULL
    _sv = (MASS, EXT, N_FULL)
    # out_var 의 훅도 함께 끊는다. 안 끊으면 평가 중 step_once 가 **풀에서 마지막
    # 으로 다루던 상태**의 출력 변수를 돌려주고, 노드 수가 달라 색인이 터진다.
    _ov = _OV_CUR[0]
    _OV_CUR[0] = None
    MASS = traj_mass(d)
    EXT = float(d.get("_ext", EXT))
    N_FULL = d["x"].shape[1]
    try:
        yield
    finally:
        MASS, EXT, N_FULL = _sv
        _OV_CUR[0] = _ov


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


CRITIC = None
OPT_C = None
SCHED = None


class Critic(torch.nn.Module):
    """상태의 가치 V(s) -- 이 상태에서 앞으로 쌓일 i-PG 비용의 추정치.

    학생이 보는 것과 **같은 셀 특징**을 받아 MLP 로 누른 뒤 전역 평균을 낸다.
    구조에 의존하지 않게 하려고 격자·앵커 어느 쪽이든 [M, F] 만 받는다.
    """

    def __init__(self, n_feat, hidden=128, n_extra=2):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(n_feat, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + n_extra, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, feat, extra, frozen=False):
        """frozen 이면 **가중치로는 기울기를 보내지 않는다** (입력으로만 보낸다).

        액터 손실에 V(s') 이 들어가는데, 거기서 가중치까지 학습하면 정책이 V 를
        낮추는 쪽으로 착취한다 -- 실제로 정책 손실이 음수로 달아났다.
        """
        if frozen:
            ps = [q.detach() for q in self.parameters()]
            h = torch.nn.functional.silu(
                torch.nn.functional.linear(feat, ps[0], ps[1]))
            h = torch.nn.functional.silu(
                torch.nn.functional.linear(h, ps[2], ps[3]))
            h = h.mean(0, keepdim=True)
            z = torch.cat([h, extra.reshape(1, -1)], -1)
            z = torch.nn.functional.silu(
                torch.nn.functional.linear(z, ps[4], ps[5]))
            out = torch.nn.functional.linear(z, ps[6], ps[7])
        else:
            h = self.f(feat).mean(0, keepdim=True)
            out = self.head(torch.cat([h, extra.reshape(1, -1)], -1))
        # 비용이 0 이상이므로 가치도 0 이상이어야 한다
        return torch.nn.functional.softplus(out).reshape(())


def rl_state_feat(d, t, gsel, x, v, fe, p=None):
    """크리틱 입력. 학생과 같은 셀 특징을 쓰고, 앵커 판본이면 이탈 정도를 덧붙인다."""
    if a.arch == "attn" and p is not None:
        # 학생이 받는 것과 **같은 폭**으로 맞춘다 (집계 + 물성/경계/손잡이)
        cfg = d["cfg"]
        idx, dist = anchor_knn(x, p, a.k)
        feat, _ = aggregate(x, v / VEL_SCALE, take(d["x"][0], gsel), MASS[gsel],
                            idx, p.shape[0], H, pa=p)
        feat = _san(feat)
        extra = torch.cat([
            mat_feat(cfg).reshape(1, N_MAT).expand(p.shape[0], N_MAT),
            bc_features(p, cfg) / H], -1)
        if a.control:
            extra = torch.cat([extra, ctrl_feat(d, t, p)], -1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        extra = torch.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
        drift = float(dist[:, 0].mean())
        return torch.cat([feat, extra], -1), drift
    _in = node_feats(d, t, gsel, x, v, fe=fe)[0]
    return _in, 0.0


def rl_episode(d, t0, E, gsel, gen):
    """에피소드 하나. 매 스텝 한 번만 미분하고 미래는 V 가 본다.

        정책 손실 =  E_iPG(s, pi(s))  -  gamma * V(s')
        크리틱 손실 = ( V(s) - [ E_iPG + gamma * V(s').detach() ] )^2

    상태는 스텝 사이에서 detach 한다 -- 그래서 메모리가 E 에 무관하고, 8 프레임을
    거슬러 미분하던 비용이 사라진다. 대신 8 프레임 뒤의 영향은 V 가 실어 나른다.
    """
    if _OBJ_PTS:
        raise SystemExit("--rl 경로는 격자 목적함수만 지원한다 (--obj grid)")
    mass_full = traj_mass(d)
    mass = mass_full[gsel]
    ext = d.get("_ext", EXT)
    cfg = d["cfg"]
    vol = mass / float(cfg["density"])
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    h = FRAME_DT
    norm = float(mass.sum()) * (ext ** 2) / (h * h)
    ng, gl = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))

    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / h
    F = take(traj_F(d)[t0], gsel).float()
    p = (x[fps(x, a.n_anchors, a.seed)] if a.arch == "attn" else
         take(d["x"][t0], AIDX))
    fe = None
    la = lc = 0.0
    cost_sum = 0.0
    n_st = 0
    drift0 = None
    for i in range(E):
        t = t0 + i
        feat_s, dr = rl_state_feat(d, t, gsel, x, v, fe, p)
        if drift0 is None and dr > 0:
            drift0 = dr
        v_s = CRITIC(feat_s.detach(),
                     torch.tensor([float(i) / E, dr / max(ext, 1e-9)],
                                  device=dev))
        x2, p2, v2, _J, _dp, _ai, _dmg, _cr, fe2, _Jd = step_once(
            d, t, gsel, p, x, v, need_J=False, fe=fe)
        # 상태가 무효가 되면(비유한, 또는 격자 밖으로 이탈) 에피소드를 끝내고
        # 벌점을 준다. 자르지 않는다 -- 자르면 정책이 그 경계를 이용한다.
        _fin = bool(torch.isfinite(x2).all())
        _mx = float(x2.abs().max()) if _fin else float("nan")
        _bad = (not _fin) or _mx > 4.0 * gl
        if _bad and not _RL_MSG:
            _RL_MSG.append(1)
            _fx = bool(torch.isfinite(x).all())
            _fdp = bool(torch.isfinite(_dp).all()) if _dp is not None else True
            _fv = bool(torch.isfinite(v).all())
            _ff = bool(torch.isfinite(F).all())
            _fp = bool(torch.isfinite(p).all()) if torch.is_tensor(p) else True
            print(f"[RL 무효상태] 스텝 {i} 유한 {_fin} 최대 {_mx:.3e} "
                  f"(x {_fx}, v {_fv}, F {_ff}, p {_fp}, dp {_fdp}, "
                  f"태그 {d.get('tag')})", flush=True)
            print(f"           x 최대 {float(x.abs().max()):.3e}  "
                  f"v 최대 {float(v.abs().max()):.3e}  "
                  f"dp 최대 {float(_dp.abs().max()):.3e}", flush=True)
        if _bad:
            la = la + torch.as_tensor(a.rl_term_pen, device=dev)
            cost_sum += a.rl_term_pen
            n_st += 1
            break
        fm = free_mask(d, x2.shape[0], dev, gsel, x, t) if a.control else None
        E_ip, dlog, F_tr, _pt = phys_resid.grid_ip_energy(
            x, x2 - x, v, F, mass, vol, cfg, h, ng, gl, g=g, norm=norm, free=fm)
        if a.rl_reward == "residual":
            # 정류 잔차: 정답에서 0 이고 아래로 유계다. 질량으로 나눠 길이 단위로
            # 만든 뒤 물체 크기로 정규화한다 (에너지처럼 스케일이 재질에 끌려가지
            # 않게).
            _gx, = torch.autograd.grad(E_ip * norm, x2, create_graph=True)
            _rr = _gx * (h * h) / mass.unsqueeze(-1).clamp_min(1e-20) / ext
            E_ip = ((_rr[fm] if fm is not None else _rr) ** 2).sum(-1).mean()
        # 다음 상태의 가치 (정책으로 기울기가 흐른다 -- 미래 영향의 경로)
        feat_n, dr_n = rl_state_feat(d, t + 1, gsel, x2, v2, fe2, p2)
        term = (drift0 is not None and dr_n > a.rl_term * drift0)
        if term:
            v_next = torch.zeros((), device=dev)
            v_next_c = torch.zeros((), device=dev)
            pen = torch.tensor(a.rl_term_pen, device=dev)
        else:
            _ex = torch.tensor([float(i + 1) / E, dr_n / max(ext, 1e-9)],
                               device=dev)
            v_next = CRITIC(feat_n, _ex, frozen=True)     # 액터용 (가중치 동결)
            v_next_c = CRITIC(feat_n.detach(), _ex)       # 크리틱 목표용
            pen = torch.zeros((), device=dev)
        cost = E_ip + pen                       # 비용 = -보상
        la = la + (cost + a.rl_gamma * v_next)
        with torch.no_grad():
            y = cost.detach() + a.rl_gamma * v_next_c.detach()
        lc = lc + (v_s - y) ** 2
        cost_sum = cost_sum + float(cost)
        n_st += 1
        if term:
            break
        x = x2.detach()
        v = v2.detach()
        p = p2.detach() if torch.is_tensor(p2) else p2
        fe = fe2.detach() if torch.is_tensor(fe2) else fe2
        with torch.no_grad():
            F = phys_resid.plastic_step(F_tr, dlog).detach()
    return la / n_st, lc / n_st, n_st, cost_sum / n_st


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
    _g = torch.arange(N_FULL, device=dev)
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
        n_feat = node_feats(_d, 1, _g, _x, _v, fe=_fe0)[0].shape[-1]
if a.voxel:
    VOX_CELL = H
    # 격자는 모든 궤적을 덮도록 공간에 고정한다 (프레임마다 새로 잡으면 물체가
    # 떨어지는 것만으로 모든 복셀 키가 바뀐다)
    VOX_LO = (torch.stack([dd["x"].reshape(-1, 3).float().min(0).values
                           for _t, dd in TR + held]).min(0).values
              - 4 * H).to(dev)
    _hi = torch.stack([dd["x"].reshape(-1, 3).float().max(0).values
                       for _t, dd in TR + held]).max(0).values.to(dev)
    _mx = (((_hi - VOX_LO) / (H * (max(a.vox_ens, 1) ** (1.0 / 3.0))))
           .floor().long() + 4).tolist()
    VOX_DIMS = (_mx[1] + 3, _mx[2] + 3,
                (_mx[0] + 3) * (_mx[1] + 3) * (_mx[2] + 3))
    with torch.no_grad():
        _g = torch.arange(N_FULL, device=dev)
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
                             device=dev).sort().values
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
            _s, _cr = node_feats(_dd, _t, _gs, _x, _v, fe=_fes)[0], None
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
    _miss, _unex = net.load_state_dict(ck["net"], strict=False)
    _miss = [k for k in _miss]
    if _unex:
        raise SystemExit(f"[재개] 체크포인트에 모르는 키가 있다: {_unex[:5]}")
    if _miss:
        # --dt_cond 를 뒤늦게 켠 경우가 여기다. DtFiLM 은 0 초기화라 새로
        # 붙여도 그 순간에는 항등이고, 학습이 진행되며 dt 의존을 배운다.
        if all(k.startswith("dtfilm.") for k in _miss):
            print(f"[재개] dt 조건화 모듈 {len(_miss)} 개를 새로 붙였다 "
                  f"(0 초기화라 지금은 항등)", flush=True)
        else:
            raise SystemExit(f"[재개] 없는 가중치: {_miss[:5]}")
    _ckdt = (ck.get("args") or {})
    for _k in ("dt_cond", "dt_scale", "dt_sub", "dt_sub_set"):
        if _k in _ckdt and _ckdt[_k] != getattr(a, _k):
            print(f"[재개] 경고: {_k} 가 체크포인트({_ckdt[_k]!r})와 지금"
                  f"({getattr(a, _k)!r}) 이 다르다", flush=True)
    if a.resume_fresh:
        print(f"[이어받음] {a.resume} 의 가중치만 (스텝 0 에서 새로 시작)",
              flush=True)
    else:
        opt.load_state_dict(ck["opt"])
        if a.rl and ck.get("critic") is not None and CRITIC is not None:
            CRITIC.load_state_dict(ck["critic"])
            if ck.get("opt_c") is not None:
                OPT_C.load_state_dict(ck["opt_c"])
            print("[재개] 크리틱도 이어받았다", flush=True)
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
    # 보조 지표였던 지도 손실은 지웠다 -- 학습이 교사를 안 쓰므로 그 창도 없다.
    obj = 0.0
    for _tag, d in _VAL:
        # 전역 MASS/EXT/N_FULL 을 이 궤적 값으로 (풀 루프가 씬 값으로 덮어쓴다)
        _ts = traj_scope(d); _ts.__enter__()
        gsel = torch.arange(d["x"].shape[1], device=dev)
        t0 = 3
        x = take(d["x"][t0], gsel)
        v = (x - take(d["x"][t0 - 1], gsel)) / FRAME_DT
        p = take(d["x"][t0], AIDX)
        x_still = x.clone()
        for i in range(a.val_len):
            with torch.enable_grad(), dt_scope(FRAME_DT / EVAL_SUB):
                for _sb in range(EVAL_SUB - 1):
                    x, p, v, _, _, _, _, _, _, _ = step_once(
                        d, t0 + i, gsel, p, x, v, need_J=False)
                    x, p, v = x.detach(), p.detach(), v.detach()
                x2, p, v, _, _, _, _, _, _, _ = step_once(
                    d, t0 + i, gsel, p, x, v, need_J=False)
            x2, p, v = x2.detach(), p.detach(), v.detach()
            gt = take(d["x"][t0 + i + 1], gsel)
            fm = (free_mask(d, x2.shape[0], dev, gsel, x2, t0 + i)
                  if a.control else slice(None))
            tot += float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
            ref += float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
            x = x2
        _ts.__exit__(None, None, None)
    net.train()
    return tot / max(ref, 1e-20), obj


def save_ck(name, step):
    # 난수 상태를 함께 남긴다. 이게 없으면 재개한 뒤 창 표본이 다른 흐름을 타서
    # 곡선이 이어지지 않는다 (TensorBoard 에서 바로 보인다).
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                # 크리틱도 함께 남긴다 -- 없으면 재개할 때 가치함수가 0 에서
                # 다시 시작해 정책이 한동안 미래를 못 본다
                "critic": (CRITIC.state_dict() if CRITIC is not None else None),
                "opt_c": (OPT_C.state_dict() if OPT_C is not None else None),
                "step": step, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                "n_feat": n_feat, "args": vars(a),
                "best": _best,
                "rng_gen": gen.get_state(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "rng_np": np.random.get_state(),
                "rng_py": random.getstate(),
                # 풀 자체와 누적 집계. 없으면 재개할 때 빈 풀에서 0 부터 다시
                # 세게 되어 쌓아 둔 상태를 잃고 TB 누적 곡선도 끊긴다.
                "pool": (POOL.state_dict() if POOL is not None else None),
                "pool_drop_win": list(_PL_DROP),
                "pool_rms": list(_PL_RMS),
                # 프레임별 출력 변수 (--out_var). 렌더에서 같은 값을 쓰려면 필요하다
                # out_var 변수는 **상태에 붙어** 있고 상태는 저장하지 않으므로
                # 체크포인트에 넣지 않는다 (프레임 키 시절의 유물이었다).
                "out_var": None},
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
        gsel = torch.arange(d["x"].shape[1], device=dev)
        mass = traj_mass(d)[gsel]
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
        fm = free_mask(d, x1_.shape[0], dev, gsel, x0_, int(t0)) if a.control else None
        E, _dl, parts = phys_resid.ip_energy(
            x1_, x0_ + h * v0_, F1_, mass, vol, cfg, h, free=fm, g=gvec,
            norm=norm)
        rr = phys_resid.residual(E * norm, x1_, mass, ext) * h * h
        _fr = float(rr[fm].mean()) if fm is not None else float(rr.mean())
        print(f"  {tag} t0={int(t0):3d} [{phys_resid.mat_name(cfg)}] "
              f"E={float(E):.4e} (관성 {parts[0]:.3e} 탄성 {parts[1]:.3e} "
              f"중력 {parts[2]:.3e})  잔차 {100*_fr:.4f}%", flush=True)
    raise SystemExit(0)

POOL = None
SCENE_DS = []
if a.pool:
    from anchorflow.scene_pool import StatePool, load_scenes
    _ALL_COMBOS = "hotdog_clayC,hotdog_elD,hotdog_viscoplastic,lego_clayC,lego_elD,lego_viscoplastic,mic_clayC,mic_elD,mic_viscoplastic,wolf_clayC,wolf_elD,wolf_viscoplastic".split(",")
    _combos = (_ALL_COMBOS if a.pool_combos.strip() in ("", "all")
               else [c for c in a.pool_combos.split(",") if c])
    _W = os.environ.get("AF_WORK", "/home/dkta/work")
    # 채움 디렉토리는 따로 줄 수 있다 -- 교사(i-PG)가 쓴 입자 집합과 **같은
    # 것**을 쓰려면 그쪽 캐시를 가리켜야 한다.
    _FD = a.pool_fill or _W
    _sc = load_scenes(_FD, os.path.join(_W, "wmats"), _combos,
                      0, dev, seed=a.seed)
    if not _sc:
        raise SystemExit("풀에 넣을 씬이 없다")
    _R = 0.15
    _gl0 = float(_sc[0][1]["cfg"].get("grid_lim", 2.0))
    # 목표점 마진은 **손잡이 반경보다 커야** 한다. 반경만큼만 두면 손잡이가
    # 목표에 닿는 순간 무리의 바깥쪽 입자가 경계에 걸리고, 딸려 오는 부분까지
    # 생각하면 그 자리에서 영역을 벗어난다.
    POOL = StatePool(_sc, a.pool_size, a.n_ctrl, _R, dev, gen,
                     frames=a.pool_frames, thresh=a.pool_thresh,
                     window=a.pool_window,
                     domain=_gl0, margin=_R + 0.15,
                     start_mid=a.pool_start_mid, keep_prob=a.pool_keep,
                     acc=a.ctrl_acc, vmax=a.ctrl_vmax, zbias=a.ctrl_zbias,
                     n_side=a.pool_targets)
    for _tag, _s in _sc:
        SCENE_DS.append(dict(x=_s["x0"].unsqueeze(0), cfg=_s["cfg"],
                             sel=torch.arange(_s["x0"].shape[0], device=dev),
                             n_full=_s["x0"].shape[0], tag=_tag,
                             _mass=_s["mass"], _ext=_s["ext"],
                             ctrl_R=torch.tensor([0.15], device=dev)))
    if a.pool_grid == "domain":
        _gl = float(_sc[0][1]["cfg"].get("grid_lim", 2.0))
        FIXED_GRID = (torch.zeros(3, device=dev),
                      _gl / a.vox_res,
                      torch.tensor([a.vox_res] * 3, device=dev))
        print(f"[풀] 격자 고정: 영역 [0,{_gl}]^3 을 {a.vox_res}^3 으로, "
              f"셀 크기 {_gl / a.vox_res:.4f}", flush=True)
    print(f"[풀] 씬 {len(_sc)} 개, 크기 {a.pool_size}, 신규 비율 "
          f"{a.pool_fresh:.2f}, 문턱 {a.pool_thresh}, 유예 {a.pool_keep}, "
          f"창 {a.pool_window}, 목표 후보 {POOL.cand.shape[0]} 개, "
          f"가속 {a.ctrl_acc} 최고속도 {a.ctrl_vmax} z편향 {a.ctrl_zbias}",
          flush=True)
    _pck = (_rng_ck or {}).get("pool") if not a.resume_fresh else None
    if _pck:
        _nl, _ns = POOL.load_state_dict(_pck)
        _PL_DROP[:] = list((_rng_ck or {}).get("pool_drop_win") or [])
        _rms = (_rng_ck or {}).get("pool_rms")
        if _rms:
            _PL_RMS[0] = float(_rms[0])
        print(f"[재개] 풀 {_nl} 개 상태 이어받음 (씬이 달라 버린 것 {_ns} 개), "
              f"폐기누적 {POOL.n_drop}, 유예누적 {POOL.n_keep}", flush=True)

ap_dummy = None
if a.oracle_roll:
    # ---- 오라클 롤아웃: **학습·평가 코드를 그대로 쓰고** 망 출력만 자유 변수로 --
    # fit_rollout.py 처럼 따로 짜면 손잡이 적용·바닥·질량·상태 전진 중 하나가
    # 어긋난다 (실측: 무게중심 z 가 0.942 -> 0.525 로 내려앉고 손잡이 추종이
    # 명령 0.725 대비 0.140 이었다). 여기서는 step_once 를 그대로 호출하고
    # _DP_HOOK 으로 격자점 변위만 갈아끼운다.
    import time as _time
    _tag, _d = TR[0]
    _t0 = a.eval_t0[0] if a.eval_t0 else 3
    _L = min(a.eval_len, _d["x"].shape[0] - _t0 - 1)
    _gsel = torch.arange(_d["x"].shape[1], device=dev)
    _mass = traj_mass(_d)[_gsel]
    _ext = _d.get("_ext", EXT)
    _cfg = _d["cfg"]
    _vol = _mass / float(_cfg["density"])
    _g = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
    _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
    _norm = float(_mass.sum()) * (_ext ** 2) / (FRAME_DT ** 2)
    x = take(_d["x"][_t0], _gsel)
    v = (x - take(_d["x"][max(_t0 - 1, 0)], _gsel)) / FRAME_DT
    p = x[fps(x, a.n_anchors, a.seed)] if a.refps else take(_d["x"][_t0], AIDX)
    F = take(traj_F(_d)[_t0], _gsel).float()
    x_still = x.clone()
    _CURVES = []
    import numpy as _np0
    _SNAP = set(int(q) for q in (a.oracle_snap or "").split(",") if q)
    PRED, GT = [], []
    errs, stills = [], []
    net.eval()
    print(f"[오라클] {_tag} t0={_t0} {_L} 프레임, 입자 {_gsel.numel()}, "
          f"스텝 {a.oracle_steps}", flush=True)
    _wall = _time.time()
    _K = max(int(a.oracle_sub), 1)
    _hs = FRAME_DT / _K
    # 예전에는 손잡이 명령을 _CTRL_SCALE 로 따로 줄였다. 이제 스텝 dt 자체를
    # 줄이면 d_cmd = dt * v_cmd 가 알아서 같은 값이 된다 (FRAME_DT*(1/K) == _hs).
    _CTRL_SCALE[0] = 1.0
    _DT[0] = _hs
    if _K > 1:
        print(f"[오라클] 프레임을 {_K} 서브스텝으로 나눈다 (h={_hs:.3e})",
              flush=True)
    for i in range(_L * _K):
        _fi = i // _K                     # 이 서브스텝이 속한 프레임
        _last = ((i + 1) % _K == 0)       # 프레임 끝인가 (기록·보고는 여기서만)
        _hold = {}

        def _hook(out, _h=_hold):
            # 출력 전체를 변수로 둔다. 초기값은 **그 시점 망의 출력** 이라
            # (학습 안 된 망이면 그 난수 출력) 학습이 출발하는 자리와 같다.
            if "z" not in _h:
                _h["z"] = [o.detach().clone().requires_grad_(True)
                           for o in (out if isinstance(out, (tuple, list))
                                     else (out,))]
                _h["n"] = len(_h["z"])
            return tuple(_h["z"]) if _h["n"] > 1 else _h["z"][0]

        _DP_HOOK[0] = _hook
        # 변수 모양·초기값을 만들기 위한 첫 호출 (망 출력을 그대로 받는다)
        step_once(_d, _t0 + i, _gsel, p, x, v, need_J=False)
        zs = _hold["z"]
        z = zs[0]
        opt = torch.optim.Adam(
            [{"params": [zs[0]], "lr": a.oracle_lr * float(_ext)}]
            + ([{"params": zs[1:], "lr": a.oracle_lr_shape}] if len(zs) > 1
               else []))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.oracle_steps)
        _best, _bz = float("inf"), [q.detach().clone() for q in zs]
        _curve = []
        for _s in range(a.oracle_steps):
            opt.zero_grad(set_to_none=True)
            x2, _p2, _v2, _J, _dp, _ai, _dm, _cr, _fe, _Jd = step_once(
                _d, _t0 + _fi, _gsel, p, x, v, need_J=False)
            fm = (free_mask(_d, x2.shape[0], dev, _gsel, x, _t0 + _fi)
                  if a.control else None)
            E, _dl, _Ft, _pt = ip_of(
                x, x2 - x, v, F, _mass, _vol, _cfg, _hs, _ng, _gl,
                g=_g, norm=_norm, free=fm, jac=_Jd)
            E.backward()
            opt.step()
            sch.step()
            if float(E) < _best:
                _best = float(E)
                _bz = [q.detach().clone() for q in zs]
            if a.oracle_curve and (_s % max(a.oracle_steps // 20, 1) == 0
                                   or _s == a.oracle_steps - 1):
                with torch.no_grad():
                    _gt = take(_d["x"][_t0 + i + 1], _gsel)
                    _fmc = (free_mask(_d, x2.shape[0], dev, _gsel, x, _t0 + i)
                            if a.control else slice(None))
                    _ec = (float((x2.detach()[_fmc] - _gt[_fmc]).norm(dim=-1)
                                 .mean()) / _ext)
                _curve.append((_s, float(E), _ec))
        with torch.no_grad():
            for _q, _b in zip(zs, _bz):
                _q.copy_(_b)
        if a.oracle_curve:
            _CURVES.append((i + 1, _curve))
            print(f"  [커브 {i + 1}] " + "  ".join(
                f"{q[0]}:E{q[1]:.2e}/오차{100 * q[2]:.2f}%"
                for q in _curve[::max(len(_curve) // 6, 1)]), flush=True)
        with torch.no_grad():
            x2, p2, v2, _J, _dp, _ai, _dm, _cr, _fe, _Jd = step_once(
                _d, _t0 + _fi, _gsel, p, x, v, need_J=False)
            fm = (free_mask(_d, x2.shape[0], dev, _gsel, x, _t0 + _fi)
                  if a.control else slice(None))
            gt = take(_d["x"][_t0 + _fi + 1], _gsel)
            e = float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / _ext
            st = float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / _ext
            _E2, dlog, F_tr, _pt = ip_of(
                x, x2 - x, v, F, _mass, _vol, _cfg, _hs, _ng, _gl,
                g=_g, norm=_norm, free=(fm if a.control else None), jac=_Jd)
            F = phys_resid.plastic_step(F_tr, dlog).detach()
            if _last:
                PRED.append(x2.detach().cpu()); GT.append(gt.detach().cpu())
                errs.append(e); stills.append(st)
            if _last and ((i + 1) // _K) in _SNAP:
                # 진행 중 스냅샷: 그 시점까지의 롤아웃을 따로 저장해 렌더한다
                _sp = (a.oracle_out or "oracle_roll.pt").replace(
                    ".pt", f"_f{i + 1:02d}.pt")
                torch.save({"pred": torch.stack(PRED), "gt": torch.stack(GT),
                            "x0": take(_d["x"][_t0], _gsel).cpu(),
                            "ctrl_pos": _d.get("ctrl_pos"), "t0": _t0,
                            "tag": _tag, "EXT": _ext}, _sp)
                print(f"  [스냅샷] {i + 1} 프레임 -> {_sp}  누적 비 "
                      f"{float(_np0.mean(errs)) / max(float(_np0.mean(stills)), 1e-12):.3f}",
                      flush=True)
            x, p, v = x2.detach(), p2.detach(), v2.detach()
        if _last:
            _e, _st = errs[-1], stills[-1]
            print(f"  프레임 {i // _K + 1:2d}  오라클 {100 * _e:.4f}%  정지 "
                  f"{100 * _st:.4f}%  비 {_e / max(_st, 1e-20):.3f}  "
                  f"E {_best:.4e}", flush=True)
    _DP_HOOK[0] = None
    import numpy as _np
    print(f"[오라클] {_L} 프레임 평균 {100 * _np.mean(errs):.3f}% "
          f"(정지 {100 * _np.mean(stills):.3f}%, 비 "
          f"{_np.mean(errs) / max(_np.mean(stills), 1e-12):.3f})  "
          f"{_time.time() - _wall:.1f}s", flush=True)
    if a.oracle_curve:
        _cd = (a.oracle_out or "oracle_roll.pt").replace(".pt", "_curve.pt")
        torch.save({"curves": _CURVES, "tag": _tag, "t0": _t0}, _cd)
        print(f"[커브] 저장 {_cd}", flush=True)
    _dst = a.oracle_out or "oracle_roll.pt"
    torch.save({"pred": torch.stack(PRED), "gt": torch.stack(GT),
                "x0": take(_d["x"][_t0], _gsel).cpu(),
                "ctrl_pos": _d.get("ctrl_pos"), "t0": _t0, "tag": _tag,
                "EXT": _ext}, _dst)
    print(f"[오라클] 저장 {_dst}", flush=True)
    raise SystemExit(0)

if a.roll_scen:
    # ---- 벤치 롤아웃: 시나리오 파일의 지령 속도로만 굴린다 -------------------
    # 교사 궤적을 읽지 않는다. 초기 상태는 씬의 정지 자세이고, 손잡이는 PG·i-PG
    # 와 **같은 파일**에서 읽은 지령 속도를 쓴다.
    import time as _time
    _sn = np.load(a.roll_scen)
    _hid_full = torch.as_tensor(_sn["hid"], dtype=torch.long, device=dev)
    _vel = torch.as_tensor(_sn["vel"], dtype=torch.float32, device=dev)
    _tag, _sc = POOL.scenes[0]
    ds = SCENE_DS[0]
    x = _sc["x0"].clone()
    n_p = x.shape[0]
    gsel = torch.arange(n_p, device=dev)
    v = torch.zeros_like(x)
    F = torch.eye(3, device=dev).expand(n_p, 3, 3).contiguous()
    # 시나리오의 제어 입자는 **전체 채우기** 색인이다. 부분표본에서 그 자리에
    # 가장 가까운 입자로 옮긴다 (부분표본이 2 만이면 옮김 거리가 셀보다 작다).
    _xfull = torch.from_numpy(np.load(
        os.path.join(os.environ.get("AF_WORK", "/root/work"),
                     f"pgfill_{_tag.split('_')[0]}.npy"))).float().to(dev)
    _cpos = _xfull[_hid_full]
    _d2 = torch.cdist(_cpos, x)
    _loc = _d2.argmin(1)
    print(f"[롤아웃] {_tag} 제어 입자 {_hid_full.tolist()} -> 부분표본 "
          f"{_loc.tolist()}, 옮김 거리 {float(_d2.min(1).values.max()):.5f}",
          flush=True)
    p_st = (x[fps(x, a.n_anchors, a.seed)] if a.arch == "attn"
            else x[torch.arange(0, n_p, max(1, n_p // a.n_anchors),
                                device=dev)[:a.n_anchors]])
    XS, FS = [x.detach().cpu().half()], [F.detach().cpu().half()]
    _t0 = _time.time()
    with torch.no_grad():
        _hr = FRAME_DT / EVAL_SUB
        for _f in range(_vel.shape[0]):
            _vc = _vel[_f]
            for _sb in range(EVAL_SUB):
                ds["ctrl_id"] = _loc.reshape(1, -1)
                ds["ctrl_vel"] = _vc.reshape(1, -1, 3)
                _cc = x[_loc]
                ds["ctrl_pos"] = torch.stack([_cc, _cc + _hr * _vc], 0)
                ds.pop("_ca20", None); ds.pop("_ca", None)
                ds.pop("_ca_key", None)
                # jac=auto 일 때만 autograd 야코비안 때문에 grad 를 켠다
                # (analytic 은 닫힌 형식이라 no_grad 로 충분하고 메모리도 준다)
                with dt_scope(_hr), (torch.enable_grad()
                                     if a.jac == "auto"
                                     else contextlib.nullcontext()):
                    (x2, p2, v2, _J, _dp, _ai, _dmg, _cr, _fe,
                     _Jd) = step_once(ds, 0, gsel, p_st, x, v, need_J=False)
                _fm = (free_mask(ds, n_p, dev, gsel, x, 0) if a.control
                       else None)
                _cfg = _sc["cfg"]
                _E, _dlog, _Ftr, _ = ip_of(
                    x, x2 - x, v, F, _sc["mass"],
                    _sc["mass"] / float(_cfg["density"]),
                    _cfg, _hr, int(_cfg["n_grid"]),
                    float(_cfg.get("grid_lim", 2.0)),
                    g=torch.tensor(_cfg["g"], device=dev), norm=1.0, free=_fm,
                    jac=_Jd)
                F = phys_resid.plastic_step(_Ftr, _dlog).detach()
                x, v = x2.detach(), v2.detach()
                p_st = p2.detach() if torch.is_tensor(p2) else p_st
            XS.append(x.cpu().half()); FS.append(F.cpu().half())
            if not bool(torch.isfinite(x).all()):
                print(f"[롤아웃] {_f} 프레임에서 비유한 -- 멈춘다", flush=True)
                break
    _wall = _time.time() - _t0
    _dst = a.roll_out or (a.roll_scen.replace(".npz", "_stu.pt"))
    torch.save(dict(x=torch.stack(XS), F=torch.stack(FS), cfg=_sc["cfg"],
                    n=n_p, wall_s=_wall, fps=(len(XS) - 1) / max(_wall, 1e-9)),
               _dst)
    print(f"[롤아웃] 저장 {_dst} {len(XS)}프레임, {_wall:.2f}s, "
          f"{(len(XS) - 1) / max(_wall, 1e-9):.2f} FPS", flush=True)
    raise SystemExit(0)

def _mk_ov_opt(vs):
    if a.ov_opt == "lbfgs":
        return torch.optim.LBFGS(vs, lr=1.0, max_iter=max(a.ov_roll, 1),
                                 history_size=50, tolerance_grad=0.0,
                                 tolerance_change=0.0,
                                 line_search_fn="strong_wolfe")
    return torch.optim.Adam(vs, lr=a.out_var_lr)


if a.out_var or a.ov_roll > 0:
    # **상태마다** 노드 출력을 직접 최적화한다. 망은 초기화에만 쓰인다.
    #
    # 예전에는 교사 프레임 번호로 변수를 키로 잡았다. 풀 경로에서는 t 가 항상
    # 0 이라 그 키가 무의미하고(모든 상태가 하나로 뭉갠다) 노드 집합이 상태의
    # 바운딩박스로 정해져 크기도 어긋난다. 그래서 변수를 **상태 사전에** 붙이고
    # (풀의 sample 이 슬롯을 재정렬하므로 슬롯 키는 불안정하다), out_var 일 때는
    # 상태를 전진시키지 않아 같은 상태를 계속 최적화하게 둔다.
    #
    # 재는 것: 이 매개화(노드 변위 + barycentric, 셀 내부 항등)가 한 스텝의
    # 증분 포텐셜을 얼마나 낮출 수 있는가 -- 즉 망이 아무리 좋아도 못 넘는 하한.
    def _ov_hook(out):
        st = _OV_CUR[0]
        outs = out if isinstance(out, (tuple, list)) else (out,)
        if st is None:                      # 풀 밖(초기화·검증) 에서는 그대로
            return out
        if st.get("ov") is None:
            st["ov"] = [o.detach().clone().requires_grad_(True) for o in outs]
            st["ov_opt"] = _mk_ov_opt(st["ov"])
            st["ov_tuple"] = isinstance(out, (tuple, list))
        # **입력과 같은 모양으로** 돌려준다. 출력이 하나일 때 맨 텐서를 주면
        # 호출부의 out[0] 이 첫 노드 행을 집어 색인이 범위를 벗어난다 (옛 conv
        # 경로는 출력이 3 개라 드러나지 않았다).
        return (tuple(st["ov"]) if isinstance(out, (tuple, list))
                else st["ov"][0])

    def _ov_fast(M=None, dtype=None, device=None):
        """망을 건너뛰고 자유변수를 바로 돌려준다 (없으면 만든다).

        --ov_init net 일 때만 망을 한 번 통과시켜 초기값으로 쓴다. zero/randn
        이면 망은 **한 번도 쓰이지 않는다** -- 재는 것은 이 매개화의 한계이지
        망의 학습 상태가 아니므로 그쪽이 기본이다.
        """
        st = _OV_CUR[0]
        if st is None:
            return None
        if st.get("ov") is None:
            if a.ov_init == "net" or M is None:
                return None                      # 망 출력으로 초기화한다
            # **손잡이 노드는 자유변수에서 아예 뺀다.** 값이 명령으로 덮이므로
            # 변수로 남겨두면 기울기가 0 인 채 L-BFGS 의 곡률쌍만 더럽힌다.
            _hm = _NOD_M[0]
            st["ov_idx"] = (None if _hm is None else
                            (~_hm).nonzero(as_tuple=True)[0])
            z = torch.zeros(M, 3, dtype=dtype, device=device)
            if a.ov_init in ("target", "predict"):
                if not torch.is_tensor(_OV_TGT[0]):
                    raise SystemExit(f"--ov_init {a.ov_init} 은 --ov_roll "
                                     "에서만 쓴다")
                z = z + _OV_TGT[0].to(dtype).reshape(-1, 3)
            if a.ov_init == "randn":
                z = a.ov_init_std * torch.randn(M, 3, dtype=dtype,
                                                device=device)
            st["ov_M"] = M
            if st["ov_idx"] is not None:
                z = z[st["ov_idx"]]
            st["ov"] = [z.requires_grad_(True)]
            st["ov_opt"] = _mk_ov_opt(st["ov"])
            # 호출부는 out[0] 만 읽는다 -- 늘 묶음으로 돌려주면 안전하다
            st["ov_tuple"] = True
        if st.get("ov_idx") is not None:
            _z = torch.zeros(st["ov_M"], 3, dtype=st["ov"][0].dtype,
                             device=st["ov"][0].device)
            return (_z.index_copy(0, st["ov_idx"], st["ov"][0]),)
        return tuple(st["ov"]) if st["ov_tuple"] else st["ov"][0]

    _DP_HOOK[0] = _ov_hook
    _DP_FAST[0] = _ov_fast
    print(f"[출력변수] 노드 출력을 직접 최적화한다 (lr {a.out_var_lr}, 초기값 "
          f"{a.ov_init}"
          + (f" std {a.ov_init_std:g}" if a.ov_init == "randn" else "")
          + ("). --ov_roll 이라 프레임마다 변수를 새로 만들고 상태는 정상 "
             f"전진한다 (프레임당 {a.ov_roll} 반복)" if a.ov_roll > 0 else
             "). 상태는 전진하지 않는다 -- 한 스텝 하한만 잰다"), flush=True)

TBW = None
if a.tb:
    from torch.utils.tensorboard import SummaryWriter
    # 재개하면 같은 폴더에 새 이벤트 파일이 생기는데, 이전 파일에 그 지점 이후
    # 기록이 남아 있으면 TensorBoard 가 두 곡선을 겹쳐 그려 스텝이 꼬인다.
    # purge_step 으로 재개 지점 이후를 버리게 한다.
    TBW = SummaryWriter(os.path.join(a.tb, a.tag),
                        purge_step=step0 if step0 else None)
    print(f"[TB] {os.path.join(a.tb, a.tag)}", flush=True)

if not a.pool and a.iters > 0:
    raise SystemExit("--no_pool 은 학습 경로가 없다. 교사 궤적 학습 경로는 "
                     "삭제됐다 (학습은 정지 상태에서 출발해 모델이 상태를 "
                     "갱신하는 상태 풀만 쓴다)")

pbar = tqdm(range(step0, a.iters), desc="학습", ncols=90)
for it in pbar:
    L = a.unroll            # 학습 중 펼치기 길이는 바꾸지 않는다
    opt.zero_grad(set_to_none=True)
    lx = lJ = la = ldm = ldet = 0.0
    still = arel = dmean = 0.0
    if a.pool:
        # ---- 상태 풀 한 스텝 ----------------------------------------------
        # 교사 궤적을 읽지 않는다. 풀에서 일부, 새 초기 상태 일부로 배치를
        # 짜고 한 스텝 전진시킨 뒤 그 물리 잔차로 갱신한다. 상태는 최근 창의
        # 평균 잔차가 문턱을 넘으면 버린다.
        n_fresh = max(1, int(round(a.batch * a.pool_fresh)))
        lres = 0.0
        picks = POOL.sample(a.batch - n_fresh, n_fresh)
        for kind, slot in picks:
            if kind == "pool" and POOL.items[slot] is None:
                continue          # 이 배치 안에서 이미 버려진 자리
            st = POOL.items[slot] if kind == "pool" else POOL.fresh()
            tag, sc = POOL.scenes[st["si"]]
            ds = SCENE_DS[st["si"]]
            x, v, F = st["x"], st["v"], st["F"]
            n_p = x.shape[0]
            gsel = torch.arange(n_p, device=dev)
            MASS, EXT, N_FULL = sc["mass"], sc["ext"], n_p
            plan = st["plan"]
            # 이번 상태를 밟을 dt. 풀은 교사 프레임에 묶이지 않으므로 배수가
            # 정수일 필요가 없다. 계획의 경과 시간은 **프레임 단위**로 세므로
            # 한 스텝이 1/sub 프레임만큼 흐른 것으로 넘긴다.
            _sub = sample_sub(gen)
            _hp = FRAME_DT / _sub
            _vcmd = plan.velocity(x, st["elapsed"])            # [K,3]
            ds["ctrl_id"] = plan.idx.reshape(1, -1)
            ds["ctrl_vel"] = _vcmd.reshape(1, -1, 3)
            # 손잡이 중심과 **다음 스텝** 위치. 이게 없으면 손잡이 특징이 통째로
            # 0 으로 들어가 학생이 무엇을 잡고 어디로 끄는지 모르게 된다.
            _cc = x[plan.idx]
            ds["ctrl_pos"] = torch.stack([_cc, _cc + _hp * _vcmd], 0)
            ds.pop("_ca20", None); ds.pop("_ca", None); ds.pop("_ca_key", None)
            p_st = (st["p"] if st["p"] is not None else
                    (x[fps(x, a.n_anchors, a.seed)] if a.arch == "attn"
                     else x[torch.arange(0, n_p,
                                         max(1, n_p // a.n_anchors),
                                         device=dev)[:a.n_anchors]]))
            _OV_CUR[0] = st if a.out_var else None
            with dt_scope(_hp), _tsec("step_once"):
                x2, p2, v2, _J, _dp, _ai, _dmg, _cr, _fe, _Jd = step_once(
                    ds, 0, gsel, p_st, x, v, need_J=False)
            with _tsec("free_mask"):
                fm = (free_mask(ds, n_p, dev, gsel, x, 0)
                      if a.control else None)
            # 구속 일치 항: 망 출력이 하드 사영값과 어긋난 만큼. 자유 입자는 0.
            # 하드로 박힌 입자는 증분 포텐셜에서 뺀다 (그 잔차는 반력이 실어
            # 나르는 것이라 학생이 정할 양이 아니다).
            _bc = torch.zeros((), device=dev)
            if _BC_LAST[0] is not None:
                _corr, _actf = _BC_LAST[0]
                _bc = (_corr * _corr).sum(-1).mean() / (sc["ext"] ** 2)
                _am = _actf.reshape(-1) > 0.5
                fm = (~_am) if fm is None else (fm & ~_am)
                with torch.no_grad():
                    _pen0 = 0.0
                    for _b in (sc["cfg"].get("boundary_conditions") or []):
                        if _b.get("type") != "surface_collider":
                            continue
                        _pt0 = torch.as_tensor(_b["point"], device=dev,
                                               dtype=x2.dtype)
                        _nr0 = torch.as_tensor(_b["normal"], device=dev,
                                               dtype=x2.dtype)
                        _nr0 = _nr0 / _nr0.norm().clamp_min(1e-12)
                        _sd0 = ((x2 - _pt0) * _nr0).sum(-1)
                        _pen0 = max(_pen0, float((-_sd0).clamp_min(0).max())
                                    / sc["ext"])
                    _BC_DIAG.append((
                        float(_actf.reshape(-1).gt(0.5).float().mean()),
                        float(_corr.norm(dim=-1).max()) / sc["ext"], _pen0,
                        _CTRL_ERR[0]))
                    if len(_BC_DIAG) > 200:
                        del _BC_DIAG[:-200]
            # det 가 문턱 아래인 사면체는 탄성에서 빼고 복구 손실을 크게 준다
            _emask, _dpen = det_take()
            cfg = sc["cfg"]
            vol = sc["mass"] / float(cfg["density"])
            gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
            ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
            if "_msum" not in sc:                 # 씬 상수 -- 한 번만 내린다
                sc["_msum"] = float(sc["mass"].sum())
            nrm = sc["_msum"] * (sc["ext"] ** 2) / (_hp ** 2)
            with _tsec("에너지"):
                E_ip, dlog, F_tr, _pt = ip_of(
                    x, x2 - x, v, F, sc["mass"], vol, cfg, _hp, ng_, gl_,
                    g=gv, norm=nrm, free=fm, jac=_Jd,
                    elastic_mask=_emask)
            # 폐기 판정용 길이 단위 잔차.
            if _OBJ_PTS:
                # 입자 목적함수에서는 dE/dx2 가 **탄성항을 담지 못한다** --
                # 탄성은 F=∇Φ·F 를 거치고 ∇Φ 는 x2 의 함수가 아니라 망 출력의
                # 함수다 (자유변수가 x2 가 아니라 망 출력이라 학습 기울기는
                # 온전하다). 그래서 폐기 판정은 닫힌 형식의 관성 잔차로 잰다:
                #   r = (Δu − h v − h² g) / ext   = (h²/m)·∂E_관성/∂x2 / ext
                _rr = (x2 - x - _hp * v - (_hp ** 2) * gv) / sc["ext"]
                if a.pool_loss == "residual":
                    raise SystemExit(
                        "--obj pts 에서는 --pool_loss residual 을 쓸 수 없다 "
                        "(잔차가 탄성항을 담지 못한다). --pool_loss energy 로.")
                loss_p = a.phys_w * E_ip + _dpen + a.lambda_bc * _bc
            else:
                # 손실이 E 값이든 잔차든 **같은 그래프에서 한 번만** dE/dx2 를
                # 뽑아 쓴다 (재평가 없음).
                _need_g = (a.pool_loss == "residual")
                _gx, = torch.autograd.grad(E_ip * nrm, x2, retain_graph=True,
                                           create_graph=_need_g)
                _rr = _gx * (_hp ** 2) / sc["mass"].unsqueeze(
                    -1).clamp_min(1e-20) / sc["ext"]
                if _need_g:
                    loss_p = (((_rr[fm] if fm is not None else _rr) ** 2
                               ).sum(-1).mean() + _dpen + a.lambda_bc * _bc)
                else:
                    loss_p = a.phys_w * E_ip + _dpen + a.lambda_bc * _bc
            # 스칼라들을 **한 번에** 내린다. float()/bool() 하나가 파이프라인을
            # 세우므로 배치 원소마다 3~4 회면 스텝당 30 회 가까이 멈춘다.
            _rl = _rr.detach().norm(dim=-1)
            _res_t = (_rl[fm] if fm is not None else _rl).mean()
            if FIXED_GRID is not None:
                _glo, _ghh, _gn3 = FIXED_GRID
                _ghi = _glo + _ghh * _gn3.to(x2.dtype)
                _oob_t = ((x2 < _glo) | (x2 > _ghi)).any().to(_res_t.dtype)
            else:
                _oob_t = torch.zeros_like(_res_t)
            res, _oob, _lv = torch.stack(
                [_res_t, _oob_t, loss_p.detach().to(_res_t.dtype)]).tolist()
            if a.pool_whiten:
                _PL_RMS[0] = 0.99 * _PL_RMS[0] + 0.01 * _lv ** 2
                loss_p = loss_p / max(_PL_RMS[0] ** 0.5, 1e-12)
            if math.isfinite(_lv):
                with _tsec("역전파"):
                    (loss_p / a.batch).backward()
                if a.out_var and st.get("ov_opt") is not None:
                    # 망 파라미터가 아니라 **이 상태의 출력 변수**를 갱신한다
                    st["ov_opt"].step()
                    st["ov_opt"].zero_grad(set_to_none=True)
            elif _PL_MSG == []:
                _PL_MSG.append(1)
                print(f"[풀] 씬 {tag} 손실이 비유한 -- 이 배치는 건너뛴다",
                      flush=True)
            lx = lx + _lv / a.batch
            lres = lres + res / a.batch
            if _oob > 0.5:
                res = float("inf")            # 영역을 벗어난 상태는 버린다
            # 문턱을 넘었을 때 되돌릴 **전진 전** 사본. 이번 스텝을 없던 일로
            # 하려면 프레임 색인과 잔차 이력까지 그대로여야 한다.
            _prev = (dict(si=st["si"], x=x, v=v, F=F, p=st["p"], plan=plan,
                          elapsed=st["elapsed"], hist=list(st["hist"]),
                          age=st["age"]) if a.pool_keep > 0 else None)
            with torch.no_grad(), _tsec("소성·풀"):
                if not a.out_var:
                    st["x"] = x2.detach()
                    st["v"] = v2.detach()
                    st["F"] = phys_resid.plastic_step(F_tr, dlog).detach()
                    st["p"] = p2.detach() if torch.is_tensor(p2) else None
                # out_var 는 상태를 **전진시키지 않는다** -- 같은 상태의 노드
                # 출력을 계속 최적화해 하한까지 내린다 (노드 집합도 고정된다)
            POOL.put_back(slot if kind == "pool" else None, st, res,
                          prev=_prev, dt_frac=1.0 / _sub)
        # 폐기율은 **최근 100 반복에서 배치 대비 몇 %가 죽었는가** 로 둔다.
        # 누적 수를 전체 반복으로 나누면 추세가 안 보이고 값도 오해를 부른다.
        _PL_DROP.append(POOL.n_drop)
        if len(_PL_DROP) > 101:
            _PL_DROP.pop(0)
        still = ((_PL_DROP[-1] - _PL_DROP[0])
                 / max(len(_PL_DROP) - 1, 1) / a.batch)
        # 버린 자리는 None 으로 비워 둔다 -- 통계에서 걸러야 한다
        _alive = [q for q in POOL.items if q is not None]
        _ages = np.asarray([q["age"] for q in _alive] or [0.0],
                           dtype=np.float64)
        _res = np.asarray([float(np.mean(q["hist"])) if q["hist"] else 0.0
                           for q in _alive] or [0.0], dtype=np.float64)
        arel = float(_ages.mean())
        # 기울기가 망가진 배치는 갱신을 건너뛴다 (풀 상태는 이미 전진했다)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if bool(torch.isfinite(gn)):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            POOL.n_nan = getattr(POOL, "n_nan", 0) + 1
        if it % 20 == 0:
            pbar.set_postfix(E=f"{lx:.3e}", 잔차=f"{lres:.3e}",
                             폐기=f"{POOL.n_drop}",
                             프레임=f"{arel:.1f}", gn=f"{float(gn):.1e}")
            if TBW is not None:
                TBW.add_scalar("학습/목적함수", lx, it)
                TBW.add_scalar("학습/잔차", lres, it)
                TBW.add_scalar("풀/폐기누적", POOL.n_drop, it)
                TBW.add_scalar("풀/폐기율", still, it)
                if a.pool_keep > 0:
                    TBW.add_scalar("풀/유예누적", POOL.n_keep, it)
                TBW.add_scalar("풀/NaN배치", getattr(POOL, "n_nan", 0), it)
                TBW.add_scalar("풀/채움", len(_alive) / POOL.size, it)
                # 나이·프레임·누적잔차의 분포
                # 전진 프레임 = 초기 상태에서 지금까지 굴러온 프레임 수.
                # 계획 안 시점(0~59)은 규칙적으로 돌아 정보가 없어 빼둔다.
                TBW.add_scalar("풀/전진프레임_평균", arel, it)
                TBW.add_scalar("풀/전진프레임_중앙", float(np.median(_ages)), it)
                TBW.add_scalar("풀/전진프레임_최대", float(_ages.max()), it)
                TBW.add_scalar("풀/누적잔차_중앙", float(np.median(_res)), it)
                TBW.add_scalar("풀/누적잔차_상위10%",
                               float(np.quantile(_res, 0.9)), it)
                if it % 200 == 0:
                    TBW.add_histogram("풀분포/전진프레임", _ages, it)
                    TBW.add_histogram("풀분포/누적잔차", _res, it)
                    if len(POOL.scenes) > 1:
                        _si = np.asarray([q["si"] for q in _alive] or [0])
                        TBW.add_histogram("풀분포/씬", _si, it)
        if a.val_every and (it + 1) % a.val_every == 0:
            _v, _vo = quick_val()
            if TBW is not None:
                TBW.add_scalar("검증/비", _v, it)
            if _v < _best:
                _best = _v
                save_ck("best", it + 1)
                print(f"  [검증 {it+1}] 비 {_v:.4f} -- best 갱신", flush=True)
            else:
                print(f"  [검증 {it+1}] 비 {_v:.4f} (best {_best:.4f})",
                      flush=True)
        if (it + 1) % a.save_every == 0 or it == a.iters - 1:
            save_ck("last", it + 1)
        if _PROF_ON and it > 0 and it % 10 == 0:
            _tot = sum(_PROF.values())
            _msg = "  ".join(
                f"{k} {v/it*1000:.1f}ms({100*v/max(_tot,1e-9):.0f}%)"
                for k, v in sorted(_PROF.items(), key=lambda z: -z[1]))
            print(f"[구간 {it}] 합 {_tot/it*1000:.0f}ms/it  {_msg}", flush=True)
        # det·구속 진단도 여기서 낸다. 아래 공통 자리에 두면 이 continue 뒤라
        # 풀 경로에서 영원히 도달하지 않는다 (프로파일 출력도 같은 함정이었다).
        if a.det_every and (it + 1) % a.det_every == 0:
            if _DET_BAD:
                print(f"  [det {it+1}] {det_report()}", flush=True)
            _br = bc_report()
            if _br:
                print(f"  [구속 {it+1}] {_br}", flush=True)
            if TBW is not None and _DET_BAD:
                TBW.add_scalar("det/무효비율",
                               sum(_DET_BAD) / len(_DET_BAD), it)
        continue
    if a.rl and OPT_C is not None:
        torch.nn.utils.clip_grad_norm_(CRITIC.parameters(), 1.0)
        OPT_C.step()
        OPT_C.zero_grad(set_to_none=True)
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
    if not a.out_var:
        opt.step()
        if SCHED is not None:
            SCHED.step()
    if os.environ.get("AF_DIAG") and it < int(os.environ["AF_DIAG"]):
        _d = [(n, float((q.detach() - _prev[n]).norm())) for n, q in
              net.named_parameters()]
        print(f"    변화 합 {sum(v for _, v in _d):.3e}  "
              f"상위 {[(n.split('.')[-2:], round(v, 8)) for n, v in sorted(_d, key=lambda z: -z[1])[:3]]}",
              flush=True)
    hist.append((lx, lJ, still, la, arel, ldm, dmean))
    if TBW is not None and it % 20 == 0:
        if a.rl:
            TBW.add_scalar("rl/즉시비용", lx, it)
            TBW.add_scalar("rl/정책손실", la, it)
            TBW.add_scalar("rl/크리틱손실", still, it)
            TBW.add_scalar("rl/에피소드길이", arel, it)
        elif a.phase2:
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
        if a.rl:
            pbar.set_postfix(비용=f"{lx:.3e}", 정책=f"{la:.3e}",
                             크리틱=f"{still:.3e}",
                             스텝=f"{arel:.1f}", gn=f"{float(gn):.1e}")
        elif a.phase2:
            pbar.set_postfix(E=f"{lx:.3e}", 자유잔차=f"{100*arel:.3f}%",
                             구속잔차=f"{100*still:.3f}%",
                             무효=(f"{100*sum(_DET_BAD)/len(_DET_BAD):.2f}%"
                                 if _DET_BAD else "-"),
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
    if a.det_every and (it + 1) % a.det_every == 0 and _DET_BAD:
        # 진행바는 폭에 잘려 못 믿는다 -- det 통계를 로그로 남긴다
        print(f"  [det {it+1}] {det_report()}", flush=True)
        _br = bc_report()
        if _br:
            print(f"  [구속 {it+1}] {_br}", flush=True)
        if TBW is not None:
            TBW.add_scalar("det/무효비율", sum(_DET_BAD) / len(_DET_BAD), it)
            TBW.add_scalar("det/최소", min(_DET_MIN), it)
            TBW.add_scalar("det/중앙", sum(_DET_MED) / len(_DET_MED), it)
    if a.val_every and ((it + 1) % a.val_every == 0 or it == a.iters - 1):
        _v, _vo = quick_val()
        if TBW is not None:
            TBW.add_scalar("검증/비", _v, it)
            TBW.add_scalar("목적함수/검증", _vo, it)
        if _v < _best:
            _best = _v
            save_ck("best", it + 1)
            print(f"  [검증 {it+1}] 비 {_v:.4f} 목적 {_vo:.3e} -- best 갱신"
                  + (f"  (관성 {float(_pt[0]):.3e} 탄성 {float(_pt[1]):.3e} "
                     f"중력 {float(_pt[2]):.3e})"
                     if a.phase2 and _pt is not None else ""), flush=True)
        else:
            print(f"  [검증 {it+1}] 비 {_v:.4f} 목적 {_vo:.3e} "
                  f"(best {_best:.4f})", flush=True)
    if (it + 1) % a.save_every == 0 or it == a.iters - 1:
        save_ck("last", it + 1)

# ---------------------------------------------------------------- 평가
_ROLLDUMP = None


def _ov_frame(d, t, gsel, p, x, v, F):
    """프레임 하나를 **노드 출력 직접 최적화**로 전진시킨다 -> (x2, F_next).

    --out_var 는 상태를 전진시키지 않아 한 스텝 하한만 잰다. 그 값으로 롤아웃을
    말할 수 없다 (평가 중에는 훅이 끊겨 **초기 망**이 굴러간다). 여기서는 매
    프레임 변수를 새로 만들어 망 출력에서 시작해 증분 포텐셜을 내리고, 내려간
    출력으로 상태를 전진시킨다 -- 이것이 롤아웃에서 성립하는 하한이다.
    """
    cfg = d["cfg"]
    m = MASS[gsel]
    vol = m / float(cfg["density"])
    gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
    nrm = float(m.sum()) * (EXT ** 2) / (FRAME_DT ** 2)
    st, E0, E1 = {}, None, None
    _E0, _tr = [None], []
    _trp = []                  # 항별 곡선 [(관성, 탄성, 접촉), ...]

    def _fwd():
        with torch.enable_grad(), dt_scope(FRAME_DT):
            x2, _, v2, _, _, _, _, _, _, _Jd = step_once(
                d, t, gsel, p, x, v, need_J=False)
        fm = (free_mask(d, x2.shape[0], x2.device, gsel, x, t)
              if a.control else None)
        if _BC_LAST[0] is not None:
            _am = _BC_LAST[0][1].reshape(-1) > 0.5
            fm = (~_am) if fm is None else (fm & ~_am)
        E, dlog, F_tr, _pt = ip_of(x, x2 - x, v, F, m, vol, cfg, FRAME_DT,
                                   ng_, gl_, g=gv, norm=nrm, free=fm, jac=_Jd)
        return x2, v2, E, dlog, F_tr, _pt, _Jd

    _ov = _OV_CUR[0]
    _OV_CUR[0] = st
    _OV_TGT[0] = ("want" if a.ov_init == "predict" else
                  (FRAME_DT * v.mean(0) + (FRAME_DT ** 2) * gv).detach())
    _NF_HOLD[0] = "want"          # 이 프레임의 격자·간선·노드특징을 한 번만
    try:
      with torch.enable_grad():
        if a.ov_opt == "lbfgs":
            _n = [0]

            def _closure():
                st["ov_opt"].zero_grad(set_to_none=True)
                _, _, E, _, _, _pq, _ = _fwd()
                if _n[0] == 0:
                    _E0[0] = float(E)
                _n[0] += 1
                if _CURVE is not None:
                    _tr.append(float(E))
                    _trp.append((float(_pq[0]), float(_pq[1]),
                                 float(_pq[3])))
                E.backward()
                return E
            _fwd()                    # 변수를 만든다 (훅이 첫 호출에서 만든다)
            if st.get("ov_opt") is None:
                raise SystemExit("--ov_roll: 출력 훅이 변수를 못 만들었다")
            st["ov_opt"].step(_closure)
            E0 = _E0[0]
        else:
            for _k in range(a.ov_roll):
                _, _, E, _, _, _pt, _ = _fwd()
                if st.get("ov_opt") is None:
                    raise SystemExit("--ov_roll: 출력 훅이 변수를 못 만들었다")
                if E0 is None:
                    E0 = float(E)
                if _CURVE is not None:
                    _tr.append(float(E))
                    _trp.append((float(_pt[0]), float(_pt[1]), float(_pt[3])))
                st["ov_opt"].zero_grad(set_to_none=True)
                E.backward()
                st["ov_opt"].step()
        x2, v2, E, dlog, F_tr, _pt, _Jd = _fwd()
        E1 = float(E)
        if os.environ.get("AF_OV_DIAG"):
            _tg = (FRAME_DT * v + (FRAME_DT ** 2) * gv)
            _du = x2 - x
            print(f"    [항] t={t} 관성 {float(_pt[0]):.4e} 탄성 "
                  f"{float(_pt[1]):.4e} 접촉 {float(_pt[3]):.4e} | "
                  f"dz 평균 {float(_du[:, 2].mean()):+.3e} 목표 "
                  f"{float(_tg[:, 2].mean() if _tg.dim() > 1 else _tg[2]):+.3e}"
                  f" | detJ {float(torch.linalg.det(_Jd).mean()):.4f} "
                  f"F {float(F.reshape(-1, 9).abs().max()):.3f}", flush=True)
    finally:
        _OV_CUR[0] = _ov
        _NF_HOLD[0] = None
        _OV_TGT[0] = None
    _OV_LOG.append((E0 if E0 is not None else E1, E1))
    if _CURVE is not None:
        _CURVE.append((int(t), _tr, _trp))
    with torch.no_grad():
        F_next = phys_resid.plastic_step(F_tr, dlog).detach()
    return x2.detach(), v2.detach(), F_next


@torch.no_grad()        # <- _ov_frame 을 위에 끼우면서 이 데코레이터를 빼앗았다.
def rollout(d, t0, L, gsel=None):   #    평가가 그래프를 쌓아 메모리가 터진다.
    with traj_scope(d):
        return _rollout(d, t0, L, gsel)


def _rollout(d, t0, L, gsel=None):
    # 드롭아웃이 켜져 있으면 롤아웃이 확률적이 된다 -- 평가는 항상 eval 로.
    net.eval()
    if gsel is None:
        gsel = torch.arange(d["x"].shape[1], device=dev)
    x = take(d["x"][t0], gsel)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel)) / FRAME_DT
    p = x[fps(x, a.n_anchors, a.seed)] if a.refps else take(d["x"][t0], AIDX)
    errs, stills = [], []
    x_still = x.clone()
    x0e, p0e = x.clone(), p.clone()
    dmg_e, idx_e = None, None
    fe_r = (take(traj_F(d)[t0], gsel).float() if a.fe_state else None)
    cds, ems, cds_s, ems_s = [], [], [], []
    _RO_CTRL = []
    _NODE_CAP[0] = _ROLLDUMP is not None
    F_ov = take(traj_F(d)[t0], gsel).float() if a.ov_roll > 0 else None
    for i in range(L):
        if a.ov_roll > 0:
            # 출력만 최적화: 서브스텝·jvp 없이 프레임마다 변수를 내린다
            x2, v, F_ov = _ov_frame(d, t0 + i, gsel, p, x, v, F_ov)
        else:
            with torch.enable_grad(), dt_scope(FRAME_DT / EVAL_SUB):
                for _sb in range(EVAL_SUB - 1):
                    x, p, v, _, _, _, dmg_e, idx_e, fe_r, _ = step_once(
                        d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                        idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
                    x, p, v = x.detach(), p.detach(), v.detach()
                x2, p, v, _, _, _, dmg_e, idx_e, fe_r, _ = step_once(
                    d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                    idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
            x2 = x2.detach(); p = p.detach(); v = v.detach()
        gt = take(d["x"][t0 + i + 1], gsel)
        # 평가에서도 손잡이가 명령대로 끌리는지 본다. 학습 경로만 보고 "손잡이는
        # 정확하다" 고 판단했다가, 평가에서는 ctrl_id 가 저장 표본에 없어 대체
        # 입자가 중심이 되는 것을 놓쳤다.
        _RO_CTRL.append((_CTRL_ERR[0], _CTRL_ERR[1], _CTRL_ERR[2]))
        fm = free_mask(d, x2.shape[0], x2.device, gsel, x2, t0 + i) if a.control else slice(None)
        errs.append(float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / EXT)
        stills.append(float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / EXT)
        if _ROLLDUMP is not None:
            _ROLLDUMP.append((x2.detach().cpu(), gt.detach().cpu()))
            if _NODE_LAST[0] is not None:
                _NODEDUMP.append(_NODE_LAST[0])
            if F_ov is not None:
                # 입자별 변형 스칼라. C = FᵀF (우 코시-그린), E = (C−I)/2
                # (그린-라그랑주). 정지 상태면 det 1, trC 3, ‖E‖ 0 이다.
                _C = F_ov.transpose(-1, -2) @ F_ov
                _E = 0.5 * (_C - torch.eye(3, device=_C.device,
                                           dtype=_C.dtype))
                _DETDUMP.append(torch.stack([
                    torch.linalg.det(F_ov),
                    _C.diagonal(dim1=-2, dim2=-1).sum(-1),
                    _E.reshape(_E.shape[0], -1).norm(dim=-1),
                ], -1).detach().cpu())
        if a.metrics:
            cds.append(chamfer(x2, gt) / (EXT ** 2))
            ems.append(emd(x2, gt, t0 * 1000 + i) / EXT)
            cds_s.append(chamfer(x_still, gt) / (EXT ** 2))
            ems_s.append(emd(x_still, gt, t0 * 1000 + i) / EXT)
        x = x2
    net.train()
    _NODE_CAP[0] = False
    if not errs:
        raise SystemExit("롤아웃이 한 프레임도 재지 못했다 -- 분기 구조를 "
                         "확인할 것 (nan 을 평균해 비를 내지 않는다)")
    if _CURVE:
        _cp = os.environ["AF_OV_CURVE"]
        with open(_cp, "w") as _f:
            json.dump({"tag": d.get("tag", ""), "t0": t0,
                       "frames": [{"t": _t, "E": _e,
                                   "in": [q[0] for q in _p],
                                   "el": [q[1] for q in _p],
                                   "bc": [q[2] for q in _p]}
                                  for _t, _e, _p in _CURVE]},
                      _f)
        print(f"  [곡선] {_cp}  프레임 {len(_CURVE)}", flush=True)
        del _CURVE[:]
    if _OV_LOG:
        _b = sum(q[0] for q in _OV_LOG) / len(_OV_LOG)
        _af = sum(q[1] for q in _OV_LOG) / len(_OV_LOG)
        print(f"  [출력만최적화] 프레임 {len(_OV_LOG)}, 증분포텐셜 평균 "
              f"{_b:.4e} -> {_af:.4e} ({100*(1-_af/max(_b,1e-30)):.1f}% 감소, "
              f"프레임당 {a.ov_roll} 반복)", flush=True)
        del _OV_LOG[:]
    if _RO_CTRL and a.control:
        _er = max(q[0] for q in _RO_CTRL)
        _nn = [q[1] for q in _RO_CTRL]
        _cm = sum(q[2] for q in _RO_CTRL) / len(_RO_CTRL)
        print(f"  [롤아웃 손잡이] 오차최대 {_er:.2e}  구속입자 "
              f"{min(_nn)}~{max(_nn)}  명령변위평균 {_cm:.4f}", flush=True)
    return errs, stills, cds, ems, cds_s, ems_s


def chamfer(p_, q_, chunk=4096):
    """양방향 평균 최근접 **제곱** 거리의 합 (Spring-Gaus 관례)."""
    def one(u, w_):
        s_, n_ = 0.0, u.shape[0]
        for i_ in range(0, n_, chunk):
            # 기본 cdist 는 |a|^2-2ab+|b|^2 행렬곱 꼴이라 가까운 점에서 자릿수가
            # 날아간다 (같은 점집합에서도 5e-05 가 남는다). 정확한 경로를 쓴다.
            dd = torch.cdist(u[i_:i_ + chunk], w_,
                             compute_mode="donot_use_mm_for_euclid_dist")
            s_ += float((dd.min(1).values ** 2).sum())
        return s_ / n_
    return one(p_, q_) + one(q_, p_)


def emd(p_, q_, seed):
    """최적 일대일 대응의 평균 이동량. 두 구름에서 **같은 인덱스**를 뽑는다."""
    from scipy.optimize import linear_sum_assignment
    g_ = torch.Generator().manual_seed(int(seed))
    i_ = torch.randperm(p_.shape[0], generator=g_)[:a.cd_pts].to(p_.device)
    dd = torch.cdist(p_[i_], q_[i_],
                     compute_mode="donot_use_mm_for_euclid_dist"
                     ).double().cpu().numpy()
    r_, c_ = linear_sum_assignment(dd)
    return float(dd[r_, c_].mean())


rows = {}
for tag, d in TR + held:
    T = d["x"].shape[0]
    rows[tag] = dict(held=(tag in hold), windows={})
    for t0 in a.eval_t0:
        L = min(a.eval_len, T - t0 - 1)
        if L < 1:                     # 한 스텝(L=1)도 재게 둔다 -- 롤아웃 누적을
            continue                  # 빼고 순수한 한 스텝 오차를 보려면 필요하다
        _rd = os.environ.get("AF_ROLL_DUMP")
        _want = (_rd and tag.endswith(os.environ.get("AF_ROLL_TAG", "")) 
                 and t0 == int(os.environ.get("AF_ROLL_T0", "3")))
        _ROLLDUMP = [] if _want else None
        _NODEDUMP, _DETDUMP = [], []
        # gsel 은 **그 궤적의** 입자 수로 만든다 (궤적마다 다를 수 있고, 풀
        # 루프가 전역 N_FULL 을 씬 값으로 덮어써 전역을 믿을 수 없다).
        e, st, cd, em, cds_, ems_ = rollout(d, t0, L)
        if _want and _ROLLDUMP:
            torch.save({"pred": torch.stack([a_ for a_, _ in _ROLLDUMP]),
                        "gt": torch.stack([b_ for _, b_ in _ROLLDUMP]),
                        "x0": d["x"][t0].to(dev).float().cpu(),
                        "ctrl_pos": d.get("ctrl_pos"),
                        # 반경을 안 담으면 렌더가 기본값 0.15 로 원을 그려
                        # **실제 손잡이와 다른 크기**가 영상에 나온다
                        "ctrl_R": d.get("ctrl_R"),
                        "t0": t0, "tag": tag,
                        "nodes": _NODEDUMP or None,
                        # [T,N,3] = (det F, tr C, ‖E‖_F)
                        "fscal": (torch.stack(_DETDUMP) if _DETDUMP else None),
                        # 풀 루프가 전역 EXT 를 씬 값으로 덮어쓰므로 **그 궤적의**
                        # 것을 쓴다 (덤프를 렌더할 때 길이 단위가 된다)
                        "EXT": float(d.get("_ext", EXT))}, _rd)
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
