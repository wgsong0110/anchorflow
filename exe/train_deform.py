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
import subprocess
import threading
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
ap.add_argument("--elast", choices=["full", "lin"], default="full",
                help="탄성항. full=고정 공회전 Psi 그대로(비선형), "
                     "lin=**현재 상태 둘레 2차 근사** "
                     "<tau, dJ> + mu|sym dJ|² + (lam/2)tr(dJ)² -- 이때 전체 "
                     "목적함수가 노드 변위의 2차형식이라 최적해가 선형계 한 번")
ap.add_argument("--cg_iters", type=int, default=200,
                help="--ov_opt cg 의 공액기울기 반복 수")
ap.add_argument("--ctrl_barrier", type=float, default=0.0,
                help="손잡이 구에 **비침투 장벽**을 건다 (0 이면 끔). 초기에 "
                     "구 안이던 입자는 제외하고, 밖이던 입자가 구 안으로 못 "
                     "들어오게 막는다. 값은 kappa (바닥 장벽과 같은 척도)")
ap.add_argument("--ctrl_barrier_dhat", type=float, default=0.0,
                help="손잡이 장벽이 켜지는 거리. 0 이면 반경의 1/4")
ap.add_argument("--bc_mpm", action="store_true",
                help="**접선 처리를 MPM 과 맞춘다**: sticky 접선 구속을 면에서 "
                     "dx 안 띠가 아니라 **면 아래** 입자에만 건다 (MPM 은 면 "
                     "아래 격자점만 묶는다). 비관통은 그대로 둔다. "
                     "--bc_level node 면 노드 속도를 세 성분 0 으로도 바꾼다")
ap.add_argument("--bc_ipg", action="store_true",
                help="**i-PG(MPM) 와 같은 바닥**: 장벽·접선 벌점 없이, 격자점(노드) 위치가 면 "
                     "아래인 노드의 이번 스텝 변위를 세 성분 모두 0 으로 박는다 (collide 커널 "
                     "surface_type 0 과 같은 판정). --bc_mode project --bc_level node 와 함께, "
                     "--bc_mpm·--bc_soft 없이 쓴다")
ap.add_argument("--bc_softplus", type=float, default=0.0,
                help="**softplus 바닥 매핑** (0 이면 끔): 노드는 사영하지 않고, g2p 로 구한 입자 위치의 바닥 법선 "
                     "좌표 s 를 s'=ε·log(1+exp(s/ε)) 로 옮기고 F 에 그 야코비안 I+(σ(s/ε)-1)nnᵀ 를 곱한다. 값이 ε. "
                     "--bc_mode project --bc_level node 와 함께 (--bc_ipg/--bc_mpm/--bc_soft 없이) -- 장벽·접선 벌점 없음")
ap.add_argument("--bc_stick", type=float, default=0.0,
                help="**미끄러짐 방지 매핑** (0 이면 끔, --bc_softplus 와 함께): 스텝 시작 x_n, g2p 결과 x̃ 에서 "
                     "Δ=x̃-x_n 의 접선 성분을 w=σ(n·(x̃-p)/ε) 배로 줄이고(법선 성분 그대로) 그 뒤 softplus. "
                     "F 에 접선 축소의 ∂x/∂x̃ (w 의 s̃ 의존 항 포함) 를 곱한다. 값이 ε. 증분 포텐셜은 그대로")
ap.add_argument("--bc_stick_xn", action="store_true",
                help="--bc_stick 의 F 에 x_n 명시적 의존 항 (1-w)(I-nnᵀ) 를 더한다 (스텝 사상의 x_n 전미분). 기본 끔")
ap.add_argument("--bc_slip", action="store_true",
                help="바닥의 **접선 고정(no-slip) 항을 끈다**. 비관통만 남고 "
                     "닿은 입자가 면을 따라 미끄러진다")
ap.add_argument("--smooth_nb", type=int, default=0, choices=[0, 6, 12, 14],
                help="변형구배를 만들 때 쓸 **노드 변위 스무딩**의 이웃 수. "
                     "6=격자축만, 12=축+면대각, 14=몸대각까지. 위치는 스무딩하지 "
                     "않은 변위로 정하고, J 만 스무딩한 변위로 만든다")
ap.add_argument("--smooth_a", type=float, default=0.5,
                help="스무딩 세기. d~ = (1-a)·d + a·(이웃 평균)")
ap.add_argument("--ens_off", type=int, default=0,
                help="**오프셋이 다른 격자로 각각 따로 최적화**해 입자 변위를 "
                     "평균한다 (0/1 이면 끈다). 오프셋마다 빈 셀·고아 셀 패턴이 "
                     "달라 그 격자 의존적 결함이 평균에서 상쇄된다. 비용은 "
                     "오프셋 수만큼 곱절이다")
ap.add_argument("--bc_mode", choices=["project", "barrier"], default="project",
                help="바닥을 어떻게 세우는지. project=하드 사영(기본), "
                     "barrier=**IPC 꼴 장벽 에너지** b(d)=-(d-d̂)²ln(d/d̂) 를 "
                     "증분 포텐셜에 더한다 (사영을 끈다 -- 관통이 원천적으로 "
                     "불가능하고 죽은 기울기가 없다)")
ap.add_argument("--bc_kappa", type=float, default=1.0)
ap.add_argument("--inv_barrier", type=float, default=0.0,
                help="det F 뒤집힘 장벽의 세기 (0 이면 끔). 에너지밀도에 "
                     "kappa*mu*b(det F) 를 더한다")
ap.add_argument("--inv_jhat", type=float, default=0.3,
                help="그 장벽이 작동하기 시작하는 det F")
ap.add_argument("--inv_ext", choices=["quad", "linear"], default="linear",
                help="det <= 0 쪽 연장 방식")
ap.add_argument("--ov_tol", type=float, default=0.0,
                help="L-BFGS 수렴 허용오차 (기울기·변화량). 0 이면 항상 "
                     "--ov_roll 회를 다 돈다 (예전 기본값)")
ap.add_argument("--fuse", action="store_true",
                help="셀 다항식 평가를 torch.compile 로 묶어 커널 발사 수를 "
                     "줄인다 (작은 씬에서 지배적인 비용이다)")
ap.add_argument("--fused", action="store_true",
                help="증분 포텐셜 평가(에너지+기울기)를 **커널 한 번**으로 한다 "
                     "(근사 없음). 탄성·관성·바닥 장벽을 한 커널에서 푼다")
ap.add_argument("--cmp_lin", action="store_true",
                help="매 프레임 **같은 상태에서** 원본(근사 없음)과 2차 근사를 "
                     "각각 풀어 차이를 잰다. 전진은 원본 해로 한다")
ap.add_argument("--lin_outer", type=int, default=1,
                help="2차 근사를 **현재 상태에서 다시 전개**하는 바깥 반복 수. "
                     "1 이면 J=I 둘레 한 번만 (한 프레임 변위가 크면 깨진다)")
ap.add_argument("--cell_poly", action="store_true",
                help="입자 기여를 그 사면체 네 꼭짓점 변위(12 자유도)의 "
                     "다항식으로 바꿔 **셀마다 계수를 합산**해 둔다. 평가가 "
                     "입자 수가 아니라 셀 수에 비례한다 (--elast lin 필요)")
ap.add_argument("--lin_psd", action="store_true",
                help="선형(2차) 근사의 입자별 헤시안을 고유값 0 으로 잘라 "
                     "양반정부호로 만든다 (CG 가 음곡률에서 멈추지 않게)")
ap.add_argument("--cell_barrier", type=float, default=0.0,
                help="이번 스텝 셀 사상 det(grad Phi) 의 뒤집힘 장벽 세기")
ap.add_argument("--cell_jhat", type=float, default=0.3)
ap.add_argument("--cell_ext", choices=["quad", "linear"], default="linear")
ap.add_argument("--hex_hg", type=float, default=0.0,
                help="육면체 모래시계 안정화 세기 (0 이면 끔). 셀마다 "
                     "영에너지 모드 네 개의 진폭에 0.5*k*mu*V*(q/h)^2 을 "
                     "더한다")
ap.add_argument("--hex_jac", action="store_true",
                help="육면체 셀의 **삼선형** 사상으로 입자 변위와 야코비안을 "
                     "함께 만든다 (사면체 중심좌표 대신)")
ap.add_argument("--snap_inv", type=int, default=0,
                help="최적화 뒤 det<0 인 사면체를 det>=snap_eps 로 되돌리는 "
                     "**최소 이동** 사영을 이 횟수만큼 돌린다")
ap.add_argument("--snap_eps", type=float, default=0.05,
                help="사영이 목표로 하는 det 하한")
ap.add_argument("--ov_clip", type=float, default=0.0,
                help="출력만 최적화에서 변수 기울기의 전체 노름을 이 값으로 "
                     "자른다 (0 이면 안 자른다). 장벽이 터질 때 한 스텝이 "
                     "통째로 날아가는 것을 막는다")
ap.add_argument("--bc_ext", choices=["quad", "linear"], default="quad",
                help="장벽의 d<=eps 연장. quad=값·기울기·곡률을 잇는다(기본, "
                     "깊이에 비례해 복원력이 커진다), linear=접선만 잇는다"
                     "(복원력이 깊이와 무관한 상수다)")
ap.add_argument("--bc_dhat", type=float, default=0.0,
                help="장벽이 켜지는 거리. 0 이면 격자 dx 의 절반")
ap.add_argument("--bc_level", choices=["node", "particle"], default="node",
                help="바닥·경계를 어디에 거는지. node=**격자점 변위**에 사영"
                     "(기본, 손잡이와 같은 자리다), particle=입자 위치에 사영"
                     "(옛 방식 -- 변형장 자체는 바닥을 모른다)")
ap.add_argument("--node_h", type=float, default=0.0,
                help="노드 간격을 **절대값으로 고정**한다 (0 이면 "
                     "바운딩박스/n_nodes). 셀 부피가 실행 중에 변하지 않는다")
ap.add_argument("--hz_ratio", type=float, default=1.0 / 6 ** 0.5,
                help="층 간격 / 면내 간격. 기본 1/sqrt6 이면 셀이 정육면체다. "
                     "셀 부피는 (sqrt3/2)·h²·hz 라, 부피를 고정한 채 이 값만 "
                     "바꾸려면 h 를 (2V/(sqrt3·hz_ratio))^(1/3) 로 함께 잡는다")
ap.add_argument("--n_nodes", type=float, default=32,
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
ap.add_argument("--det_keep_elastic", action="store_true",
                help="det < --det_eps 인 사면체를 **탄성항에서 빼지 않는다** (모든 사면체가 탄성항에 "
                     "들어간다). 기본은 뺀다. 복구 손실은 --det_w 로 따로 끈다")
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
ap.add_argument("--infer", action="store_true",
                help="추론만: --resume 체크포인트로 홀드아웃 롤아웃·평가만 한다. 표준화 통계 재계산(체크포인트가 덮어씀)·"
                     "풀 상태 복원(128 상태 ~2.4 GB)·학습·같은 궤적 중복 평가를 건너뛴다")
ap.add_argument("--probe_every", type=int, default=0,
                help="이 반복마다 고정 확인 상태(첫 검증 궤적의 정답 프레임들)에서 한 스텝 증분 포텐셜·관성 잔차를 재서 TB 에 쓴다")
ap.add_argument("--probe_frames", default="0,5,10,20,40,60,79")
ap.add_argument("--pe_only", type=int, default=0,
                help="대조군: 노드 입력을 노드 위치·프레임 번호의 다주파수 인코딩(sin/cos(2^k π ·), k<K)만으로 (물리 특징 없음)")
ap.add_argument("--interp", choices=["bary", "bspline", "nodeavg", "c1wf"], default="bary",
                help="노드 변위 -> 입자 변위·한 스텝 야코비안 보간. bary=Kuhn 사면체 barycentric (기본; 야코비안이 사면체마다 "
                     "상수, 경계에서 불연속), bspline=같은 노드 격자의 2 차 B-스플라인 27 노드 (MPM 가중치, C1; 변위·야코비안 "
                     "같은 보간), nodeavg=변위는 barycentric 그대로, 야코비안만 사면체별 grad u 를 노드에 부피 가중 평균한 뒤 "
                     "barycentric 보간 (F 누적·탄성 에너지 모두 이 값), c1wf=Worsey–Farin C1 3 차 매크로 요소 (사면체를 "
                     "12 조각, 노드 출력 54 채널 = 변위 3 + 격자좌표 기울기 9 + 변 가로 미분 42; 입자 F 는 요소 다항식의 실제 "
                     "기울기, lib/anchorflow/wf.py)")
ap.add_argument("--c1_barrier", type=float, default=0.0,
                help="--interp c1wf: 점유 사면체 모든 조각의 det(I+∇u) 6 차 Bernstein 계수에 진짜 로그 장벽 "
                     "Σ (V_조각/84) κ μ b(c), b(c)=-(c-ĉ)² ln(c/ĉ) (c<ĉ), c<=0 이면 +inf (연장 없음). 0 이면 끔. "
                     "켜면 출력만 최적화의 L-BFGS 가 유한 에너지를 지키는 되추적 판(feas_lbfgs)으로 바뀐다")
ap.add_argument("--c1_jhat", type=float, default=0.3, help="--c1_barrier 가 작동하기 시작하는 계수 ĉ")
ap.add_argument("--c1_base", choices=["pl", "none"], default="pl",
                help="--interp c1wf 기울기·변 채널의 매개화: pl=노드 값의 PL 장에서 만든 C1 자유도 + 출력 (기본; 출력 0 이 매끄럽게 한 PL), "
                     "none=출력이 곧 자유도")
ap.add_argument("--agg", choices=["bary", "bspline"], default="bary",
                help="노드 특징 집계 가중치: bary=자기 사면체 4 꼭짓점 barycentric (기본), bspline=격자 좌표 2 차 B-스플라인 27 노드 (MPM P2G 식)")
ap.add_argument("--rel_pe", type=int, default=0,
                help="노드 특징에 입자-노드 상대 위치 (x_p-x_j)/h 의 다주파수 인코딩 sin/cos(2^k π ·), k<K 를 집계 가중으로 더한다")
ap.add_argument("--bary_pe", type=int, default=0,
                help="노드 특징에 셀 안 barycentric 좌표의 다주파수 위치 인코딩 sin/cos(2^k π λ), k<K 를 질량 가중으로 더한다 (0 이면 끔)")
ap.add_argument("--gregory", action="store_true",
                help="셀 안 로컬 변환: 출력만 최적화(rep_ip ours)와 같은 Gregory·방사형 스키닝 (lib/anchorflow/skin.py). "
                     "망 출력이 노드당 4 채널(변위 3 + 반경 1)이 된다. --node_h 고정 셀 크기가 필요하다")
ap.add_argument("--pool_gate", type=float, default=0.0,
                help="0 보다 크면 새 풀 규칙: 증분 포텐셜을 노드 변위 출력으로 미분한 노드별 벡터의 크기 × h²/노드질량 ÷ L 의 "
                     "노드 평균(출력 공간 정류 잔차)이 이 값 이하일 때만 상태를 전진시킨다. 넘으면 전진 없이 풀에 남긴다. "
                     "관성 잔차 문턱 폐기·되돌리기는 쓰지 않는다")
ap.add_argument("--pool_gate_mode", choices=["grad", "objdec", "pgrad"], default="grad",
                help="전진 기준: grad = 증분 포텐셜을 노드 출력으로 미분한 기울기 전체 L2 노름 (h²/평균노드질량/L 로 환산) ≤ --pool_gate, "
                     "pgrad = 이 상태 손실의 **망 파라미터** 기울기 전체 L2 노름 ≤ --pool_gate (역전파 전후 차이로 구해 역전파를 더 하지 않는다), "
                     "objdec = 같은 상태를 다시 볼 때 직전 방문 대비 목적함수 상대 감소 < --pool_gate (정체하면 전진)")
ap.add_argument("--ov_seq", action="store_true",
                help="출력만 최적화 궤적 만들기: --seq_frames 순차 모드에서 망 학습 없이 프레임마다 --ov_ref 와 같은 L-BFGS 로 "
                     "학습과 같은 증분 포텐셜을 수렴시키고 그 해로 전진한다. 궤적은 <out>/<tag>_seq_traj.pt (x, v, F 프레임별)")
ap.add_argument("--ov_ref", action="store_true",
                help="순차 보정 모드에서 학습 전에 같은 상태의 출력만 최적화 기준해를 L-BFGS 로 구하고 (망 고정, 노드 출력 탐침이 변수), "
                     "학습 중 그 해와의 위치·F 차이를 TB '기준해오차/' 로 기록한다")
ap.add_argument("--pool_stuck_max", type=int, default=0,
                help="--pool_gate 에서 연속 이 횟수만큼 방문하는 동안 한 번도 전진하지 못한 상태를 버린다 (0 이면 끔)")
ap.add_argument("--pool_rdrop", type=float, default=0.0,
                help="--pool_gate 에서 풀 상태를 방문할 때마다 이 확률로 무작위로 버린다 (빈자리는 새 상태로)")
ap.add_argument("--pool_expire", action="store_true",
                help="--pool_gate 에서 나이 --pool_frames 에 닿은 상태를 만기로 뺀다. 기본은 만기 없음 "
                     "(사용자 2026-10-12: 같은 시기에 들어온 상태가 한꺼번에 빠지는 문제 때문에 기본에서 뺀다)")
ap.add_argument("--pool_adv", choices=["model", "langevin"], default="model",
                help="풀 상태 전진: model = 모델 출력으로 (기존), langevin = 노드 출력 공간에서 증분 포텐셜 MALA 한 걸음 "
                     "(로그 장벽 계량, 게이트 없이 항상 전진; 무작위 버림 --pool_rdrop·만기 --pool_expire 는 그대로)")
ap.add_argument("--lgv_start", choices=["model", "inertial"], default="model",
                help="--pool_adv langevin 의 시작점: model = 모델 출력 (Z0=0, 학습 역전파의 기울기 재활용), "
                     "inertial = 관성 예측 노드 변위 h·v_n + h²g (그 점에서 기울기 한 번 더)")
ap.add_argument("--lgv_rho", type=float, default=0.5,
                help="랑제빈 온도 T = 2·rho·E(u0)/d (d = 자유 노드 출력 수)")
ap.add_argument("--lgv_eta0", type=float, default=1e-2,
                help="랑제빈 걸음 크기 eta 시작값 (전역, 수락 ×--lgv_up, 기각 ×--lgv_down 으로 적응)")
ap.add_argument("--lgv_up", type=float, default=1.1, help="랑제빈 수락 때 eta 배수")
ap.add_argument("--lgv_down", type=float, default=0.7, help="랑제빈 기각 때 eta 배수")
ap.add_argument("--lgv_backtrack", type=int, default=5,
                help="제안에서 det J <= --det_eps 인 셀이 있으면 걸음(기울기·잡음)을 반으로 줄이는 최대 횟수 (그래도 안 되면 u0 로 전진)")
ap.add_argument("--lgv_lam", type=float, default=1e-2,
                help="계량 블록 G_n 에 더하는 정규화 lam·I 의 lam (전체 노드 블록 대각 평균에 대한 비)")
ap.add_argument("--pool_init_frame", type=int, default=0,
                help="--pool_init_traj 의 몇 번째 프레임을 고정 초기 상태로 쓸지 (문턱 보정용 출력만 최적화에 쓴다)")
ap.add_argument("--seq_frames", type=int, default=0,
                help="프레임 순차 과적합: 풀 대신 상태 하나(--pool_init_traj 0 프레임)를 두고, 모델 출력이 수렴할 때까지 "
                     "그 상태에서만 학습한 뒤 수렴한 출력으로 한 프레임 전진, 이전 가중치를 초기값으로 다음 프레임 -- 이 수만큼. "
                     "출력만 최적화(--out_var)와 같은 순서이고 최적화 대상만 망 파라미터다")
ap.add_argument("--seq_tol", type=float, default=3e-6,
                help="수렴 판정: 학습 한 반복 사이 출력 위치 변화의 최대값이 이 값 * h/frame_dt 아래 (rep_ip --newton_tol 과 같은 기준)")
ap.add_argument("--seq_max", type=int, default=2000, help="프레임당 반복 상한 (Adam 잡음으로 수렴 판정이 안 날 때)")
ap.add_argument("--gate_log_every", type=int, default=20,
                help="게이트/·기준후보/·실제오차/ TB 기록 간격 (문턱 보정에서는 1 로 매 반복)")
ap.add_argument("--seq_save_every", type=int, default=1,
                help="순차 궤적을 이 스텝마다 한 번만 저장한다 (--dt_sub N 이면 N 으로 두어 프레임 경계만 남긴다)")
ap.add_argument("--diag_pool", default="",
                help="진단: --resume 체크포인트의 풀 상태를 하나씩, 그 체크포인트 가중치로 한 스텝씩만 계산해 (나이, 물리잔차, 증분 포텐셜)을 "
                     "이 json 에 쓰고 끝낸다. 학습 경로의 계산을 그대로 쓰고 갱신·전진·폐기는 하지 않는다")
ap.add_argument("--keep_ckpts", action="store_true", help="save_every 마다 체크포인트를 따로 남긴다 (<tag>_itNNNNNN.pt)")
ap.add_argument("--net_inertial", action="store_true",
                help="망 출력을 노드별 관성 예측 τ·v_n + τ²g 위의 **보정**으로 둔다 (출력만 최적화가 출발하는 그 점). "
                     "없으면 망이 초기 속도·중력 이동을 스스로 만들어야 해 첫 스텝부터 틀렸다 (평가 첫 프레임 비 0.82)")
ap.add_argument("--v0_traj", action="store_true",
                help="롤아웃 **초기 속도**를 궤적에 저장된 v 로 쓴다. 기본은 "
                     "(x[t0]-x[t0-1])/h 차분이라 t0=0 에서 0 이 되는데, 초기 "
                     "속도가 0 이 아닌 씬(GF 수박은 -6)에서는 그게 틀렸다")
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
ap.add_argument("--cfg_bc", action="store_true",
                help="씬 config 의 공식 경계조건(enforce_particle_translation·enforce_particle_velocity_rotation·cuboid, "
                     "시간창 포함)과 grid_v_damping_scale 을 **노드 하드 구속**으로 옮긴다 (손잡이와 같은 경로: 노드 변위를 "
                     "τ·v_bc 로 덮고 최적화에서 뺀다). 입자 구속은 PG 처럼 0 프레임 위치로 입자를 고르고, 노드는 그 입자의 "
                     "질량 가중 비율이 절반을 넘으면 묶는다. cuboid 는 PG 처럼 상자 안 격자(=노드) 속도를 박는다")
ap.add_argument("--pg_region_density", action="store_true",
                help="PG config 의 additional_material_params 영역별 밀도를 질량에 넣는다 (PG apply_additional_params 와 같은 "
                     "0 프레임 상자 판정). 부피(= 질량/밀도)는 입자별 밀도로 나눠 그대로 둔다. 기본은 cfg density 하나")
ap.add_argument("--pg_impulse", action="store_true",
                help="PG config 의 particle_impulse 를 0 프레임 초기 속도에 더한다: PG add_impulse_on_particles 그대로 "
                     "상자(point 기본 [1,1,1], size 기본 [1,1,1], |x0-point|<size) 안 입자에 v += force/m · substep_dt · num_dt "
                     "(start_time 0 만). --v0_traj 필요 (궤적 v[0] 에 더한다)")
ap.add_argument("--pg_vdamp", action="store_true",
                help="PG grid_v_damping_scale 을 PG 처럼 **PG 서브스텝마다** 곱한 만큼 우리 한 스텝 뒤 속도에 곱한다: "
                     "v *= s^(h/substep_dt) (ficus: 0.9999^400 = 0.961/프레임). 출력만 최적화·망 롤아웃·상태 적합 롤아웃 모두. "
                     "--cfg_bc 의 스텝당 한 번 곱하기 대신 쓴다")
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
ap.add_argument("--pool_cfg", default="", help="물성 json 디렉토리 ({형상}_{물성}[_t].json). 비우면 $AF_WORK/wmats")
ap.add_argument("--pool_start_k", type=int, default=0,
                help="낙하 모드에서 새 시작 상태를 고정 시드로 이 개수만큼 미리 뽑아 두고 그중에서만 고른다 (1 = 한 상태로 고정, 0 = 매번 새로)")
ap.add_argument("--pool_init_traj", default="",
                help="이 궤적 .pt 의 0 프레임 상태(x, v, F)를 풀의 유일한 새 상태로 쓴다 -- 한 궤적 과적합 실험")
ap.add_argument("--pool_data", default="",
                help="학습 궤적 .pt 디렉토리 (lb_to_pt.py). 새 상태를 이 궤적들의 무작위 (궤적, 프레임) 에서 뽑는다 -- "
                     "다른 학습 기반 방법과 같은 학습 궤적을 쓰는 설정. --pool_fresh 1 --pool_size 0 이면 매 배치가 데이터 상태뿐")
ap.add_argument("--pool_impulse", default="",
                help="충격 모드 'lo,hi' (PG 공식 config 씬, lego 낙하 모드의 대응): 풀 새 상태 = 정지 자세(x0, F=I) + "
                     "config 의 particle_impulse 를 PG 규칙(상자 그대로, v += force/m·substep_dt·num_dt)으로 준 초기 속도. "
                     "힘 방향은 위쪽 축(mpm_space_vertical_upward_axis) 둘레로 균일 회전, 크기는 공식의 U[lo,hi] 배")
ap.add_argument("--pool_drop", default="",
                help="낙하 모드 'h0,h1,v0,v1' (학습 비교 공통 데이터셋 exe/lb_gen.py 와 같은 초기 조건 분포). "
                     "새 상태를 무작위 회전·바닥 위 높이·초기 병진 속도로 뽑는다")
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
ap.add_argument("--ov_vars", choices=["out", "net"], default="out",
                help="--ov_roll 의 변수: out=노드 출력(출력만 최적화, 기본), net=망 파라미터 -- 같은 증분 포텐셜·장벽을 "
                     "프레임마다 망 파라미터로 내린다 (한 궤적 과적합; 파라미터는 프레임 사이에 이어진다)")
ap.add_argument("--ov_obj", choices=["ip", "l2", "node"], default="ip",
                help="--ov_roll 의 프레임별 목적함수: ip=증분 포텐셜(기본), l2=다음 프레임 정답 위치와의 평균 L2 / EXT² (장벽 없음), "
                     "node=노드 출력을 다른 실행(출력만 최적화)의 노드 출력에 맞춘다 (AF_OV_NODE_TGT=그 롤아웃 덤프; 상태도 그 덤프로 매 프레임 맞춘다)")
ap.add_argument("--ov_gd_lr", type=float, default=1e-4, help="--ov_opt gd 의 고정 학습률 (경사하강)")
ap.add_argument("--ov_opt", choices=["adam", "lbfgs", "cg", "gd"], default="lbfgs",
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
if a.interp != "bary":
    _bad = [k for k, on in (("hex_jac", a.hex_jac), ("gregory", getattr(a, "gregory", False)), ("fused", a.fused),
                            ("ens_off>1", int(a.ens_off) > 1), ("elast lin", a.elast == "lin"),
                            ("smooth_nb", int(a.smooth_nb) > 0), ("cell_poly", a.cell_poly)) if on]
    if _bad:
        raise SystemExit(f"--interp {a.interp} 는 {_bad} 와 함께 쓸 수 없다")
    print(f"[보간] --interp {a.interp}", flush=True)
if a.c1_barrier > 0 and a.interp != "c1wf":
    raise SystemExit("--c1_barrier 는 --interp c1wf 에서만")

dev = "cuda"
torch.manual_seed(a.seed)

import torch.autograd.forward_ad as _fwAD                       # noqa: E402
from anchorflow import deform                                  # noqa: E402
from anchorflow import deform_model as DM                       # noqa: E402
from anchorflow import phys_resid                                # noqa: E402
from anchorflow import simplex as SX                            # noqa: E402
from anchorflow import wf as WF                                 # noqa: E402
from anchorflow.feas_lbfgs import FeasLBFGS                     # noqa: E402
from anchorflow import simplex_gnn as _SGM                      # noqa: E402
from anchorflow.simplex_gnn import (SimplexGNN, node_moments,    # noqa: E402
                                    scatter_to_nodes)
from anchorflow import voxel                                    # noqa: E402
from anchorflow.deform import (DeformNet, aggregate, anchor_knn,  # noqa: E402
                               bc_features, bond_stretch, bures_w2_sq,
                               fps, gauss_stretch, grid_knn,
                               jacobian_of)
DM.REGION_DENSITY[0] = bool(a.pg_region_density)

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
    if not a.v0_traj:
        d.pop("v", None)      # --v0_traj 면 초기 속도로 쓰므로 남긴다
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
    _GRID[0] = SX.grid_for_nodes(x, a.n_nodes, hz_ratio=a.hz_ratio,
                                 off=_GRID_OFF[0],
                                 h_fix=(a.node_h or None))
    try:
        yield
    finally:
        _GRID[0] = _o


if not a.bc_soft and a.bc_mode == "project":
    # 하드 사영을 쓰면 같은 경계를 벌점으로 또 세지 않는다.
    # **--bc_mode barrier 는 사영을 끄고 장벽 항으로만 막으므로 여기서 끄면
    # 안 된다.** 이 조건을 빠뜨려 장벽이 통째로 무효가 됐었다 (로프가 바닥을
    # 그냥 통과했다).
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
# 망 출력 크기(0.02·EXT)와 속도 단위(VEL_SCALE)를 정한 값. 풀 루프가 전역 EXT 를
# 씬 값으로 덮어쓰므로 따로 남겨 체크포인트에 넣는다 (추론이 같은 모델을 세우게).
_EXT0 = EXT
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
_RHO0 = DM.region_rho(cfg0, X0d.float())
MASS = ((dx ** 3) / cnt[flat]) * (float(cfg0["density"]) if _RHO0 is None else _RHO0)
_RHO = {}                                   # id(cfg) -> 입자별 밀도 [N_full] (--pg_region_density)
if _RHO0 is not None:
    _RHO[id(cfg0)] = _RHO0
    for _t, _d in TR + held:
        DM.traj_mass(_d, dev)
        if "_rho" in _d:
            _RHO[id(_d["cfg"])] = _d["_rho"]
    print(f"[밀도] 영역별 밀도 (PG additional_material_params): " + ", ".join(
        f"{float(r_):g}: {int((_RHO0 == r_).sum())} 입자" for r_ in torch.unique(_RHO0)) +
        f" | 질량 합 {float(MASS.sum()):.5g}", flush=True)


def _pvol(m, cfg):
    """입자 부피 = 질량 / 밀도. --pg_region_density 면 입자별 밀도로 나눈다 (부분표본이면 멈춘다)."""
    r = _RHO.get(id(cfg))
    if r is None:
        return m / float(cfg["density"])
    if r.shape[0] != m.shape[0]:
        raise SystemExit(f"--pg_region_density: 부피 계산의 입자 수 {m.shape[0]} 가 밀도 배열 {r.shape[0]} 와 다르다 (부분표본 미지원)")
    return m / r.to(m.device, m.dtype)


def _pg_vdamp(v, d, h):
    """--pg_vdamp: PG 가 서브스텝마다 격자 속도에 곱하는 grid_v_damping_scale 을 우리 스텝 h 만큼 (s^(h/substep_dt))."""
    if not a.pg_vdamp:
        return v
    s_ = float(d["cfg"].get("grid_v_damping_scale", 1.0))
    if s_ >= 1.0:
        return v
    return v * (s_ ** (float(h) / float(d["cfg"]["substep_dt"])))


if a.pg_impulse:
    # PG add_impulse_on_particles: 시각 [start, start + substep_dt·num_dt) 의 매 서브스텝 v += force/m · substep_dt.
    # 0 프레임(시각 0) 에서 시작하는 것만 -- 우리 한 스텝이 프레임이라 그 서브스텝들의 합을 초기 속도에 싣는다.
    if not a.v0_traj:
        raise SystemExit("--pg_impulse 는 --v0_traj 와 같이 (궤적 v[0] 에 더한다)")
    for _t, _d in TR + held:
        if _d.get("_imp_done"):
            continue
        _c = _d["cfg"]; _x0 = _d["x"][0].to(dev).float(); _m = DM.traj_mass(_d, dev)
        _dv = torch.zeros_like(_x0)
        for _bc in (_c.get("boundary_conditions") or []):
            if _bc.get("type") != "particle_impulse":
                continue
            if float(_bc.get("start_time", 0.0)) != 0.0:
                raise SystemExit(f"--pg_impulse: start_time 0 이 아닌 충격은 아직 없다 ({_bc})")
            _q = torch.as_tensor(_bc.get("point", [1, 1, 1]), device=dev, dtype=torch.float32)
            _sz = torch.as_tensor(_bc.get("size", [1, 1, 1]), device=dev, dtype=torch.float32)
            _in = ((_x0 - _q).abs() < _sz).all(-1)
            _f = torch.as_tensor(_bc["force"], device=dev, dtype=torch.float32)
            _sdt = float(_c["substep_dt"]) * int(_bc.get("num_dt", 1))
            _dv[_in] += _f * _sdt / _m[_in].unsqueeze(-1)
            print(f"[충격] {_t}: particle_impulse 힘 {_bc['force']} x {_sdt:g} s, 상자 안 입자 {int(_in.sum())}/{_in.numel()}", flush=True)
        _v0 = _d["v"][0].to(dev).float()
        _d["v"][0] = (_v0 + _dv).to(_d["v"].dtype).cpu() if _d["v"].device.type == "cpu" else (_v0 + _dv).to(_d["v"].dtype)
        _d["_imp_done"] = True
        _vn = _dv.norm(dim=-1)
        _msg = (f"[충격] {_t}: 초기 속도 |v0| 평균 {float(_vn.mean()):.4f} 최대 {float(_vn.max()):.4f}, "
                f"운동량 Σm·v0 {[round(float(z), 5) for z in (_m.unsqueeze(-1) * _dv).sum(0)]}")
        if _d["v"].shape[0] > 1:
            _v1 = _d["v"][1].to(dev).float()
            _msg += (f" | 정답 1 프레임 |v| 평균 {float(_v1.norm(dim=-1).mean()):.4f} 최대 {float(_v1.norm(dim=-1).max()):.4f}, "
                     f"Σm·v1 {[round(float(z), 5) for z in (_m.unsqueeze(-1) * _v1).sum(0)]}")
        print(_msg, flush=True)

# 물성은 앵커 특징에 들어간다. 궤적마다 다르므로 궤적별로 만든다.
def mat_feat(cfg):
    """물성 특징 (lib/anchorflow/deform_model.mat_feat)."""
    return DM.mat_feat(cfg, a.no_mat, dev)


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


DM.apply_arch_flags(a)
if a.gnn_no_norm:
    print("[구조] 메시지 패싱 LayerNorm 제거", flush=True)
if a.gnn_sum:
    print("[구조] 메시지 집계를 합으로", flush=True)


def build(n_feat):
    global net, opt
    net = DM.build_net(a, n_feat, _EXT0, FRAME_DT, N_FILM, dev)
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
    return DM.take(t, idx_gpu, dev)


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


_NODE_FIT_DONE = [False]
_DPN_LIVE = [None]         # (노드 위치, 노드 출력) -- 마지막 step_once 의 것, 기울기 유지
_NODE_TGT = None           # --ov_obj node: 목표 롤아웃 덤프 (pred·v·F·nodes)
if os.environ.get("AF_OV_NODE_TGT"):
    _NODE_TGT = torch.load(os.environ["AF_OV_NODE_TGT"], map_location="cpu", weights_only=False)
_PIN = [None]              # (손잡이 노드 마스크, 명령 변위) -- 풀 학습 경로에서도 on_pin 이 쓰므로 여기서 먼저 만든다
_SIM_T = [0.0]             # --cfg_bc: 롤아웃 시작부터의 물리 시각 (서브스텝마다 갱신)


def _cfg_bc_rot_v(pos, bc):
    """PG enforce_particle_velocity_rotation 의 속도: 축 n 둘레 각속도 rotation_scale + 축 방향 translation_scale.
    PG 식 (axis1 = -d sinθ ω, axis2 = d cosθ ω, h2 = h1 × n) 은 v = ω (r_⊥ × n) 와 같다."""
    n = torch.tensor(bc["normal"], device=pos.device, dtype=pos.dtype)
    n = n / n.norm()
    q = torch.tensor(bc["point"], device=pos.device, dtype=pos.dtype)
    off = pos - q
    rp = off - (off @ n).unsqueeze(-1) * n
    return (float(bc["rotation_scale"]) * torch.cross(rp, n.expand_as(rp), dim=-1)
            + float(bc["translation_scale"]) * n)


def _cfg_bc_sel(d, gsel, dev, dtype):
    """입자 구속의 입자 선택 (PG 처럼 0 프레임 위치로 한 번) -> [(bc, mask[N])]."""
    key = "_cfgbc_sel"
    if key not in d:
        x0 = take(d["x"][0], gsel).to(dev, dtype)
        out = []
        for bc in (d["cfg"].get("boundary_conditions") or []):
            tp = bc.get("type")
            if tp == "enforce_particle_translation":
                q = torch.tensor(bc["point"], device=dev, dtype=dtype)
                sz = torch.tensor(bc["size"], device=dev, dtype=dtype)
                out.append((bc, ((x0 - q).abs() < sz).all(-1)))
            elif tp == "enforce_particle_velocity_rotation":
                n = torch.tensor(bc["normal"], device=dev, dtype=dtype); n = n / n.norm()
                q = torch.tensor(bc["point"], device=dev, dtype=dtype)
                off = x0 - q
                vd = (off @ n).abs()
                hd = (off - (off @ n).unsqueeze(-1) * n).norm(dim=-1)
                hh, rr = (float(z) for z in bc["half_height_and_radius"])
                out.append((bc, (vd < hh) & (hd < rr)))
        d[key] = out
        for bc, m in out:
            print(f"[경계조건] {bc['type']} 입자 {int(m.sum())} 개, 시간 [{bc.get('start_time', 0)}, {bc.get('end_time', 1e3)})",
                  flush=True)
    return d[key]


def cfg_bc_nodes(d, gsel, x, npos, rows, lam, mass, now):
    """--cfg_bc: 시각 now 에 묶이는 노드와 속도 -> (mask[M] bool, vel[M,3]) 또는 (None, None)."""
    M = npos.shape[0]
    dev, dt_ = npos.device, npos.dtype
    msk = torch.zeros(M, dtype=torch.bool, device=dev)
    vel = torch.zeros(M, 3, device=dev, dtype=dt_)
    hit = False
    wl = (lam * mass.to(dt_).unsqueeze(1))                    # [N,4] 질량 가중
    den = None
    for bc, pm in _cfg_bc_sel(d, gsel, dev, dt_):
        if not (float(bc.get("start_time", 0.0)) <= now < float(bc.get("end_time", 1e3))):
            continue
        if den is None:
            den = torch.zeros(M, device=dev, dtype=dt_).index_add_(0, rows.reshape(-1), wl.reshape(-1))
        num = torch.zeros(M, device=dev, dtype=dt_).index_add_(
            0, rows.reshape(-1), (wl * pm.to(dt_).unsqueeze(1)).reshape(-1))
        nm = num > 0.5 * den.clamp_min(1e-30)
        if bc["type"] == "enforce_particle_translation":
            vv = torch.tensor(bc["velocity"], device=dev, dtype=dt_).expand(M, 3)
        else:
            vv = _cfg_bc_rot_v(npos, bc)
        vel = torch.where(nm.unsqueeze(-1), vv, vel); msk = msk | nm; hit = True
    for bc in (d["cfg"].get("boundary_conditions") or []):
        if bc.get("type") != "cuboid":
            continue
        t0_, t1_ = float(bc.get("start_time", 0.0)), float(bc.get("end_time", 1e3))
        if not (t0_ <= now < t1_):
            continue                                          # reset 은 끝시각 뒤 15 dt 동안 격자 전체 속도 0 (아래)
        vb = torch.tensor(bc["velocity"], device=dev, dtype=dt_)
        q = torch.tensor(bc["point"], device=dev, dtype=dt_) + (now - t0_) * vb    # PG 는 상자를 dt·v 씩 옮긴다
        sz = torch.tensor(bc["size"], device=dev, dtype=dt_)
        nm = ((npos - q).abs() < sz).all(-1)
        vel = torch.where(nm.unsqueeze(-1), vb.expand(M, 3), vel); msk = msk | nm; hit = True
    for bc in (d["cfg"].get("boundary_conditions") or []):
        if bc.get("type") == "cuboid" and int(bc.get("reset", 0)) == 1:
            t1_ = float(bc.get("end_time", 1e3))
            if t1_ <= now < t1_ + 15.0 * float(_DT[0]):
                msk = torch.ones(M, dtype=torch.bool, device=dev); vel = torch.zeros(M, 3, device=dev, dtype=dt_); hit = True
    return (msk, vel) if hit else (None, None)


def release_held(d, x, t, dt=None):
    """PG 의 `release_particles_sequentially` 를 그대로 옮긴다 -> 아직 **묶여
    있는** 입자 마스크 [N] (True = 아직 못 움직인다).

    PG 는 50 겹의 "상자 안 속도를 0 으로" 구속을 **끝시각만 엇갈리게** 걸어 둔다
    (mpm_solver_warp.release_particles_sequentially). 상자 i 는 축 좌표
    end_position 을 중심으로 반폭 |start-end|/50*(50-i), 끝시각은
    end_time*(i+1)/50 이다. 그 입자를 담는 가장 큰 i 가 마지막까지 붙들므로

        t_rel(q) = end_time * (50 - floor(|q - end| / (|start-end|/50))) / 50

    이고, 벽 자리(end_position)에 가까울수록 늦게 풀린다. 모래성 씬(wolf)이
    이 구속으로 "벽을 치우면 무너진다" 를 만든다 -- 빼먹으면 첫 프레임부터
    전부 무너져 GT 와 아예 다른 장면이 된다.
    """
    cfg = d["cfg"]
    held = None
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") != "release_particles_sequentially":
            continue
        nrm = bc.get("normal", [1, 0, 0])
        ax = max(range(3), key=lambda i: abs(float(nrm[i])))
        sp, ep = float(bc["start_position"]), float(bc["end_position"])
        nl = int(bc.get("num_layers", 50)) or 50
        nl = 50                      # PG 가 인자를 무시하고 50 으로 덮어쓴다
        t0 = float(bc.get("start_time", 0.0))
        t1 = float(bc.get("end_time", 0.0))
        w = abs(sp - ep) / nl
        now = float(t) * (float(dt) if dt is not None else float(_DT[0]))
        if now < t0:
            m = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            lay = ((x[:, ax] - ep).abs() / max(w, 1e-12)).floor()
            t_rel = t1 * (nl - lay).clamp_min(0.0) / nl
            m = now < t_rel
        held = m if held is None else (held | m)
    return held


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
from anchorflow import cellpoly as CPOLY_K      # noqa: E402
from anchorflow import fused_step as FUSED      # noqa: E402


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
    return (None if a.det_keep_elastic else ~bad), a.det_w * pen.mean()


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
if os.environ.get("AF_SYNC_DEBUG"):
    # 진단: GPU→CPU 동기화가 일어나는 코드 위치를 센다 (exe/·lib/ 안의 가장 안쪽 줄 기준). 끝날 때 많은 순으로 찍는다
    import warnings as _wn, traceback as _tb, collections as _co, atexit as _ae
    _SYNC_N = _co.Counter()

    def _sync_show(msg, cat, fn, ln, file=None, line=None):
        if "synchroniz" not in str(msg):
            return
        fr = [f for f in _tb.extract_stack()[:-1] if ("/exe/" in f.filename or "/lib/anchorflow/" in f.filename)]
        _SYNC_N[" <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in fr[-3:][::-1])] += 1
    _wn.showwarning = _sync_show
    _wn.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    _ae.register(lambda: print("[동기화 위치]\n" + "\n".join(f"  {n:7d}  {k}" for k, n in _SYNC_N.most_common(30)), flush=True))

_PROF = {}
_CUR_T = [0]              # step_once 가 남기는 현재 프레임
_CTRL_SCALE = [1.0]       # 손잡이 명령 변위 배수 (서브스텝이면 1/K)


def _setup_inv_barrier(a):
    """뒤집힘 장벽을 켠다 (켤 때만 알린다)."""
    if getattr(a, "inv_barrier", 0.0) > 0:
        phys_resid.set_inv_barrier(a.inv_barrier, a.inv_jhat, a.inv_ext)
        print(f"[뒤집힘] det F 로그 장벽 kappa {a.inv_barrier:g}, "
              f"Ĵ {a.inv_jhat:g}, det<=0 연장 {a.inv_ext}", flush=True)
    else:
        phys_resid.set_inv_barrier(None)
    if getattr(a, "cell_barrier", 0.0) > 0:
        phys_resid.set_cell_barrier(a.cell_barrier, a.cell_jhat, a.cell_ext)
        print(f"[뒤집힘] 셀 사상 det(grad Phi) 장벽 kappa {a.cell_barrier:g}, "
              f"Ĵ {a.cell_jhat:g}, 연장 {a.cell_ext}", flush=True)
    else:
        phys_resid.set_cell_barrier(None)
_DP_HOOK = [None]
# 망을 아예 부르지 않고 바로 출력을 내는 경로. None 을 돌려주면 망을 쓴다.
_DP_FAST = [None]
# "want" 를 넣으면 다음 셀집계 결과를 붙잡아 이후 호출에서 재사용한다
_NF_HOLD = [None]
_SNAP_N = [0]       # 사영이 손본 사면체 최대 개수 (보고용)
_OV_TGT = [None]           # --ov_init target 이 쓸 자유낙하 평행이동 [3]
# --pool_adv langevin: 전역 걸음 크기와 TB 창 집계 (수락·백트래킹·E 증가·시간)
_PLGV = {"eta": None, "acc": [], "bt": [], "dE": [], "dEc": [], "T": [], "fail": 0,
         "t_it": [], "t_pool": [], "t_cur": 0.0}
# AF_OV_CURVE=경로 를 주면 프레임별 (반복, E) 곡선을 그 경로에 남긴다
_CURVE = [] if os.environ.get("AF_OV_CURVE") else None
# 롤아웃 덤프에 격자점을 함께 담을지. (노드 위치, 노드 변위, 유효 여부)
_NOD_M = [None]            # 손잡이로 덮는 노드 (최적화에서 뺀다)
_SM_HOLD = [None]          # 노드 이웃 색인 (스무딩용)
_GRID_OFF = [None]         # 격자 원점을 셀의 분수만큼 민다 (앙상블)
_ENS_K = [0]               # 지금 다루는 앙상블 격자 번호
_ENS_N = [1]               # 앙상블 격자 수
_NODE_CAP = [False]
_NODE_LAST = [None]
_BC_LAST = [None]          # (구속 보정량 [N,3], 활성집합 [N,1]) -- 최근 step_once
# AF_INV_LOG=경로: 롤아웃 프레임마다 이번 스텝 det(grad Phi) <= 0 인 사면체·입자 수와 누적 det F <= 0 입자 수를 한 줄씩 남긴다
_INV_LOG = os.environ.get("AF_INV_LOG") or None
_INV_CUR = [None]
_C1_P1 = [0]               # --c1_barrier: 이번 프레임 실현 가능 시작점 1 단계 외곽 횟수
_OV_CUR_OPT = [None]       # 마지막으로 쓴 출력만 최적화 옵티마이저 (feas_lbfgs 통계 기록용)
# AF_TUNNEL_LOG=경로: 롤아웃 프레임마다 활성 사면체 중 스텝 시작에 꼭지점 하나라도 바닥면 위였다가
# 스텝 끝에 네 꼭지점이 모두 바닥면 아래가 된 것의 수를 한 줄씩 남긴다
_TUN_LOG = os.environ.get("AF_TUNNEL_LOG") or None
_TUN_CUR = [None]
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
    """사면체 복합체의 **노드** 특징 (lib/anchorflow/deform_model.node_feats).

    -> (_in [M,F], npos [M,3], (lo,lat,nn), rows [N,4], lam [N,4], uniq [M],
        간선 (src,dst,cls), aux)
    """
    cfg = d["cfg"]

    def _parts(hn):
        parts = []
        if a.fe_state:
            parts.append(fe_invariants(fe) if fe is not None
                         else torch.zeros(x.shape[0], 3, device=dev, dtype=x.dtype))
        if a.control:
            parts.append(ctrl_feat_pts(d, t, x, hn))
        return parts
    _o = DM.node_feats(
        a, cfg, take(d["x"][0], gsel), MASS[gsel], x, v, VEL_SCALE,
        grid=_GRID[0], off=_GRID_OFF[0], pt_parts=_parts,
        mat_vec=(mat_feat(cfg) if (N_MAT and not N_FILM) else None))
    if a.pe_only > 0:
        # **대조군**: 물리 특징을 다 버리고 노드 세계 위치와 프레임 번호의 다주파수 인코딩만 입력으로 (한 궤적이면
        # (위치, 시각) 이 목표를 유일하게 가리킨다 -- 순수 암기 능력의 기준)
        _K = int(a.pe_only)
        _fr = (2.0 ** torch.arange(_K, device=x.device, dtype=x.dtype)) * math.pi
        _p = (_o[1] / 2.0).unsqueeze(-1) * _fr                                   # 영역 [0,2] -> [0,1]
        _tt = torch.full((_o[1].shape[0], 1), float(t) / 80.0, device=x.device, dtype=x.dtype) * _fr
        _pe = torch.cat([torch.sin(_p).flatten(1), torch.cos(_p).flatten(1), torch.sin(_tt), torch.cos(_tt)], -1)
        _o = (_pe,) + tuple(_o[1:])
    return _o


_ens_offsets = DM.ens_offsets


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
            #
            # --ens_off N 이면 **오프셋이 다른 격자 N 개**를 만든다. 변위장은
            # 그 N 개의 평균이고 목적함수는 그 평균 하나로 세므로, N 벌의 노드
            # 변수가 **함께** 최적화된다 (따로 풀고 나중에 평균하면 아무도
            # 최적화하지 않은 장이 된다).
            if _NF_HOLD[0] is not None and _NF_HOLD[0] != "want":
                _ENS = _NF_HOLD[0]
            else:
                _ENS = []
                for _ok in _ens_offsets(max(int(a.ens_off), 1)):
                    _GRID_OFF[0] = _ok
                    _ENS.append(node_feats(d, t, gsel, x, v, fe=fe))
                _GRID_OFF[0] = None
                if _NF_HOLD[0] == "want":
                    _NF_HOLD[0] = _ENS
        _ENS_N[0] = len(_ENS)
        (_in, npos, (lo, lat, nn), rows, lam, uniq,
         (esrc, edst, ecls), _lax) = _ENS[0]
        _mv = (mat_feat(d["cfg"]).reshape(1, N_MAT) if N_FILM else None)
        # 손잡이는 **노드 변위를 명령값으로 덮는다** (i-PG 가 격자 속도를 박는
        # 것과 같다). 격자마다 노드가 다르므로 오프셋별로 따로 만든다.
        _NM, _NV, _TG, _VNL = [], [], [], []
        for _nfk in _ENS:
            _npk, _rwk, _lmk = _nfk[1], _nfk[3], _nfk[4]
            _nmk = _nvk = None
            if a.control and "ctrl_id" in d and "ctrl_vel" in d:
                _tt = min(t, d["ctrl_vel"].shape[0] - 1)
                _cvc = d["ctrl_vel"].to(x.device, x.dtype)[_tt]
                _cen = ctrl_center(d, gsel, _tt, x)
                _Rc = d["ctrl_R"].to(x.device, x.dtype)
                _Rc = _Rc[min(_tt, _Rc.numel() - 1)].clamp_min(1e-6)
                _dn = (_npk.unsqueeze(1) - _cen.unsqueeze(0)).norm(dim=-1)
                _dmin, _kmin = _dn.min(1)
                _nmk = _dmin < _Rc
                _nvk = _cvc[_kmin]
            if a.cfg_bc and _nmk is None:
                _mwk = (MASS[gsel] if MASS.numel() != x.shape[0] else MASS)
                _nmk, _nvk = cfg_bc_nodes(d, gsel, x, _npk, _rwk, _lmk, _mwk, _SIM_T[0])
            _NM.append(_nmk); _NV.append(_nvk)
            if a.ov_init == "predict" or a.net_inertial:
                # **노드별 관성 예측자** h·v_n + h²g (barycentric 가중 질량평균). --net_inertial 은 망 출력에 더하므로
                # --ov_init net(망 출력에서 출발하는 출력만 최적화)일 때도 만든다
                # **노드별 관성 예측자** h·v_n + h²g (barycentric 가중 질량평균)
                _Mn = _npk.shape[0]
                _mw = (MASS[gsel] if MASS.numel() != x.shape[0] else MASS)
                # 육면체면 **여덟 꼭짓점 삼선형 가중**으로 뿌린다. 사면체 네
                # 개로만 뿌리면 나머지 네 노드가 0 으로 남아 첫 프레임부터
                # 셀이 접힌다 (실측: 프레임 0 에 det<0 14.7%).
                if a.hex_jac:
                    _p8, _, _N8, _, _ = _hex_struct(x, _nfk[2][0],
                                                    _nfk[2][1], _nfk[2][2],
                                                    _nfk[5])
                    _rr, _ll = _p8, _N8
                else:
                    _rr, _ll = _rwk, _lmk
                _tgk, _vnlk = DM.node_inertial(x, v, _mw, _rr, _ll, _Mn,
                                               d["cfg"]["g"], float(_DT[0]))
                _TG.append(_tgk)
                _VNL.append(_vnlk)
            else:
                _TG.append(None)
                _VNL.append(None)
        _nod_m, _nod_v = _NM[0], _NV[0]
        _NOD_M[0] = _nod_m

        # ---- 학습 쪽 실험 경로 훅 (추론 라이브러리에는 없다) ----------------
        def _pre_k(_k):
            _ENS_K[0] = _k
            _NOD_M[0] = _NM[_k]
            _OV_TGT[0] = _TG[_k]

        def _on_pin(tau):
            _PIN[0] = (_NM[0], (tau * _CTRL_SCALE[0] * _NV[0]).detach())
            with torch.no_grad():
                _CTRL_ERR[1] = int(_NM[0].sum())
                _CTRL_ERR[2] = float(
                    (tau * _NV[0])[_NM[0]].norm(dim=-1).mean()
                    if bool(_NM[0].any()) else 0.0)

        def _post_jac(_k, _nfk, dpn_k, _duk, _Jk):
            (_ink, _npk, (_lok, _latk, _nnk), _rwk, _lmk, _uqk,
             (_srk, _dsk, _clk), _lxk) = _nfk
            if a.hex_jac:
                # **육면체 삼선형**: 입자 변위와 야코비안을 같은 사상에서
                # 뽑는다 (사면체 중심좌표 대신). 여덟 꼭짓점이 다 살아있지
                # 않은 입자만 사면체 값을 그대로 쓴다.
                _uh, _Jh, _hok = _hex_field(x, dpn_k, _lok, _latk, _nnk,
                                            _uqk)
                if os.environ.get("AF_HEX_DIAG") and _HXD[0] < 3:
                    _HXD[0] += 1
                    with torch.no_grad():
                        _I3d = torch.eye(3, device=_Jk.device,
                                         dtype=_Jk.dtype)
                        _dt = torch.linalg.det(_I3d + _Jk)
                        _dh2 = torch.linalg.det(_I3d + _Jh)
                        print(f"      [육면체진단] 노드 {dpn_k.shape[0]} "
                              f"변위 std {float(dpn_k.std()):.3e} "
                              f"max {float(dpn_k.abs().max()):.3e} | "
                              f"detJ 사면체 {float(_dt.mean()):.4f}"
                              f"(min {float(_dt.min()):.3f}) 육면체 "
                              f"{float(_dh2.mean()):.4f}"
                              f"(min {float(_dh2.min()):.3f}) | "
                              f"|u차| {float((_uh - _duk).abs().max()):.3e} "
                              f"쓸수있음 {int(_hok.sum())}/{_hok.numel()}",
                              flush=True)
                if _k == 0 and a.hex_hg > 0:
                    _HG_LAST[0] = (dpn_k, _hex_struct(
                        x, _lok, _latk, _nnk, _uqk)[4], _latk)
                if not bool(_hok.all()):
                    raise SystemExit(
                        f"--hex_jac: 여덟 꼭짓점이 없는 입자 "
                        f"{int((~_hok).sum())} 개 -- 활성 노드 구성이 "
                        f"셀 기준이 아니다")
                _duk, _Jk = _uh, _Jh
            if _k == 0 and a.smooth_nb > 0:
                # **노드 변위를 이웃과 섞어** 야코비안을 만든다. 위치는
                # 스무딩하지 않은 변위로 정한다 -- 변형 측정만 넓힌다.
                _ds = _smooth_nodes(dpn_k, _uqk, _nnk)
                _u3, _Jk = SX.g2p_jac_pre(_rwk, _lmk, _lxk, _latk, _ds)
            if _k == 0 and a.cell_poly and _TAU_H[0] is not None:
                # **셀 다항식**: 입자 기여를 12 자유도 다항식으로 바꿔
                # 셀마다 계수를 합산해 두고, 평가는 셀 수만큼만 돈다.
                if _CPOLY[0] is None:
                    _tp, _hp2 = _TAU_H[0]
                    _uqt, _invt = torch.unique(_rwk, dim=0,
                                               return_inverse=True)
                    _Tn = int(_uqt.shape[0])
                    _rept = torch.zeros(_Tn, dtype=torch.long, device=dev)
                    _rept.scatter_(0, _invt,
                                   torch.arange(_invt.numel(), device=dev))
                    _tc2 = torch.zeros(_Tn, 3, 3, device=dev,
                                       dtype=_tp.dtype
                                       ).index_add_(0, _invt, _tp)
                    _hc2 = torch.zeros(_Tn, 3, 3, 3, 3, device=dev,
                                       dtype=_hp2.dtype
                                       ).index_add_(0, _invt, _hp2)
                    _gq2 = torch.as_tensor(d["cfg"]["g"], device=dev,
                                           dtype=x.dtype)
                    _tgt2 = FRAME_DT * v + (FRAME_DT ** 2) * _gq2
                    _wf2 = torch.ones(x.shape[0], device=dev,
                                      dtype=x.dtype)
                    if _NM[0] is not None:
                        _wf2 = 1.0 - _NM[0][_rwk].all(1).to(x.dtype)
                    _mw2 = (MASS[gsel] if MASS.numel() != x.shape[0]
                            else MASS).to(x.dtype) / (FRAME_DT ** 2)
                    _CPOLY[0] = _cell_poly_build(
                        _rwk, _lmk, _lxk, _latk, _tc2, _hc2, _invt, _rept,
                        _mw2, _tgt2, _wf2)
                    _CP_INV[0] = _invt
                _ein2, _eel2, _gu2 = _cell_poly_energy(dpn_k, _CPOLY[0])
                _CP_E[0] = (_ein2, _eel2)
                _Jk = _gu2[_CP_INV[0]]
            return _duk, _Jk

        _need_post = a.hex_jac or a.smooth_nb > 0 or a.cell_poly
        # **한 스텝의 본체는 라이브러리 하나** (lib/anchorflow/deform_model.advance).
        # 추론(exe/lb_rollout.py)도 같은 함수를 부른다.
        x2, v_next, dpn, Jf, _corr, _act = DM.advance(
            net, a, _ENS, x, _DT[0], d["cfg"], inertial=_VNL, node_mask=_NM,
            node_vel=_NV, ctrl_scale=_CTRL_SCALE[0], mat=_mv, sec=_tsec,
            dp_fast=((lambda _n, _dt: _DP_FAST[0](_n, _dt, dev))
                     if _DP_FAST[0] is not None else None),
            dp_hook=_DP_HOOK[0], pre_k=_pre_k, on_pin=_on_pin,
            post_jac=(_post_jac if _need_post else None))
        _DPN_LIVE[0] = (npos, dpn)                     # --ov_obj node: 기울기가 붙은 노드 출력
        if _NODE_CAP[0]:
            _sup = torch.zeros(npos.shape[0], dtype=torch.bool, device=dev)
            _sup.index_fill_(0, rows.reshape(-1), True)       # 값 텐서를 장치로 복사하지 않는다 (동기화 없음)
            if _nod_m is None:
                _pc = torch.zeros(x.shape[0], dtype=torch.uint8, device=dev)
            else:
                _hit = _nod_m[rows]
                _pc = (_hit.any(1).to(torch.uint8)
                       + _hit.all(1).to(torch.uint8))
            # GPU 에 둔다 -- 덤프할 때(프레임 끝) 한 번만 CPU 로 옮긴다. 매 평가마다 .cpu() 하면 동기화가 생긴다
            _NODE_LAST[0] = (npos.detach(), dpn.detach(), _sup, _pc)
        _DET_LAST[0] = torch.linalg.det(Jf)
        if _INV_LOG is not None:                       # AF_INV_LOG: 프레임별 뒤집힌 사면체 수 (마지막 step_once 의 것)
            _INV_CUR[0] = (_DET_LAST[0].detach(), rows.detach())
        if _TUN_LOG is not None:                       # AF_TUNNEL_LOG: 바닥을 통째로 통과한 사면체 (마지막 step_once 의 것)
            _TUN_CUR[0] = (npos.detach(), dpn.detach(), rows.detach())
        _BC_LAST[0] = (_corr, _act)
        return (x2, p, v_next, None, dpn, None, dmg, None, fe, Jf)
def traj_F(d):
    """탄성 변형구배 [T,N,3,3]. 궤적에 든 것은 전부 항등이라 위치에서 되살린다."""
    def _msg(sec):
        if not _F_MSG:
            print(f"[Phase2] 궤적의 F 가 항등이라 위치에서 복원한다 "
                  f"(궤적당 {sec:.1f}초)", flush=True)
            _F_MSG.append(1)
    return DM.traj_F(d, FRAME_DT, dev, keep_gpu=bool(a.gpu_data),
                     on_rebuild=_msg)


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
    return DM.traj_mass(d, dev)


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
    vol = _pvol(mass, cfg)
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
with torch.no_grad() if not a.infer else contextlib.nullcontext():
  if not a.infer:                 # 추론 모드: 체크포인트의 in_mu/in_sd 가 덮어쓰므로 재계산하지 않는다
    # 학습이 실제로 넣는 것과 **같은 경로**로 통계를 잡는다.   예전에는 여기만
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
        # 체크포인트를 map_location=cuda 로 읽어 난수 상태(ByteTensor)가 GPU 에 올라와 있다 -- CPU 로 돌린다
        if torch.is_tensor(_v):
            _v = _v.cpu()
        elif isinstance(_v, (list, tuple)) and _v and torch.is_tensor(_v[0]):
            _v = [t_.cpu() for t_ in _v]
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


def probe_ip():
    """고정 확인 상태(첫 검증 궤적의 정답 프레임)에서 한 스텝: 학습과 같은 증분 포텐셜과 관성 잔차 (기울기 없음).
    학습 풀이 모델 자신의 롤아웃으로 오염돼도 이 값은 고정된 상태에서 재므로, 최적화 자체가 되는지 따로 읽는다."""
    net.eval()
    _tag, d = _VAL[0]
    _ts = traj_scope(d); _ts.__enter__()
    try:
        gsel = torch.arange(d["x"].shape[1], device=dev)
        cfg = d["cfg"]; mass = traj_mass(d)[gsel]; vol = _pvol(mass, cfg)
        gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
        ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
        h = FRAME_DT; nrm = float(mass.sum()) * (EXT ** 2) / (h ** 2)
        Es, Rs, Gs, Ps = [], [], [], {}
        T = d["x"].shape[0]
        for t in [int(q) for q in a.probe_frames.split(",") if int(q) < T - 1]:
            x, v = DM.init_state(d, t, gsel, FRAME_DT, a.v0_traj, dev)
            x, v = x.float(), v.float()
            F = take(traj_F(d)[t], gsel).float()
            p = take(d["x"][t], AIDX)
            with torch.enable_grad(), dt_scope(h):
                x2, _, _, _, _, _, _, _, _, _Jd = step_once(d, t, gsel, p, x, v, need_J=False)
            with torch.no_grad():
                x2 = x2.detach(); _Jd = _Jd.detach() if torch.is_tensor(_Jd) else _Jd
                E, _, _, parts = ip_of(x, x2 - x, v, F, mass, vol, cfg, h, ng_, gl_, g=gv, norm=nrm, jac=_Jd)
                rr = ((x2 - x - h * v - (h ** 2) * gv) / EXT).norm(dim=-1).mean()
                Gs.append(DM.frame_err(x2, take(d["x"][t + 1], gsel), EXT))
            Es.append(float(E)); Rs.append(float(rr))
            for k_, v_ in zip(("관성", "탄성", "중력", "경계"), parts):
                Ps.setdefault(k_, []).append(float(v_))
        Ps["한스텝GT오차"] = Gs
        return float(np.mean(Es)), float(np.mean(Rs)), {k_: float(np.mean(v_)) for k_, v_ in Ps.items()}
    finally:
        _ts.__exit__(None, None, None)
        net.train()


def quick_val():
    """홀드아웃 지표를 함께 낸다 -> (비, 목적함수, 출력잔차).

    (1) 짧은 롤아웃의 정지기준 대비 비 -- best 체크포인트의 기준
    (2) **학습과 똑같은 목적함수**(증분 포텐셜, 학습과 같은 정규화)를 홀드아웃 롤아웃의 매 서브스텝에서 잰 평균
    (3) 학습 게이트와 같은 출력잔차(노드별 |dE/dU| 평균의 무차원 환산)의 평균
    (2)(3) 은 2026-09-30 교사 경로를 지울 때 0 으로 남겨둔 것을 되살린 것이다 (2026-10-16).
    롤아웃 상태 F 는 정답 t0 의 F 에서 시작해 학습과 같이 plastic_step 으로 전진한다.
    """
    net.eval()
    # 검증은 탐침(노드 출력)에 대한 기울기만 쓴다 -- 망 파라미터 기울기를 끄면 GNN 활성값을 그래프에 남기지 않는다
    # (값은 같다). 켜 둔 채로는 8층 GNN 활성값이 서브스텝마다 쌓여 44GB 카드에서 OOM 이 났다 (m6lgv8_inertial_fd, 10-19).
    _vreq = [q.requires_grad for q in net.parameters()]
    net.requires_grad_(False)
    try:
        return _quick_val_body()
    finally:
        for q, r_ in zip(net.parameters(), _vreq):
            q.requires_grad_(r_)
        net.train()


def _quick_val_body():
    tot = ref = 0.0
    obj_s = rout_s = 0.0
    n_obj = n_rout = 0
    for _tag, d in _VAL:
        # 전역 MASS/EXT/N_FULL 을 이 궤적 값으로 (풀 루프가 씬 값으로 덮어쓴다)
        _ts = traj_scope(d); _ts.__enter__()
        gsel = torch.arange(d["x"].shape[1], device=dev)
        # 평가 롤아웃(_rollout)과 같은 시작점·같은 초기 속도로 잰다 -- 다르면 검증으로 고른 체크포인트가 평가에서 뒤집힌다
        t0 = int(a.eval_t0[0]) if (a.v0_traj and a.eval_t0) else 3
        x, v = DM.init_state(d, t0, gsel, FRAME_DT, a.v0_traj, dev)
        F = take(traj_F(d)[t0], gsel).float()
        p = take(d["x"][t0], AIDX)
        x_still = x.clone()
        cfg = d["cfg"]
        m = MASS[gsel]
        vol = _pvol(m, cfg)
        gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
        ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
        _hs = FRAME_DT / EVAL_SUB
        nrm = float(m.sum()) * (EXT ** 2) / (_hs ** 2)
        for i in range(min(a.val_len, d["x"].shape[0] - t0 - 1)):
            for _sb in range(EVAL_SUB):
                with torch.enable_grad(), dt_scope(_hs):
                    DM.PROBE[0] = "want"
                    x2, p2, v2, _, _, _, _, _, _, _Jd = step_once(
                        d, t0 + i, gsel, p, x, v, need_J=False)
                    DM.PROBE[0] = None
                    fm = (free_mask(d, x2.shape[0], dev, gsel, x, t0 + i)
                          if a.control else None)
                    if _BC_LAST[0] is not None:
                        _am = _BC_LAST[0][1].reshape(-1) > 0.5
                        fm = (~_am) if fm is None else (fm & ~_am)
                    _emask, _ = det_take()
                    E_ip, dlog, F_tr, _ = ip_of(
                        x, x2 - x, v, F, m, vol, cfg, _hs, ng_, gl_,
                        g=gv, norm=nrm, free=fm, jac=_Jd, elastic_mask=_emask)
                    if math.isfinite(float(E_ip)):
                        obj_s += float(E_ip); n_obj += 1
                    if DM.PROBE[1]:
                        _pz = DM.PROBE[1][0]
                        _gU, = torch.autograd.grad(E_ip * nrm, _pz, retain_graph=True)
                        _mN, = torch.autograd.grad((m.unsqueeze(-1) * x2).sum(), _pz)
                        _mN = _mN[:, 0]
                        _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                        _r = float(_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hs ** 2) / _mN[_sup].mean() / EXT)
                        if math.isfinite(_r):
                            rout_s += _r; n_rout += 1
                    DM.PROBE[1] = None
                with torch.no_grad():
                    F = phys_resid.plastic_step(F_tr, dlog).detach()
                x, p, v = x2.detach(), p2.detach(), v2.detach()
            gt = take(d["x"][t0 + i + 1], gsel)
            fm = (free_mask(d, x.shape[0], dev, gsel, x, t0 + i)
                  if a.control else slice(None))
            tot += float((x[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
            ref += float((x_still[fm] - gt[fm]).norm(dim=-1).mean()) / EXT
        _ts.__exit__(None, None, None)
    return (tot / max(ref, 1e-20), obj_s / max(n_obj, 1),
            (rout_s / n_rout) if n_rout else float("nan"))


def save_ck(name, step):
    # 난수 상태를 함께 남긴다. 이게 없으면 재개한 뒤 창 표본이 다른 흐름을 타서
    # 곡선이 이어지지 않는다 (TensorBoard 에서 바로 보인다).
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                # 크리틱도 함께 남긴다 -- 없으면 재개할 때 가치함수가 0 에서
                # 다시 시작해 정책이 한동안 미래를 못 본다
                "critic": (CRITIC.state_dict() if CRITIC is not None else None),
                "opt_c": (OPT_C.state_dict() if OPT_C is not None else None),
                "step": step, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                # 망을 세운 기준 (출력 크기 0.02·net_ext0, 속도 단위
                # net_ext0/frame_dt). 추론(lb_rollout)이 같은 모델을 세운다
                "net_ext0": _EXT0, "frame_dt": FRAME_DT,
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
                "pool_lgv_eta": _PLGV["eta"],
                "seq": ({k_: (v_.detach().cpu() if torch.is_tensor(v_) else v_) for k_, v_ in SEQ.items()
                         if k_ != "cand"} if a.seq_frames else None),
                # 프레임별 출력 변수 (--out_var). 렌더에서 같은 값을 쓰려면 필요하다
                # out_var 변수는 **상태에 붙어** 있고 상태는 저장하지 않으므로
                # 체크포인트에 넣지 않는다 (프레임 키 시절의 유물이었다).
                "out_var": None},
               os.path.join(a.out, f"{a.tag}_{name}.pt"))
    if a.r2:
        _r2_upload_request()


# --r2 업로드는 **한 번에 하나만** 돈다. 저장마다 rclone 을 떼어 띄우면 2.4 GB 체크포인트
# 업로드가 저장 간격보다 길 때 같은 파일 업로드가 수십 개 쌓여 대역을 다 먹는다 (10-20 실측:
# 풀 학습 12 개·랑제빈 11 개). 업로드 중에 새 저장이 오면 "다음에 올릴 것 있음" 만 표시하고,
# 진행 중인 업로드가 끝나는 순간 그때의 최신 파일을 한 번 더 올린다. 진행 중인 것을 끊고
# 새로 시작하면 업로드가 저장 간격보다 길 때 아무것도 끝나지 않으므로 끊지 않는다.
# 스레드는 daemon 이 아니라서 학습이 끝나도 마지막 업로드를 마치고 프로세스가 끝난다.
_R2_UP = {"lock": threading.Lock(), "running": False, "pending": False}


def _r2_upload_loop():
    while True:
        subprocess.run(f"rclone copy {a.out} {a.r2} --include '*.pt' --include '*.json'",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with _R2_UP["lock"]:
            if not _R2_UP["pending"]:
                _R2_UP["running"] = False
                return
            _R2_UP["pending"] = False


def _r2_upload_request():
    with _R2_UP["lock"]:
        if _R2_UP["running"]:
            _R2_UP["pending"] = True
            return
        _R2_UP["running"] = True
    threading.Thread(target=_r2_upload_loop, daemon=False).start()


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
        vol = _pvol(mass, cfg)
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
    _sc = load_scenes(_FD, a.pool_cfg or os.path.join(_W, "wmats"), _combos,
                      0, dev, seed=a.seed)
    if not _sc:
        raise SystemExit("풀에 넣을 씬이 없다")
    _R = 0.15
    _gl0 = float(_sc[0][1]["cfg"].get("grid_lim", 2.0))
    _PINIT = None
    if a.pool_init_traj:
        _pi = torch.load(a.pool_init_traj, map_location="cpu", weights_only=False)
        _PINIT = tuple(_pi[k][int(a.pool_init_frame)].to(dev).float().contiguous() for k in ("x", "v", "F"))
        print(f"[풀] 고정 초기 상태: {a.pool_init_traj} {a.pool_init_frame} 프레임 (속도 크기 평균 {float(_PINIT[1].norm(dim=-1).mean()):.3f})", flush=True)
    _PDATA = None
    if a.pool_data:
        _PDATA = []
        for _f in sorted(glob.glob(os.path.join(a.pool_data, "*.pt"))):
            _dd = torch.load(_f, map_location="cpu", weights_only=False, mmap=True)
            _PDATA.append((_dd["x"], _dd["v"], _dd["F"]))
        print(f"[풀] 데이터 모드: 학습 궤적 {len(_PDATA)} 개 ({a.pool_data}), 각 {_PDATA[0][0].shape[0]} 프레임", flush=True)
    if a.pg_region_density:
        for _tg, _s in _sc:
            _rr = DM.region_rho(_s["cfg"], _s["x0"].float())
            if _rr is not None:
                _s["mass"] = _s["mass"] / float(_s["cfg"]["density"]) * _rr
                _RHO[id(_s["cfg"])] = _rr
                print(f"[풀] {_tg}: 영역별 밀도 질량 합 {float(_s['mass'].sum()):.5g}", flush=True)
    _DROP = None
    if a.pool_impulse:
        _lo, _hi = (float(q) for q in a.pool_impulse.split(","))
        _c0 = _sc[0][1]["cfg"]; _x00 = _sc[0][1]["x0"].float(); _m00 = _sc[0][1]["mass"].float()
        _imps = [b for b in (_c0.get("boundary_conditions") or []) if b.get("type") == "particle_impulse"]
        if len(_imps) != 1 or float(_imps[0].get("start_time", 0.0)) != 0.0:
            raise SystemExit(f"--pool_impulse: start_time 0 인 particle_impulse 가 하나여야 한다 ({_imps})")
        _bi = _imps[0]
        _q = torch.as_tensor(_bi.get("point", [1, 1, 1]), device=dev, dtype=torch.float32)
        _sz = torch.as_tensor(_bi.get("size", [1, 1, 1]), device=dev, dtype=torch.float32)
        _in = ((_x00 - _q).abs() < _sz).all(-1)
        _sdt = float(_c0["substep_dt"]) * int(_bi.get("num_dt", 1))
        _up = torch.as_tensor(_c0.get("mpm_space_vertical_upward_axis", [0, 0, 1]), device=dev, dtype=torch.float32)
        _DROP = dict(kind="impulse", s=(_lo, _hi), f=torch.as_tensor(_bi["force"], device=dev, dtype=torch.float32),
                     up=_up / _up.norm(), coef=torch.where(_in, _sdt / _m00, torch.zeros_like(_m00)))
        print(f"[풀] 충격 모드: 공식 힘 {_bi['force']} 를 위쪽 축 {_up.tolist()} 둘레 균일 회전, 크기 x U[{_lo}, {_hi}], "
              f"상자 안 {int(_in.sum())}/{_in.numel()} 입자, {_sdt:g} s", flush=True)
    elif a.pool_drop:
        _h0, _h1, _v0, _v1 = (float(q) for q in a.pool_drop.split(","))
        _c0 = _sc[0][1]["cfg"]
        _fl = ([b for b in _c0["boundary_conditions"] if b["type"] == "surface_collider"] or [None])[0]
    if a.pool_impulse:
        pass
    elif a.pool_drop and _fl is None:
        print("[풀] 바닥(surface_collider)이 없는 씬이라 낙하 모드를 끈다", flush=True)
    elif a.pool_drop:
        _DROP = dict(h=(_h0, _h1), v=(_v0, _v1), zf=float(_fl["point"][2]), gz=float(_c0["g"][2]),
                     margin=3 * _gl0 / int(_c0["n_grid"]))
        print(f"[풀] 낙하 모드: 높이 여유 {_DROP['h']}, 속도 {_DROP['v']}, 바닥 z {_DROP['zf']}", flush=True)
    # 목표점 마진은 **손잡이 반경보다 커야** 한다. 반경만큼만 두면 손잡이가
    # 목표에 닿는 순간 무리의 바깥쪽 입자가 경계에 걸리고, 딸려 오는 부분까지
    # 생각하면 그 자리에서 영역을 벗어난다.
    POOL = StatePool(_sc, a.pool_size, a.n_ctrl, _R, dev, gen,
                     frames=a.pool_frames, thresh=a.pool_thresh,
                     window=a.pool_window,
                     domain=_gl0, margin=_R + 0.15,
                     start_mid=a.pool_start_mid, keep_prob=a.pool_keep,
                     acc=a.ctrl_acc, vmax=a.ctrl_vmax, zbias=a.ctrl_zbias,
                     n_side=a.pool_targets, drop=_DROP, data=_PDATA, init=_PINIT)
    if a.pool_start_k > 0:
        POOL.fix_start_set(a.pool_start_k)
        print(f"[풀] 시작 상태 고정 집합 {a.pool_start_k} 개 (고정 시드)", flush=True)
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
    _pck = (_rng_ck or {}).get("pool") if not (a.resume_fresh or a.infer) else None
    if _pck:
        _nl, _ns = POOL.load_state_dict(_pck)
        _PL_DROP[:] = list((_rng_ck or {}).get("pool_drop_win") or [])
        _rms = (_rng_ck or {}).get("pool_rms")
        if (_rng_ck or {}).get("pool_lgv_eta") is not None:
            _PLGV["eta"] = float(_rng_ck["pool_lgv_eta"])
            print(f"[재개] 랑제빈 eta {_PLGV['eta']:.3e} 이어받음", flush=True)
        if _rms:
            _PL_RMS[0] = float(_rms[0])
        print(f"[재개] 풀 {_nl} 개 상태 이어받음 (씬이 달라 버린 것 {_ns} 개), "
              f"폐기누적 {POOL.n_drop}, 유예누적 {POOL.n_keep}", flush=True)

# ---- 프레임 순차 과적합 상태 (--seq_frames) ------------------------------------
# t: 지금 학습 중인 프레임(0 부터), it: 이 프레임에서 돈 반복 수, prev: 직전 반복의 출력 위치,
# err/iters: 프레임별 (전진한 상태의 GT 오차, 걸린 반복 수), traj: 전진한 위치들 (half, cpu).
SEQ = dict(t=0, it=0, prev=None, err=[], iters=[], capped=0, traj=[])
_DIAG, _DIAG_I = [], [0]
_GATE_LOG, _GATE_N = [], [0, 0]
_OVREF = {}                                                  # --ov_ref 기준해 (위치·F·목적함수)
_LGV = {"eta": None, "T": None}                              # AF_OV_LANGEVIN: 프레임 사이에 이어 쓰는 MALA 걸음 크기
_MINSM = {}                                                  # AF_OV_MIN_MULTI: 궤적 태그 -> 상태별 최솟값
_MINS = {}                                                   # AF_OV_MIN: 상태별 출력만 최적화 최솟값 (상태 적합 초과% 기록)
_GAP = []                                                    # AF_GAP_*: 같은 상태의 망 증분 포텐셜 vs 출력만 최적화 최솟값


def _lgv_eval(ds, gsel, p_st, x, v, F, sc, h, ens, U0, Z, need_grad, t=0):
    """노드 출력 u = U0 + Z 에서 학습과 같은 증분 포텐셜(정규화된 E, ip_of 그대로)과 need_grad 면 ∂E/∂Z (= ∂E/∂u).

    망은 부르지 않는다: U0 (이 상태의 망 출력, 관성 예측 포함·고정) 를 _DP_FAST 로 돌려주고 탐침 "fixed" 로 Z 를 더한다.
    격자·노드특징은 학습 forward 때 만든 것(ens)을 그대로 쓴다. det 통계(_DET_BAD 등)는 건드리지 않는다.
    """
    cfg = sc["cfg"]
    vol = _pvol(sc["mass"], cfg)
    gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
    nrm = sc["_msum"] * (sc["ext"] ** 2) / (h ** 2)
    Zv = Z.detach().clone().requires_grad_(bool(need_grad))
    _sv = (_DP_FAST[0], _NF_HOLD[0], DM.PROBE[0], DM.PROBE[1])
    _DP_FAST[0] = lambda _n, _dt, _dev: (U0,)
    _NF_HOLD[0] = ens
    DM.PROBE[0], DM.PROBE[1] = "fixed", [Zv]
    try:
        with (torch.enable_grad() if need_grad else torch.no_grad()), dt_scope(h):
            x2, p2, v2, _, _, _, _, _, _, Jd = step_once(ds, t, gsel, p_st, x, v, need_J=False)
            fm = free_mask(ds, x.shape[0], dev, gsel, x, t) if a.control else None
            if _BC_LAST[0] is not None:
                _am = _BC_LAST[0][1].reshape(-1) > 0.5
                fm = (~_am) if fm is None else (fm & ~_am)
            _dt_ = _DET_LAST[0]
            _DET_LAST[0] = None
            _emask = (_dt_ >= float(a.det_eps)) if _dt_ is not None else None   # det_take 과 같은 탄성 마스크
            E, dlog, F_tr, _ = ip_of(x, x2 - x, v, F, sc["mass"], vol, cfg, h, ng_, gl_,
                                     g=gv, norm=nrm, free=fm, jac=Jd, elastic_mask=_emask)
            g = torch.autograd.grad(E, Zv)[0].detach() if need_grad else None
    finally:
        _DP_FAST[0], _NF_HOLD[0], DM.PROBE[0], DM.PROBE[1] = _sv
    return dict(Z=Z.detach(), E=E.detach(), g=g, J=Jd.detach(),
                out=(x2.detach(), v2.detach(), p2.detach() if torch.is_tensor(p2) else None,
                     F_tr.detach(), dlog.detach() if torch.is_tensor(dlog) else dlog))


def _lgv_metric(ens0, J, M, free):
    """셀 뒤집힘 로그 장벽 φ(u) = −Σ_c log det J_c 의 헤시안을 노드별 3×3 블록 대각으로: G_n = Σ_{c∋n} w wᵀ + lam·I,
    w = J_c^{-T} g_n (g_n = 사면체 c 에서 노드 n 의 형상함수 기울기). 셀 = 입자가 든 사면체 (tet_id 로 중복 제거).
    -> (G, P = G^{-1}, P^{1/2}) 각 [M,3,3]."""
    lo, lat, nn = ens0[2]
    rows, aux = ens0[3], ens0[7]
    N = rows.shape[0]
    tid = SX.tet_id(lo, lat, nn, aux)
    ut, inv = torch.unique(tid, return_inverse=True)
    rep = torch.zeros(ut.shape[0], dtype=torch.long, device=rows.device).scatter_(
        0, inv, torch.arange(N, device=rows.device))              # 사면체마다 대표 입자 하나 (J 는 사면체 안에서 상수)
    # Kuhn 사슬 꼭짓점 r 의 형상함수 기울기: J = I + Σ_r u_r g_rᵀ, g_r = Ai[perm[r-1]] − Ai[perm[r]] (끝은 한쪽만)
    Ar = lat.Ai.to(J.dtype)[aux[2][rep]]                          # [T,3(r),3]
    gs = torch.zeros(rep.shape[0], 4, 3, device=J.device, dtype=J.dtype)
    gs[:, 1:] += Ar
    gs[:, :3] -= Ar
    Ji, _info = torch.linalg.inv_ex(J[rep])
    Ji = torch.nan_to_num(Ji, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.einsum("tji,trj->tri", Ji, gs)                      # J^{-T} g_r  [T,4,3]
    G = torch.zeros(M, 3, 3, device=J.device, dtype=J.dtype).index_add_(
        0, rows[rep].reshape(-1), (w.unsqueeze(-1) * w.unsqueeze(-2)).reshape(-1, 3, 3))
    dg = G.diagonal(dim1=-2, dim2=-1).mean(-1)
    lam = float(a.lgv_lam) * dg[free].mean().clamp_min(1e-30)
    G = G + lam * torch.eye(3, device=J.device, dtype=J.dtype)
    e, Q = torch.linalg.eigh(G)
    e = e.clamp_min(lam)
    P = (Q * (1.0 / e).unsqueeze(-2)) @ Q.transpose(-1, -2)
    Ph = (Q * e.rsqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
    return G, P, Ph


def _lgv_step(ds, gsel, p_st, x, v, F, sc, h, ens, U0, s0, nod_m, t=0):
    """노드 출력 공간 MALA 한 걸음 (블록 대각 계량 P 를 u0 의 것으로 고정한 근사 MALA). s0 = 시작점 _lgv_eval 결과.

    T = 2·rho·E(u0)/d,  u' = u0 − η P∇E + √(2ηT) P^{1/2} ξ.  제안의 어떤 셀이든 det J <= --det_eps 면 걸음(기울기·잡음 둘 다)을
    반으로 (최대 --lgv_backtrack 번), 그래도 안 되면 u0. 수락:
      log α = −(E'−E0)/T − |u0−u'+aP∇E'|²_G/(2b) + |u'−u0+aP∇E0|²_G/(2b),  a = η·s, b = 2ηT·s² (s = 백트래킹 배율; s=1 이면 4ηT)
    기각이면 u0 로 전진. -> 전진에 쓸 out (x2, v2, p2, F_tr, dlog)."""
    ens0 = ens[0]                                                # 계량은 첫 격자, 평가는 격자 목록 전체 (step_once 가 목록을 받는다)
    M = U0.shape[0]
    free = (torch.ones(M, dtype=torch.bool, device=dev) if nod_m is None else ~nod_m)
    fz = free.unsqueeze(-1).to(U0.dtype)
    d = 3 * int(free.sum())
    E0 = float(s0["E"])
    if not (math.isfinite(E0) and d > 0 and s0["g"] is not None and bool(torch.isfinite(s0["g"]).all())):
        _PLGV["fail"] += 1
        _PLGV["last"] = dict(acc=0, nbt=-1, E0=E0, E1=float("nan"), eta=float(_PLGV["eta"]), J=s0["J"])
        return s0["out"]
    T = 2.0 * float(a.lgv_rho) * max(E0, 1e-30) / d
    G, P, Ph = _lgv_metric(ens0, s0["J"], M, free)
    eta = float(_PLGV["eta"])
    Pg0 = (P @ s0["g"][:, :3].unsqueeze(-1)).squeeze(-1) * fz
    xi = torch.randn(M, 3, device=dev, dtype=U0.dtype, generator=gen)
    nz = (Ph @ xi.unsqueeze(-1)).squeeze(-1) * fz * math.sqrt(2.0 * eta * T)
    s, nbt, r, Zp = 1.0, 0, None, None
    while True:
        Zp = s0["Z"].clone()
        Zp[:, :3] = Zp[:, :3] + s * (-eta * Pg0 + nz)
        r = _lgv_eval(ds, gsel, p_st, x, v, F, sc, h, ens, U0, Zp, True, t=t)
        _dmin, _ef = torch.stack([torch.linalg.det(r["J"]).min(), r["E"].to(r["J"].dtype)]).tolist()
        if math.isfinite(_ef) and _dmin > float(a.det_eps) and bool(torch.isfinite(r["g"]).all()):
            break
        if nbt >= int(a.lgv_backtrack):
            r = None
            break
        s *= 0.5
        nbt += 1
    _PLGV["bt"].append(nbt)
    _PLGV["T"].append(T)
    acc = False
    if r is None:
        _PLGV["fail"] += 1
    else:
        E1 = float(r["E"])
        Pg1 = (P @ r["g"][:, :3].unsqueeze(-1)).squeeze(-1) * fz
        du = (Zp[:, :3] - s0["Z"][:, :3]) * fz
        a_, b_ = eta * s, 2.0 * eta * T * s * s
        fw = du + a_ * Pg0
        bw = -du + a_ * Pg1
        qf = float((fw.unsqueeze(-2) @ G @ fw.unsqueeze(-1)).sum())
        qb = float((bw.unsqueeze(-2) @ G @ bw.unsqueeze(-1)).sum())
        la = -(E1 - E0) / T - qb / (2.0 * b_) + qf / (2.0 * b_)
        _PLGV["dE"].append(E1 / max(E0, 1e-30) - 1.0)
        _u = float(torch.rand(1, device=dev, generator=gen))
        acc = math.isfinite(la) and math.log(max(_u, 1e-300)) < la
    _PLGV["acc"].append(float(acc))
    _PLGV["dEc"].append((float(r["E"]) / max(E0, 1e-30) - 1.0) if acc else 0.0)
    _PLGV["eta"] = min(max(eta * (float(a.lgv_up) if acc else float(a.lgv_down)), 1e-12), 1e6)
    # 마지막 걸음 요약 (AF_ROLL_LGV 롤아웃 로그·F 누적용; J = 실제로 전진한 출력의 야코비안)
    _PLGV["last"] = dict(acc=int(acc), nbt=nbt, E0=E0, E1=(float(r["E"]) if r is not None else float("nan")),
                         eta=eta, J=(r["J"] if acc else s0["J"]))
    _PLGV["n"] = _PLGV.get("n", 0) + 1
    if _PLGV["n"] <= 24:                                          # 처음 몇 걸음은 한 줄씩 (시험 확인용)
        print(f"    [랑제빈 걸음 {_PLGV['n']}] d {d}  E0 {E0:.4e}  T {T:.3e}  eta {eta:.3e}  백트래킹 {nbt}  "
              f"E' {(float(r['E']) if r is not None else float('nan')):.4e}  "
              f"logα {(la if r is not None else float('nan')):+.3f}  수락 {int(acc)}  "
              f"|Δu|최대 {(float(du.norm(dim=-1).max()) if r is not None else 0.0):.3e}", flush=True)
    return r["out"] if acc else s0["out"]


def _ov_reference(ds, gsel, p_st, x, v, F, sc, h, t=0):
    """같은 상태의 **출력만 최적화** 기준해: 망을 고정하고 노드 출력에 더하는 탐침 텐서(변위 3 + 반경 1)를 변수로 학습과 같은
    증분 포텐셜을 L-BFGS 로 수렴시킨다 (망 출력 0 에서 출발 -- 학습 전 망의 출력층은 0 초기화). 학습 경로와 같은 step_once·ip_of."""
    cfg = sc["cfg"]
    vol = _pvol(sc["mass"], cfg)
    gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
    nrm = float(sc["mass"].sum()) * (sc["ext"] ** 2) / (h ** 2)
    _req = [q.requires_grad for q in net.parameters()]
    net.requires_grad_(False)
    DM.PROBE[0] = "want"
    with dt_scope(h):
        step_once(ds, t, gsel, p_st, x, v, need_J=False)
    Z = DM.PROBE[1][0].detach().clone().requires_grad_(True)

    def _ev():
        DM.PROBE[0], DM.PROBE[1] = "fixed", [Z]
        with dt_scope(h):
            _x2, _, _v2, _, _, _, _, _, _, _Jd = step_once(ds, t, gsel, p_st, x, v, need_J=False)
        _E, _dl, _Ft, _ = ip_of(x, _x2 - x, v, F, sc["mass"], vol, cfg, h, ng_, gl_, g=gv, norm=nrm, jac=_Jd)
        _ev.v2 = _v2
        return _E, _x2, phys_resid.plastic_step(_Ft, _dl)

    opt_ = torch.optim.LBFGS([Z], lr=1.0, max_iter=100, history_size=50, tolerance_grad=0.0, tolerance_change=0.0,
                             line_search_fn="strong_wolfe")

    def _clo():
        opt_.zero_grad()
        _E = _ev()[0]
        _E.backward()
        return _E
    hist = []
    for _k in range(40):
        hist.append(float(opt_.step(_clo)))
        if _k > 3 and abs(hist[-2] - hist[-1]) <= 1e-7 * abs(hist[-1]):
            break
    _E, _x2, _Fr = _ev()
    DM.PROBE[0], DM.PROBE[1] = None, None
    for q, r in zip(net.parameters(), _req):
        q.requires_grad_(r)
    print(f"[기준해] 출력만 최적화 L-BFGS {len(hist)} 번(각 최대 100 반복): 목적함수 {hist[0]:.3e} -> {float(_E):.3e}", flush=True)
    return dict(x2=_x2.detach(), v2=_ev.v2.detach(), F=_Fr.detach(), E=float(_E), hist=hist)                              # (나이, 출력잔차) 기록, [전진, 정체] 횟수 (TB 기록 뒤 비움)                                     # --diag_pool 기록과 지금 계산할 풀 자리
if a.seq_frames:
    if not a.pool or not a.pool_init_traj:
        raise SystemExit("--seq_frames 는 풀 경로와 --pool_init_traj 가 필요하다")
    _sq = (_rng_ck or {}).get("seq") if not (a.resume_fresh or a.infer) else None
    if _sq:
        SEQ.update({k_: (v_.to(dev) if (k_ == "prev" and torch.is_tensor(v_)) else v_) for k_, v_ in _sq.items()})
        SEQ["traj"] = [t_.cpu() for t_ in SEQ["traj"]]          # 체크포인트를 GPU 로 읽어 왔다
        for _kk in ("traj_v", "traj_F"):
            if SEQ.get(_kk):
                SEQ[_kk] = [t_.cpu() for t_ in SEQ[_kk]]
        print(f"[순차] 재개: 프레임 {SEQ['t']} 의 {SEQ['it']} 번째 반복부터, 끝난 프레임 {len(SEQ['err'])}", flush=True)
    if not POOL.items:
        POOL.items = [POOL.fresh()]
    POOL.items = POOL.items[:1]
    print(f"[순차] 프레임 {a.seq_frames} 개, 수렴 기준 출력 변화 최대 < {a.seq_tol:g}·h/frame_dt, "
          f"프레임당 상한 {a.seq_max} 반복, 배치는 상태 하나", flush=True)

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
    _vol = _pvol(_mass, _cfg)
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
                    _pvol(_sc["mass"], _cfg),
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

def _snap_inverted(st, iters, eps):
    """det<0 인 사면체를 det>=eps 로 되돌리는 **최소 이동** 사영.

    사면체마다 det 은 그 네 노드의 변위만으로 정해지므로, 제약
    g = eps - det >= 0 을 1차로 푸는 최소 노름 보정은

        d(dp_i) = g / ||grad g||^2 * grad_i g

    이다 (PBD 사영과 같은 식). 노드를 공유하는 사면체들은 평균한다. 손잡이
    노드는 뒤에서 명령으로 다시 덮이므로 따로 빼지 않는다.
    """
    ens = _NF_HOLD[0]
    if iters <= 0 or not ens or ens == "want":
        return 0
    n_fix = 0
    for _k in range(len(ens)):
        var = (st.get("ov") or [])[_k] if st.get("ov") else None
        if var is None:
            continue
        (_ink, _npk, (_lok, _latk, _nnk), _rwk, _lmk, _uqk, _e, _lxk) = ens[_k]
        _ci, rank, _perm = _lxk
        Ai = _latk.Ai
        I3 = torch.eye(3, device=var.device, dtype=var.dtype)
        rows = _rwk.reshape(-1)
        # 변수는 **손잡이를 뺀 자유 노드만** 담고 있다. 사면체는 전체 노드를
        # 가리키므로 전체 장으로 펴서 보정하고, 자유 노드 자리만 되돌려 쓴다.
        _idx = (st.get("ov_idx") or [None] * len(ens))[_k]
        _M = (st.get("ov_M") or [0] * len(ens))[_k] or var.shape[0]

        def _full(v):
            if _idx is None:
                return v
            _z = torch.zeros(_M, 3, dtype=v.dtype, device=v.device)
            return _z.index_copy(0, _idx, v)

        for _ in range(int(iters)):
            with torch.enable_grad():
                dpc = _full(var.detach())[_rwk].requires_grad_(True)
                dd = (dpc[:, 1:] - dpc[:, :-1]).transpose(1, 2)
                Dy = dd.gather(2, rank.unsqueeze(1).expand(-1, 3, -1))
                det = torch.linalg.det(I3 + Dy @ Ai.to(dpc.dtype))
                need = (eps - det).clamp_min(0.0)
                bad = int((det < eps).sum())
                if bad == 0:
                    break
                g = torch.autograd.grad(det.sum(), dpc)[0]      # [N,4,3]
            with torch.no_grad():
                gn2 = (g * g).sum((1, 2)).clamp_min(1e-12)
                corr = (need / gn2).view(-1, 1, 1) * g
                num = torch.zeros(_M, 3, device=var.device, dtype=var.dtype)
                cnt = torch.zeros(_M, 1, device=var.device, dtype=var.dtype)
                num.index_add_(0, rows, corr.reshape(-1, 3).to(var.dtype))
                cnt.index_add_(0, rows,
                               (need > 0).view(-1, 1).repeat(1, 4).reshape(-1, 1)
                               .to(var.dtype))
                upd = num / cnt.clamp_min(1.0)
                var.data.add_(upd if _idx is None else upd[_idx])
            n_fix = max(n_fix, bad)
    return n_fix


def _ov_clip_grads(st):
    """변수 기울기의 전체 노름을 --ov_clip 으로 자른다.

    장벽이 깊이 들어간 셀에서 기울기가 수십 배로 튀면 line search 가 그 방향에
    끌려가 한 스텝을 통째로 버린다. 방향은 그대로 두고 크기만 자른다.
    """
    if a.ov_clip <= 0:
        return
    _vs = [q for q in (st.get("ov") or []) if q is not None]
    if _vs:
        torch.nn.utils.clip_grad_norm_(_vs, a.ov_clip)


def _mk_ov_opt(vs):
    if a.ov_vars == "net":
        vs = [q for q in net.parameters() if q.requires_grad]     # 출력 대신 망 파라미터
    if a.ov_opt == "gd":
        return torch.optim.SGD(vs, lr=a.ov_gd_lr)
    if a.ov_opt == "lbfgs" and a.c1_barrier > 0:
        # 진짜 로그 장벽(+inf)과 함께: 시도점이 무한이면 반감하는 되추적 L-BFGS (같은 반복 수·이력·첫 걸음 규칙)
        _me = os.environ.get("AF_OV_LBFGS_MAX_EVAL")
        return FeasLBFGS(vs, lr=float(os.environ.get("AF_OV_LBFGS_LR", 1.0)), max_iter=max(a.ov_roll, 1),
                         max_eval=(int(_me) if _me else None), history_size=50,
                         tolerance_grad=float(a.ov_tol), tolerance_change=float(a.ov_tol))
    if a.ov_opt == "lbfgs":
        # AF_OV_LBFGS_LR: 초기 걸음 배율(기본 1). 첫 걸음은 곡률 이력이 없어 t = lr·min(1, 1/|g|₁) 로
        # 시작하고 strong_wolfe 선 탐색(최대 25 평가, max_eval 과 무관)이 늘리거나 줄인다.
        # AF_OV_LBFGS_MAX_EVAL: 한 step 의 함수 평가 상한(기본 torch 값 max_iter*5/4)
        _me = os.environ.get("AF_OV_LBFGS_MAX_EVAL")
        return torch.optim.LBFGS(vs, lr=float(os.environ.get("AF_OV_LBFGS_LR", 1.0)),
                                 max_iter=max(a.ov_roll, 1),
                                 max_eval=(int(_me) if _me else None),
                                 history_size=50,
                                 tolerance_grad=float(a.ov_tol),
                                 tolerance_change=float(a.ov_tol),
                                 line_search_fn="strong_wolfe")
    return torch.optim.Adam(vs, lr=a.out_var_lr)


if a.bc_ipg:
    if a.bc_mode != "project" or a.bc_level != "node" or a.bc_mpm or a.bc_soft or a.bc_slip:
        raise SystemExit("--bc_ipg 는 --bc_mode project --bc_level node 와만 (--bc_mpm/--bc_soft/--bc_slip 없이)")
    print("[바닥] i-PG 방식: 면 아래 격자점(노드) 변위 세 성분 0 사영 -- 장벽·접선 벌점 없음", flush=True)

if a.bc_softplus > 0:
    if a.bc_mode != "project" or a.bc_level != "node" or a.bc_ipg or a.bc_mpm or a.bc_soft:
        raise SystemExit("--bc_softplus 는 --bc_mode project --bc_level node 와만 (--bc_ipg/--bc_mpm/--bc_soft 없이)")
    print(f"[바닥] softplus 매핑 ε {a.bc_softplus:g}: 노드 사영 없음, 입자 법선 좌표 softplus + F 에 매핑 야코비안 "
          f"-- 장벽·접선 벌점 없음 (AF_NO_BC={os.environ.get('AF_NO_BC')})", flush=True)

if a.bc_stick > 0:
    if not a.bc_softplus > 0:
        raise SystemExit("--bc_stick 은 --bc_softplus 와 함께만")
    print(f"[바닥] 미끄러짐 방지 매핑 ε {a.bc_stick:g}: 접선 변위 ×σ(s̃/ε) (s̃ = 스텝 끝 법선 거리) -> softplus, "
          f"F 에 ∂x/∂x̃{' + x_n 명시 항' if a.bc_stick_xn else ''} -- 증분 포텐셜 그대로", flush=True)

if a.bc_mpm:
    os.environ["AF_BC_NODE_MPM"] = "1"
    print("[바닥] 접선 구속을 MPM 과 맞춘다 (면 아래만, dx 띠 아님)",
          flush=True)

if a.bc_slip:
    os.environ["AF_BC_SLIP"] = "1"
    print("[바닥] 접선 고정(no-slip) 을 끈다 -- 비관통만 남는다", flush=True)

if a.bc_mode == "barrier":
    _dh = a.bc_dhat if a.bc_dhat > 0 else \
        0.5 * float(cfg0.get("grid_lim", 2.0)) / int(cfg0["n_grid"])
    phys_resid.set_barrier(a.bc_kappa, _dh, a.bc_ext)
    print(f"[바닥] 장벽 에너지 b(d)=-(d-d̂)²ln(d/d̂), kappa {a.bc_kappa:g}, "
          f"d̂ {_dh:.5f}, 연장 {a.bc_ext} -- 하드 사영은 끈다", flush=True)

_setup_inv_barrier(a)


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
        if a.ov_vars == "net":              # 변수 = 망 파라미터: 망 출력을 그대로 쓰고 최적화기만 만든다
            if st.get("ov_opt") is None:
                st["ov"] = [q for q in net.parameters() if q.requires_grad]
                st["ov_opt"] = _mk_ov_opt(st["ov"])
            return out
        if st.get("ov") is None or any(o is None for o in st["ov"]):   # _ov_fast 가 빈 자리만 만들어 둔 경우도
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

        앙상블이면 격자마다 변수 한 벌씩 들고 있는다 (`_ENS_K` 로 고른다).
        최적화기는 **모든 격자의 변수가 만들어진 뒤** _ov_once 가 만든다.
        """
        st = _OV_CUR[0]
        if st is None or a.ov_vars == "net":     # 망을 건너뛰지 않는다
            return None
        _k = _ENS_K[0]
        if st.get("ov") is None:
            st["ov"] = [None] * _ENS_N[0]
            st["ov_idx"] = [None] * _ENS_N[0]
            st["ov_M"] = [0] * _ENS_N[0]
        if st["ov"][_k] is None:
            if a.ov_init == "net" or M is None:
                return None                      # 망 출력으로 초기화한다
            # **손잡이 노드는 자유변수에서 아예 뺀다.**
            _hm = _NOD_M[0]
            st["ov_idx"][_k] = (None if _hm is None else
                                (~_hm).nonzero(as_tuple=True)[0])
            _no = 4 if a.gregory else (WF.C1_NOUT if a.interp == "c1wf" else 3)   # --gregory: 변위 3 + 반경 1, c1wf: 54
            z = torch.zeros(M, _no, dtype=dtype, device=device)
            if a.ov_init in ("target", "predict"):
                if not torch.is_tensor(_OV_TGT[0]):
                    raise SystemExit(f"--ov_init {a.ov_init} 은 --ov_roll 에서만")
                z[:, :3] = z[:, :3] + _OV_TGT[0].to(dtype).reshape(-1, 3)
            if a.ov_init == "randn":
                z = a.ov_init_std * torch.randn(M, _no, dtype=dtype,
                                                device=device)
            st["ov_M"][_k] = M
            if st["ov_idx"][_k] is not None:
                z = z[st["ov_idx"][_k]]
            st["ov"][_k] = z.requires_grad_(True)
        _v = st["ov"][_k]
        if st["ov_idx"][_k] is not None:
            _z = torch.zeros(st["ov_M"][_k], _v.shape[-1], dtype=_v.dtype,
                             device=_v.device)
            return (_z.index_copy(0, st["ov_idx"][_k], _v),)
        return (_v,)

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
    # 실행마다 따로 (태그 + --out 이름). 예전에는 모든 실행이 tb/MAIN 하나에
    # 섞여 들어갔다.
    _TBD = os.path.join(a.tb, f"{a.tag}_{os.path.basename(os.path.normpath(a.out))}")
    TBW = SummaryWriter(_TBD, purge_step=step0 if step0 else None)
    print(f"[TB] {_TBD}", flush=True)

if not a.pool and a.iters > 0:
    raise SystemExit("--no_pool 은 학습 경로가 없다. 교사 궤적 학습 경로는 "
                     "삭제됐다 (학습은 정지 상태에서 출발해 모델이 상태를 "
                     "갱신하는 상태 풀만 쓴다)")

if a.infer:                           # 추론만: 체크포인트 단계에서 바로 평가로 (예전엔 --iters 까지 남은 반복을 학습한 뒤 평가했다)
    print(f"[추론] 학습 없이 평가만 (체크포인트 단계 {step0})", flush=True)
pbar = tqdm(range(step0, step0 if a.infer else a.iters), desc="학습", ncols=90)
if a.pool_adv == "langevin":
    if _PLGV["eta"] is None:
        _PLGV["eta"] = float(a.lgv_eta0)
    print(f"[랑제빈] 풀 전진 = 증분 포텐셜 MALA 한 걸음 (시작점 {a.lgv_start}, rho {a.lgv_rho:g}, eta {_PLGV['eta']:.3e}, "
          f"백트래킹 {a.lgv_backtrack}, lam {a.lgv_lam:g}, det_w {a.det_w:g})", flush=True)
for it in pbar:
    L = a.unroll            # 학습 중 펼치기 길이는 바꾸지 않는다
    if a.pool_adv == "langevin":
        torch.cuda.synchronize()
        _PLGV["t_it0"], _PLGV["t_cur"] = time.perf_counter(), 0.0
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
        picks = ([("pool", 0)] if a.seq_frames else
                 [("pool", _DIAG_I[0])] if a.diag_pool else POOL.sample(a.batch - n_fresh, n_fresh))
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
            if os.environ.get("AF_GAP_POOL"):
                # 진단 (사용자 10-17): 학습 풀 상태에서 같은 상태의 망 증분 포텐셜(탐침 0 = 망 출력 그대로)과
                # 망 출력에서 출발한 출력만 최적화(L-BFGS) 최솟값. 학습·전진 없이 N 개 모아 기록하고 끝낸다
                _r = _ov_reference(ds, gsel, p_st, x, v, F, sc, _hp)
                _GAP.append(dict(kind="pool", age=int(st["age"]), h=float(_hp), E_model=_r["hist"][0],
                                 E_min=_r["E"], hist=_r["hist"]))
                if len(_GAP) >= int(os.environ.get("AF_GAP_N", "40")):
                    json.dump(_GAP, open(os.environ["AF_GAP_POOL"], "w"))
                    print(f"[간극 기록] {os.environ['AF_GAP_POOL']} 상태 {len(_GAP)}", flush=True)
                    raise SystemExit(0)
                continue
            if a.ov_seq:
                # 출력만 최적화 궤적: 이 상태의 기준해를 구해 후보로 두고(아래 순차 전진이 그대로 쓴다) 망 학습은 건너뛴다
                _r = _ov_reference(ds, gsel, p_st, x, v, F, sc, _hp)
                SEQ["cand"] = (_r["x2"], _r["v2"], _r["F"], None, float(_hp))
                SEQ["prev"] = _r["x2"]                       # 출력 변화 0 -> 바로 수렴 판정
                if TBW is not None:
                    TBW.add_scalar("출력만최적화궤적/목적함수", _r["E"], SEQ["t"])
                    TBW.add_scalar("출력만최적화궤적/LBFGS외곽횟수", len(_r["hist"]), SEQ["t"])
                continue
            if a.ov_ref and not _OVREF:
                _OVREF.update(_ov_reference(ds, gsel, p_st, x, v, F, sc, _hp))
                if TBW is not None:
                    for _i, _e in enumerate(_OVREF["hist"]):
                        TBW.add_scalar("기준해/목적함수_LBFGS", _e, _i)
            _LGV_ON = (a.pool_adv == "langevin")
            DM.PROBE[0] = "want" if (a.pool_gate > 0 or a.out_var or _LGV_ON) else None
            if _LGV_ON:
                _NF_HOLD[0] = "want"                     # 격자·노드특징을 붙잡아 랑제빈 평가에서 재사용
            with dt_scope(_hp), _tsec("step_once"):
                x2, p2, v2, _J, _dp, _ai, _dmg, _cr, _fe, _Jd = step_once(
                    ds, 0, gsel, p_st, x, v, need_J=False)
            DM.PROBE[0] = None
            if _LGV_ON:
                _lgv_ens, _NF_HOLD[0] = _NF_HOLD[0], None
                _lgv_nodm = _NOD_M[0]
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
            vol = _pvol(sc["mass"], cfg)
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
            # 출력 공간 정류 잔차 (--pool_gate 판정·기록): 증분 포텐셜을 노드 변위 출력 U 로 미분한 노드별 벡터의 크기에
            # h²/노드질량 을 곱해 길이로 바꾸고 L 로 나눈 노드 평균. 노드질량 = ∂(Σ m x₂)/∂U = Σ_p m_p W_pn (스키닝 가중치 그대로).
            # 탄성·접촉·관성이 모두 들어 있고, 출력이 증분 포텐셜의 최소점이면 0 이다.
            _rout_t = None
            _gnorm_t = None
            _gU = None
            if (a.pool_gate > 0 or a.out_var or _LGV_ON) and DM.PROBE[1]:
                _pz = DM.PROBE[1][0]                         # 격자 하나 (ens_off 1) 기준
                _gU, = torch.autograd.grad(E_ip * nrm, _pz, retain_graph=True)
                _mN, = torch.autograd.grad((sc["mass"].unsqueeze(-1) * x2).sum(), _pz, retain_graph=True)
                _mN = _mN[:, 0]
                _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                # 노드별 기울기 크기의 **단순 평균**에 고정 상수(평균 노드 질량)로 단위만 맞춘다. 노드마다 자기 질량으로 나누면
                # 질량이 거의 없는 가장자리 노드가 평균을 지배해 수렴해도 값이 커진다 (보정 실험 2026-10-12)
                _mbar = _mN[_sup].mean()
                _rout_t = (_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hp ** 2) / _mbar / sc["ext"]).detach()
                _gnorm_t = (_gU[:, :3][_sup].norm() * (_hp ** 2) / _mbar / sc["ext"]).detach()
                if TBW is not None and it % a.gate_log_every == 0:   # 진단: 기울기 전체 L2 노름(원값·같은 단위 환산)과 노드 최댓값
                    _gn_all = _gU[:, :3][_sup].norm()
                    _gmax = _gU[:, :3].norm(dim=-1)[_sup].max()
                    _cv = (_hp ** 2) / _mbar / sc["ext"]
                    TBW.add_scalar("게이트/출력기울기_전체노름_원값", float(_gn_all), it)
                    TBW.add_scalar("게이트/출력기울기_전체노름_환산", float(_gn_all * _cv), it)
                    TBW.add_scalar("게이트/출력기울기_노드최대_환산", float(_gmax * _cv), it)
                    TBW.add_scalar("게이트/노드수", float(_sup.sum()), it)
                    if _gU.shape[1] > 3:                       # 셀 변환 반경 채널의 기울기
                        TBW.add_scalar("게이트/반경기울기_전체노름", float(_gU[:, 3][_sup].norm()), it)
                    # 기준 후보를 상태의 첫 값 대비 비율로도 (같은 상태를 계속 보는 순차·보정 모드에서 뜻이 있다)
                    _first = st.setdefault("_diag_first", dict(r=float(_rout_t), g=float(_gn_all), E=float(E_ip.detach())))
                    TBW.add_scalar("기준후보/출력잔차_처음대비", float(_rout_t) / max(_first["r"], 1e-30), it)
                    TBW.add_scalar("기준후보/전체노름_처음대비", float(_gn_all) / max(_first["g"], 1e-30), it)
                    TBW.add_scalar("기준후보/목적함수_처음대비", float(E_ip.detach()) / max(_first["E"], 1e-30), it)
                    TBW.add_scalar("기준후보/목적함수", float(E_ip.detach()), it)
                    for _nm, _pv in zip(("관성", "탄성", "중력", "경계"), _pt):
                        TBW.add_scalar(f"목적함수항/{_nm}", float(_pv), it)
                    # 실제 오차: 고정 정답 상태(--pool_init_traj 의 --pool_init_frame)면 다음 프레임 정답과 비교
                    if a.pool_init_traj and a.seq_frames and _VAL:
                        _gd = _VAL[0][1]; _tn = int(a.pool_init_frame) + int(SEQ["t"]) + 1
                        if _tn < _gd["x"].shape[0] and _hp == FRAME_DT:
                            _gsel = torch.arange(x2.shape[0], device=dev)
                            _xg = take(_gd["x"][_tn], _gsel).float()
                            TBW.add_scalar("실제오차/한스텝위치_L대비", float((x2.detach() - _xg).norm(dim=-1).mean()) / sc["ext"], it)
                            _Fg = take(traj_F(_gd)[_tn], _gsel).float()
                            _Fm = phys_resid.plastic_step(F_tr, dlog).detach()
                            TBW.add_scalar("실제오차/한스텝F_평균", float((_Fm - _Fg).flatten(1).norm(dim=1).mean()), it)
                    if _OVREF and SEQ["t"] == 0:                # 출력만 최적화 기준해와의 차이 (사용자: 정답은 이쪽)
                        _Fm2 = phys_resid.plastic_step(F_tr, dlog).detach()
                        TBW.add_scalar("기준해오차/위치_L대비", float((x2.detach() - _OVREF["x2"]).norm(dim=-1).mean()) / sc["ext"], it)
                        TBW.add_scalar("기준해오차/F_평균", float((_Fm2 - _OVREF["F"]).flatten(1).norm(dim=1).mean()), it)
                        TBW.add_scalar("기준해오차/목적함수_초과비", float(E_ip.detach()) / max(_OVREF["E"], 1e-30), it)
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
            _rout = float(_rout_t) if _rout_t is not None else float("nan")
            _GATE_LOG.append((int(st["age"]), _rout))
            # 전진 판정값: grad = 기울기 전체 노름(환산), objdec = 직전 방문 대비 목적함수 상대 감소 (첫 방문은 전진 안 함)
            _Enow = float(E_ip.detach())
            if a.pool_gate_mode == "objdec":
                _Eprev = st.get("_E_last")
                _gate_val = (float("inf") if _Eprev is None else (_Eprev - _Enow) / max(abs(_Eprev), 1e-30))
            else:
                _gate_val = float(_gnorm_t) if _gnorm_t is not None else float("nan")
            st["_E_last"] = _Enow
            _gate_ok = bool(_gate_val < a.pool_gate) if a.pool_gate_mode == "objdec" else bool(_gate_val <= a.pool_gate)
            if a.pool_gate_mode == "pgrad":
                _gate_val, _gate_ok = float("nan"), False     # 역전파 뒤에 정한다 (손실이 유한하지 않으면 전진 안 함)
            if a.diag_pool:
                # 진단: 기록만 하고 역전파·전진·폐기 없이 다음 상태로
                _DIAG.append(dict(age=int(st["age"]), res=float(res), hist=(float(np.mean(st["hist"])) if st["hist"] else None),
                                  E=float(E_ip.detach()), parts=[float(q) for q in _pt], oob=float(_oob),
                                  elapsed=float(st["elapsed"]), rout=_rout))
                continue
            if a.pool_whiten:
                _PL_RMS[0] = 0.99 * _PL_RMS[0] + 0.01 * _lv ** 2
                loss_p = loss_p / max(_PL_RMS[0] ** 0.5, 1e-12)
            if math.isfinite(_lv):
                _pg0 = ([q.grad.detach().clone() if q.grad is not None else None for q in net.parameters()]
                        if a.pool_gate_mode == "pgrad" else None)
                with _tsec("역전파"):
                    (loss_p / a.batch).backward()
                if _pg0 is not None:
                    # 이 상태 하나의 파라미터 기울기 = 역전파 뒤 - 앞 (배치 평균이라 batch 배 되돌린다)
                    _pgs = torch.zeros((), device=dev)
                    for q, g0 in zip(net.parameters(), _pg0):
                        if q.grad is not None:
                            _dg = q.grad.detach() - (g0 if g0 is not None else 0.0)
                            _pgs = _pgs + (_dg * _dg).sum()
                    _gate_val = float(_pgs.sqrt()) * a.batch
                    _gate_ok = bool(_gate_val <= a.pool_gate)
                if a.out_var and st.get("ov_opt") is not None:
                    # 망 파라미터가 아니라 **이 상태의 출력 변수**를 갱신한다
                    st["ov_opt"].step()
                    st["ov_opt"].zero_grad(set_to_none=True)
            elif _PL_MSG == []:
                _PL_MSG.append(1)
                print(f"[풀] 씬 {tag} 손실이 비유한 -- 이 배치는 건너뛴다",
                      flush=True)
            _lgv_out = None
            if _LGV_ON:
                # 풀 상태 전진 = 노드 출력 공간 증분 포텐셜 MALA 한 걸음 (게이트 없음, 항상 전진)
                if dev == "cuda" or str(dev).startswith("cuda"):
                    torch.cuda.synchronize()
                _tp0 = time.perf_counter()
                with _tsec("랑제빈"):
                    _U0 = _dp.detach()
                    _Zm = torch.zeros_like(_U0)
                    if a.lgv_start == "model":
                        # 시작점 = 모델 출력 (Z0 = 0). 기울기는 위에서 구한 _gU (= ∂(E·nrm)/∂u) 를 재활용
                        _s0 = dict(Z=_Zm, E=E_ip.detach(),
                                   g=(_gU / nrm if _gU is not None else None), J=_Jd.detach(),
                                   out=(x2.detach(), v2.detach(), p2.detach() if torch.is_tensor(p2) else None,
                                        F_tr.detach(), dlog.detach() if torch.is_tensor(dlog) else dlog))
                    else:
                        # 시작점 = 관성 예측 노드 변위 h·v_n + h²g (Z0 = u_inertial − u_θ), 그 점에서 기울기 한 번 더 (망 기울기 없음)
                        _e0 = _lgv_ens[0]
                        _uin = DM.node_inertial(x, v, sc["mass"], _e0[3], _e0[4], _e0[1].shape[0],
                                                sc["cfg"]["g"], _hp)[0]
                        _Zi = _Zm.clone()
                        _Zi[:, :3] = (_uin - _U0[:, :3]).detach()
                        _s0 = _lgv_eval(ds, gsel, p_st, x, v, F, sc, _hp, _lgv_ens, _U0, _Zi, True)
                    _lgv_out = _lgv_step(ds, gsel, p_st, x, v, F, sc, _hp, _lgv_ens, _U0, _s0, _lgv_nodm)
                if dev == "cuda" or str(dev).startswith("cuda"):
                    torch.cuda.synchronize()
                _PLGV["t_cur"] += time.perf_counter() - _tp0
                del _lgv_ens
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
                if a.seq_frames:
                    # 전진하지 않고 이번 출력을 후보로만 둔다 -- 수렴 판정 뒤 이것으로 전진
                    SEQ["cand"] = (x2.detach(), v2.detach(),
                                   phys_resid.plastic_step(F_tr, dlog).detach(),
                                   p2.detach() if torch.is_tensor(p2) else None, float(_hp))
                elif _lgv_out is not None:
                    _lx2, _lv2, _lp2, _lFt, _ldl = _lgv_out
                    st["x"], st["v"] = _lx2, _lv2
                    st["F"] = phys_resid.plastic_step(_lFt, _ldl).detach()
                    st["p"] = _lp2
                elif a.pool_gate > 0 and not _gate_ok:
                    _GATE_N[1] += 1                          # 기준 못 넘음: 전진 없이 같은 상태로 남긴다
                elif not a.out_var:
                    if a.pool_gate > 0:
                        _GATE_N[0] += 1
                        st.pop("_E_last", None)                  # 새 상태 -- 감소율 비교를 처음부터
                    st["x"] = x2.detach()
                    st["v"] = _pg_vdamp(v2.detach(), ds, _hp)
                    st["F"] = phys_resid.plastic_step(F_tr, dlog).detach()
                    st["p"] = p2.detach() if torch.is_tensor(p2) else None
                # out_var 는 상태를 **전진시키지 않는다** -- 같은 상태의 노드
                # 출력을 계속 최적화해 하한까지 내린다 (노드 집합도 고정된다)
            if _LGV_ON and not a.seq_frames:
                POOL.put_back_gate(slot if kind == "pool" else None, st, advanced=True,
                                   rdrop=a.pool_rdrop, expire=a.pool_expire, dt_frac=1.0 / _sub,
                                   stuck_max=0)
            elif a.pool_gate > 0 and not a.seq_frames:
                POOL.put_back_gate(slot if kind == "pool" else None, st,
                                   advanced=_gate_ok and not a.out_var,
                                   rdrop=a.pool_rdrop, expire=a.pool_expire, dt_frac=1.0 / _sub,
                                   stuck_max=a.pool_stuck_max)
            elif not a.seq_frames:
                POOL.put_back(slot if kind == "pool" else None, st, res,
                              prev=_prev, dt_frac=1.0 / _sub)
        # 폐기율은 **최근 100 반복에서 배치 대비 몇 %가 죽었는가** 로 둔다.
        # 누적 수를 전체 반복으로 나누면 추세가 안 보이고 값도 오해를 부른다.
        if a.diag_pool:
            opt.zero_grad(set_to_none=True)
            _DIAG_I[0] += 1
            if _DIAG_I[0] >= len(POOL.items):
                json.dump(dict(step=int(step0), n=len(_DIAG), items=_DIAG), open(a.diag_pool, "w"))
                print(f"[진단] {a.resume} 단계 {step0}: 풀 {len(_DIAG)} 개 -> {a.diag_pool}", flush=True)
                raise SystemExit(0)
            continue
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
        if a.seq_frames and SEQ.get("cand") is not None:
            _cx, _cv, _cF, _cp, _ch = SEQ.pop("cand")
            _dmax = (float((_cx - SEQ["prev"]).abs().max()) if SEQ["prev"] is not None else float("inf"))
            SEQ["it"] += 1
            if TBW is not None and it % 20 == 0:
                TBW.add_scalar("순차/출력변화최대", _dmax, it)
                TBW.add_scalar("순차/프레임", SEQ["t"], it)
            if _dmax < a.seq_tol * _ch / FRAME_DT or SEQ["it"] >= a.seq_max:
                # 수렴(또는 상한) -- 수렴한 출력으로 한 프레임 전진, 가중치는 그대로 다음 프레임의 초기값
                _st0 = POOL.items[0]
                _st0["x"], _st0["v"], _st0["F"], _st0["p"] = _cx, _cv, _cF, _cp
                _st0["age"] += 1
                SEQ["t"] += 1
                _gtd = _VAL[0][1]
                _e = (DM.frame_err(_cx, take(_gtd["x"][SEQ["t"]], torch.arange(_cx.shape[0], device=dev)),
                                   float(_gtd.get("_ext", EXT))) if SEQ["t"] < _gtd["x"].shape[0] else float("nan"))
                SEQ["err"].append(_e); SEQ["iters"].append(SEQ["it"]); SEQ["capped"] += int(_dmax >= a.seq_tol * _ch / FRAME_DT)
                if SEQ["t"] % a.seq_save_every == 0:
                    SEQ["traj"].append(_cx.half().cpu())
                    if a.ov_seq:                             # 출력만 최적화 궤적은 v·F 까지 (다음 실험의 상태로 쓴다)
                        SEQ.setdefault("traj_v", []).append(_cv.cpu()); SEQ.setdefault("traj_F", []).append(_cF.cpu())
                        SEQ["traj"][-1] = _cx.cpu()
                print(f"  [순차 프레임 {SEQ['t']}] 반복 {SEQ['it']}{' (상한)' if _dmax >= a.seq_tol * _ch / FRAME_DT else ''}  "
                      f"출력변화 {_dmax:.2e}  GT 오차 {_e * 100:.3f}%", flush=True)
                if TBW is not None:
                    TBW.add_scalar("순차/GT오차_프레임별", _e, SEQ["t"])
                    TBW.add_scalar("순차/반복수_프레임별", SEQ["it"], SEQ["t"])
                    TBW.add_scalar("순차/상한도달누적", SEQ["capped"], SEQ["t"])
                SEQ["it"], SEQ["prev"] = 0, None
                if SEQ["t"] >= a.seq_frames:
                    save_ck("last", it + 1)
                    torch.save({"x": torch.stack(SEQ["traj"]), "err": SEQ["err"], "iters": SEQ["iters"],
                                "v": (torch.stack(SEQ["traj_v"]) if SEQ.get("traj_v") else None),
                                "F": (torch.stack(SEQ["traj_F"]) if SEQ.get("traj_F") else None),
                                "init_frame": int(a.pool_init_frame), "traj": a.pool_init_traj},
                               os.path.join(a.out, f"{a.tag}_seq_traj.pt"))
                    print(f"[순차] {a.seq_frames} 프레임 끝 -- 학습 중 궤적 평균 GT 오차 "
                          f"{float(np.nanmean(SEQ['err'])) * 100:.3f}%, 상한 도달 {SEQ['capped']} 프레임", flush=True)
                    break
            else:
                SEQ["prev"] = _cx
        if a.pool_adv == "langevin":
            torch.cuda.synchronize()
            _PLGV["t_it"].append(time.perf_counter() - _PLGV["t_it0"])
            _PLGV["t_pool"].append(_PLGV["t_cur"])
        if (it % 20 == 0 or it == a.iters - 1) and a.pool_adv == "langevin" and _PLGV["acc"]:
            _tit, _tpl = float(np.mean(_PLGV["t_it"])), float(np.mean(_PLGV["t_pool"]))
            _tbw = TBW if TBW is not None else type("_N", (), {"add_scalar": staticmethod(lambda *q, **k: None)})
            _tbw.add_scalar("랑제빈/수락률", float(np.mean(_PLGV["acc"])), it)
            _tbw.add_scalar("랑제빈/백트래킹_평균", float(np.mean(_PLGV["bt"])) if _PLGV["bt"] else 0.0, it)
            _tbw.add_scalar("랑제빈/백트래킹_실패누적", _PLGV["fail"], it)
            if _PLGV["dE"]:
                _tbw.add_scalar("랑제빈/E증가_비율", float(np.mean(_PLGV["dE"])), it)
                _tbw.add_scalar("랑제빈/E증가_비율_중앙", float(np.median(_PLGV["dE"])), it)
            _tbw.add_scalar("랑제빈/E증가_비율_채택", float(np.mean(_PLGV["dEc"])), it)
            _tbw.add_scalar("랑제빈/eta", float(_PLGV["eta"]), it)
            if _PLGV["T"]:
                _tbw.add_scalar("랑제빈/온도_중앙", float(np.median(_PLGV["T"])), it)
            _tbw.add_scalar("속도/반복당_초", _tit, it)
            _tbw.add_scalar("속도/모델업데이트_초", _tit - _tpl, it)
            _tbw.add_scalar("속도/풀업데이트_초", _tpl, it)
            if it % 100 == 0 or it == a.iters - 1:
                print(f"  [랑제빈 {it}] 수락 {np.mean(_PLGV['acc']):.3f}  백트래킹 {np.mean(_PLGV['bt'] or [0]):.2f}  "
                      f"E증가 {np.mean(_PLGV['dE'] or [float('nan')]):+.3f}  eta {_PLGV['eta']:.3e}  실패누적 {_PLGV['fail']}  "
                      f"반복당 {_tit:.3f}초 (모델 {_tit - _tpl:.3f} / 풀 {_tpl:.3f})  전진프레임 평균 {arel:.1f}", flush=True)
            for _k in ("acc", "bt", "dE", "dEc", "T", "t_it", "t_pool"):
                _PLGV[_k].clear()
        if it % 20 == 0:
            pbar.set_postfix(E=f"{lx:.3e}", 잔차=f"{lres:.3e}",
                             폐기=f"{POOL.n_drop}",
                             프레임=f"{arel:.1f}", gn=f"{float(gn):.1e}")
            if TBW is not None:
                TBW.add_scalar("학습/목적함수", lx, it)
                TBW.add_scalar("학습/잔차", lres, it)
                TBW.add_scalar("학습/기울기노름", float(gn), it)
                if _GATE_LOG:
                    _gr = np.asarray([q[1] for q in _GATE_LOG], dtype=np.float64)
                    _gr = _gr[np.isfinite(_gr)]
                    if len(_gr):
                        TBW.add_scalar("게이트/출력잔차_중앙", float(np.median(_gr)), it)
                        TBW.add_scalar("게이트/출력잔차_상위10%", float(np.quantile(_gr, 0.9)), it)
                    if a.pool_gate > 0:
                        TBW.add_scalar("게이트/통과율", _GATE_N[0] / max(_GATE_N[0] + _GATE_N[1], 1), it)
                        TBW.add_scalar("풀/무작위폐기누적", getattr(POOL, "n_rdrop", 0), it)
                        TBW.add_scalar("풀/정체폐기누적", getattr(POOL, "n_stuckdrop", 0), it)
                        TBW.add_scalar("풀/만기누적", getattr(POOL, "n_expire", 0), it)
                        _stk = [q.get("stuck", 0) for q in POOL.items if q is not None] or [0]
                        TBW.add_scalar("풀/정체_최대", max(_stk), it)
                        TBW.add_scalar("풀/정체_평균", float(np.mean(_stk)), it)
                    if it % 200 == 0 and len(_gr):
                        TBW.add_histogram("게이트분포/출력잔차", np.log10(np.maximum(_gr, 1e-12)), it)
                    _GATE_LOG.clear(); _GATE_N[0] = _GATE_N[1] = 0
                TBW.add_scalar("학습/학습률", float(opt.param_groups[0]["lr"]), it)
                TBW.add_scalar("풀/폐기누적", POOL.n_drop, it)
                TBW.add_scalar("풀/폐기율", still, it)
                if a.pool_keep > 0:
                    TBW.add_scalar("풀/유예누적", POOL.n_keep, it)
                TBW.add_scalar("풀/NaN배치", getattr(POOL, "n_nan", 0), it)
                TBW.add_scalar("풀/채움", len(_alive) / max(POOL.size, 1), it)
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
            _v, _vo, _vr = quick_val()
            if TBW is not None:
                TBW.add_scalar("검증/비", _v, it)
                TBW.add_scalar("검증/증분포텐셜", _vo, it)
                TBW.add_scalar("검증/출력잔차", _vr, it)
            if _v < _best:
                _best = _v
                save_ck("best", it + 1)
                print(f"  [검증 {it+1}] 비 {_v:.4f} 증분포텐셜 {_vo:.3e} 출력잔차 {_vr:.3e} -- best 갱신", flush=True)
            else:
                print(f"  [검증 {it+1}] 비 {_v:.4f} 증분포텐셜 {_vo:.3e} 출력잔차 {_vr:.3e} (best {_best:.4f})",
                      flush=True)
        if a.probe_every and (it == 0 or (it + 1) % a.probe_every == 0):
            _pe, _pr, _pp = probe_ip()
            print(f"  [확인 {it+1}] E {_pe:.4e}  관성잔차 {_pr:.3e}  한스텝GT {_pp['한스텝GT오차']:.3e}", flush=True)
            if TBW is not None:
                TBW.add_scalar("확인/증분포텐셜", _pe, it)
                TBW.add_scalar("확인/관성잔차", _pr, it)
                for _k, _v in _pp.items():
                    TBW.add_scalar(f"확인/{_k}", _v, it)
        if (it + 1) % a.save_every == 0 or it == a.iters - 1:
            save_ck("last", it + 1)
            if a.keep_ckpts:
                save_ck(f"it{it + 1:06d}", it + 1)
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
        _v, _vo, _vr = quick_val()
        if TBW is not None:
            TBW.add_scalar("검증/비", _v, it)
            TBW.add_scalar("검증/증분포텐셜", _vo, it)
            TBW.add_scalar("검증/출력잔차", _vr, it)
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
_FCUM, _FDUMP = [None], []
_VDUMP = []                 # 롤아웃 덤프: 프레임 경계 속도 (--ov_roll)
_ROLL_IP = [] if os.environ.get("AF_ROLL_IP") else None   # 롤아웃 프레임별 증분 포텐셜 (망: 서브스텝별 E·출력잔차, 출력만 최적화: 최적화 전·후 E)


# AF_OV_SNAP="t:반복:간격:경로" -- 프레임 t 한 번만 반복을 늘리고, 목적함수 평가 '간격' 번마다 그 시점 위치·F 를 저장
#   (최적화 진행을 렌더로 보기 위함; 덤프 형식은 AF_ROLL_DUMP 와 같아 lb_roll_to_h5 로 바로 렌더한다)
_OV_SNAP = None
if os.environ.get("AF_OV_SNAP"):
    _sp = os.environ["AF_OV_SNAP"].split(":")
    _OV_SNAP = dict(t=int(_sp[0]), iters=int(_sp[1]), every=int(_sp[2]), path=_sp[3], done=False)


# AF_GRAD_PROBE=출력디렉토리:프레임,프레임,...[:추가반복] (사용자 10-20 "정지한 끝 프레임에서 셀 탄성 에너지 기울기가 0 인가"):
# 그 프레임의 출력만 최적화에서 u = 관성 예측 / 0 / L-BFGS 해 (/ 추가 반복 해) 각각의
#   노드별 증분 포텐셜 기울기 ∂(E·nrm)/∂u, 탄성항만의 기울기 ∂E_el/∂u (자유 노드, 고정 노드는 변수 밖),
#   셀(사면체)별 자기 탄성 에너지의 꼭지점 기울기 Σ_{p∈셀} (∂E_el/∂J_p) ∇λ_pk, 셀 탄성 에너지·부피, 노드 질량
# 을 저장한다. 계산은 _fwd (출력만 최적화가 쓰는 바로 그 경로: step_once → pts_ip_energy, 화분 고정·영역 밀도·부피 그대로).
_GRAD_PROBE = None
if os.environ.get("AF_GRAD_PROBE"):
    _gpa = os.environ["AF_GRAD_PROBE"].split(":")
    _GRAD_PROBE = {"out": _gpa[0], "frames": set(int(q) for q in _gpa[1].split(",") if q),
                   "extra": int(_gpa[2]) if len(_gpa) > 2 else 0, "skip": bool(os.environ.get("AF_GRAD_PROBE_SKIP"))}
    os.makedirs(_GRAD_PROBE["out"], exist_ok=True)


def _grad_probe(st, _fwd, F, m, vol, cfg, x, v, gv, tag, rec):
    _vs = [q for q in st["ov"] if q is not None]
    with torch.enable_grad():
        x2, _, E, _, _, _pq, Jd = _fwd()
        Jd = Jd if Jd.requires_grad else Jd.requires_grad_(True)
        h_ = _DT[0]
        F_tr = Jd.to(F.dtype) @ F
        psi = phys_resid.psi_of(F_tr, cfg, h_)[0]
        e_p = vol * psi
        E_el = e_p.sum()
        du_ = x2 - x
        _d = du_ - h_ * v - (h_ * h_) * gv
        terms = {"in": (0.5 * m / (h_ * h_) * (_d * _d).sum(-1)).sum()}
        _ip = phys_resid._inv_pen(F_tr, cfg)
        if _ip is not None:
            terms["det"] = (vol * _ip).sum()
        _cp = phys_resid._cell_pen(Jd, cfg)
        if _cp is not None:
            terms["cell"] = (vol * _cp).sum()
        # 바닥: bc_energy 와 같은 식을 장벽(비관통)·접선 벌점(sticky)으로 나눠서
        gl_ = float(cfg.get("grid_lim", 2.0)); dx_ = gl_ / int(cfg["n_grid"])
        k_ = float(os.environ.get("AF_BC_STIFF", 1000.0)); c_ = 0.5 * k_ * m / (h_ * h_)
        for bc in (cfg.get("boundary_conditions") or []):
            if bc.get("type") != "surface_collider" or os.environ.get("AF_NO_BC"):
                continue
            pt = torch.as_tensor(bc["point"], device=x.device, dtype=x.dtype)
            nr = torch.as_tensor(bc["normal"], device=x.device, dtype=x.dtype); nr = nr / nr.norm().clamp_min(1e-12)
            sd = ((x2 - pt) * nr).sum(-1)
            if phys_resid._BARRIER[0] is not None:
                _kap, _dh, _ek = phys_resid._BARRIER[0]
                terms["floor"] = terms.get("floor", 0.0) + (_kap * m / (h_ * h_) * phys_resid._barrier_b(sd, _dh, ext_kind=_ek)).sum()
            else:
                terms["floor"] = terms.get("floor", 0.0) + (c_ * sd.clamp_max(0.0) ** 2).sum()
            if str(bc.get("surface", "sticky")) == "sticky" and not os.environ.get("AF_BC_SLIP"):
                _thr = 0.0 if os.environ.get("AF_BC_NODE_MPM") else dx_
                touch = (((x - pt) * nr).sum(-1) < _thr).to(x.dtype).detach()
                du_t = du_ - (du_ * nr).sum(-1, keepdim=True) * nr
                terms["tan"] = terms.get("tan", 0.0) + (c_ * touch * (du_t * du_t).sum(-1)).sum()
        g_ip = torch.autograd.grad(E * rec["nrm"], _vs[0], retain_graph=True)[0]
        g_terms = {k: torch.autograd.grad(t_, _vs[0], retain_graph=True, allow_unused=True)[0]
                   for k, t_ in terms.items() if torch.is_tensor(t_) and t_.requires_grad}
        g_el, G = torch.autograd.grad(E_el, [_vs[0], Jd])
    ens = _NF_HOLD[0][0] if isinstance(_NF_HOLD[0], list) else None
    _tsum = float(sum(float(t_) for t_ in terms.values())) + float(E_el)
    out = {"E": float(E) * rec["nrm"], "E_el": float(E_el), "E_in": float(_pq[0]), "E_el_ip": float(_pq[1]),
           "du_mean": float((x2 - x).norm(dim=-1).mean()), "E_terms_sum": _tsum,
           "g_ip": g_ip[:, :3].detach().cpu(), "g_el": g_el[:, :3].detach().cpu()}
    for k_t, t_ in terms.items():
        out[f"E_{k_t}"] = float(t_)
        if k_t in g_terms and g_terms[k_t] is not None:
            out[f"g_{k_t}"] = g_terms[k_t][:, :3].detach().cpu()
    if ens is not None:
        npos, rows, lam = ens[1], ens[3], ens[4]
        Xv = npos[rows].to(torch.float64)
        Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
        dlam = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)               # [N,4,3] (barycentric ∇λ)
        gpk = torch.einsum("nab,nkb->nka", G.to(torch.float64), dlam)          # [N,4,3] 입자 p 가 꼭지점 k 에 주는 힘
        cells, inv = torch.unique(rows, dim=0, return_inverse=True)
        nc = cells.shape[0]
        gc = torch.zeros(nc, 4, 3, dtype=torch.float64, device=G.device).index_add_(0, inv, gpk)
        ec = torch.zeros(nc, dtype=torch.float64, device=G.device).index_add_(0, inv, e_p.detach().double())
        vc = torch.zeros(nc, dtype=torch.float64, device=G.device).index_add_(0, inv, vol.detach().double())
        # 검산: 셀 꼭지점 힘을 노드로 모으면 ∂E_el/∂u (자유 노드) 와 같아야 한다
        gn = torch.zeros(npos.shape[0], 3, dtype=torch.float64, device=G.device).index_add_(
            0, rows.reshape(-1), gpk.reshape(-1, 3))
        idx = (st.get("ov_idx") or [None])[0]
        gn_free = gn if idx is None else gn[idx]
        chk = float((gn_free - g_el[:, :3].double()).norm() / g_el[:, :3].double().norm().clamp_min(1e-30))
        mn = torch.zeros(npos.shape[0], dtype=torch.float64, device=G.device).index_add_(
            0, rows.reshape(-1), (lam.double() * m.double().unsqueeze(1)).reshape(-1))
        out.update({"cell_g": gc.norm(dim=-1).float().cpu(), "cell_e": ec.float().cpu(), "cell_vol": vc.float().cpu(),
                    "node_m": (mn if idx is None else mn[idx]).float().cpu(), "chk_rel": chk,
                    "n_nodes": int(npos.shape[0]), "n_free": int(g_el.shape[0]), "n_cells": int(nc)})
    rec[tag] = out


# AF_GD_PROBE=출력디렉토리:프레임:반복:덤프간격 (사용자 10-20 "멈춘 상태에서 경사하강으로 변형"): 그 프레임에서 노드 변위 u 를
# u=0 에서 시작해 u <- u - α ∂IP/∂u (매 반복 Armijo 백트래킹, α 는 직전 값의 2 배에서 시작) 로 내리며 반복마다 IP·탄성·|g|·이동·α 를,
# 덤프간격마다 입자 위치·ψ·F 를 저장한다. 또 같은 u=0 에서 L-BFGS 첫 걸음과 같은 방향(-g)·크기(t0 = min(1, 1/|g|₁))의 시험 걸음
# 여러 배율로 IP 변화를 재고 float32 합 반올림 크기와 비교한다. 끝나면 u 를 원래 값으로 되돌린다 (롤아웃은 그대로).
_GD_PROBE = None
if os.environ.get("AF_GD_PROBE"):
    _gda = os.environ["AF_GD_PROBE"].split(":")
    _GD_PROBE = {"out": _gda[0], "t": int(_gda[1]), "iters": int(_gda[2]), "every": int(_gda[3]) if len(_gda) > 3 else 1}
    os.makedirs(_GD_PROBE["out"], exist_ok=True)


def _gd_run(st, _fwd, F, vol, cfg, x, t, nrm):
    _vs = [q for q in st["ov"] if q is not None]
    u = _vs[0]
    keep = u.detach().clone()

    def ev(need_g=True, keep_state=False):
        with torch.enable_grad():
            x2, _, E, _, _, _pq, Jd = _fwd()
            Et = E * nrm
            g = torch.autograd.grad(Et, u)[0] if need_g else None
        with torch.no_grad():
            F_tr = Jd.detach().to(F.dtype) @ F
            psi = phys_resid.psi_of(F_tr, cfg, _DT[0])[0]
            Eel = float((vol * psi).sum())
        out = dict(E=float(Et), E_el=Eel, du=float((x2 - x).norm(dim=-1).mean()), g=g)
        if keep_state:
            out.update(x2=x2.detach().float().cpu(), psi=psi.detach().float().cpu(), F=F_tr.detach().float().cpu())
        return out
    with torch.no_grad():
        u[:, :3] = 0.0
    r0 = ev(keep_state=True)
    g0 = r0["g"]
    # L-BFGS 첫 걸음 재현: 방향 -g, 크기 t0 = lr·min(1, 1/|g|₁) (torch LBFGS 첫 반복), 배율 여러 개
    t0 = min(1.0, 1.0 / float(g0.abs().sum()))
    ls = []
    E_terms = abs(r0["E"])
    rnd = float(torch.finfo(torch.float32).eps) * E_terms
    for f_ in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        with torch.no_grad():
            u[:, :3] = -(t0 * f_) * g0[:, :3]
        r_ = ev(need_g=False)
        ls.append(dict(scale=f_, step=t0 * f_, dE=r_["E"] - r0["E"], du=r_["du"]))
    print(f"    [경사하강] t={t} u=0: IP {r0['E']:.6e} 탄성 {r0['E_el']:.6e} |g| {float(g0.norm()):.3e} |g|₁ {float(g0.abs().sum()):.3e}, "
          f"L-BFGS 첫 걸음 t0 {t0:.3e}, float32 반올림 크기(eps·IP) {rnd:.2e}", flush=True)
    for q_ in ls:
        print(f"      시험 걸음 t0x{q_['scale']:g} ({q_['step']:.3e}): ΔIP {q_['dE']:+.3e} (반올림 대비 {q_['dE'] / max(rnd, 1e-30):+.1f}), 이동 {q_['du']:.3e}", flush=True)
    # 경사하강 -- 뒤집힘 로그 장벽 매니폴드 위 (사용자 정정 10-20): Δu = -α G⁻¹∇IP,
    #   G = Σ_c ‖J_c⁻¹ dJ_c‖_F² (-Σ log det J_c 의 정확 헤시안 tr(J⁻¹dJ J⁻¹dJ) 은 부정부호일 수 있어 prw 와 같은 PSD 계량) + eps·I,
    #   J_c = I + Σ_k u_k ⊗ ∇λ_k (셀 = 입자가 든 사면체, 변형장이 셀마다 선형). 노드 공간 정확 행렬(밀집, float64)로 푼다.
    #   ‖Δu‖_G ≤ r (Dikin: 셀마다 ‖J⁻¹dJ‖_F ≤ r < 1 이라 det 가 양수로 남는다), 그 안에서 Armijo 백트래킹. 바닥 항 없음 (ficus).
    ens = _NF_HOLD[0][0]
    npos, rows = ens[1], ens[3]
    cells = torch.unique(rows, dim=0)
    Xv = npos[cells].double()
    Minv = torch.linalg.inv((Xv[:, 1:] - Xv[:, :1]).transpose(1, 2))
    B = torch.cat([-Minv.sum(1, keepdim=True), Minv], 1)                    # [C,4,3] ∇λ_k
    M_ = npos.shape[0]
    idx = (st.get("ov_idx") or [None])[0]
    free = torch.zeros(M_, dtype=torch.bool, device=npos.device)
    free[idx if idx is not None else slice(None)] = True
    pos_of = torch.full((M_,), -1, dtype=torch.long, device=npos.device)
    pos_of[free] = torch.arange(int(free.sum()), device=npos.device)
    nf = int(free.sum())
    r_dk = float(os.environ.get("AF_GD_R", 0.5)); eps_rel = float(os.environ.get("AF_GD_EPS", 1e-6))

    def node_u():
        U = torch.zeros(M_, 3, dtype=torch.float64, device=npos.device)
        U[free] = u.detach()[:, :3].double()
        return U

    def cellJ(U):
        return torch.eye(3, dtype=torch.float64, device=U.device) + torch.einsum("cka,ckb->cab", U[cells], B)

    def metric(U):
        A = torch.linalg.inv(cellJ(U)); AtA = A.transpose(1, 2) @ A          # [C,3,3]
        bb = torch.einsum("cka,cla->ckl", B, B)                                # [C,4,4]
        blk = bb[:, :, :, None, None] * AtA[:, None, None]                     # [C,4,4,3,3]
        G = torch.zeros(nf * 3, nf * 3, dtype=torch.float64, device=U.device)
        pk = pos_of[cells]                                                     # [C,4]
        ok = (pk >= 0)
        for k in range(4):
            for l in range(4):
                m_ = ok[:, k] & ok[:, l]
                if not bool(m_.any()):
                    continue
                ri = (pk[m_, k] * 3)[:, None, None] + torch.arange(3, device=U.device)[None, :, None]
                ci = (pk[m_, l] * 3)[:, None, None] + torch.arange(3, device=U.device)[None, None, :]
                G.index_put_((ri.expand(-1, 3, 3).reshape(-1), ci.expand(-1, 3, 3).reshape(-1)),
                             blk[m_, k, l].reshape(-1), accumulate=True)
        G = 0.5 * (G + G.T)
        G += eps_rel * float(G.diagonal().mean()) * torch.eye(nf * 3, dtype=torch.float64, device=U.device)
        return G

    def detmin(U):
        return float(torch.linalg.det(cellJ(U)).min())

    hist, X, PS, FF, ks = [], [r0["x2"]], [r0["psi"]], [r0["F"]], [0]
    with torch.no_grad():
        u[:, :3] = 0.0
    cur, g = r0, g0
    a_ = None
    print(f"      [다양체 경사하강] 셀 {cells.shape[0]}, 자유 노드 {nf}, Dikin r {r_dk}, eps {eps_rel:g}·평균대각, 밀집 {3 * nf}²", flush=True)
    for k in range(1, _GD_PROBE["iters"] + 1):
        U0 = node_u()
        G = metric(U0)
        gv_ = g[:, :3].double().reshape(-1)
        d_ = torch.linalg.solve(G, gv_)                                        # G⁻¹∇IP
        nG = float(torch.sqrt((gv_ * d_).sum()).clamp_min(1e-300))           # ‖G⁻¹g‖_G
        amax = r_dk / nG
        a_ = amax if a_ is None else min(amax, 2.0 * a_)
        u0 = u.detach().clone()
        acc = False
        for _bt in range(60):
            with torch.no_grad():
                u.copy_(u0); u[:, :3] -= (a_ * d_).reshape(-1, 3).to(u.dtype)
            dm_ = detmin(node_u())
            if dm_ > 0:
                r_ = ev(need_g=False)
                if r_["E"] <= cur["E"] - 1e-4 * a_ * float((gv_ * d_).sum()):
                    acc = True
                    break
            a_ *= 0.5
        if not acc:
            with torch.no_grad():
                u.copy_(u0)
            hist.append(dict(k=k, E=cur["E"], E_el=cur["E_el"], gnorm=float(g.norm()), gG=nG, du=cur["du"], alpha=0.0,
                             stepG=0.0, detmin=detmin(U0), acc=False))
            print(f"      [다양체 경사하강] 반복 {k}: 백트래킹 60 회 실패 -- 멈춤 (IP {cur['E']:.6e}, ‖g‖_G* {nG:.3e})", flush=True)
            break
        last = (k % _GD_PROBE["every"] == 0) or k == _GD_PROBE["iters"]
        cur = ev(keep_state=last)
        g = cur["g"]
        dmk = detmin(node_u())
        hist.append(dict(k=k, E=cur["E"], E_el=cur["E_el"], gnorm=float(g.norm()), gG=nG, du=cur["du"], alpha=a_,
                         stepG=a_ * nG, detmin=dmk, acc=True))
        if last:
            X.append(cur["x2"]); PS.append(cur["psi"]); FF.append(cur["F"]); ks.append(k)
        if k % max(1, _GD_PROBE["iters"] // 20) == 0 or k <= 3:
            print(f"      [다양체 경사하강] 반복 {k}: IP {cur['E']:.6e} (시작 대비 {cur['E'] - r0['E']:+.3e}) 탄성 {cur['E_el']:.6e} "
                  f"|g| {float(g.norm()):.3e} ‖g‖_G* {nG:.3e} 걸음 ‖Δu‖_G {a_ * nG:.3f} 이동 {cur['du']:.3e} 셀 det 최소 {dmk:.4f}", flush=True)
    torch.save({"t": t, "E0": r0["E"], "Eel0": r0["E_el"], "t0": t0, "round": rnd, "linesearch": ls, "hist": hist, "k": ks,
                "pred": torch.stack(X), "psi": torch.stack(PS), "F": torch.stack(FF), "x": x.detach().float().cpu()},
               os.path.join(_GD_PROBE["out"], f"gd_t{t:04d}.pt"))
    print(f"    [경사하강] t={t} 저장 {_GD_PROBE['out']}/gd_t{t:04d}.pt ({len(ks)} 장, 마지막 반복 {hist[-1]['k'] if hist else 0})", flush=True)
    with torch.no_grad():
        u.copy_(keep)


def _ov_once(d, t, gsel, p, x, v, F, off=None):
    if _OV_SNAP is not None and int(t) == _OV_SNAP["t"] and not _OV_SNAP["done"]:
        _OV_SNAP["done"] = True
        _keep = a.ov_roll
        a.ov_roll = _OV_SNAP["iters"]
        _OV_SNAP["active"], _OV_SNAP["P"], _OV_SNAP["F"], _OV_SNAP["E"], _OV_SNAP["k"] = True, [], [], [], []
        _OV_SNAP["F0"] = F
        try:
            out = _ov_once(d, t, gsel, p, x, v, F, off)
        finally:
            a.ov_roll = _keep
            _OV_SNAP["active"] = False
        torch.save({"pred": torch.stack(_OV_SNAP["P"]), "F": torch.stack(_OV_SNAP["F"]),
                    "evals": _OV_SNAP["k"], "E": _OV_SNAP["E"], "t": int(t)}, _OV_SNAP["path"])
        print(f"    [중간 저장] t={t} 평가 {len(_OV_SNAP['k'])} 장 -> {_OV_SNAP['path']}", flush=True)
        return out
    """한 프레임을 **한 격자(오프셋 off)** 에서 최적화한다 -> (x2, v2, J).

    --out_var 는 상태를 전진시키지 않아 한 스텝 하한만 잰다. 그 값으로 롤아웃을
    말할 수 없다 (평가 중에는 훅이 끊겨 **초기 망**이 굴러간다). 여기서는 매
    프레임 변수를 새로 만들어 망 출력에서 시작해 증분 포텐셜을 내리고, 내려간
    출력으로 상태를 전진시킨다 -- 이것이 롤아웃에서 성립하는 하한이다.
    """
    cfg = d["cfg"]
    m = MASS[gsel]
    vol = _pvol(m, cfg)
    gv = torch.tensor(cfg["g"], device=dev, dtype=torch.float32)
    ng_, gl_ = int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0))
    nrm = float(m.sum()) * (EXT ** 2) / (_DT[0] ** 2)
    st, E0, E1 = {}, None, None
    _E0, _tr = [None], []
    _trp = []                  # 항별 곡선 [(관성, 탄성, 접촉), ...]

    def _fwd():
        with torch.enable_grad(), dt_scope(_DT[0]):
            x2, _, v2, _, _, _, _, _, _, _Jd = step_once(
                d, t, gsel, p, x, v, need_J=False)
        _hd = release_held(d, x, t, _DT[0])
        if _hd is not None:
            # 아직 안 풀린 입자는 **제자리에 박는다** (PG 는 속도를 0 으로 둔다)
            x2 = torch.where(_hd.unsqueeze(-1), x, x2)
            v2 = torch.where(_hd.unsqueeze(-1), torch.zeros_like(v2), v2)
        fm = (free_mask(d, x2.shape[0], x2.device, gsel, x, t)
              if a.control else None)
        if _hd is not None:
            fm = (~_hd) if fm is None else (fm & ~_hd)
        if _BC_LAST[0] is not None:
            _am = _BC_LAST[0][1].reshape(-1) > 0.5
            fm = (~_am) if fm is None else (fm & ~_am)
        if a.ctrl_barrier > 0 and "ctrl_id" in d:
            # **손잡이 비침투**: 초기에 구 밖이던 입자가 구 안으로 못 들어온다.
            # PG 에는 없는 항이다 (우리 쪽에만 더하는 구속이다).
            if "_cb_out" not in d:
                _c0 = ctrl_center(d, gsel, 0, take(d["x"][0], gsel))
                _R0 = float(d["ctrl_R"].reshape(-1)[0])
                _dm = (take(d["x"][0], gsel).unsqueeze(1)
                       - _c0.unsqueeze(0)).norm(dim=-1).min(1).values
                d["_cb_out"] = (_dm >= _R0).detach()
            _tt2 = min(t, d["ctrl_vel"].shape[0] - 1)
            _cc2 = ctrl_center(d, gsel, _tt2, x)
            _R2 = float(d["ctrl_R"].reshape(-1)[0])
            _dh2 = (a.ctrl_barrier_dhat if a.ctrl_barrier_dhat > 0
                    else 0.25 * _R2)
            _dd = (x2.unsqueeze(1) - _cc2.unsqueeze(0)).norm(dim=-1).min(1
                                                                        ).values
            _sd2 = _dd - _R2                       # 구 밖이면 양수
            _bb = phys_resid._barrier_b(_sd2, _dh2, ext_kind="linear")
            _ecb = (a.ctrl_barrier * m / (_DT[0] ** 2) * _bb
                    * d["_cb_out"].to(_bb.dtype)).sum()
        else:
            _ecb = None
        if _TAU is not None:
            # **2차 근사 탄성**: <tau, dJ> + V[mu|sym dJ|² + (lam/2) tr(dJ)²]
            # 사면체마다 대표 입자 하나의 dJ 면 충분하다 (셀 안에서 상수)
            _rp, _tc, _hc = _CELL
            _dJ = _Jd[_rp] - torch.eye(3, device=dev, dtype=_Jd.dtype)
            _eel = ((_tc * _dJ).sum((-1, -2))
                    + 0.5 * torch.einsum("tabcd,tab,tcd->t", _hc, _dJ, _dJ)
                    ).sum()
            _du = x2 - x
            _w = (torch.ones_like(m) if fm is None else fm.to(m.dtype))
            _ein = (0.5 * _w * m / (_DT[0] ** 2)
                    * ((_du - _DT[0] * v - (_DT[0] ** 2) * gv) ** 2
                       ).sum(-1)).sum()
            _ebc = phys_resid.bc_energy(x, _du, m, cfg, _DT[0], gl_, ng_)
            E = (_ein + _eel + _ebc) / nrm
            dlog, F_tr = None, _Jd.to(F.dtype) @ F
            _pt = (_ein.detach(), _eel.detach(), torch.zeros(()),
                   _ebc.detach() if torch.is_tensor(_ebc)
                   else torch.zeros(()))
        else:
            E, dlog, F_tr, _pt = ip_of(x, x2 - x, v, F, m, vol, cfg, _DT[0],
                                       ng_, gl_, g=gv, norm=nrm, free=fm,
                                       jac=_Jd)
        if _ecb is not None:
            E = E + _ecb / nrm
        if a.c1_barrier > 0:
            # --interp c1wf: 점유 사면체 모든 조각의 det(I+∇u) Bernstein 계수에 진짜 로그 장벽 (advance 가 계수를 남긴다)
            if DM.C1_LAST[0] is None:
                raise SystemExit("--c1_barrier: advance 가 Bernstein 계수를 남기지 않았다")
            _c1c, _c1v = DM.C1_LAST[0][0], DM.C1_LAST[0][1]
            _mu1, _ = phys_resid.lame(cfg["E"], cfg["nu"])
            _mu1 = float(_mu1.float().mean()) if torch.is_tensor(_mu1) else float(_mu1)
            E = E + WF.barrier_energy(_c1c, _c1v, a.c1_barrier * _mu1, a.c1_jhat) / nrm
        if a.ov_obj == "node":                  # 노드 출력을 목표 실행의 같은 프레임 노드 출력에 (같은 상태라 격자가 같다)
            _np, _dp = _DPN_LIVE[0]
            _tn = _NODE_TGT["nodes"][int(t) - int(_NODE_TGT.get("t0", 0))]
            _tp, _td = _tn[0].to(dev), _tn[1].to(dev)
            if _tp.shape != _np.shape or float((_tp - _np.detach()).abs().max()) > 1e-5:
                raise SystemExit(f"--ov_obj node: t={t} 노드가 목표와 다르다 ({tuple(_np.shape)} vs {tuple(_tp.shape)})")
            E = ((_dp[:, :3] - _td[:, :3].to(_dp.dtype)) ** 2).sum(-1).mean() / (EXT ** 2)
        if a.ov_obj == "l2":                    # 정답 다음 프레임 위치와의 L2 (장벽·물리 없음)
            _xt = take(d["x"][min(t + 1, d["x"].shape[0] - 1)], gsel).to(x2.dtype)
            E = ((x2 - _xt) ** 2).sum(-1).mean() / (EXT ** 2)
        if a.hex_hg > 0 and _HG_LAST[0] is not None:
            _dh, _c8, _lt = _HG_LAST[0]
            E = E + _hex_hg_energy(_dh, _c8, _lt, cfg, a.hex_hg) / nrm
        return x2, v2, E, dlog, F_tr, _pt, _Jd

    _TAU = _HES = _CELL = None
    _TAU_H[0] = None
    if a.elast == "lin":
        # **입자별 2차 테일러**: 1차와 2차 계수를 같은 방식으로 (각 입자의
        # 실제 F^n 에서 미분해) 구하고 더한다. 예전에는 2차항만 F=I 의 등방
        # 선형탄성 상수(mu, lambda)로 고정했는데, F 가 I 에서 멀어지면 그
        # 곡률이 실제보다 작아 1차항이 이겨 에너지가 음수로 발산했다
        # (실측: 탄성항이 t=7 에 -5.7e-3, t=14 에 -0.198, 끝내 -1e19).
        def _tau_at(Jv):
            """V psi 의 J 에 대한 1차 미분 [N,3,3] (자동미분 1 회)."""
            with torch.enable_grad():
                _Jx = Jv.detach().clone().requires_grad_(True)
                _ps, _ = phys_resid.psi_of(_Jx @ F, cfg, _DT[0])
                return torch.autograd.grad((vol * _ps).sum(), _Jx)[0].detach()

        _I3L = torch.eye(3, device=dev, dtype=F.dtype).expand(
            F.shape[0], 3, 3).contiguous()
        _JREF = (_JR_HOLD[0] if _JR_HOLD[0] is not None else _I3L)
        _TAU = _tau_at(_JREF)
        # 헤시안은 **tau 의 중심차분**으로 만든다. 이중 역전파로 뽑으면
        # psi 가 eigvalsh 를 거치는 탓에 F ~ I 처럼 고유값이 겹칠 때
        # 1/(l_i - l_j) 가 터져 전부 NaN 이 된다 (실측: 비 0.92 = 제자리).
        _eps = 1e-3
        _cols = []
        for _a in range(3):
            for _b in range(3):
                _E = torch.zeros(3, 3, device=dev, dtype=F.dtype)
                _E[_a, _b] = _eps
                _cols.append((_tau_at(_JREF + _E) - _tau_at(_JREF - _E))
                             / (2.0 * _eps))
        _HES = torch.stack(_cols, 1).reshape(-1, 3, 3, 3, 3)
        # 헤시안은 대칭이어야 한다 (H_abcd = H_cdab) -- 차분 잔차를 없앤다
        _H9 = _HES.reshape(-1, 9, 9)
        _HES = (0.5 * (_H9 + _H9.transpose(-1, -2))).reshape(-1, 3, 3, 3, 3)
        # **셀별 합산**: 한 사면체 안 입자들은 같은 dJ 를 받으므로 계수를 미리
        # 더해 두면 평가가 입자 수가 아니라 셀 수에 비례한다.
        _lo_c, _lat_c, _nn_c = SX.grid_for_nodes(
            x, a.n_nodes, hz_ratio=a.hz_ratio, off=None,
            h_fix=(a.node_h or None))
        _idx_c, _, _ = SX.locate(x, _lo_c, _lat_c, _nn_c)
        _uq_c, _inv_c = torch.unique(_idx_c, dim=0, return_inverse=True)
        _nT = int(_uq_c.shape[0])
        _rep_c = torch.zeros(_nT, dtype=torch.long, device=dev)
        _rep_c.scatter_(0, _inv_c, torch.arange(_inv_c.numel(), device=dev))
        _TAU_C = torch.zeros(_nT, 3, 3, device=dev, dtype=_TAU.dtype
                             ).index_add_(0, _inv_c, _TAU)
        _HES_C = torch.zeros(_nT, 3, 3, 3, 3, device=dev, dtype=_HES.dtype
                             ).index_add_(0, _inv_c, _HES)
        _CELL = (_rep_c, _TAU_C, _HES_C)
        _TAU_H[0] = (_TAU, _HES)
        _CPOLY[0] = None
        _CP_E[0] = None
        if os.environ.get("AF_OV_DIAG"):
            print(f"    [셀집계] 입자 {x.shape[0]} -> 사면체 {_nT} "
                  f"(평가 비용 {x.shape[0] / max(_nT, 1):.1f} 배 감소)",
                  flush=True)
        # 전개 중심이 I 가 아니면 1 차항을 옮긴다:
        #   dJ = (I + grad u) - Jref = grad u - D,  D = Jref - I
        #   <tau, dJ> + 1/2 dJ:H:dJ = <tau - H:D, grad u> + 1/2 grad u:H:grad u
        if _JR_HOLD[0] is not None:
            _Dref = (_JREF - _I3L)
            _TAU = _TAU - torch.einsum("nabcd,ncd->nab", _HES, _Dref)
        _mu, _lm = phys_resid.lame(cfg["E"], cfg["nu"])
        if a.lin_psd:
            _H9 = _HES.reshape(-1, 9, 9)
            _w9, _V9 = torch.linalg.eigh(_H9.double())
            _HES = (_V9 @ torch.diag_embed(_w9.clamp_min(0.0))
                    @ _V9.transpose(-1, -2)).to(_HES.dtype).reshape(
                        -1, 3, 3, 3, 3)
        if os.environ.get("AF_OV_DIAG"):
            _ev = torch.linalg.eigvalsh(
                0.5 * (_HES.reshape(-1, 9, 9)
                       + _HES.reshape(-1, 9, 9).transpose(-1, -2)))
            print(f"    [2차근사] |tau| 최대 {float(_TAU.abs().max()):.3e}, "
                  f"헤시안 고유값 최소 {float(_ev.min()):.3e} 최대 "
                  f"{float(_ev.max()):.3e}, 음수 비율 "
                  f"{100 * float((_ev < 0).float().mean()):.1f}%", flush=True)
    _ov = _OV_CUR[0]
    _OV_CUR[0] = st
    _GRID_OFF[0] = off
    _SM_HOLD[0] = "want"
    _OV_TGT[0] = ("want" if a.ov_init == "predict" else
                  (_DT[0] * v.mean(0) + (_DT[0] ** 2) * gv).detach())
    _HX_HOLD[0] = {}              # 육면체 색인도 프레임당 한 번만
    _NF_HOLD[0] = "want"          # 이 프레임의 격자·간선·노드특징을 한 번만
    try:
      with torch.enable_grad():
        if a.fused:
            # **커널 한 번 경로**: 구조를 한 번 만든 뒤 반복마다 커널만 돈다.
            _fwd()                      # 격자·변수·손잡이 마스크를 만든다
            _ens0 = _NF_HOLD[0][0]
            (_, _np0, (_lo0, _lat0, _nn0), _rw0, _lm0, _uq0, _e0, _lx0) = _ens0
            _N0 = x.shape[0]
            _Ai0 = _lat0.Ai.to(x.dtype)
            _b0 = torch.zeros(_N0, 4, 3, device=dev, dtype=x.dtype)
            _ar0 = torch.arange(_N0, device=dev)
            for _j in range(3):
                _r = _lx0[1][:, _j]
                _b0[_ar0, _r + 1] += _Ai0[_j]
                _b0[_ar0, _r] -= _Ai0[_j]
            _tg0 = _DT[0] * v + (_DT[0] ** 2) * gv
            if release_held(d, x, t, _DT[0]) is not None:
                raise SystemExit("--fused 는 release 구속(모래성 씬)을 아직 "
                                 "지원하지 않는다. 그 씬은 --fused 없이 돌릴 것")
            _fm0 = (free_mask(d, x.shape[0], dev, gsel, x, t)
                    if a.control else None)
            _w0 = (torch.ones_like(m) if _fm0 is None
                   else _fm0.to(m.dtype))
            _mh0 = _w0 * m / (_DT[0] ** 2)      # 관성 (구속 입자 제외)
            _mb0 = m / (_DT[0] ** 2)            # 접촉·장벽 (전체 질량)
            _bc0 = None
            for _bcd in (cfg.get("boundary_conditions") or []):
                if _bcd.get("type") == "surface_collider":
                    _nn_ = torch.as_tensor(_bcd["normal"], dtype=torch.float64)
                    _nn_ = _nn_ / _nn_.norm().clamp_min(1e-12)
                    _dh0 = (a.bc_dhat if a.bc_dhat > 0 else
                            0.5 * gl_ / float(ng_))
                    _bc0 = (tuple(_nn_.tolist()),
                            tuple(float(q_) for q_ in _bcd["point"]),
                            float(a.bc_kappa), float(_dh0),
                            float(os.environ.get("AF_BC_STIFF", 1000.0)))
                    break
            _cb0 = None
            if a.ctrl_barrier > 0 and "ctrl_id" in d:
                if "_cb_out" not in d:
                    _c00 = ctrl_center(d, gsel, 0, take(d["x"][0], gsel))
                    _R00 = float(d["ctrl_R"].reshape(-1)[0])
                    _dm0 = (take(d["x"][0], gsel).unsqueeze(1)
                            - _c00.unsqueeze(0)).norm(dim=-1).min(1).values
                    d["_cb_out"] = (_dm0 >= _R00).detach()
                _tt0 = min(t, d["ctrl_vel"].shape[0] - 1)
                _cc0 = ctrl_center(d, gsel, _tt0, x)[0]
                _R0c = float(d["ctrl_R"].reshape(-1)[0])
                _cb0 = (d["_cb_out"].to(x.dtype).contiguous(),
                        tuple(float(q_) for q_ in _cc0.tolist()), _R0c,
                        float(a.ctrl_barrier),
                        float(a.ctrl_barrier_dhat if a.ctrl_barrier_dhat > 0
                              else 0.25 * _R0c))
            _box0 = None
            for _bcd in (cfg.get("boundary_conditions") or []):
                if _bcd.get("type") == "bounding_box":
                    _bnd = float(cfg.get("bound", 3)) * (gl_ / float(ng_))
                    _box0 = (_bnd, gl_ - _bnd,
                             float(os.environ.get("AF_BC_STIFF", 1000.0)))
                    break
            _vs0 = [q for q in (st.get("ov") or []) if q is not None]
            _idx0 = (st.get("ov_idx") or [None])[0]
            _M0 = (st.get("ov_M") or [0])[0] or _vs0[0].shape[0]
            _pm0, _pv0 = (_PIN[0] if _PIN[0] is not None else (None, None))
            _mu0f, _lm0f = phys_resid.lame(cfg["E"], cfg["nu"])

            def _Efused():
                _z = torch.zeros(_M0, 3, device=dev, dtype=x.dtype)
                _z = (_z.index_copy(0, _idx0, _vs0[0]) if _idx0 is not None
                      else _vs0[0])
                if _pm0 is not None:
                    _z = torch.where(_pm0.unsqueeze(-1), _pv0, _z)
                return FUSED.energy(_z, _rw0, _lm0, _b0, F, _mh0, vol, _tg0,
                                    _mu0f, _lm0f, x, _bc0, _cb0,
                                    _box0, _mb0) / nrm

            _cmp_on = bool(os.environ.get("AF_FUSE_CMP"))
            if _cmp_on or (os.environ.get("AF_FUSE_CHECK") and t == int(
                    os.environ["AF_FUSE_CHECK"])):
                with torch.no_grad():
                    _zc = torch.zeros(_M0, 3, device=dev, dtype=x.dtype)
                    _zc = (_zc.index_copy(0, _idx0, _vs0[0])
                           if _idx0 is not None else _vs0[0])
                    if _pm0 is not None:
                        _zc = torch.where(_pm0.unsqueeze(-1), _pv0, _zc)
                    _ef = float(FUSED.energy(_zc, _rw0, _lm0, _b0, F, _mh0,
                                             vol, _tg0, _mu0f, _lm0f, x,
                                             _bc0, _cb0, _box0, _mb0) / nrm)
                _vs0[0].grad = None
                _Ef2 = _Efused(); _Ef2.backward()
                _gf = _vs0[0].grad.detach().clone()
                _vs0[0].grad = None
                _x2c, _v2c, _Ec, _, _, _ptc, _ = _fwd()
                _Ec.backward()
                _gr2 = _vs0[0].grad.detach().clone()
                _vs0[0].grad = None
                print(f"    [기울기대조] t={t} 상대오차 "
                      f"{float((_gf - _gr2).norm() / _gr2.norm().clamp_min(1e-30)):.3e}"
                      f"  크기 융합 {float(_gf.norm()):.3e} 기준 "
                      f"{float(_gr2.norm()):.3e}", flush=True)
                _x2c, _v2c, _Ec, _, _, _ptc, _ = _fwd()
                print(f"    [융합대조] t={t} 기준 {float(_Ec):.6e} 융합 "
                      f"{_ef:.6e} 상대차 "
                      f"{abs(float(_Ec) - _ef) / max(abs(float(_Ec)), 1e-30):.2e}"
                      f" | 기준 항 관성 {float(_ptc[0]):.3e} 탄성 "
                      f"{float(_ptc[1]):.3e} 접촉 {float(_ptc[3]):.3e}",
                      flush=True)
            if _cmp_on:
                # 비교 모드: 전진은 **기준 경로**로 해서 상태를 같게 유지한다
                _optR = torch.optim.LBFGS(_vs0, lr=1.0,
                                          max_iter=max(a.ov_roll, 1),
                                          history_size=50,
                                          tolerance_grad=0.0,
                                          tolerance_change=0.0,
                                          line_search_fn="strong_wolfe")

                def _clR():
                    _optR.zero_grad(set_to_none=True)
                    _, _, _E, _, _, _, _ = _fwd()
                    _E.backward()
                    return _E
                _optR.step(_clR)
                E0 = None
            else:
                _opt0 = torch.optim.LBFGS(
                    _vs0, lr=1.0, max_iter=max(a.ov_roll, 1),
                    history_size=50, tolerance_grad=0.0,
                    tolerance_change=0.0, line_search_fn="strong_wolfe")
                _E0f = [None]

                def _cl0():
                    _opt0.zero_grad(set_to_none=True)
                    _E = _Efused()
                    if _E0f[0] is None:
                        _E0f[0] = float(_E)
                    _E.backward()
                    return _E
                _opt0.step(_cl0)
                E0 = _E0f[0]
        elif a.ov_opt == "cg":
            # 목적함수가 2차형식이므로 **공액기울기로 바로 최소점**에 간다.
            # 헤시안-벡터 곱은 이중 역전파로 행렬 없이 만든다.
            _fwd()
            _vs = [q for q in (st.get("ov") or []) if q is not None]
            if not _vs:
                raise SystemExit("--ov_opt cg: 변수를 못 만들었다")

            def _grad(create=False):
                _, _, E, _, _, _, _ = _fwd()
                return torch.autograd.grad(E, _vs, create_graph=create), E
            (_g0,), _E = _grad()
            E0 = float(_E)
            _r = [-q.detach() for q in (_g0,)] if False else None
            _gs, _ = _grad()
            _r = [-q.detach() for q in _gs]
            _pv = [q.clone() for q in _r]
            _rs = sum(float((q * q).sum()) for q in _r)
            for _it in range(a.cg_iters):
                if _rs < 1e-30:
                    break
                _gs2, _ = _grad(create=True)
                _hv = torch.autograd.grad(
                    sum((g_ * p_).sum() for g_, p_ in zip(_gs2, _pv)), _vs)
                _pHp = sum(float((p_ * h_).sum())
                           for p_, h_ in zip(_pv, _hv))
                if _pHp <= 0:
                    break
                _al = _rs / _pHp
                with torch.no_grad():
                    for _v2, _p2 in zip(_vs, _pv):
                        _v2 += _al * _p2
                _r = [q - _al * h_ for q, h_ in zip(_r, _hv)]
                _rs2 = sum(float((q * q).sum()) for q in _r)
                _be = _rs2 / max(_rs, 1e-30)
                _pv = [q + _be * p_ for q, p_ in zip(_r, _pv)]
                _rs = _rs2
        elif a.ov_opt == "lbfgs":
            _n = [0]

            def _closure():
                st["ov_opt"].zero_grad(set_to_none=True)
                _x2s, _, E, _, _, _pq, _Jds = _fwd()
                if _n[0] == 0:
                    _E0[0] = float(E)
                if _OV_SNAP is not None and _OV_SNAP.get("active") and _n[0] % _OV_SNAP["every"] == 0:
                    with torch.no_grad():
                        _OV_SNAP["P"].append(_x2s.detach().float().cpu())
                        _F0s = _OV_SNAP["F0"]
                        _OV_SNAP["F"].append((_Jds.detach().float() @ _F0s.float()).cpu() if _F0s is not None
                                             else _Jds.detach().float().cpu())
                        _OV_SNAP["E"].append(float(E)); _OV_SNAP["k"].append(_n[0])
                _n[0] += 1
                if _CURVE is not None:
                    _tr.append(float(E))
                    _trp.append((float(_pq[0]), float(_pq[1]),
                                 float(_pq[3])))
                E.backward()
                _ov_clip_grads(st)
                return E
            _fwd()                    # 변수를 만든다 (훅이 첫 호출에서 만든다)
            if st.get("ov_opt") is None:
                _vs = [q for q in (st.get("ov") or []) if q is not None]
                if not _vs:
                    raise SystemExit("--ov_roll: 출력 훅이 변수를 못 만들었다")
                st["ov_opt"] = _mk_ov_opt(_vs)
            _gpr = _GRAD_PROBE is not None and int(t) in _GRAD_PROBE["frames"]
            if _gpr:
                _gp_rec = {"t": int(t), "nrm": float(nrm), "h": float(_DT[0])}
                _gp_z0 = [q.detach().clone() for q in st["ov"] if q is not None]
                _grad_probe(st, _fwd, F, m, vol, cfg, x, v, gv, "pred", _gp_rec)          # 관성 예측 u = h v (출발점)
                with torch.no_grad():
                    for q in st["ov"]:
                        if q is not None:
                            q[:, :3] = 0.0
                _grad_probe(st, _fwd, F, m, vol, cfg, x, v, gv, "zero", _gp_rec)          # u = 0 (현재 배치 그대로)
                with torch.no_grad():
                    for q, z0 in zip([q for q in st["ov"] if q is not None], _gp_z0):
                        q.copy_(z0)
            if _GD_PROBE is not None and int(t) == _GD_PROBE["t"]:
                _gd_run(st, _fwd, F, vol, cfg, x, int(t), nrm)
            _tw = time.time()
            if not (_GRAD_PROBE is not None and _GRAD_PROBE.get("skip") and not _gpr):
                _OV_CUR_OPT[0] = st["ov_opt"]
                _p1 = 0
                if a.c1_barrier > 0:
                    # **실현 가능 시작점 찾기** (1 단계): 시작점에서 Bernstein det 계수가 0 이하이면 장벽 에너지가 +inf 라
                    # 되추적 L-BFGS 가 한 걸음도 못 뗀다. 계수 벌점 Σ relu(ĉ/2 - c)² 만 내려 모든 계수 > 0 으로 만든 뒤 시작한다
                    _vs1 = [q for q in st["ov"] if q is not None]
                    with torch.no_grad():
                        _fwd()
                    if bool((DM.C1_LAST[0][0] <= 0).any()):
                        _o1 = torch.optim.LBFGS(_vs1, lr=1.0, max_iter=20, history_size=20, line_search_fn="strong_wolfe")

                        def _cl1():
                            _o1.zero_grad(set_to_none=True)
                            _fwd()
                            _P = torch.relu(0.5 * a.c1_jhat - DM.C1_LAST[0][0]).pow(2).sum()
                            _P.backward()
                            return _P
                        for _p1 in range(1, 51):
                            _o1.step(_cl1)
                            with torch.no_grad():
                                _fwd()
                            if not bool((DM.C1_LAST[0][0] <= 0).any()):
                                break
                        print(f"    [C1 실현 가능 시작점] t={t} 1 단계 {_p1} 회 -> 계수 최소 {float(DM.C1_LAST[0][0].min()):.3e}", flush=True)
                    _C1_P1[0] = _p1
                st["ov_opt"].step(_closure)       # AF_GRAD_PROBE_SKIP: 상태를 덤프로 맞추는 탐침 실행에서 탐침 밖 프레임은 최적화 생략
            if _gpr:
                _grad_probe(st, _fwd, F, m, vol, cfg, x, v, gv, "opt", _gp_rec)           # L-BFGS ov_roll 회 뒤 (실제로 쓴 해)
                _ext = int(_GRAD_PROBE.get("extra", 0))
                if _ext > 0:
                    # 같은 프레임을 더 오래: 이어서 _ext 회 더 (새 L-BFGS), 기록 뒤 원래 해로 되돌린다 (롤아웃은 그대로)
                    _gp_z1 = [q.detach().clone() for q in st["ov"] if q is not None]
                    _vsx = [q for q in st["ov"] if q is not None]
                    _optx = torch.optim.LBFGS(_vsx, lr=1.0, max_iter=_ext, history_size=50, tolerance_grad=0.0,
                                              tolerance_change=0.0, line_search_fn="strong_wolfe")

                    def _clx():
                        _optx.zero_grad(set_to_none=True)
                        _, _, _Ex, _, _, _, _ = _fwd()
                        _Ex.backward()
                        return _Ex
                    _optx.step(_clx)
                    _grad_probe(st, _fwd, F, m, vol, cfg, x, v, gv, f"opt{_ext + a.ov_roll}", _gp_rec)
                    with torch.no_grad():
                        for q, z1 in zip(_vsx, _gp_z1):
                            q.copy_(z1)
                torch.save(_gp_rec, os.path.join(_GRAD_PROBE["out"], f"probe_t{int(t):04d}.pt"))
                print(f"    [기울기 탐침] t={t} -> {_GRAD_PROBE['out']}/probe_t{int(t):04d}.pt  " + "  ".join(
                    f"{k}: E {_gp_rec[k]['E']:.4e} 탄성 {_gp_rec[k]['E_el']:.4e} |du| {_gp_rec[k]['du_mean']:.3e}"
                    for k in _gp_rec if isinstance(_gp_rec[k], dict)), flush=True)
            if os.environ.get("AF_OV_ITERS"):
                torch.cuda.synchronize()
                print(f"    [반복] t={t} 평가 {_n[0]} 회, {time.time() - _tw:.2f} s", flush=True)
            if a.snap_inv > 0:
                _nb = _snap_inverted(st, a.snap_inv, a.snap_eps)
                _SNAP_N[0] = max(_SNAP_N[0], _nb)
            E0 = _E0[0]
        else:
            for _k in range(a.ov_roll):
                _, _, E, _, _, _pt, _ = _fwd()
                if st.get("ov_opt") is None:
                    _vs = [q for q in (st.get("ov") or []) if q is not None]
                    if not _vs:
                        raise SystemExit("--ov_roll: 변수를 못 만들었다")
                    st["ov_opt"] = _mk_ov_opt(_vs)
                if E0 is None:
                    E0 = float(E)
                if _CURVE is not None:
                    _tr.append(float(E))
                    _trp.append((float(_pt[0]), float(_pt[1]), float(_pt[3])))
                st["ov_opt"].zero_grad(set_to_none=True)
                E.backward()
                _ov_clip_grads(st)
                st["ov_opt"].step()
            if a.snap_inv > 0:
                _SNAP_N[0] = max(_SNAP_N[0],
                                 _snap_inverted(st, a.snap_inv, a.snap_eps))
        if os.environ.get("AF_OV_LANGEVIN") and a.ov_vars == "out" and st.get("ov"):
            # 증분 포텐셜 랑제빈 표본 (사용자 10-18): 최적화가 끝난 노드 출력 u 에서 p(u|s) ∝ exp(-E(u;s)/T) 를 MALA 로 걷는다.
            # T = 2·rho·E_min / d (2 차 근사에서 평형 <E - E_min> = d·T/2 = rho·E_min, d = 자유 노드 출력 수).
            # 걸음 크기는 수락률 ~0.57 로 맞춰 프레임 사이에 이어 쓴다. 마지막 표본으로 상태를 전진시킨다
            _lg = os.environ["AF_OV_LANGEVIN"].split(":")
            _ns, _rho, _sd = int(_lg[0]), float(_lg[1]), int(_lg[2]) if len(_lg) > 2 else 0
            _vs = [q for q in st["ov"] if q is not None]
            _gen = torch.Generator(device=dev).manual_seed(_sd * 100003 + int(t) * 7 + int(_E0[0] is not None))

            def _EG():
                with torch.enable_grad():
                    _Ev = _fwd()[2]
                    _g = torch.autograd.grad(_Ev, _vs)
                return float(_Ev), [gi.detach() for gi in _g]
            _Ec, _gc = _EG()
            _dof = sum(q.numel() for q in _vs)
            _T = 2.0 * _rho * max(_Ec, 1e-30) / _dof
            if _LGV["eta"] is None:                    # 첫 걸음: 잡음 표준편차 1e-4·EXT 에서 시작
                _LGV["eta"] = (1e-4 * EXT) ** 2 / (2.0 * _T)
            _eta = _LGV["eta"] * (_T / _LGV["T"] if _LGV["T"] else 1.0)
            _acc, _E_start = 0, _Ec
            for _it in range(_ns):
                _u0 = [q.detach().clone() for q in _vs]
                _nz = [torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=_gen) for q in _vs]
                with torch.no_grad():
                    for q, g_, z_ in zip(_vs, _gc, _nz):
                        q.add_(-_eta * g_ + (2.0 * _eta * _T) ** 0.5 * z_)
                _Ep, _gp = _EG()
                # MALA 수락: log a = -(E'-E)/T - |u-u'+eta g'|²/(4 eta T) + |u'-u+eta g|²/(4 eta T)
                with torch.no_grad():
                    _fw = sum(float(((q - u0 + _eta * g_) ** 2).sum()) for q, u0, g_ in zip(_vs, _u0, _gc))
                    _bw = sum(float(((u0 - q + _eta * gp_) ** 2).sum()) for q, u0, gp_ in zip(_vs, _u0, _gp))
                _la = -(_Ep - _Ec) / _T - _bw / (4 * _eta * _T) + _fw / (4 * _eta * _T)
                if math.isfinite(_Ep) and math.log(max(float(torch.rand(1, generator=_gen, device=dev)), 1e-30)) < _la:
                    _Ec, _gc, _acc = _Ep, _gp, _acc + 1
                    _eta *= 1.1
                else:
                    with torch.no_grad():
                        for q, u0 in zip(_vs, _u0):
                            q.copy_(u0)
                    _eta *= 0.7
            _LGV["eta"], _LGV["T"] = _eta, _T
            print(f"    [랑제빈] t={t} 걸음 {_ns} 수락 {_acc}/{_ns}  E {_E_start:.3e} -> {_Ec:.3e} "
                  f"(+{100 * (_Ec / max(_E_start, 1e-30) - 1):.1f}%, 목표 +{100 * _rho:.0f}%)  eta {_eta:.2e}", flush=True)
        x2, v2, E, dlog, F_tr, _pt, _Jd = _fwd()
        E1 = float(E)
        if a.ov_vars == "net" and os.environ.get("AF_OVNET_DIAG"):
            _pv = torch.cat([q.detach().reshape(-1) for q in net.parameters()])
            _gn = sum(float(q.grad.norm()) ** 2 for q in net.parameters() if q.grad is not None) ** 0.5
            print(f"    [망 변수] t={t} E0 {E0} E1 {E1:.6e} |θ| {float(_pv.norm()):.6e} |∇θ| {_gn:.3e} "
                  f"요구기울기 {sum(int(q.requires_grad) for q in net.parameters())}/{len(list(net.parameters()))}", flush=True)
        if os.environ.get("AF_OV_DIAG"):
            _tg = (_DT[0] * v + (_DT[0] ** 2) * gv)
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
        _HX_HOLD[0] = None
        _SM_HOLD[0] = None
        _OV_TGT[0] = None
        _GRID_OFF[0] = None
    _OV_LOG.append((E0 if E0 is not None else E1, E1))
    if _CURVE is not None:
        _CURVE.append((int(t), _tr, _trp))
    return x2.detach(), v2.detach(), _Jd.detach()


# 앙상블 오프셋 (셀의 분수). 첫 번째는 원점이고 나머지는 고르게 흩어 놓는다.
def _smooth_nodes(dpn, uniq, nn):
    """노드 변위를 격자 이웃과 섞는다 -> [M,3].

    d~_n = (1-a)·d_n + a·(이웃 평균). 이웃은 격자축 6, +면대각 12, +몸대각 14 다.
    실제로 존재하는 이웃만 평균에 넣는다. 색인은 프레임 안에서 한 번만 만든다.
    """
    if isinstance(_SM_HOLD[0], tuple):
        _idx, _msk = _SM_HOLD[0]
    else:
        _n0, _n1, _n2 = int(nn[0]), int(nn[1]), int(nn[2])
        _k2 = uniq % _n2
        _k1 = (uniq // _n2) % _n1
        _k0 = uniq // (_n1 * _n2)
        _po = SX.EDGE_OFFSETS[:7]
        _sel = ({0, 1, 2} if a.smooth_nb == 6 else
                {0, 1, 2, 3, 4, 5} if a.smooth_nb == 12 else set(range(7)))
        _offs = []
        for _i in sorted(_sel):
            _o = _po[_i]
            _offs.append(_o)
            _offs.append((-_o[0], -_o[1], -_o[2]))
        _I, _M = [], []
        for _o in _offs:
            _j0, _j1, _j2 = _k0 + _o[0], _k1 + _o[1], _k2 + _o[2]
            _ok = ((_j0 >= 0) & (_j1 >= 0) & (_j2 >= 0) & (_j0 < _n0)
                   & (_j1 < _n1) & (_j2 < _n2))
            _fl = (_j0 * _n1 + _j1) * _n2 + _j2
            _p = torch.searchsorted(uniq, _fl.clamp_min(0)
                                    ).clamp(max=uniq.numel() - 1)
            _ok = _ok & (uniq[_p] == _fl)
            _I.append(_p); _M.append(_ok)
        _idx = torch.stack(_I, 1)                     # [M,K]
        _msk = torch.stack(_M, 1).to(dpn.dtype)
        if _SM_HOLD[0] is not None:
            _SM_HOLD[0] = (_idx, _msk)
    _nb = (dpn[_idx] * _msk.unsqueeze(-1)).sum(1)
    _cnt = _msk.sum(1, keepdim=True).clamp_min(1.0)
    _avg = _nb / _cnt
    _a = float(a.smooth_a)
    return (1.0 - _a) * dpn + _a * _avg


class _FuseCmpDone(Exception):
    pass


_TAU_H = [None]     # (_TAU, _HES) -- _ov_once 가 _fieldS 로 넘긴다
_JR_HOLD = [None]   # 2차 근사를 전개하는 중심 J (바깥 반복용)
_CPOLY = [None]     # 셀 다항식 계수 (프레임마다 한 번 만든다)
_CP_INV = [None]    # 입자 -> 사면체 색인
_CP_E = [None]      # 이번 평가의 셀 다항식 에너지


def _cell_poly_build(rows, lam, aux, lat, tau_c, hes_c, inv, rep, mass, tgt,
                     wfree):
    """입자 기여를 사면체 12 자유도 다항식으로 바꿔 셀마다 합산한다.

      dx_p = sum_i lam_i u_i,      dJ = sum_i u_i (x) b_i
      관성  m_p/(2h²)|dx_p - t_p|²      -> M_ij = m_p lam_i lam_j / h², f_i
      탄성  <tau,dJ> + ½ dJ:H:dJ        -> g_i = tau b_i, K_ij = H : b_i b_j

    돌려주는 것: (rows_t [T,4], b [T,4,3], M [T,4,4], f [T,4,3],
                  g [T,4,3], K [T,4,3,4,3])
    """
    dev_ = rows.device
    rows_t = rows[rep]                                  # [T,4]
    rank_t = aux[1][rep]                                # [T,3]
    T = rows_t.shape[0]
    Ai = lat.Ai.to(tau_c.dtype)
    b = torch.zeros(T, 4, 3, device=dev_, dtype=tau_c.dtype)
    ar = torch.arange(T, device=dev_)
    for j in range(3):
        r = rank_t[:, j]
        b[ar, r + 1] += Ai[j]
        b[ar, r] -= Ai[j]
    # 관성 (손잡이에 구속된 입자는 가중치 0)
    mw = (wfree * mass).reshape(-1, 1)                  # [N,1]
    M = torch.zeros(T, 4, 4, device=dev_, dtype=tau_c.dtype)
    f = torch.zeros(T, 4, 3, device=dev_, dtype=tau_c.dtype)
    for i in range(4):
        _fi = torch.zeros(T, 3, device=dev_, dtype=tau_c.dtype)
        _fi.index_add_(0, inv, (mw * lam[:, i:i + 1] * tgt).to(tau_c.dtype))
        f[:, i] = _fi
        for j in range(4):
            _mij = torch.zeros(T, device=dev_, dtype=tau_c.dtype)
            _mij.index_add_(0, inv,
                            (mw[:, 0] * lam[:, i] * lam[:, j]).to(tau_c.dtype))
            M[:, i, j] = _mij
    # 탄성
    g = torch.einsum("tab,tib->tia", tau_c, b)
    K = torch.einsum("tacbd,tic,tjd->tiajb", hes_c, b, b)
    return rows_t, b, M, f, g, K


def _cp_eval(dpn, rows_t, b, M, f, g, K):
    """셀 다항식 에너지 (관성, 탄성) 와 사면체별 grad u."""
    U = dpn[rows_t]                                     # [T,4,3]
    e_in = (0.5 * torch.einsum("tij,tia,tja->t", M, U, U)
            - torch.einsum("tia,tia->t", f, U)).sum()
    e_el = (torch.einsum("tia,tia->t", g, U)
            + 0.5 * torch.einsum("tiajb,tia,tjb->t", K, U, U)).sum()
    gu = torch.einsum("tia,tib->tab", U, b)
    return e_in, e_el, gu


_CP_FN = [None]


def _cell_poly_energy(dpn, cp):
    """셀 다항식 에너지와 사면체별 grad u 를 돌려준다.

    --fuse 면 torch.compile 로 묶는다. 작은 씬에서는 연산량이 아니라 **커널
    발사 수**가 비용이라, 작은 einsum 여러 개를 하나로 묶는 것이 효과가 크다.
    """
    rows_t, b, M, f, g, K = cp
    if a.fuse and os.environ.get("AF_CP_CHECK") and _CP_FN[0] is None:
        _CP_FN[0] = 1
        with torch.enable_grad():
            _d1 = dpn.detach().clone().requires_grad_(True)
            _e1 = (_cp_eval(_d1, rows_t, b, M, f, g, K)[0]
                   + _cp_eval(_d1, rows_t, b, M, f, g, K)[1])
            _g1, = torch.autograd.grad(_e1, _d1)
            _d2 = dpn.detach().clone().requires_grad_(True)
            _e2 = CPOLY_K.energy(_d2, rows_t.to(torch.int32), M, g - f, K)
            _g2, = torch.autograd.grad(_e2, _d2)
            print(f"    [커널대조] 에너지 {float(_e1):.6e} 대 {float(_e2):.6e} "
                  f"(상대차 {abs(float(_e1-_e2))/max(abs(float(_e1)),1e-30):.2e}), "
                  f"기울기 상대오차 "
                  f"{float((_g1-_g2).norm()/_g1.norm().clamp_min(1e-30)):.2e}",
                  flush=True)
    if a.fuse:
        # **커널 한 번**: 1 차항을 g - f 로 미리 합쳐 Triton 에 넘긴다.
        # grad u 는 에너지와 무관하게 필요하므로 따로 구한다 (가벼운 einsum).
        E = CPOLY_K.energy(dpn, rows_t.to(torch.int32), M, g - f, K)
        # grad u 는 그래프를 끊지 않는다 (야코비안이 다른 항·상태 전진에 쓰인다)
        gu = torch.einsum("tia,tib->tab", dpn[rows_t], b)
        return E, torch.zeros((), device=dpn.device, dtype=dpn.dtype), gu
    return _cp_eval(dpn, *cp)


_HX_HOLD = [None]   # 프레임 안에서 육면체 색인을 재사용한다
_HG_LAST = [None]   # (노드 변위, 점유 셀 꼭짓점, 격자) -- 모래시계 항용
_HXD = [0]          # 육면체 진단 출력 횟수


def _hex_struct(x, lo, lat, nn, uniq):
    """육면체 삼선형의 (꼭짓점 색인 [N,8], 쓸 수 있나 [N], N [N,8], dN [N,8,3]).

    한 프레임 안에서 x 가 고정이라 격자마다 한 번만 만든다.
    """
    _key = int(uniq.data_ptr())
    _h = _HX_HOLD[0]
    if isinstance(_h, dict) and _key in _h:
        return _h[_key]
    nnl = [int(nn[k]) for k in range(3)]
    y = (x - lo) @ lat.Ai.T
    ci = y.floor().long()
    ci = torch.stack([ci[:, k].clamp(0, nnl[k] - 2) for k in range(3)], -1)
    xi = (y - ci).clamp(0.0, 1.0)                      # [N,3]
    _o = torch.tensor([[i_, j_, k_] for i_ in (0, 1) for j_ in (0, 1)
                       for k_ in (0, 1)], device=x.device, dtype=torch.long)
    v8 = ci.unsqueeze(1) + _o.unsqueeze(0)             # [N,8,3]
    _inb = ((v8[..., 0] < nnl[0]) & (v8[..., 1] < nnl[1])
            & (v8[..., 2] < nnl[2]))
    flat = (v8[..., 0] * nnl[1] + v8[..., 1]) * nnl[2] + v8[..., 2]
    _pos = torch.searchsorted(uniq, flat.reshape(-1).clamp_min(0)
                              ).reshape(flat.shape).clamp(max=uniq.numel() - 1)
    _ok = ((uniq[_pos] == flat) & _inb).all(1)
    of = _o.to(xi.dtype)
    w = of.unsqueeze(0) * xi.unsqueeze(1) + \
        (1.0 - of).unsqueeze(0) * (1.0 - xi).unsqueeze(1)      # [N,8,3]
    dw = (2.0 * of - 1.0).unsqueeze(0).expand_as(w)
    _N = w.prod(-1)
    _dN = torch.stack([dw[..., 0] * w[..., 1] * w[..., 2],
                       w[..., 0] * dw[..., 1] * w[..., 2],
                       w[..., 0] * w[..., 1] * dw[..., 2]], -1)
    # 모래시계 안정화는 **셀마다** 한 번 걸어야 하므로 점유 셀을 추린다
    _cflat = (ci[:, 0] * nnl[1] + ci[:, 1]) * nnl[2] + ci[:, 2]
    _uc, _first = torch.unique(_cflat, return_inverse=True)
    _sel = torch.zeros(_uc.numel(), dtype=torch.long, device=x.device)
    _sel.scatter_(0, _first, torch.arange(_cflat.numel(), device=x.device))
    _cell8 = _pos[_sel]                                # [C,8]
    out = (_pos, _ok, _N, _dN, _cell8)
    if isinstance(_h, dict):
        _h[_key] = out
    return out


def _hex_field(x, dpn, lo, lat, nn, uniq):
    """육면체 셀의 **삼선형** 사상: (u [N,3], grad u [N,3,3], 쓸 수 있나 [N]).

        u(xi)    = sum_m N_m(xi) u_m
        grad_x u = sum_m u_m (dN_m/dxi)^T Ai,     xi = Ai (x - lo) - ci

    입자 변위와 변형구배가 **같은 사상**에서 나온다. 사면체(중심좌표) 쪽은 셀을
    6 조각으로 쪼개 조각마다 J 가 상수였다.
    """
    _pos, _ok, _N, _dN, _ = _hex_struct(x, lo, lat, nn, uniq)
    U = dpn[_pos]                                      # [N,8,3]
    u = (U * _N.to(U.dtype).unsqueeze(-1)).sum(1)
    # **grad u 를 그대로 돌려준다** (I 는 부르는 쪽에서 더한다 -- 사면체
    # g2p_jac_pre 와 같은 규약이다. 여기서 I 를 더해 돌려주면 I 가 두 번
    # 들어가 정지 상태에서 det ~ 8 이 된다).
    G = torch.einsum("nmi,nmj->nij", U, _dN.to(U.dtype)) @ lat.Ai.to(U.dtype)
    return u, G, _ok


_HG_GAMMA = [None]


def _hex_hg_energy(dpn, cell8, lat, cfg, kappa):
    """삼선형 육면체의 **모래시계(영에너지) 모드** 안정화 에너지.

    꼭짓점 부호 s = 2o - 1 로 만든 네 벡터

        G1 = sx sy,  G2 = sy sz,  G3 = sx sz,  G4 = sx sy sz

    는 강체·선형 변위장과 직교한다 (sum_m G_m = 0, sum_m G_m x_m = 0). 그래서
    이 성분은 변형을 하나도 내지 않으면서 셀을 일그러뜨릴 수 있고, 변형을
    입자 위치에서만 재는 우리 설정에서 최적화가 공짜로 키운다. 진폭
    q_a = sum_m G_a,m u_m 에 벌점을 건다:

        E = 0.5 k mu V sum_a |q_a / h|^2 / 8
    """
    if _HG_GAMMA[0] is None:
        _o = torch.tensor([[i_, j_, k_] for i_ in (0, 1) for j_ in (0, 1)
                           for k_ in (0, 1)], dtype=dpn.dtype,
                          device=dpn.device)
        sx, sy, sz = (2.0 * _o - 1.0).unbind(-1)
        _HG_GAMMA[0] = torch.stack([sx * sy, sy * sz, sx * sz, sx * sy * sz])
    G = _HG_GAMMA[0].to(dpn.dtype)                      # [4,8]
    U = dpn[cell8]                                      # [C,8,3]
    q = torch.einsum("am,cmi->cai", G, U)               # [C,4,3]
    mu, _ = phys_resid.lame(cfg["E"], cfg["nu"])
    V = float(abs(torch.linalg.det(lat.A.double())))
    h = float(lat.s)
    return 0.5 * kappa * mu * V * (q / h).pow(2).sum() / 8.0


def _gt_fscal(d, t0, L):
    """교사 궤적의 F 로 (det, trC, ‖E‖, C_zz, psi) 를 만든다 -> [L,N,5]."""
    if "F" not in d:
        return None
    _F = traj_F(d)
    out = []
    for _i in range(L):
        Fi = _F[min(t0 + _i + 1, _F.shape[0] - 1)].to(dev).float()
        C = Fi.transpose(-1, -2) @ Fi
        E = 0.5 * (C - torch.eye(3, device=dev))
        psi, _ = phys_resid.psi_of(Fi, d["cfg"], FRAME_DT)
        out.append(torch.stack([
            torch.linalg.det(Fi), C.diagonal(dim1=-2, dim2=-1).sum(-1),
            E.reshape(E.shape[0], -1).norm(dim=-1), C[:, 2, 2], psi], -1).cpu())
    return torch.stack(out)


def _ov_frame(d, t, gsel, p, x, v, F):
    """프레임 하나를 전진시킨다 -> (x2, v2, F_next).

    --ens_off N 이면 오프셋이 다른 격자 N 벌의 노드 변수를 **함께** 최적화한다
    (변위장은 그 평균이고 목적함수는 그 평균 하나로 센다). 평균이 선형이라
    평균한 변위장의 기울기 = 기울기의 평균이므로, 야코비안도 평균이 된다.
    """
    cfg = d["cfg"]
    # **바깥 반복**: 2차 근사를 현재 상태에서 다시 전개한다. 한 프레임 변위가
    # 크면 J=I 둘레 한 번으로는 모자라다 (실측: 89 프레임에서 비 0.88).
    _JR_HOLD[0] = None
    if a.cmp_lin:
        # **같은 상태에서 두 번 푼다**: 원본 탄성 / 2차 근사. 전진은 원본으로.
        _keep = a.elast
        a.elast = "full"
        x2, v2, Jd = _ov_once(d, t, gsel, p, x, v, F)
        _s2 = _OV_CUR[0]
        if _s2 is not None:
            _s2["ov"] = None
            _s2["ov_opt"] = None
        a.elast = "lin"
        _x2l, _v2l, _Jl = _ov_once(d, t, gsel, p, x, v, F)
        a.elast = _keep
        with torch.no_grad():
            _df = (x2 - x)
            _dl = (_x2l - x)
            _mv = float(_df.norm(dim=-1).mean())
            _er = float((_df - _dl).norm(dim=-1).mean())
            _hh = float(_GRID[0][1].s) if _GRID[0] is not None else float("nan")
            print(f"    [근사비교] t={t} 원본 평균이동 {_mv:.5e}  차이 "
                  f"{_er:.5e}  이동 대비 {100 * _er / max(_mv, 1e-30):.2f}%  "
                  f"셀간격 대비 {100 * _er / _hh:.2f}% (h={_hh:.5f})",
                  flush=True)
        if _s2 is not None:
            _s2["ov"] = None
            _s2["ov_opt"] = None
    else:
        _no = max(int(a.lin_outer), 1) if a.elast == "lin" else 1
        for _oi in range(_no):
            x2, v2, Jd = _ov_once(d, t, gsel, p, x, v, F)
            if _no > 1 and _oi + 1 < _no:
                _JR_HOLD[0] = Jd.detach()
    _JR_HOLD[0] = None
    with torch.no_grad():
        F_tr = Jd.to(F.dtype) @ F
        _, dlog = phys_resid.psi_of(F_tr, cfg, _DT[0])
        F_next = phys_resid.plastic_step(F_tr, dlog).detach()
    return x2.detach(), v2.detach(), F_next


@torch.no_grad()        # <- _ov_frame 을 위에 끼우면서 이 데코레이터를 빼앗았다.
def rollout(d, t0, L, gsel=None):   #    평가가 그래프를 쌓아 메모리가 터진다.
    with traj_scope(d):
        if os.environ.get("AF_TORCH_PROF"):
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as _pf:
                _r = _rollout(d, t0, L, gsel)
            _ka = _pf.key_averages()
            print(_ka.table(sort_by="cuda_time_total", row_limit=16),
                  flush=True)
            print(_ka.table(sort_by="self_cpu_time_total", row_limit=25),     # CPU 에서 오래 걸린 연산 (장치 오가기·동기화 포함)
                  flush=True)
            _n = sum(int(e.count) for e in _ka)
            _cu = sum(float(e.self_device_time_total) for e in _ka) / 1e3
            _cp = sum(float(e.self_cpu_time_total) for e in _ka) / 1e3
            print(f"[프로파일] 연산 호출 {_n} 회, GPU 자체시간 {_cu:.1f} ms, "
                  f"CPU 자체시간 {_cp:.1f} ms", flush=True)
            return _r
        if _PROF_ON:
            _PROF.clear(); torch.cuda.synchronize(); _tw0 = time.time()
            _r = _rollout(d, t0, L, gsel)
            torch.cuda.synchronize(); _tw = time.time() - _tw0
            print(f"[롤아웃 시간] {L} 프레임 {_tw:.2f} s (프레임당 {_tw / max(L, 1):.3f} s) | "
                  + " ".join(f"{k} {v:.2f}s" for k, v in sorted(_PROF.items(), key=lambda z: -z[1])), flush=True)
            return _r
        return _rollout(d, t0, L, gsel)


def _rollout(d, t0, L, gsel=None):
    # 드롭아웃이 켜져 있으면 롤아웃이 확률적이 된다 -- 평가는 항상 eval 로.
    net.eval()
    if gsel is None:
        gsel = torch.arange(d["x"].shape[1], device=dev)
    x, v = DM.init_state(d, t0, gsel, FRAME_DT, a.v0_traj, dev)
    if a.v0_traj and "v" in d:
        print(f"    [초기속도] 궤적의 v 를 쓴다 (크기 평균 "
              f"{float(v.norm(dim=-1).mean()):.4f})", flush=True)
    p = x[fps(x, a.n_anchors, a.seed)] if a.refps else take(d["x"][t0], AIDX)
    if os.environ.get("AF_STATE_EVAL"):
        # 상태 적합 감시 (사용자 10-17): AF_STATE_EVAL=체크포인트:TB디렉토리:끝반복. 체크포인트가 바뀔 때마다 그 망으로
        # AF_IP_STATES 덤프의 상태 전부(0 = 정답 시작)에서 한 스텝 굴려
        #   출력오차% = mean|x2_망 - x_출력만최적화 다음 상태| / EXT x 100,  증분 포텐셜,  (AF_OV_MIN 있으면) 최솟값 대비 초과%
        # 를 TB 에 쓴다. 학습 실행은 건드리지 않는다
        from torch.utils.tensorboard import SummaryWriter as _SW
        _ckp, _tbd, _end = os.environ["AF_STATE_EVAL"].rsplit(":", 2); _end = int(_end)
        _D = torch.load(os.environ["AF_IP_STATES"], map_location="cpu", weights_only=False)
        _mins = ({r["state"]: r["E_min"] for r in json.load(open(os.environ["AF_OV_MIN"]))}
                 if os.environ.get("AF_OV_MIN") and os.path.exists(os.environ["AF_OV_MIN"]) else {})
        _F0 = take(traj_F(d)[t0], gsel).float()
        _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
        _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
        _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
        _hs = FRAME_DT; _nrm = float(_m.sum()) * (EXT ** 2) / (_hs ** 2)
        _nst = _D["pred"].shape[0]
        _tb = _SW(_tbd); _seen = None
        net.requires_grad_(False)
        while True:
            if not os.path.exists(_ckp):
                time.sleep(60); continue
            _mt = os.path.getmtime(_ckp)
            if _mt == _seen:
                time.sleep(60); continue
            time.sleep(5)
            try:
                _c = torch.load(_ckp, map_location=dev, weights_only=False)
            except Exception as _e:
                print(f"    [감시] 읽기 실패 ({_e}) -- 다음에", flush=True); time.sleep(30); continue
            _seen = _mt; _k = int(_c["k"])
            net.load_state_dict(_c["net"]); net.eval()
            _errs, _Es, _pc = [], [], []
            for i in range(_nst):
                if i == 0:
                    xs, vs, Fs = x, v, _F0
                else:
                    xs, vs = _D["pred"][i - 1].float().to(dev), _D["v"][i - 1].float().to(dev)
                    Fs = _D["F"][i - 1].float().to(dev).reshape(-1, 3, 3)
                ps = xs[AIDX] if (torch.is_tensor(AIDX) and int(AIDX.max()) < xs.shape[0]) else p
                with torch.enable_grad(), dt_scope(_hs):
                    x2, _, _, _, _, _, _, _, _, _jf = step_once(d, t0 + i, gsel, ps, xs, vs, need_J=False)
                    _E, _, _, _ = ip_of(xs, x2 - xs, vs, Fs, _m, _vol, _cfg, _hs, _ng, _gl, g=_gv, norm=_nrm, jac=_jf)
                _DET_LAST[0] = None
                _er = float((x2.detach() - _D["pred"][i].float().to(dev)).norm(dim=-1).mean()) / EXT
                _errs.append(_er); _Es.append(float(_E))
                _tb.add_scalar(f"출력오차%/{i:02d}", 100 * _er, _k)
                _tb.add_scalar(f"증분포텐셜/{i:02d}", float(_E), _k)
                if i in _mins:
                    _pc.append(100 * (float(_E) / _mins[i] - 1))
                    _tb.add_scalar(f"최솟값대비_초과%/{i:02d}", _pc[-1], _k)
            _tb.add_scalar("요약/출력오차%_평균", 100 * float(np.mean(_errs)), _k)
            _tb.add_scalar("요약/출력오차%_최대", 100 * float(np.max(_errs)), _k)
            _tb.add_scalar("요약/증분포텐셜_평균", float(np.mean(_Es)), _k)
            if _pc:
                _tb.add_scalar("요약/최솟값대비_초과%_평균", float(np.mean(_pc)), _k)
                _tb.add_scalar("요약/최솟값대비_초과%_중앙", float(np.median(_pc)), _k)
            _tb.flush()
            print(f"    [감시] 반복 {_k}: 출력오차 평균 {100 * np.mean(_errs):.4f}%  증분포텐셜 평균 {np.mean(_Es):.3e}"
                  + (f"  초과 중앙 {np.median(_pc):.1f}%" if _pc else ""), flush=True)
            if _k >= _end:
                break
        _tb.close()
        raise SystemExit(0)
    if os.environ.get("AF_OV_MIN_STATES"):
        # 상태별 출력만 최적화 최솟값 (사용자 10-17): AF_IP_STATES 덤프의 상태(0 = 정답 시작, i = 덤프 프레임 i)마다
        # 롤아웃과 **같은 경로**(_ov_frame, --ov_init·--ov_roll·--ov_opt 그대로)로 한 스텝을 최적화해 최적화 전·후 증분 포텐셜을 남긴다
        if a.ov_roll <= 0:
            raise SystemExit("AF_OV_MIN_STATES 는 --out_var --ov_roll 설정이 필요하다")
        _D = torch.load(os.environ["AF_IP_STATES"], map_location="cpu", weights_only=False)
        _F0 = take(traj_F(d)[t0], gsel).float()
        _res = []
        for i in range(_D["pred"].shape[0]):
            if i == 0:
                xs, vs, Fs = x, v, _F0
            else:
                xs, vs = _D["pred"][i - 1].float().to(dev), _D["v"][i - 1].float().to(dev)
                Fs = _D["F"][i - 1].float().to(dev).reshape(-1, 3, 3)
            ps = xs[AIDX] if (torch.is_tensor(AIDX) and int(AIDX.max()) < xs.shape[0]) else p
            with dt_scope(FRAME_DT / EVAL_SUB):
                _ov_frame(d, t0 + i, gsel, ps, xs, vs, Fs)
            _e0, _e1 = _OV_LOG[-1]
            _res.append(dict(state=i, E_init=float(_e0), E_min=float(_e1)))
            print(f"    [상태 최솟값] 상태 {i}: {_e0:.4e} -> {_e1:.4e}", flush=True)
            json.dump(_res, open(os.environ["AF_OV_MIN_STATES"], "w"))
        print(f"[상태 최솟값 기록] {os.environ['AF_OV_MIN_STATES']} 상태 {len(_res)}", flush=True)
        raise SystemExit(0)
    if os.environ.get("AF_GAP_ROLL"):
        # 진단 (사용자 10-17): 같은 상태에서 망 증분 포텐셜 vs 망 출력에서 출발한 출력만 최적화 최솟값 (_ov_reference).
        # 상태 = (가) 이 망 자신의 롤아웃이 지나간 상태, (나) AF_IP_STATES 덤프(출력만 최적화 궤적)의 상태. 프레임 AF_GAP_FRAMES
        _fr = [int(q) for q in os.environ.get("AF_GAP_FRAMES", ",".join(str(k) for k in range(4, 81, 4))).split(",")]
        _sc = dict(cfg=d["cfg"], mass=MASS[gsel], ext=EXT)
        _hs = FRAME_DT / EVAL_SUB
        _out = []
        xc, vc, pc = x.clone(), v.clone(), p
        Fc = take(traj_F(d)[t0], gsel).float()
        _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
        _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
        _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
        for i in range(max(_fr) + 1):
            if i in _fr:                                    # 프레임 i 상태 (i=0 이 시작) 에서 잰다
                _r = _ov_reference(d, gsel, pc, xc, vc, Fc, _sc, _hs, t=t0 + i)
                _out.append(dict(kind="model_roll", frame=i, E_model=_r["hist"][0], E_min=_r["E"], hist=_r["hist"]))
            if i == max(_fr):
                break
            with torch.enable_grad(), dt_scope(_hs):
                x2, p2, v2, _, _, _, _, _, _, _jf = step_once(d, t0 + i, gsel, pc, xc, vc, need_J=False)
                _E, _dl, _Ftr, _ = ip_of(xc, x2 - xc, vc, Fc, _m, _vol, _cfg, _hs, _ng, _gl, g=_gv,
                                         norm=float(_m.sum()) * (EXT ** 2) / (_hs ** 2), jac=_jf)
            _DET_LAST[0] = None
            with torch.no_grad():
                Fc = phys_resid.plastic_step(_Ftr, _dl).detach()
            xc, vc, pc = x2.detach(), v2.detach(), p2.detach()
        if os.environ.get("AF_IP_STATES"):
            _D = torch.load(os.environ["AF_IP_STATES"], map_location="cpu", weights_only=False)
            for i in _fr:
                if i < 1 or i > _D["pred"].shape[0]:
                    continue
                xs, vs = _D["pred"][i - 1].float().to(dev), _D["v"][i - 1].float().to(dev)
                Fs = _D["F"][i - 1].float().to(dev).reshape(-1, 3, 3)
                ps = xs[AIDX] if (torch.is_tensor(AIDX) and int(AIDX.max()) < xs.shape[0]) else p
                _r = _ov_reference(d, gsel, ps, xs, vs, Fs, _sc, _hs, t=t0 + i)
                _out.append(dict(kind="ov_traj", frame=i, E_model=_r["hist"][0], E_min=_r["E"], hist=_r["hist"]))
        json.dump(_out, open(os.environ["AF_GAP_ROLL"], "w"))
        print(f"[간극 기록] {os.environ['AF_GAP_ROLL']} 상태 {len(_out)}", flush=True)
        raise SystemExit(0)
    if os.environ.get("AF_IP_INPERT"):
        # 망 입력 섭동 민감도 (사용자 10-17): 출력만 최적화 롤아웃 덤프(AF_IP_STATES=덤프 경로)에서 바닥에 닿은 뒤의
        # 프레임을 무작위로 골라 그 상태(x, v, F)에서 망 한 스텝의 증분 포텐셜을 잰다.
        #   E0 = 그대로, E(σ) = 표준화된 입력 공간에서 σ 크기 가우시안 섭동 (feat + σ·in_sd·n),
        #   E_rand = 입력 자체를 무작위로 (in_mu + in_sd·n) -- 같은 물리 상태에서 여러 번 재서 평균
        # 비율 = (E(σ) − E0) / mean(E_rand)
        _D = torch.load(os.environ["AF_IP_STATES"], map_location="cpu", weights_only=False)
        _sg = [float(q) for q in os.environ.get("AF_IP_SIGMAS", "1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3").split(",")]
        _ns = int(os.environ.get("AF_IP_SEEDS", "4")); _nr = int(os.environ.get("AF_IP_NRAND", "8"))
        _nst = int(os.environ.get("AF_IP_NSTATES", "5"))
        _P, _V, _FF = _D["pred"].float(), _D["v"].float(), _D["F"].float()
        _floor = [b for b in (d["cfg"].get("boundary_conditions") or []) if b.get("type") == "surface_collider"][0]
        _fz = float(_floor["point"][2])
        _zmin = _P[:, :, 2].min(1).values
        _hit = [k for k in range(_P.shape[0]) if float(_zmin[k]) <= _fz + 1e-3]
        _k0 = _hit[0] if _hit else 0
        _rg = np.random.RandomState(0)
        _cand = list(range(_k0 + 1, _P.shape[0] - 1))
        _pick = sorted(_rg.choice(_cand, size=min(_nst, len(_cand)), replace=False).tolist())
        print(f"  [입력섭동] 바닥 z={_fz}, 처음 닿은 프레임 {_k0 + 1}, 고른 상태(프레임) {[k + 1 for k in _pick]}", flush=True)
        _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
        _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
        _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
        _hs = FRAME_DT / EVAL_SUB
        _nrm = float(_m.sum()) * (EXT ** 2) / (_hs ** 2)

        def _ip1(t, xs, vs, Fs):
            Fc, xc, vc, Es = Fs, xs, vs, []
            pc = xs[AIDX] if (torch.is_tensor(AIDX) and AIDX.numel() and int(AIDX.max()) < xs.shape[0]) else p
            for _sb in range(EVAL_SUB):
                with torch.enable_grad(), dt_scope(_hs):
                    x2, p2, v2, _, _, _, _, _, _, _jf = step_once(d, t, gsel, pc, xc, vc, need_J=False)
                    _fmi = (free_mask(d, x2.shape[0], dev, gsel, xc, t) if a.control else None)
                    if _BC_LAST[0] is not None:
                        _am = _BC_LAST[0][1].reshape(-1) > 0.5
                        _fmi = (~_am) if _fmi is None else (_fmi & ~_am)
                    _E, _dl, _Ftr, _ = ip_of(xc, x2 - xc, vc, Fc, _m, _vol, _cfg, _hs, _ng, _gl,
                                             g=_gv, norm=_nrm, free=_fmi, jac=_jf)
                    Es.append(float(_E))
                with torch.no_grad():
                    Fc = phys_resid.plastic_step(_Ftr, _dl).detach()
                xc, vc, pc = x2.detach(), v2.detach(), p2.detach()
            return float(np.mean(Es))

        def _hook_pert(sig, seed):
            def h(f, nt):
                g = torch.Generator(device=f.device).manual_seed(seed)
                return f + sig * nt.in_sd.to(f.dtype) * torch.randn(f.shape, device=f.device, dtype=f.dtype, generator=g)
            return h

        def _hook_rand(seed):
            def h(f, nt):
                g = torch.Generator(device=f.device).manual_seed(seed)
                return nt.in_mu.to(f.dtype) + nt.in_sd.to(f.dtype) * torch.randn(f.shape, device=f.device, dtype=f.dtype, generator=g)
            return h

        _res = {"sigmas": _sg, "floor_frame": _k0 + 1, "states": []}
        for k in _pick:
            xs, vs, Fs = _P[k].to(dev), _V[k].to(dev), _FF[k].to(dev).reshape(-1, 3, 3)
            t = t0 + k + 1
            DM.INPUT_HOOK[0] = None
            E0 = _ip1(t, xs, vs, Fs)
            Er = []
            for r in range(_nr):
                DM.INPUT_HOOK[0] = _hook_rand(5000 + r); Er.append(_ip1(t, xs, vs, Fs))
            Ep = []
            for sgm in _sg:
                row = []
                for sd in range(_ns):
                    DM.INPUT_HOOK[0] = _hook_pert(sgm, 1000 + sd); row.append(_ip1(t, xs, vs, Fs))
                Ep.append(row)
            DM.INPUT_HOOK[0] = None
            _mr = float(np.mean(Er))
            _res["states"].append(dict(frame=k + 1, E0=E0, E_rand=Er, E_pert=Ep))
            print(f"  [입력섭동] 프레임 {k + 1}: E0 {E0:.3e}  무작위입력 평균 {_mr:.3e}  "
                  + " ".join(f"σ={s_:g}:{(np.mean(r_) - E0) / _mr:+.2e}" for s_, r_ in zip(_sg, Ep)), flush=True)
        json.dump(_res, open(os.environ["AF_IP_INPERT"], "w"))
        print(f"[입력섭동 기록] {os.environ['AF_IP_INPERT']}", flush=True)
        raise SystemExit(0)
    if os.environ.get("AF_IP_PERTURB"):
        # 상태 섭동 민감도: 시작 상태(t0)에 크기 σ 의 가우시안 섭동을 주고 망 한 스텝의 증분 포텐셜·출력잔차를 잰다.
        # 위치: x + σ·EXT·n (앵커도 같은 섭동), 속도: v + σ·mean|v|·n. F 는 정답 t0 그대로. 값 = 서브스텝 평균
        _sg = [float(q) for q in os.environ.get("AF_IP_SIGMAS", "0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1").split(",")]
        _ns = int(os.environ.get("AF_IP_SEEDS", "4"))
        _F0 = take(traj_F(d)[t0], gsel).float()
        _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
        _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
        _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
        _hs = FRAME_DT / EVAL_SUB
        _nrm = float(_m.sum()) * (EXT ** 2) / (_hs ** 2)
        _vm = float(v.norm(dim=-1).mean())
        _aidx = AIDX if torch.is_tensor(AIDX) else None

        def _ip_at(xs, vs, ps):
            Fc, xc, vc, pc, Es, Rs = _F0, xs, vs, ps, [], []
            for _sb in range(EVAL_SUB):
                with torch.enable_grad(), dt_scope(_hs):
                    DM.PROBE[0] = "want"
                    x2, p2, v2, _, _, _, _, _, _, _jf = step_once(d, t0, gsel, pc, xc, vc, need_J=False)
                    DM.PROBE[0] = None
                    _fmi = (free_mask(d, x2.shape[0], dev, gsel, xc, t0) if a.control else None)
                    if _BC_LAST[0] is not None:
                        _am = _BC_LAST[0][1].reshape(-1) > 0.5
                        _fmi = (~_am) if _fmi is None else (_fmi & ~_am)
                    _E, _dl, _Ftr, _ = ip_of(xc, x2 - xc, vc, Fc, _m, _vol, _cfg, _hs, _ng, _gl,
                                             g=_gv, norm=_nrm, free=_fmi, jac=_jf)
                    _r = float("nan")
                    if DM.PROBE[1]:
                        _pz = DM.PROBE[1][0]
                        _gU, = torch.autograd.grad(_E * _nrm, _pz, retain_graph=True)
                        _mN, = torch.autograd.grad((_m.unsqueeze(-1) * x2).sum(), _pz)
                        _mN = _mN[:, 0]
                        _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                        _r = float(_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hs ** 2) / _mN[_sup].mean() / EXT)
                    DM.PROBE[1] = None
                    Es.append(float(_E)); Rs.append(_r)
                with torch.no_grad():
                    Fc = phys_resid.plastic_step(_Ftr, _dl).detach()
                xc, vc, pc = x2.detach(), v2.detach(), p2.detach()
            return float(np.mean(Es)), float(np.mean(Rs))

        _res = {"t0": t0, "ext": EXT, "v_mean": _vm, "sigmas": _sg, "pos": [], "vel": []}
        for _kind in ("pos", "vel"):
            for _s in _sg:
                _row = []
                for _sd in range(_ns if _s > 0 else 1):
                    _g = torch.Generator(device=dev).manual_seed(1000 + _sd)
                    _n = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=_g)
                    if _kind == "pos":
                        _xs, _vs = x + _s * EXT * _n, v
                        _ps = (_xs[_aidx] if (_aidx is not None and _aidx.numel() and int(_aidx.max()) < _xs.shape[0]) else p)
                    else:
                        _xs, _vs, _ps = x, v + _s * _vm * _n, p
                    _row.append(_ip_at(_xs, _vs, _ps))
                _res[_kind].append(_row)
                print(f"  [섭동] {_kind} σ={_s:g}: E " + " ".join(f"{q[0]:.3e}" for q in _row)
                      + " | 출력잔차 " + " ".join(f"{q[1]:.3e}" for q in _row), flush=True)
        json.dump(_res, open(os.environ["AF_IP_PERTURB"], "w"))
        print(f"[섭동 기록] {os.environ['AF_IP_PERTURB']}", flush=True)
        raise SystemExit(0)
    if os.environ.get("AF_NODE_FIT") and not _NODE_FIT_DONE[0]:
        # 망 하나를 **전 프레임 한꺼번에** 학습: 목표 실행(출력만 최적화)의 프레임 i 상태에서 망의 노드 출력이
        # 그 실행의 프레임 i 노드 출력과 같아지게 (상태가 같아 격자·노드가 같다). 끝나면 고정된 망으로 아래 롤아웃.
        _NODE_FIT_DONE[0] = True
        _sp = os.environ["AF_NODE_FIT"].split(":")
        _tg = torch.load(_sp[0], map_location="cpu", weights_only=False)
        _it, _lr = int(_sp[1]), float(_sp[2])
        _nf = len(_tg["nodes"])
        _opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=_lr)
        _sch = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_it, eta_min=_lr * 0.01)
        _gen = torch.Generator().manual_seed(0)
        _x0, _v0 = x.clone(), v.clone()
        _hist = []
        # AF_NODE_FIT_NOISE="sx:sv" -- 입력 상태에 가우스 잡음 (위치 표준편차 sx, 속도 sv). 목표는 깨끗한 상태의 출력 그대로
        # (입력 섭동에 강하게). 잡음으로 활성 노드가 조금 달라지므로 노드는 격자 정수 좌표로 짝지어 겹치는 것만 센다.
        _nz = [float(q) for q in os.environ.get("AF_NODE_FIT_NOISE", "0:0").split(":")]
        if not a.resume:
            # 새 망: --infer 는 입력 표준화를 체크포인트에서 받는다고 보고 건너뛴다 -- 여기서 목표 실행의 전 프레임 상태로 잰다
            with torch.no_grad():
                _sf = []
                for _i in range(_nf):
                    _xs = _x0 if _i == 0 else _tg["pred"][_i - 1].to(dev, x.dtype)
                    _vs = _v0 if _i == 0 else _tg["v"][_i - 1].to(dev, v.dtype)
                    _sf.append(node_feats(d, t0 + _i, gsel, _xs, _vs)[0])
                net.set_input_stats(torch.cat(_sf, 0))
            print(f"    [입력 표준화] 목표 실행 {_nf} 프레임 노드 특징으로 평균·표준편차 (채널 {net.in_mu.numel()}, "
                  f"sd 최소 {float(net.in_sd.min()):.2e} 최대 {float(net.in_sd.max()):.2e})", flush=True)
        _tw = time.time()
        for _k in range(_it):
            _i = int(torch.randint(0, _nf, (1,), generator=_gen))
            _xs = _x0 if _i == 0 else _tg["pred"][_i - 1].to(dev, x.dtype)
            _vs = _v0 if _i == 0 else _tg["v"][_i - 1].to(dev, v.dtype)
            if _nz[0] > 0:
                _xs = _xs + _nz[0] * torch.randn_like(_xs)
            if _nz[1] > 0:
                _vs = _vs + _nz[1] * torch.randn_like(_vs)
            with torch.enable_grad(), dt_scope(FRAME_DT):
                step_once(d, t0 + _i, gsel, p, _xs, _vs, need_J=False)
                _np, _dp = _DPN_LIVE[0]
                _tp, _td = _tg["nodes"][_i][0].to(dev), _tg["nodes"][_i][1].to(dev)
                if _nz[0] > 0 or _nz[1] > 0:
                    _h = float(a.node_h)                                   # 노드는 세계 고정 격자 위 -- 위치/h 반올림으로 짝짓기
                    _ka = torch.round(_np.detach() / (_h * 1e-3)).long()
                    _kb = torch.round(_tp / (_h * 1e-3)).long()
                    _key = lambda k_: (k_[:, 0] * 1000003 + k_[:, 1]) * 1000003 + k_[:, 2]
                    _ka1, _kb1 = _key(_ka), _key(_kb)
                    _sb, _ob = torch.sort(_kb1)
                    _pos = torch.searchsorted(_sb, _ka1).clamp(max=_sb.numel() - 1)
                    _hit = _sb[_pos] == _ka1
                    _ia, _ib = torch.nonzero(_hit).squeeze(1), _ob[_pos[_hit]]
                    _loss = ((_dp[_ia, :3] - _td[_ib, :3].to(_dp.dtype)) ** 2).sum(-1).mean() / (EXT ** 2)
                else:
                    if _tp.shape != _np.shape or float((_tp - _np.detach()).abs().max()) > 1e-5:
                        raise SystemExit(f"AF_NODE_FIT: 프레임 {_i} 노드가 목표와 다르다")
                    _loss = ((_dp[:, :3] - _td[:, :3].to(_dp.dtype)) ** 2).sum(-1).mean() / (EXT ** 2)
                _opt.zero_grad(set_to_none=True)
                _loss.backward()
                _opt.step(); _sch.step()
            _hist.append((_k, _i, float(_loss)))
            if _k % 100 == 0 or _k == _it - 1:
                _rec = [h[2] for h in _hist[-100:]]
                print(f"    [노드 맞춤 학습] {_k + 1}/{_it}  최근 100 평균 손실 {sum(_rec) / len(_rec):.4e}  "
                      f"lr {_sch.get_last_lr()[0]:.2e}  {time.time() - _tw:.0f} s", flush=True)
        if os.environ.get("AF_NODE_FIT_LOG"):
            json.dump(_hist, open(os.environ["AF_NODE_FIT_LOG"], "w"))
        torch.save({"net": net.state_dict()}, os.environ.get("AF_NODE_FIT_CKPT", "/tmp/node_fit.pt"))
        _DPN_LIVE[0] = None
    if os.environ.get("AF_STATE_FIT_MULTI") and not _NODE_FIT_DONE[0]:
        # 출력만 최적화 **여러 궤적**의 상태로 새 망 학습 (사용자 10-17). AF_STATE_FIT_MULTI="태그=덤프,태그=덤프,...:반복:배치".
        # 궤적마다 상태 = 프레임 0(정답 시작) + 덤프 프레임 1..L-1. 매 반복 (궤적, 프레임)을 균등하게 배치만큼 뽑아 그 궤적의
        # 질량·크기(traj_scope)로 망 한 스텝(dt = 프레임)의 phys_w·증분 포텐셜 + det 벌점을 평균해 Adam (lr = --lr), 기울기 1.0 자르기.
        # 체크포인트는 AF_STATE_FIT_EVERY(500)마다 AF_STATE_FIT_CK 에 쓰고 AF_STATE_FIT_R2 로 바로 올린다; 있으면(로컬 또는 R2) 이어 돈다.
        # AF_STATE_FIT_EVAL(1000)마다 궤적별로 정답 시작에서 굴려 위치 오차·증분 포텐셜을 TB 에 남긴다.
        _NODE_FIT_DONE[0] = True
        import subprocess as _sp_
        from torch.utils.tensorboard import SummaryWriter as _SW
        _spec, _it, _B = os.environ["AF_STATE_FIT_MULTI"].rsplit(":", 2)
        _it, _B = int(_it), int(_B)
        _ckp = os.environ["AF_STATE_FIT_CK"]; _r2 = os.environ.get("AF_STATE_FIT_R2")
        _every = int(os.environ.get("AF_STATE_FIT_EVERY", "500")); _evev = int(os.environ.get("AF_STATE_FIT_EVAL", "1000"))
        _byt = {t_: d_ for t_, d_ in TR + held}
        _TJ = []
        for _q in _spec.split(","):
            _tg_, _pth = _q.split("=", 1)
            _dj = _byt[_tg_]
            _Dj = torch.load(_pth, map_location="cpu", weights_only=False)
            _gs = torch.arange(_dj["x"].shape[1], device=dev)
            with traj_scope(_dj):
                _xj, _vj = DM.init_state(_dj, t0, _gs, FRAME_DT, a.v0_traj, dev)
                _mj = MASS[_gs].clone(); _ej = float(EXT)
            _TJ.append(dict(tag=_tg_, d=_dj, gsel=_gs, x0=_xj, v0=_vj, F0=take(traj_F(_dj)[t0], _gs).float(),
                            P=_Dj["pred"].float(), V=_Dj["v"].float(), F=_Dj["F"].float(), m=_mj, ext=_ej))
            print(f"    [다궤적 적합] {_tg_}: 덤프 {_pth} 프레임 {_Dj['pred'].shape[0]}", flush=True)

        def _st(j, i):
            T_ = _TJ[j]
            if i == 0:
                return T_["x0"], T_["v0"], T_["F0"]
            return T_["P"][i - 1].to(dev), T_["V"][i - 1].to(dev), T_["F"][i - 1].to(dev).reshape(-1, 3, 3)
        _pairs = [(j, i) for j in range(len(_TJ)) for i in range(_TJ[j]["P"].shape[0])]
        _opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=a.lr)
        _gen = torch.Generator().manual_seed(a.seed)
        _k0 = 0
        if os.path.exists(_ckp) or (_r2 and _sp_.run(["rclone", "copyto", _r2, _ckp], capture_output=True).returncode == 0
                                    and os.path.exists(_ckp)):
            _c = torch.load(_ckp, map_location=dev, weights_only=False)
            net.load_state_dict(_c["net"]); _opt.load_state_dict(_c["opt"]); _gen.set_state(_c["gen"].cpu())
            _k0 = int(_c["k"])
            print(f"    [다궤적 적합] 이어 돈다: {_ckp} 반복 {_k0}", flush=True)
        else:
            with torch.no_grad():
                _sf = []
                for (j, i) in _pairs:
                    with traj_scope(_TJ[j]["d"]):
                        xs, vs, _ = _st(j, i)
                        _sf.append(node_feats(_TJ[j]["d"], t0 + i, _TJ[j]["gsel"], xs, vs)[0])
                net.set_input_stats(torch.cat(_sf, 0)); del _sf
            print(f"    [다궤적 적합] 새 망, 입력 표준화 = {len(_TJ)} 궤적 {len(_pairs)} 상태", flush=True)
        _tb = _SW(os.environ["AF_STATE_FIT_TB"], purge_step=_k0 if _k0 > 0 else None)
        _hs = FRAME_DT

        def _consts(T_):
            cfg_ = T_["d"]["cfg"]
            return (cfg_, T_["m"], _pvol(T_["m"], cfg_), torch.tensor(cfg_["g"], device=dev, dtype=torch.float32),
                    int(cfg_["n_grid"]), float(cfg_.get("grid_lim", 2.0)), float(T_["m"].sum()) * (T_["ext"] ** 2) / (_hs ** 2))

        def _anch(T_, xs):
            return xs[AIDX] if (torch.is_tensor(AIDX) and int(AIDX.max()) < xs.shape[0]) else take(T_["d"]["x"][t0], AIDX)

        def _save(k):
            torch.save({"net": net.state_dict(), "opt": _opt.state_dict(), "gen": _gen.get_state(), "k": k,
                        "n_feat": n_feat, "args": vars(a), "step": k, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                        "net_ext0": _EXT0, "frame_dt": FRAME_DT}, _ckp + ".tmp")
            os.replace(_ckp + ".tmp", _ckp)
            if _r2:
                _r = _sp_.run(["rclone", "copyto", _ckp, _r2], capture_output=True, text=True)
                print(f"    [다궤적 적합] 체크포인트 {k} -> {_r2} ({'성공' if _r.returncode == 0 else '실패 ' + _r.stderr[-200:]})", flush=True)

        def _eval_roll(T_):
            net.eval()
            cfg_, m_, vol_, gv_, ng_, gl_, nrm_ = _consts(T_)
            xc, vc, Fc = T_["x0"].clone(), T_["v0"].clone(), T_["F0"].clone()
            pc = _anch(T_, xc)
            errs, Es = [], []
            with traj_scope(T_["d"]):
                for i in range(min(L, T_["d"]["x"].shape[0] - t0 - 1)):
                    with torch.enable_grad(), dt_scope(_hs):
                        x2, p2, v2, _, _, _, _, _, _, _jf = step_once(T_["d"], t0 + i, T_["gsel"], pc, xc, vc, need_J=False)
                        _E, _dl, _Ftr, _ = ip_of(xc, x2 - xc, vc, Fc, m_, vol_, cfg_, _hs, ng_, gl_, g=gv_, norm=nrm_, jac=_jf)
                    _DET_LAST[0] = None
                    with torch.no_grad():
                        Fc = phys_resid.plastic_step(_Ftr, _dl).detach()
                    xc, vc, pc = x2.detach(), v2.detach(), p2.detach()
                    gt = take(T_["d"]["x"][t0 + i + 1], T_["gsel"])
                    errs.append(float((xc - gt).norm(dim=-1).mean()) / T_["ext"]); Es.append(float(_E))
            net.train()
            return errs, Es

        _tw = time.time(); _win = []
        for _k in range(_k0, _it):
            _idx = torch.randint(0, len(_pairs), (_B,), generator=_gen).tolist()
            _opt.zero_grad(set_to_none=True)
            _Ls, _Es, _Ds, _Rs, _Os, _Cs = [], [], [], [], [], []
            if _k % 100 == 0 and os.environ.get("AF_OV_MIN_MULTI"):
                # 궤적별 상태 최솟값 (태그=json,...) -- 그 궤적 상태 전부가 다 나온 파일만 읽는다
                for _q in os.environ["AF_OV_MIN_MULTI"].split(","):
                    _tq, _pq = _q.split("=", 1)
                    if _tq not in _MINSM and os.path.exists(_pq):
                        _ml = json.load(open(_pq))
                        _nq = [T_["P"].shape[0] for T_ in _TJ if T_["tag"] == _tq]
                        if _nq and len(_ml) >= _nq[0]:
                            _MINSM[_tq] = {r["state"]: r["E_min"] for r in _ml}
                            print(f"    [다궤적 적합] {_tq} 최솟값 {len(_ml)} 상태 읽음 -> 초과% 기록 시작", flush=True)
            for _b, _pi in enumerate(_idx):
                j, i = _pairs[_pi]; T_ = _TJ[j]
                cfg_, m_, vol_, gv_, ng_, gl_, nrm_ = _consts(T_)
                with traj_scope(T_["d"]):
                    xs, vs, Fs = _st(j, i)
                    with torch.enable_grad(), dt_scope(_hs):
                        DM.PROBE[0] = "want" if _b == 0 else None
                        x2, _, _, _, _, _, _, _, _, _jf = step_once(T_["d"], t0 + i, T_["gsel"], _anch(T_, xs), xs, vs, need_J=False)
                        DM.PROBE[0] = None
                        _em, _dpen = det_take()
                        _E, _, _, _ = ip_of(xs, x2 - xs, vs, Fs, m_, vol_, cfg_, _hs, ng_, gl_, g=gv_, norm=nrm_,
                                            jac=_jf, elastic_mask=_em)
                        with torch.no_grad():
                            _Os.append(float((x2.detach() - T_["P"][i].to(dev)).norm(dim=-1).mean()) / T_["ext"])
                            _mq = _MINSM.get(T_["tag"], {})
                            if i in _mq:
                                _Eu = (float(_E) if (_em is None or bool(_em.all())) else
                                       float(ip_of(xs, x2.detach() - xs, vs, Fs, m_, vol_, cfg_, _hs, ng_, gl_,
                                                   g=gv_, norm=nrm_, jac=_jf.detach())[0]))
                                _Cs.append(100 * (_Eu / _mq[i] - 1))
                                _tb.add_scalar(f"다궤적적합_최솟값대비초과%/{T_['tag']}_{i:02d}", _Cs[-1], _k)
                        if _b == 0 and DM.PROBE[1]:
                            _pz = DM.PROBE[1][0]
                            _gU, = torch.autograd.grad(_E * nrm_, _pz, retain_graph=True)
                            _mN, = torch.autograd.grad((m_.unsqueeze(-1) * x2).sum(), _pz, retain_graph=True)
                            _mN = _mN[:, 0]
                            _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                            _Rs.append(float(_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hs ** 2) / _mN[_sup].mean() / T_["ext"]))
                        DM.PROBE[1] = None
                        if os.environ.get("AF_STATE_FIT_LOSS", "ip") == "l2":
                            # L2: 출력만 최적화 궤적의 다음 상태 위치와의 평균 제곱 거리 / EXT² (det 벌점 그대로)
                            _l2 = ((x2 - T_["P"][i].to(dev)) ** 2).sum(-1).mean() / (T_["ext"] ** 2)
                            _loss = (_l2 + _dpen) / _B
                        else:
                            _loss = (a.phys_w * _E + _dpen) / _B
                        if torch.isfinite(_loss):
                            _loss.backward()
                _Ls.append(float(_loss) * _B); _Es.append(float(_E)); _Ds.append(float(_dpen))
                _tb.add_scalar(f"다궤적적합_궤적/{T_['tag']}", float(_E), _k)
                _tb.add_scalar(f"다궤적적합_출력오차%/{T_['tag']}", 100 * _Os[-1], _k)
                _tb.add_scalar(f"다궤적적합_프레임/{T_['tag']}_{i:02d}", float(_E), _k)
                _tb.add_scalar(f"다궤적적합_출력오차%_프레임/{T_['tag']}_{i:02d}", 100 * _Os[-1], _k)
            _gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            _opt.step()
            _win.append(np.mean(_Ls))
            _tb.add_scalar("다궤적적합/손실", float(np.mean(_Ls)), _k)
            _tb.add_scalar("다궤적적합/증분포텐셜", float(np.mean(_Es)), _k)
            _tb.add_scalar("다궤적적합/출력오차%_평균", 100 * float(np.mean(_Os)), _k)
            if _Cs:
                _tb.add_scalar("다궤적적합/최솟값대비초과%_중앙", float(np.median(_Cs)), _k)
            _tb.add_scalar("다궤적적합/det벌점", float(np.mean(_Ds)), _k)
            _tb.add_scalar("다궤적적합/기울기노름", _gn, _k)
            if _Rs:
                _tb.add_scalar("다궤적적합/출력잔차", _Rs[0], _k)
            if _DET_BAD:
                _tb.add_scalar("다궤적적합/det무효비율", _DET_BAD[-1], _k)
            if _k % 100 == 0:
                print(f"    [다궤적 적합] {_k + 1}/{_it}  최근 손실 {np.mean(_win[-100:]):.4e}  기울기 {_gn:.2e}  "
                      f"{time.time() - _tw:.0f} s", flush=True)
            if (_k + 1) % _evev == 0 or _k == _it - 1:
                for T_ in _TJ:
                    _er, _ee = _eval_roll(T_)
                    _tb.add_scalar(f"다궤적적합_롤아웃_위치오차%/{T_['tag']}", 100 * float(np.mean(_er)), _k)
                    _tb.add_scalar(f"다궤적적합_롤아웃_증분포텐셜/{T_['tag']}", float(np.mean(_ee)), _k)
                    print(f"    [다궤적 롤아웃] {_k + 1} {T_['tag']}: 위치 오차 평균 {100 * np.mean(_er):.3f}%  "
                          f"증분포텐셜 평균 {np.mean(_ee):.3e}", flush=True)
                _tb.flush()
            if (_k + 1) % _every == 0 or _k == _it - 1:
                _save(_k + 1)
        _tb.close()
        _DPN_LIVE[0] = None
        net.requires_grad_(False)
    if os.environ.get("AF_STATE_FIT") and not _NODE_FIT_DONE[0]:
        # 출력만 최적화 궤적의 상태로 망 학습 (사용자 10-17): AF_STATE_FIT=덤프:반복:배치.
        # 상태 집합 = 프레임 0(정답 시작) + 덤프의 프레임 1..L-1 (위치·속도·F). 매 반복 배치만큼 프레임을 무작위로 뽑아
        # 그 상태에서 망 한 스텝(dt = 프레임)의 증분 포텐셜 + det 벌점(학습과 같은 손실: phys_w·E + det)을 평균해 Adam.
        # 체크포인트(망·최적화기·반복·난수)를 AF_STATE_FIT_CK 에 500 반복마다 쓰고 AF_STATE_FIT_R2 로 바로 올린다. 있으면 이어 돈다.
        _NODE_FIT_DONE[0] = True
        import subprocess as _sp_
        from torch.utils.tensorboard import SummaryWriter as _SW
        _ap = os.environ["AF_STATE_FIT"].split(":")
        _D = torch.load(_ap[0], map_location="cpu", weights_only=False)
        _it, _B = int(_ap[1]), int(_ap[2])
        _ckp = os.environ["AF_STATE_FIT_CK"]; _r2 = os.environ.get("AF_STATE_FIT_R2")
        _every = int(os.environ.get("AF_STATE_FIT_EVERY", "500"))
        _P, _V, _FF = _D["pred"].float(), _D["v"].float(), _D["F"].float()
        _F0 = take(traj_F(d)[t0], gsel).float()
        _nf = _P.shape[0]                                  # 상태 0..nf-1 (마지막 덤프 프레임은 다음 정답이 없어도 되지만 맞춰 둔다)
        _x0, _v0 = x.clone(), v.clone()

        def _state(i):
            if i == 0:
                return _x0, _v0, _F0
            return _P[i - 1].to(dev), _V[i - 1].to(dev), _FF[i - 1].to(dev).reshape(-1, 3, 3)
        _opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=a.lr)
        _gen = torch.Generator().manual_seed(a.seed)
        _k0 = 0
        if os.path.exists(_ckp) or (_r2 and _sp_.run(["rclone", "copyto", _r2, _ckp], capture_output=True).returncode == 0
                                    and os.path.exists(_ckp)):
            _c = torch.load(_ckp, map_location=dev, weights_only=False)
            net.load_state_dict(_c["net"]); _opt.load_state_dict(_c["opt"]); _gen.set_state(_c["gen"].cpu())
            _k0 = int(_c["k"])
            print(f"    [상태 적합] 이어 돈다: {_ckp} 반복 {_k0}", flush=True)
        else:
            with torch.no_grad():
                _sf = [node_feats(d, t0 + _i, gsel, *_state(_i)[:2])[0] for _i in range(_nf)]
                net.set_input_stats(torch.cat(_sf, 0))
            print(f"    [상태 적합] 새 망, 입력 표준화 = 출력만 최적화 {_nf} 상태 (sd 최소 {float(net.in_sd.min()):.2e} "
                  f"최대 {float(net.in_sd.max()):.2e})", flush=True)
        _tb = _SW(os.environ["AF_STATE_FIT_TB"], purge_step=_k0 if _k0 > 0 else None)
        _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
        _SAVE0 = (_it == 0 and _k0 == 0)                     # 반복 0: 학습 전 망(같은 입력 표준화)을 체크포인트로 남긴다 (비교용)
        _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
        _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
        _hs = FRAME_DT
        _nrm = float(_m.sum()) * (EXT ** 2) / (_hs ** 2)

        def _save(k):
            torch.save({"net": net.state_dict(), "opt": _opt.state_dict(), "gen": _gen.get_state(), "k": k,
                        "n_feat": n_feat, "args": vars(a), "step": k, "aidx": AIDX.cpu(), "H": H, "EXT": EXT,
                        "net_ext0": _EXT0, "frame_dt": FRAME_DT}, _ckp + ".tmp")
            os.replace(_ckp + ".tmp", _ckp)
            if _r2:
                _r = _sp_.run(["rclone", "copyto", _ckp, _r2], capture_output=True, text=True)
                print(f"    [상태 적합] 체크포인트 {k} -> {_r2} ({'성공' if _r.returncode == 0 else '실패 ' + _r.stderr[-200:]})", flush=True)

        def _eval_roll():
            # 같은 망으로 정답 시작 상태에서 굴린다: 프레임별 위치 오차(정답 대비, EXT 로 나눔)와 증분 포텐셜.
            # 파라미터 기울기는 끈다 -- 켜 두면 enable_grad 안에서 층마다 활성값을 들고 있어 24GB 카드에서 OOM (10-17)
            _rq = [q.requires_grad for q in net.parameters()]
            net.requires_grad_(False)
            try:
                return _eval_roll_body()
            finally:
                for q, r in zip(net.parameters(), _rq):
                    q.requires_grad_(r)

        def _eval_roll_body():
            net.eval()
            xc, vc, Fc = _x0.clone(), _v0.clone(), _F0.clone()
            pc = p
            errs, Es = [], []
            for i in range(min(L, d["x"].shape[0] - t0 - 1)):
                with torch.enable_grad(), dt_scope(_hs):
                    x2, p2, v2, _, _, _, _, _, _, _jf = step_once(d, t0 + i, gsel, pc, xc, vc, need_J=False)
                    _fmi = (free_mask(d, x2.shape[0], dev, gsel, xc, t0 + i) if a.control else None)
                    _E, _dl, _Ftr, _ = ip_of(xc, x2 - xc, vc, Fc, _m, _vol, _cfg, _hs, _ng, _gl,
                                             g=_gv, norm=_nrm, free=_fmi, jac=_jf)
                _DET_LAST[0] = None
                with torch.no_grad():
                    Fc = phys_resid.plastic_step(_Ftr, _dl).detach()
                xc, vc, pc = x2.detach(), _pg_vdamp(v2.detach(), d, _hs), p2.detach()
                gt = take(d["x"][t0 + i + 1], gsel)
                errs.append(float((xc - gt).norm(dim=-1).mean()) / EXT); Es.append(float(_E))
            net.train()
            return errs, Es

        if _SAVE0:
            _save(0)
        _tw = time.time(); _win = []
        for _k in range(_k0, _it):
            _KMAX = int(os.environ.get("AF_STATE_FIT_KMAX", "1"))
            if _KMAX > 1:
                # AF_STATE_FIT_KMAX: k ~ U{1..KMAX}, 시작 상태 i ~ U{0..nf-k}. 망으로 k-1 스텝 굴린 뒤(기울기 없음)
                # k 번째 스텝만 손실로 학습한다. L2 목표 = 출력만 최적화 궤적의 i+k 번째 상태
                _ks = torch.randint(1, _KMAX + 1, (_B,), generator=_gen).tolist()
                _ik = [(int(torch.randint(0, _nf - k_ + 1, (1,), generator=_gen)), k_) for k_ in _ks]
            else:
                _ik = [(i_, 1) for i_ in torch.randint(0, _nf, (_B,), generator=_gen).tolist()]
            _idx = [i_ + k_ - 1 for i_, k_ in _ik]
            _opt.zero_grad(set_to_none=True)
            _Ls, _Es, _Ds, _Rs, _Os, _Cs = [], [], [], [], [], []
            if not _MINS and _k % 100 == 0 and os.environ.get("AF_OV_MIN") and os.path.exists(os.environ["AF_OV_MIN"]):
                _ml = json.load(open(os.environ["AF_OV_MIN"]))
                if len(_ml) >= _nf:                         # 상태 전부의 최솟값이 다 나왔을 때만 쓴다
                    _MINS.update({r["state"]: r["E_min"] for r in _ml})
                    print(f"    [상태 적합] 최솟값 {len(_MINS)} 상태 읽음 -> 초과% 기록 시작", flush=True)
            for _j, (_i0, _kk) in enumerate(_ik):
                xs, vs, Fs = _state(_i0)
                if _kk > 1:
                    _rq = [q.requires_grad for q in net.parameters()]
                    net.requires_grad_(False)
                    for _s in range(_kk - 1):
                        with torch.enable_grad(), dt_scope(_hs):
                            _xa, _, _va, _, _, _, _, _, _, _ja = step_once(d, t0 + _i0 + _s, gsel, p, xs, vs, need_J=False)
                            _Ea, _dla, _Fta, _ = ip_of(xs, _xa - xs, vs, Fs, _m, _vol, _cfg, _hs, _ng, _gl,
                                                       g=_gv, norm=_nrm, jac=_ja)
                        _DET_LAST[0] = None
                        with torch.no_grad():
                            Fs = phys_resid.plastic_step(_Fta, _dla).detach()
                        xs, vs = _xa.detach(), _va.detach()
                    for q, r in zip(net.parameters(), _rq):
                        q.requires_grad_(r)
                _i = _i0 + _kk - 1
                with torch.enable_grad(), dt_scope(_hs):
                    DM.PROBE[0] = "want" if _j == 0 else None
                    x2, _, _, _, _, _, _, _, _, _jf = step_once(d, t0 + _i, gsel, p, xs, vs, need_J=False)
                    DM.PROBE[0] = None
                    _fmi = (free_mask(d, x2.shape[0], dev, gsel, xs, t0 + _i) if a.control else None)
                    if _BC_LAST[0] is not None:
                        _am = _BC_LAST[0][1].reshape(-1) > 0.5
                        _fmi = (~_am) if _fmi is None else (_fmi & ~_am)
                    _em, _dpen = det_take()
                    _E, _, _, _pt = ip_of(xs, x2 - xs, vs, Fs, _m, _vol, _cfg, _hs, _ng, _gl,
                                          g=_gv, norm=_nrm, free=_fmi, jac=_jf, elastic_mask=_em)
                    # 출력 오차: 출력만 최적화 궤적의 다음 상태와의 평균 위치 차 / EXT.  초과%: det 마스크 없는 같은 식의 E 로
                    with torch.no_grad():
                        _Os.append(float((x2.detach() - _P[_i].to(dev)).norm(dim=-1).mean()) / EXT)
                        if _kk == 1 and _i in _MINS:           # 최솟값은 출력만 최적화 상태에서 잰 것 -- 망이 굴린 상태(k>1)에는 안 맞는다
                            _Eu = (float(_E) if (_em is None or bool(_em.all())) else
                                   float(ip_of(xs, x2.detach() - xs, vs, Fs, _m, _vol, _cfg, _hs, _ng, _gl,
                                               g=_gv, norm=_nrm, free=_fmi, jac=_jf.detach())[0]))
                            _Cs.append(100 * (_Eu / _MINS[_i] - 1))
                        else:
                            _Cs.append(None)
                    if _j == 0 and DM.PROBE[1]:
                        _pz = DM.PROBE[1][0]
                        _gU, = torch.autograd.grad(_E * _nrm, _pz, retain_graph=True)
                        _mN, = torch.autograd.grad((_m.unsqueeze(-1) * x2).sum(), _pz, retain_graph=True)
                        _mN = _mN[:, 0]
                        _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                        _Rs.append(float(_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hs ** 2) / _mN[_sup].mean() / EXT))
                    DM.PROBE[1] = None
                    if os.environ.get("AF_STATE_FIT_LOSS", "ip") == "l2":
                        # L2: 출력만 최적화 궤적의 다음 상태 위치와의 평균 제곱 거리 / EXT² + det 벌점 (증분 포텐셜은 기록만)
                        _loss = (((x2 - _P[_i].to(dev)) ** 2).sum(-1).mean() / (EXT ** 2) + _dpen) / _B
                    else:
                        _loss = (a.phys_w * _E + _dpen) / _B
                    if torch.isfinite(_loss):
                        _loss.backward()
                _Ls.append(float(_loss) * _B); _Es.append(float(_E)); _Ds.append(float(_dpen))
            _gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            _opt.step()
            _win.append(np.mean(_Ls))
            _tb.add_scalar("상태적합/손실", float(np.mean(_Ls)), _k)
            _tb.add_scalar("상태적합/증분포텐셜", float(np.mean(_Es)), _k)
            _tb.add_scalar("상태적합/det벌점", float(np.mean(_Ds)), _k)
            _tb.add_scalar("상태적합/기울기노름", _gn, _k)
            if _Rs:
                _tb.add_scalar("상태적합/출력잔차", _Rs[0], _k)
            for _i, _e, _o, _c in zip(_idx, _Es, _Os, _Cs):
                _tb.add_scalar(f"상태적합_프레임/{_i:02d}", _e, _k)
                _tb.add_scalar(f"상태적합_출력오차%/{_i:02d}", 100 * _o, _k)
                if _c is not None:
                    _tb.add_scalar(f"상태적합_최솟값대비초과%/{_i:02d}", _c, _k)
            _tb.add_scalar("상태적합/출력오차%_평균", 100 * float(np.mean(_Os)), _k)
            if _KMAX > 1:
                _tb.add_scalar("상태적합/롤아웃스텝_평균", float(np.mean([k_ for _, k_ in _ik])), _k)
            if any(c_ is not None for c_ in _Cs):
                _tb.add_scalar("상태적합/최솟값대비초과%_중앙", float(np.median([c_ for c_ in _Cs if c_ is not None])), _k)
            if _DET_BAD:
                _tb.add_scalar("상태적합/det무효비율", _DET_BAD[-1], _k)
            if _k % 100 == 0:
                print(f"    [상태 적합] {_k + 1}/{_it}  최근 손실 {np.mean(_win[-100:]):.4e}  기울기 {_gn:.2e}  "
                      f"{time.time() - _tw:.0f} s", flush=True)
            if (_k + 1) % _every == 0 or _k == _it - 1:
                _er, _ee = _eval_roll()
                _tb.add_scalar("상태적합_롤아웃/위치오차_평균%", 100 * float(np.mean(_er)), _k)
                _tb.add_scalar("상태적합_롤아웃/증분포텐셜_평균", float(np.mean(_ee)), _k)
                _tb.add_scalar("상태적합_롤아웃/위치오차_마지막%", 100 * _er[-1], _k)
                print(f"    [상태 적합 롤아웃] {_k + 1}: 위치 오차 평균 {100 * np.mean(_er):.3f}%  증분포텐셜 평균 {np.mean(_ee):.3e}", flush=True)
                _tb.flush()
                _save(_k + 1)
        _tb.close()
        _DPN_LIVE[0] = None
        net.requires_grad_(False)
    if os.environ.get("AF_NODE_FIT") or os.environ.get("AF_ROLL_NOGRAD"):
        # 롤아웃은 고정 망 -- 파라미터 기울기를 끄지 않으면 enable_grad 안에서 층마다 활성값을 들고 있어 큰 망이 OOM
        net.requires_grad_(False)
        _DPN_LIVE[0] = None
    x0e, p0e = x.clone(), p.clone()
    # 프레임 사이에 이어지는 학습 쪽 상태 (앵커·손상·F_e·출력만 최적화의 F)
    _S = dict(p=p, dmg=None, idx=None,
              fe=(take(traj_F(d)[t0], gsel).float() if a.fe_state else None),
              F_ov=(take(traj_F(d)[t0], gsel).float() if a.ov_roll > 0 else None),
              F_ip=take(traj_F(d)[t0], gsel).float())
    cds, ems, cds_s, ems_s = [], [], [], []
    _RO_CTRL = []
    _NODE_CAP[0] = _ROLLDUMP is not None
    _FCUM[0] = (take(traj_F(d)[t0], gsel).float().clone()
                if (_ROLLDUMP is not None and os.environ.get("AF_ROLL_DUMP_F")) else None)
    if phys_resid.mat_name(d["cfg"]) == "watermelon":
        # CD-MPM 경화 상태(logJp)는 롤아웃마다 초기값 -0.04 에서 다시 시작한다
        phys_resid.cdmpm_reset(x.shape[0], x.device, x.dtype)
    _RO_T0 = time.time()
    # AF_ROLL_LGV=model|inertial: 롤아웃의 매 서브스텝을 풀 학습(--pool_adv langevin)과 **같은 _lgv_step** 으로 전진한다
    #   (시작점 model = 모델 출력과 그 점의 기울기, inertial = 관성 예측 노드 변위와 그 점의 기울기). 온도·rho·백트래킹은
    #   학습 플래그 그대로, 걸음 eta 는 AF_ROLL_LGV_ETA > 체크포인트의 pool_lgv_eta > --lgv_eta0 순. eta 는 학습처럼 걸음마다 적응한다.
    _RLGV = os.environ.get("AF_ROLL_LGV") or None
    if _RLGV:
        if _RLGV not in ("model", "inertial"):
            raise SystemExit(f"AF_ROLL_LGV 는 model 또는 inertial ({_RLGV})")
        if a.ov_roll > 0 or _ROLL_IP is not None:
            raise SystemExit("AF_ROLL_LGV 는 --ov_roll·AF_ROLL_IP 와 함께 쓰지 않는다")
        _ck_eta = (_rng_ck or {}).get("pool_lgv_eta")
        if os.environ.get("AF_ROLL_LGV_ETA"):
            _PLGV["eta"], _src = float(os.environ["AF_ROLL_LGV_ETA"]), "AF_ROLL_LGV_ETA"
        elif _ck_eta is not None:
            _PLGV["eta"], _src = float(_ck_eta), "체크포인트"
        else:
            _PLGV["eta"], _src = float(a.lgv_eta0), "--lgv_eta0"
        _PLGV["n"] = 0
        _S["F_ip"] = take(traj_F(d)[t0], gsel).float()
        _LSC = dict(cfg=d["cfg"], mass=MASS[gsel], ext=EXT, _msum=float(MASS[gsel].sum()))
        _LACC = []
        print(f"[랑제빈 롤아웃] 매 서브스텝 = 증분 포텐셜 MALA 한 걸음 (시작점 {_RLGV}, rho {a.lgv_rho:g}, "
              f"eta {_PLGV['eta']:.3e} ({_src}), 백트래킹 {a.lgv_backtrack}, lam {a.lgv_lam:g}, 서브스텝 {EVAL_SUB})", flush=True)
    # AF_ROLL_RW=r[:eps[:seed]]: 증분 포텐셜·중력·관성을 전혀 쓰지 않는 **순수 랜덤 걸음** (사용자 10-19).
    #   프레임마다 한 걸음, 노드 출력(변위) 공간에서 Δ = r · G^{-1/2} z / ‖z‖ (z ~ N(0,I)) -> ‖Δ‖_G = r (Dikin 타원 안, 선 탐색 없음).
    #   G = (뒤집힘 장벽 −Σ log det J_c 의 블록 대각 계량: _lgv_metric 그대로, lam·I 포함)
    #     + (바닥 장벽 −Σ_i log d_i 헤시안의 블록 대각 상계: G_n += Σ_i λ_in n nᵀ / d_i²  -- 코시-슈바르츠로 정확한 헤시안을 위에서 누름)
    #     + eps · (M_n / 평균 M) · (뒤집힘 계량 대각 평균) · I   (노드 질량 항, eps 는 무차원)
    #   망은 부르지 않는다 (노드 출력 = Δ 그 자체, 기준 0). 걸음 뒤 d_i>0·det J>0 를 매 프레임 기록한다.
    _RRW = os.environ.get("AF_ROLL_RW") or None
    if _RRW:
        if _RLGV or a.ov_roll > 0 or _ROLL_IP is not None:
            raise SystemExit("AF_ROLL_RW 는 AF_ROLL_LGV·--ov_roll·AF_ROLL_IP 와 함께 쓰지 않는다")
        _rwq = _RRW.split(":")
        _RW = dict(r=float(_rwq[0]), eps=(float(_rwq[1]) if len(_rwq) > 1 and _rwq[1] else 1e-2),
                   seed=(int(_rwq[2]) if len(_rwq) > 2 and _rwq[2] else int(a.seed)), rec=[])
        _RW["gen"] = torch.Generator(device=dev).manual_seed(_RW["seed"])
        _RW["fl"] = [(torch.as_tensor(bc["point"], device=dev, dtype=torch.float64),
                      torch.as_tensor(bc["normal"], device=dev, dtype=torch.float64))
                     for bc in (d["cfg"].get("boundary_conditions") or []) if bc.get("type") == "surface_collider"]
        _RW["fl"] = [(q_, n_ / n_.norm()) for q_, n_ in _RW["fl"]]
        _RW["m"] = MASS[gsel]
        print(f"[랜덤 걸음 롤아웃] 프레임마다 한 걸음 ‖Δ‖_G = r {_RW['r']:g}, eps {_RW['eps']:g}, 시드 {_RW['seed']}, "
              f"바닥 {len(_RW['fl'])} 개, lam {a.lgv_lam:g} -- 증분 포텐셜·중력·관성 없음", flush=True)
        if not _RW["fl"]:
            print("[랜덤 걸음 롤아웃] 경고: 씬에 surface_collider 가 없다 -- 바닥 장벽 없이 뒤집힘 장벽만", flush=True)

    # AF_ROLL_PRW=기저:크기:r_s[:eps[:seed]] (사용자 10-19): 증분 포텐셜·중력·관성·망 없이 **매끄러운 함수 파라미터 θ** 로
    #   시작 상태(롤아웃 t0)를 변형한다. **프레임마다 독립 표본** (매 표본 θ=0 에서 다시 출발, 누적은 표본 안에서만).
    #   B: 노드 위치를 시작 입자 경계 상자로 [0,1]³ 정규화한 s 의 기저 -- fourier:K (축별 1, cos 2πks, sin 2πks, k<=K 텐서곱),
    #      rbf:n (n×n×n 격자 중심 가우시안, σ = 간격), affine (1, s-0.5). θ = vec Θ (기저 nb × 성분 3).
    #   변형: 덧셈 u = B Θ (기본) 또는 AF_ROLL_PRW_FLOW=1 이면 속도장 v_θ(y) = B(s(y)) Θ 를 t∈[0,1] RK4 (AF_ROLL_PRW_NST=16) 적분.
    #   표본 하나 = 장벽 매니폴드 위 측지선형 걷기: 방향 z ~ N(0,I) 하나를 뽑고 서브스텝마다 **현재 θ 에서** 계량 G(θ) 를 다시 만들어
    #     Δθ = r_s · G_R^{-1/2} z_R / ‖z_R‖  (‖Δθ‖_G = r_s),  R = 강체 성분(입자 변위의 평행이동 3·회전 3 최소제곱 성분)이 0 인 부분공간.
    #   G(θ): 바닥 −Σ_i log d_i(θ) 의 정확한 헤시안 Σ_i ∇d_i ∇d_iᵀ / d_i²  (입자 = 스킨으로 옮긴 실제 입자)
    #       + (덧셈만) 뒤집힘 Σ_c ‖J_c(θ)⁻¹ dJ_c‖_F²  (−Σ log det J_c 의 정확한 헤시안은 비대칭 J 에서 부정부호라 이 상계를 쓴다;
    #         ‖Δθ‖_G < 1 이면 각 셀 ‖J⁻¹dJ‖_2 < 1 -> det > 0, 각 입자 |Δd_i| < d_i -> d > 0 이 보장된다)
    #       + eps · (θ=0 계량의 대각 평균) · I.
    #   목표: 사면체별 주신축 σ 의 max|σ−1| 평균 >= AF_ROLL_PRW_STRAIN(0.2) 그리고 최대 >= AF_ROLL_PRW_SMAX(0.5) 가 될 때까지 걷는다
    #     (상한 AF_ROLL_PRW_KMAX=500). 한 걸음에 평균이 목표의 2 배를 넘으면 그 걸음을 반씩 줄인다 (크기 맞춤).
    #     서브스텝 후보가 바닥(모든 입자 d<=0)·뒤집힘(모든 사면체 det<=0)을 어기면 반씩 줄이고 (백트래킹) 그 서브스텝 번호를 기록한다.
    _RPRW = os.environ.get("AF_ROLL_PRW") or None
    if _RPRW:
        if _RRW or _RLGV or a.ov_roll > 0 or _ROLL_IP is not None:
            raise SystemExit("AF_ROLL_PRW 는 AF_ROLL_RW·AF_ROLL_LGV·--ov_roll·AF_ROLL_IP 와 함께 쓰지 않는다")
        _pq = _RPRW.split(":")
        _PR = dict(kind=_pq[0], size=(int(_pq[1]) if len(_pq) > 1 and _pq[1] else 1), r=float(_pq[2]),
                   eps=(float(_pq[3]) if len(_pq) > 3 and _pq[3] else 1e-4),
                   seed=(int(_pq[4]) if len(_pq) > 4 and _pq[4] else int(a.seed)),
                   flow=bool(os.environ.get("AF_ROLL_PRW_FLOW")), nst=int(os.environ.get("AF_ROLL_PRW_NST", "16")),
                   s_mean=float(os.environ.get("AF_ROLL_PRW_STRAIN", "0.2")), s_max=float(os.environ.get("AF_ROLL_PRW_SMAX", "0.5")),
                   kmax=int(os.environ.get("AF_ROLL_PRW_KMAX", "500")),
                   norigid=os.environ.get("AF_ROLL_PRW_RIGID", "remove") == "remove", rec=[], st=None, info={},
                   # AF_ROLL_PRW_PUSH=w (사용자 10-19 "바닥하고도 부딪치게"): 바닥 법선 방향 평행이동만 다시 허용하고, 걸음 방향에
                   #   아래쪽(평균 법선 변위를 줄이는 계량 최급강하, 백색화 좌표) 을 비중 w 로 섞는다. 회전·수평 이동은 계속 뺀다.
                   #   목표에 바닥 띠(시작 d < AF_ROLL_PRW_BAND) 입자 중 d < d̂ 비율 >= AF_ROLL_PRW_CONTACT 를 더한다.
                   push=float(os.environ.get("AF_ROLL_PRW_PUSH", "0")), band=float(os.environ.get("AF_ROLL_PRW_BAND", "0.05")),
                   contact=float(os.environ.get("AF_ROLL_PRW_CONTACT", "0.3")))
        if _PR["kind"] not in ("affine", "fourier", "rbf"):
            raise SystemExit(f"AF_ROLL_PRW 기저는 affine|fourier|rbf ({_RPRW})")
        _PR["gen"] = torch.Generator(device=dev).manual_seed(_PR["seed"])
        _PR["fl"] = [(torch.as_tensor(bc["point"], device=dev, dtype=torch.float64),
                      torch.as_tensor(bc["normal"], device=dev, dtype=torch.float64))
                     for bc in (d["cfg"].get("boundary_conditions") or []) if bc.get("type") == "surface_collider"]
        _PR["fl"] = [(q_, n_ / n_.norm()) for q_, n_ in _PR["fl"]]
        print(f"[파라미터 랜덤 표본 롤아웃] 기저 {_PR['kind']}:{_PR['size']}  {'흐름 RK4 ' + str(_PR['nst']) if _PR['flow'] else '덧셈 u=Bθ'}  "
              f"서브스텝 ‖Δθ‖_G = r_s {_PR['r']:g}  eps {_PR['eps']:g}  시드 {_PR['seed']}  강체 성분 {'제거' if _PR['norigid'] else '유지'}  "
              f"목표 신축 평균 {_PR['s_mean']:g} 최대 {_PR['s_max']:g} (서브스텝 상한 {_PR['kmax']})  "
              f"{('아래로 밀기 비중 ' + format(_PR['push'], 'g') + ' 바닥 띠 d<' + format(_PR['band'], 'g') + ' 접촉 목표 ' + format(_PR['contact'], 'g')) if _PR['push'] > 0 else ''}  바닥 {len(_PR['fl'])} 개 -- "
              f"프레임마다 시작 상태에서 독립 표본, 증분 포텐셜·중력·관성·망 없음", flush=True)

    def _prw_basis(s):
        # s [M,3] float64 (경계 상자 정규화) -> [M, nb]
        k_, n_ = _PR["kind"], _PR["size"]
        if k_ == "affine":
            return torch.cat([torch.ones_like(s[:, :1]), s - 0.5], 1)
        if k_ == "fourier":
            fs = []
            for ax in range(3):
                c_ = [torch.ones_like(s[:, ax])]
                for kk in range(1, n_ + 1):
                    c_ += [torch.cos(2 * math.pi * kk * s[:, ax]), torch.sin(2 * math.pi * kk * s[:, ax])]
                fs.append(torch.stack(c_, 1))                           # [M, 2K+1]
            return torch.einsum("ma,mb,mc->mabc", fs[0], fs[1], fs[2]).reshape(s.shape[0], -1)
        g1 = torch.linspace(0.0, 1.0, n_, device=s.device, dtype=s.dtype)
        cen = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        sig = 1.0 / max(n_ - 1, 1)
        return torch.exp(-((s.unsqueeze(1) - cen.unsqueeze(0)) ** 2).sum(-1) / (2 * sig * sig))

    def _rw_floor_d(xq):
        # [N] 모든 바닥 중 최소 부호거리 (float64)
        ds = [((xq.double() - q_) * n_).sum(-1) for q_, n_ in _RW["fl"]]
        return torch.stack(ds, 0).min(0).values if ds else None

    def _rw_eval(i, p, x, v, ens, U):
        # 노드 출력 U 를 그대로 쓰는 한 스텝 (망 호출 없음) -> (x2, v2, p2, Jd)
        _sv = (_DP_FAST[0], _NF_HOLD[0])
        _DP_FAST[0] = lambda _n, _dt, _dev: (U,)
        _NF_HOLD[0] = ens
        try:
            with torch.enable_grad(), dt_scope(FRAME_DT):
                _x2, _p2, _v2, _, _, _, _, _, _, _Jd = step_once(d, t0 + i, gsel, p, x, v, need_J=False)
            _DET_LAST[0] = None
        finally:
            _DP_FAST[0], _NF_HOLD[0] = _sv
        return _x2.detach(), _v2.detach(), (_p2.detach() if torch.is_tensor(_p2) else _p2), _Jd.detach()

    def _rw_step(i, p, x, v):
        _SIM_T[0] = i * FRAME_DT
        _no = int(getattr(net, "n_out", 3))
        # 1) 격자·노드 구성만 만든다 (출력 0 -- 망 안 부름)
        _NF_HOLD[0] = "want"
        _sv = _DP_FAST[0]
        _DP_FAST[0] = lambda _n, _dt, _dev: (torch.zeros(_n, _no, device=_dev, dtype=x.dtype),)
        try:
            with torch.enable_grad(), dt_scope(FRAME_DT):
                step_once(d, t0 + i, gsel, p, x, v, need_J=False)
        finally:
            _DP_FAST[0] = _sv
        _ens, _NF_HOLD[0] = _NF_HOLD[0], None
        _DET_LAST[0] = None
        e0 = _ens[0]
        rows, lam_ = e0[3], e0[4]
        Mn = e0[1].shape[0]
        nm = _NOD_M[0]
        free = torch.ones(Mn, dtype=torch.bool, device=dev) if nm is None else ~nm
        with torch.no_grad():
            # 2) 계량: 뒤집힘 (J = I, 시작점 출력 0) + 바닥 + 질량
            _I = torch.eye(3, device=dev, dtype=x.dtype).expand(x.shape[0], 3, 3)
            Gd, _, _ = _lgv_metric(e0, _I, Mn, free)
            G = Gd.double()
            dg = Gd.diagonal(dim1=-2, dim2=-1).mean(-1)[free].mean().double()
            d0 = _rw_floor_d(x)
            nbad0 = 0
            if d0 is not None:
                nbad0 = int((d0 <= 0).sum())
                if nbad0 and i == 0:
                    print(f"[랜덤 걸음] 경고: 시작 상태가 바닥 장벽 밖 -- d<=0 입자 {nbad0} 개, d 최소 {float(d0.min()):.3e} "
                          f"(계량에서는 |d| 를 EXT·1e-6 아래로 안 내린다)", flush=True)
                lw = lam_.double().clamp_min(0.0)
                for q_, n_ in _RW["fl"]:
                    dq = ((x.double() - q_) * n_).sum(-1).abs().clamp_min(1e-6 * EXT)
                    c_ = (lw / (dq * dq).unsqueeze(-1))                      # [N,4]
                    s_ = torch.zeros(Mn, device=dev, dtype=torch.float64).index_add_(0, rows.reshape(-1), c_.reshape(-1))
                    G = G + s_.view(-1, 1, 1) * torch.outer(n_, n_)
            Mnode = torch.zeros(Mn, device=dev, dtype=torch.float64).index_add_(
                0, rows.reshape(-1), (lam_.double().clamp_min(0.0) * _RW["m"].double().unsqueeze(-1)).reshape(-1))
            G = G + (_RW["eps"] * dg * Mnode / Mnode[free].mean().clamp_min(1e-30)).view(-1, 1, 1) \
                * torch.eye(3, device=dev, dtype=torch.float64)
            ev, Q = torch.linalg.eigh(G)
            ev = ev.clamp_min(1e-30)
            Ph = (Q * ev.rsqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
            # 3) 랜덤 방향, 계량 길이 r
            z = torch.randn(Mn, 3, device=dev, dtype=torch.float64, generator=_RW["gen"]) * free.unsqueeze(-1)
            Dl = _RW["r"] * (Ph @ z.unsqueeze(-1)).squeeze(-1) / z.norm().clamp_min(1e-30)
            gn = float((Dl.unsqueeze(-2) @ G @ Dl.unsqueeze(-1)).sum().sqrt())
            U = torch.zeros(Mn, _no, device=dev, dtype=x.dtype)
            U[:, :3] = Dl.to(x.dtype)
        x2, v2, p2, Jd = _rw_eval(i, p, x, v, _ens, U)
        with torch.no_grad():
            dj = torch.linalg.det(Jd.double())
            jbad = dj <= 0
            tid = SX.tet_id(e0[2][0], e0[2][1], e0[2][2], e0[7])
            ntet = int(torch.unique(tid).numel())
            ntb = int(torch.unique(tid[jbad]).numel()) if bool(jbad.any()) else 0
            d1 = _rw_floor_d(x2)
            nfb = int((d1 <= 0).sum()) if d1 is not None else 0
            du = (x2 - x).norm(dim=-1)
            dun = Dl.norm(dim=-1)[free]
            fdm = float("nan")
            if _FCUM[0] is not None:
                _FCUM[0] = (Jd.float() @ _FCUM[0]).detach()
                fdm = float(torch.linalg.det(_FCUM[0].double()).min())
        rec = dict(frame=i + 1, gnorm=gn, du_mean=100 * float(du.mean()) / EXT, du_max=100 * float(du.max()) / EXT,
                   dnode_mean=100 * float(dun.mean()) / EXT, dnode_max=100 * float(dun.max()) / EXT,
                   d_min_before=(float(d0.min()) if d0 is not None else None),
                   d_min_after=(float(d1.min()) if d1 is not None else None),
                   floor_bad=nfb, floor_bad_start=nbad0, detJ_min=float(dj.min()), detJ_bad_particles=int(jbad.sum()),
                   detJ_bad_tets=ntb, n_tets=ntet, detF_cum_min=fdm, n_nodes=Mn, n_free=int(free.sum()))
        _RW["rec"].append(rec)
        print(f"    [랜덤 걸음] 프레임 {i + 1}/{L}  ‖Δ‖_G {gn:.4f}  변위 입자 평균 {rec['du_mean']:.4f}% 최대 {rec['du_max']:.4f}% "
              f"(노드 평균 {rec['dnode_mean']:.4f}% 최대 {rec['dnode_max']:.4f}%) EXT  "
              f"d최소 {rec['d_min_before'] if rec['d_min_before'] is None else format(rec['d_min_before'], '.3e')}"
              f"->{rec['d_min_after'] if rec['d_min_after'] is None else format(rec['d_min_after'], '.3e')} 바닥위반 {nfb}  "
              f"detJ최소 {rec['detJ_min']:.4f} 뒤집힘 입자 {rec['detJ_bad_particles']} 사면체 {ntb}/{ntet}  "
              f"누적detF최소 {fdm:.4f}  누적 {time.time() - _RO_T0:.1f} s", flush=True)
        del _ens
        return x2, v2, p2

    def _prw_flow(Bfun, y0, th, nst=16):
        # 속도장 v_θ(y) = B(s(y)) Θ 를 t∈[0,1] 동안 RK4 nst 스텝 적분 -> 노드 변위 y(1) − y0  (θ 에 미분 가능)
        Th = th.view(-1, 3)
        h_ = 1.0 / nst
        y = y0
        for _ in range(nst):
            k1 = Bfun(y) @ Th
            k2 = Bfun(y + 0.5 * h_ * k1) @ Th
            k3 = Bfun(y + 0.5 * h_ * k2) @ Th
            k4 = Bfun(y + h_ * k3) @ Th
            y = y + (h_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y - y0

    def _prw_U(th):
        # 노드 변위 [Mn,3] float64
        st = _PR["st"]
        if _PR["flow"]:
            return _prw_flow(st["Bfun"], st["y0"], th, _PR["nst"])
        return st["Bm"] @ th.view(-1, 3)

    def _prw_Jn(th):
        # 노드 변위 야코비안 ∂U/∂θ [Mn,3,P] (흐름: 전진 모드 autograd 정확값, 덧셈: B ⊗ I)
        st = _PR["st"]
        P = st["P"]
        if not _PR["flow"]:
            return torch.einsum("ma,ij->miaj", st["Bm"], st["I3"]).reshape(st["Mn"], 3, P)
        _fz = lambda t_: _prw_flow(st["Bfun"], st["y0"], t_, _PR["nst"])
        _Ep = torch.eye(P, device=dev, dtype=torch.float64)
        return torch.cat([torch.func.vmap(lambda e_: torch.func.jvp(_fz, (th,), (e_,))[1])(_Ep[c_:c_ + 8])
                          for c_ in range(0, P, 8)], 0).permute(1, 2, 0).contiguous()

    def _prw_tetJ(U):
        # 모든 사면체(입자가 든 것) 의 J = I + ∇u [T,3,3] -- g2p_jac_pre 와 같은 식 (사면체마다 대표 입자)
        st = _PR["st"]
        Uc = U[st["rows"][st["rep"]]]
        dd = (Uc[:, 1:] - Uc[:, :-1]).transpose(1, 2)
        Dy = dd.gather(2, st["rank_r"].unsqueeze(1).expand(-1, 3, -1))
        return st["I3"] + Dy @ st["Ai"]

    def _prw_state(th, U=None):
        # θ 에서의 입자 위치·바닥 거리·사면체 J (전부 float64, 실제 스킨 식)
        st = _PR["st"]
        if U is None:
            U = _prw_U(th)
        xp = st["xd"] + (st["lamd"].unsqueeze(-1) * U[st["rows"]]).sum(1)
        ds = [((xp - q_) * n_).sum(-1) for q_, n_ in _PR["fl"]]
        Jt = _prw_tetJ(U)
        return U, xp, ds, Jt

    def _prw_strain(Jt):
        sv = torch.linalg.svdvals(Jt)
        e_ = (sv - 1.0).abs().max(-1).values
        return float(e_.mean()), float(e_.max())

    def _prw_metric(th, xp, ds, Jt):
        # G(θ) [P,P] = 바닥 정확 헤시안 + (덧셈) 뒤집힘 Σ‖J⁻¹dJ‖_F² + eps; 그리고 강체 제약 C [6,P] (입자 변위의 강체 최소제곱 성분)
        st = _PR["st"]
        P, nb, Mn = st["P"], st["nb"], st["Mn"]
        Jn = _prw_Jn(th)                                                 # [Mn,3,P]
        rows, lamd = st["rows"], st["lamd"]
        H = torch.zeros(P, P, device=dev, dtype=torch.float64)
        for (q_, n_), dq in zip(_PR["fl"], ds):
            # 바닥: 출력만 최적화·학습과 같은 IPC 장벽 E_b = Σ_p kappa m_p b(d_p) / (Σm · EXT²)  (phys_resid._barrier_b, 연장만 끔).
            #   헤시안 Σ_p w_p b''(d_p) ∇d_p ∇d_pᵀ (d̂ 밖 입자는 0). b'' 는 phys_resid 함수를 autograd 로 두 번 미분.
            act = dq < st["bc_dh"]
            if not bool(act.any()):
                continue
            Jnn = torch.einsum("i,mip->mp", n_, Jn)                      # [Mn,P]
            ra, la = rows[act], lamd[act]
            gd = torch.zeros(ra.shape[0], P, device=dev, dtype=torch.float64)
            for k_ in range(4):
                gd += la[:, k_:k_ + 1] * Jnn[ra[:, k_]]
            with torch.enable_grad():
                dd_ = dq[act].detach().clone().requires_grad_(True)
                b_ = phys_resid._barrier_b(dd_, st["bc_dh"], ext_kind="none")
                g1_ = torch.autograd.grad(b_.sum(), dd_, create_graph=True)[0]
                b2_ = torch.autograd.grad(g1_.sum(), dd_)[0].detach()
            A_ = gd * (st["bc_w"][act] * b2_).clamp_min(0.0).sqrt().unsqueeze(-1)
            H += A_.T @ A_
            del gd, A_
        if not _PR["flow"]:
            # Σ_c ‖M_c ΔΘᵀ ψ_c‖_F²,  M_c = J_c⁻¹:  Y[c,j,l,(a,i)] = ψ[c,a,j] M[c,l,i]
            Mi = torch.linalg.inv(Jt)
            psi = st["psi"]
            for c0 in range(0, psi.shape[0], 4096):
                Y = torch.einsum("caj,cli->cjlai", psi[c0:c0 + 4096], Mi[c0:c0 + 4096]).reshape(-1, P)
                H += Y.T @ Y
                del Y
        # 강체 성분: 입자 변위장과 강체장 6 개(평행이동, 무게중심 기준 회전)의 내적 -> 노드로 모아 Jn 과 축약
        cen = xp.mean(0)
        rel = xp - cen
        E3 = st["I3"]
        _sc = lambda r_: torch.einsum("mi,mip->p", torch.zeros(Mn, 3, device=dev, dtype=torch.float64).index_add_(
            0, rows.reshape(-1), (lamd.unsqueeze(-1) * r_.unsqueeze(1)).reshape(-1, 3)), Jn)
        if _PR["push"] > 0 and _PR["fl"]:
            # 바닥 법선 평행이동은 허용 (제약에서 뺀다): 수평 두 방향 + 회전 셋만 제약
            n0 = _PR["fl"][0][1]
            t1 = torch.linalg.svd(n0.view(1, 3), full_matrices=True)[2][1:]          # n 에 수직인 두 단위벡터
            rig = [t1[0].expand_as(xp), t1[1].expand_as(xp)] + [torch.cross(E3[k_].expand_as(xp), rel, dim=-1) for k_ in range(3)]
            cn = _sc(n0.expand_as(xp)) / xp.shape[0]                    # ∂(평균 법선 변위)/∂θ
        else:
            rig = [E3[k_].expand_as(xp) for k_ in range(3)] + [torch.cross(E3[k_].expand_as(xp), rel, dim=-1) for k_ in range(3)]
            cn = None
        C = torch.stack([_sc(r_) for r_ in rig], 0)                       # [5|6,P]
        st["_cn"] = cn
        return H, C

    def _prw_dir(H, C, z, push=None):
        # 강체 성분이 0 인 부분공간 R 에서 Δθ = G_R^{-1/2} z_R / ‖z_R‖  (‖Δθ‖_G = 1)
        st = _PR["st"]
        P = st["P"]
        G = H + st["epsI"] * torch.eye(P, device=dev, dtype=torch.float64)
        if _PR["norigid"]:
            _, sC, VhC = torch.linalg.svd(C, full_matrices=True)
            rk = int((sC > sC.max() * 1e-10).sum())
            Nb = VhC[rk:].T                                              # [P, P-rk]
        else:
            Nb = torch.eye(P, device=dev, dtype=torch.float64)
        GR = Nb.T @ G @ Nb
        ev, Q = torch.linalg.eigh(GR)
        # G >= eps·I 는 식으로 보장된다 (나머지 항은 전부 반정부호). 바닥에 바짝 붙은 입자의 b''(d) ~ d̂²/d² 가 커지면
        # 고윳값 범위가 1e15 를 넘어 반올림으로 음수가 나온다 -- 그 부분만 eps 로 올린다 (원래 고윳값 최소는 기록).
        ev_raw = float(ev.min())
        ev = ev.clamp_min(st["epsI"])
        zR = z[:Nb.shape[1]]
        Ph = (Q * ev.rsqrt().unsqueeze(-2)) @ Q.T
        vR = zR / zR.norm()
        cn = st.get("_cn")
        pw = _PR["push"] if push is None else push
        if pw > 0 and cn is not None:
            gw = Ph @ (Nb.T @ cn)                                        # 백색화 좌표의 평균 법선 변위 기울기
            if float(gw.norm()) > 0:
                vR = pw * (-gw / gw.norm()) + (1.0 - pw) * vR
                vR = vR / vR.norm()
        dth = Nb @ (Ph @ vR)                                             # ‖Δθ‖_G = ‖vR‖ = 1
        return dth, G, ev_raw, float(ev.max())

    def _prw_setup(p, x, v):
        st = dict(p=p, x=x.detach().clone(), v=v.detach().clone())
        _no = int(getattr(net, "n_out", 3))
        _SIM_T[0] = 0.0
        _NF_HOLD[0] = "want"
        _sv = _DP_FAST[0]
        _DP_FAST[0] = lambda _n, _dt, _dev: (torch.zeros(_n, _no, device=_dev, dtype=x.dtype),)
        try:
            with torch.enable_grad(), dt_scope(FRAME_DT):
                step_once(d, t0, gsel, p, x, v, need_J=False)
        finally:
            _DP_FAST[0] = _sv
        _ens, _NF_HOLD[0] = _NF_HOLD[0], None
        _DET_LAST[0] = None
        if len(_ens) != 1 or _NOD_M[0] is not None or a.hex_jac or a.smooth_nb > 0 or a.cell_poly or getattr(a, "gregory", False):
            raise SystemExit("AF_ROLL_PRW: 격자 하나·손잡이 없음·사면체 선형 전달일 때만 (계량이 정확하려면)")
        e0 = _ens[0]
        npos, (lo, lat, nn), rows, lam_, aux = e0[1], e0[2], e0[3], e0[4], e0[7]
        Mn, N = npos.shape[0], x.shape[0]
        xd = st["x"].double()
        blo, bhi = xd.min(0).values, xd.max(0).values
        bsz = (bhi - blo).clamp_min(1e-12)
        Bfun = lambda y: _prw_basis((y - blo) / bsz)
        y0 = npos.double()
        Bm = Bfun(y0)
        nb = Bm.shape[1]
        P = 3 * nb
        tid = SX.tet_id(lo, lat, nn, aux)
        ut, inv = torch.unique(tid, return_inverse=True)
        rep = torch.zeros(ut.shape[0], dtype=torch.long, device=dev).scatter_(0, inv, torch.arange(N, device=dev))
        I3 = torch.eye(3, device=dev, dtype=torch.float64)
        Ai = lat.Ai.double()
        Bc = Bm[rows[rep]]
        dd = (Bc[:, 1:] - Bc[:, :-1]).transpose(1, 2)
        psi = dd.gather(2, aux[1][rep].unsqueeze(1).expand(-1, nb, -1)) @ Ai     # [T,nb,3]: J − I = Θᵀ ψ (덧셈)
        if phys_resid._BARRIER[0] is None:
            raise SystemExit("AF_ROLL_PRW: 바닥 장벽(--bc_mode barrier)이 꺼져 있다 -- 출력만 최적화와 같은 바닥을 쓰려면 켜야 한다")
        _bk, _bdh, _bext = phys_resid._BARRIER[0]
        _mw = (MASS[gsel] if MASS.numel() != x.shape[0] else MASS).double()
        st["bc_dh"] = float(_bdh)
        st["bc_wref"] = _bk * _mw / (float(_mw.sum()) * EXT * EXT)       # 정규화 증분 포텐셜(ip_of norm=Σm·EXT²/h²)에서의 가중 (값 대조용)
        # 계량용 가중 (사용자 10-19 결정): 입자마다 kappa/d̂² -- d→0 에서 b(d) ≈ d̂²(−log d) 라 −log d 와 같은 크기의 장벽이 된다.
        #   출력만 최적화 가중(위)을 그대로 쓰면 곡률이 −log d 의 d̂²·w ≈ 3e-10 배라 Dikin 걸음이 바닥을 못 막는다 (d≈5e-17 까지 붙어 멈춤).
        st["bc_w"] = torch.full_like(_mw, _bk / (float(_bdh) ** 2))
        st["bc_info"] = (float(_bk), float(_bdh), _bext)
        st.update(ens=_ens, Bm=Bm, Bfun=Bfun, y0=y0, rows=rows, lamd=lam_.double(), xd=xd, rep=rep, tid=tid,
                  rank_r=aux[1][rep], Ai=Ai, I3=I3, psi=psi, P=P, nb=nb, Mn=Mn, N=N, epsI=0.0,
                  F0=(_FCUM[0].clone() if _FCUM[0] is not None else None))
        _PR["st"] = st
        with torch.no_grad():
            th0 = torch.zeros(P, device=dev, dtype=torch.float64)
            U0, xp0, ds0, Jt0 = _prw_state(th0)
            nbad0 = sum(int((q_ <= 0).sum()) for q_ in ds0)
            if nbad0:
                raise SystemExit(f"AF_ROLL_PRW: 시작 상태가 바닥 장벽 밖 -- d<=0 입자 {nbad0} 개")
            H0, C0 = _prw_metric(th0, xp0, ds0, Jt0)
            st["epsI"] = _PR["eps"] * float(H0.diagonal().mean())
            # 확인: 내 식(입자 위치·사면체 J) vs 실제 경로(step_once) -- 무작위 작은 θ 하나
            zt = torch.randn(P, device=dev, dtype=torch.float64, generator=_PR["gen"])
            tht, _, _, _ = _prw_dir(H0, C0, zt)
            tht = 0.3 * tht
            Ut, xpt, _, Jtt = _prw_state(tht)
            x2, _, _, Jd = _rw_eval(0, st["p"], st["x"], st["v"], _ens, _prw_pad(Ut, x.dtype))
            ex = float((x2.double() - xpt).abs().max())
            eJ = float((Jd[rep].double() - Jtt).abs().max())
            rres = float((C0 @ tht).abs().max())
            _svb, _svs = phys_resid._BARRIER[0], os.environ.get("AF_BC_SLIP")
            phys_resid._BARRIER[0] = (st["bc_info"][0], st["bc_info"][1], "none")
            os.environ["AF_BC_SLIP"] = "1"
            try:
                _cfg = d["cfg"]
                eb_ref = float(phys_resid.bc_energy(st["xd"], xpt - st["xd"], _mw, _cfg, 1.0, float(_cfg.get("grid_lim", 2.0)),
                                                    int(_cfg["n_grid"]))) / (float(_mw.sum()) * EXT * EXT)
            finally:
                phys_resid._BARRIER[0] = _svb
                if _svs is None:
                    os.environ.pop("AF_BC_SLIP", None)
                else:
                    os.environ["AF_BC_SLIP"] = _svs
            eb_me = sum(float((st["bc_wref"] * phys_resid._barrier_b(((xpt - q_) * n_).sum(-1), st["bc_dh"], ext_kind="none")).sum())
                        for q_, n_ in _PR["fl"])
            nact0 = sum(int((q_ < st["bc_dh"]).sum()) for q_ in ds0)
            st["band"] = (torch.stack(ds0, 0).min(0).values < _PR["band"]) if ds0 else torch.zeros(N, dtype=torch.bool, device=dev)
            st["band_tet"] = st["band"][rep]
        info = dict(nb=nb, P=P, Mn=Mn, N=N, T=int(rep.numel()), epsI=st["epsI"], check_x=ex, check_J=eJ, rigid_resid=rres,
                    bc=st["bc_info"], bc_energy_ref=eb_ref, bc_energy_mine=eb_me, bc_active0=nact0)
        _PR["info"] = info
        print(f"[파라미터 표본 준비] 기저 수 {nb} -> θ 차원 {P}, 노드 {Mn}, 입자 {N}, 사면체 {info['T']}, "
              f"{'흐름 RK4 ' + str(_PR['nst']) if _PR['flow'] else '덧셈'}  eps 항 {st['epsI']:.3e}  "
              f"내 식 vs step_once 최대차 x {ex:.1e} J {eJ:.1e}  강체 제약 잔차 {rres:.1e}  | 바닥 = phys_resid IPC 장벽 kappa "
              f"{st['bc_info'][0]:g} d̂ {st['bc_info'][1]:.5f} (학습 연장 {st['bc_info'][2]} -> 여기서는 none), 입자 수준, "
              f"계량 가중 kappa/d̂² (값 대조는 출력만 최적화 가중 kappa·m/(Σm·EXT²)), 시작 d<d̂ 입자 {nact0}, sticky 접선 항은 계량에 안 넣음; "
              f"장벽 값 확인 bc_energy {eb_ref:.6e} vs 내 합 {eb_me:.6e}", flush=True)

    def _prw_pad(Ud, dt_):
        _no = int(getattr(net, "n_out", 3))
        U = torch.zeros(Ud.shape[0], _no, device=dev, dtype=dt_)
        U[:, :3] = Ud.to(dt_)
        return U

    def _prw_contact(ds_):
        # (d < d̂ 입자 수, 전체 비율, 바닥 띠 안 비율)
        if not ds_:
            return 0, 0.0, 0.0
        dm_ = torch.stack(ds_, 0).min(0).values
        c_ = dm_ < _PR["st"]["bc_dh"]
        nb_ = int(_PR["st"]["band"].sum())
        return int(c_.sum()), float(c_.float().mean()), (float((c_ & _PR["st"]["band"]).sum()) / nb_ if nb_ else 0.0)

    def _prw_walk():
        # 표본 하나: θ=0 에서 방향 z 고정, 서브스텝마다 현재 θ 의 계량으로 r_s 걸음 -> (θ, 기록)
        st = _PR["st"]
        P = st["P"]
        z = torch.randn(P, device=dev, dtype=torch.float64, generator=_PR["gen"])
        th = torch.zeros(P, device=dev, dtype=torch.float64)
        U, xp, ds, Jt = _prw_state(th)
        K, nbt, nfit, bad_sub, acc_bad = 0, 0, 0, [], []
        sm, sx = 0.0, 0.0
        gmin, gmax = float("inf"), 0.0
        while K < _PR["kmax"]:
            H, C = _prw_metric(th, xp, ds, Jt)
            # 밀기 모드에서 신축 목표를 이미 채웠으면 접촉 목표만 남는다: 순수 아래쪽 방향, 크기 맞춤 상한 없음 (눌려 퍼지게)
            s_done = _PR["push"] > 0 and sm >= _PR["s_mean"] and sx >= _PR["s_max"]
            dth, G, e1, e2 = _prw_dir(H, C, z, push=(1.0 if s_done else None))
            gmin, gmax = min(gmin, e1), max(gmax, e2)
            dth = _PR["r"] * dth
            K += 1
            ok = False
            for _bt in range(40):
                c_th = th + dth
                cU, cxp, cds, cJt = _prw_state(c_th)
                nfb = sum(int((q_ <= 0).sum()) for q_ in cds)
                dj = torch.linalg.det(cJt)
                ntb = int((dj <= 0).sum())
                if nfb or ntb:
                    bad_sub.append((K, nfb, ntb))
                    dth = 0.5 * dth
                    nbt += 1
                    continue
                csm, csx = _prw_strain(cJt)
                if csm > 2.0 * _PR["s_mean"] and _bt < 39 and not s_done:
                    dth = 0.5 * dth
                    nfit += 1
                    continue
                ok = True
                break
            if not ok:                                                   # 40 번 반감해도 위반 -- 이 표본은 여기서 멈춘다 (수락 안 함)
                acc_bad.append((K, nfb, ntb))
                break
            th, U, xp, ds, Jt, sm, sx = c_th, cU, cxp, cds, cJt, csm, csx
            if K % 100 == 0:
                _dm = min(float(q_.min()) for q_ in ds) if ds else float("nan")
                print(f"      [걷기] 서브스텝 {K}  신축 평균 {sm:.3f} 최대 {sx:.3f}  백트래킹 누적 {nbt} 크기맞춤 {nfit}  "
                      f"d최소 {_dm:.3e}  계량 고윳값(반올림 전) {e1:.3e}~{e2:.3e}  ‖θ‖ {float(th.norm()):.3e}", flush=True)
            if sm >= _PR["s_mean"] and sx >= _PR["s_max"] and (_PR["push"] <= 0 or _prw_contact(ds)[2] >= _PR["contact"]):
                break
        return th, U, dict(K=K, backtrack=nbt, size_fit=nfit, sub_viol=bad_sub[:50], n_sub_viol=len(bad_sub),
                           acc_viol=acc_bad, stuck=len(acc_bad), gR_eig_min=gmin, gR_eig_max=gmax, walk_strain_mean=sm, walk_strain_max=sx)

    def _prw_step(i, p, x, v):
        if _PR["st"] is None:
            _prw_setup(p, x, v)
        st = _PR["st"]
        _SIM_T[0] = 0.0
        _t = time.time()
        with torch.no_grad():
            th, Ud, wk = _prw_walk()
        x2, v2, p2, Jd = _rw_eval(0, st["p"], st["x"], st["v"], st["ens"], _prw_pad(Ud, x.dtype))
        with torch.no_grad():
            x2d = x2.double()
            dmin, nfb = float("inf"), 0
            for q_, n_ in _PR["fl"]:
                dq = ((x2d - q_) * n_).sum(-1)
                dmin, nfb = min(dmin, float(dq.min())), nfb + int((dq <= 0).sum())
            Jdd = Jd.double()
            djp = torch.linalg.det(Jdd)
            Jt = Jdd[st["rep"]]
            djt = torch.linalg.det(Jt)
            s_mean, s_max = _prw_strain(Jt)
            du = (x2d - st["xd"]).norm(dim=-1)
            ds2 = [((x2d - q_) * n_).sum(-1) for q_, n_ in _PR["fl"]]
            ncon, fcon, fband = _prw_contact(ds2)
            bt_ = st["band_tet"]
            if bool(bt_.any()):
                sb_mean, sb_max = _prw_strain(Jt[bt_])
            else:
                sb_mean = sb_max = float("nan")
            fdm = float("nan")
            if st["F0"] is not None:
                _FCUM[0] = (Jd.float() @ st["F0"]).detach()
                fdm = float(torch.linalg.det(_FCUM[0].double()).min())
        rec = dict(frame=i + 1, strain_mean=s_mean, strain_max=s_max,
                   du_mean=100 * float(du.mean()) / EXT, du_max=100 * float(du.max()) / EXT,
                   d_min=(dmin if _PR["fl"] else None), floor_bad=nfb,
                   detJ_tet_min=float(djt.min()), detJ_tet_max=float(djt.max()), tet_bad=int((djt <= 0).sum()), n_tets=int(djt.numel()),
                   detJ_part_min=float(djp.min()), part_bad=int((djp <= 0).sum()), detF_cum_min=fdm,
                   theta_norm=float(th.norm()), sec=time.time() - _t, contact_n=ncon, contact_frac=fcon, contact_band_frac=fband,
                   band_n=int(st["band"].sum()), band_strain_mean=sb_mean, band_strain_max=sb_max,
                   contact_target=(_PR["contact"] if _PR["push"] > 0 else 0.0), **wk)
        _PR["rec"].append(rec)
        print(f"    [파라미터 표본] {i + 1}/{L}  서브스텝 {wk['K']} (백트래킹 {wk['backtrack']}, 크기맞춤 {wk['size_fit']}, "
              f"후보 위반 서브스텝 {wk['n_sub_viol']}{' ' + str(wk['sub_viol'][:5]) if wk['n_sub_viol'] else ''})  "
              f"신축 max|σ−1| 평균 {s_mean:.3f} 최대 {s_max:.3f}  변위 평균 {rec['du_mean']:.2f}% 최대 {rec['du_max']:.2f}% EXT  "
              f"바닥 d최소 {'없음' if not _PR['fl'] else format(dmin, '.3e')} d<=0 입자 {nfb}  "
              f"det 사면체 {rec['detJ_tet_min']:.3f}~{rec['detJ_tet_max']:.3f} det<=0 사면체 {rec['tet_bad']}/{rec['n_tets']}  "
              f"det 입자 최소 {rec['detJ_part_min']:.3f} det<=0 입자 {rec['part_bad']}  누적detF최소 {fdm:.3f}  "
              f"접촉(d<d̂) 입자 {ncon} ({100 * fcon:.2f}%, 바닥 띠 {rec['band_n']} 중 {100 * fband:.1f}%)  "
              f"바닥 띠 신축 평균 {sb_mean:.3f} 최대 {sb_max:.3f}  {rec['sec']:.1f} s", flush=True)
        return x2, st["v"], st["p"]

    def _step(i, x, v):
        p = _S["p"]
        if _RPRW:
            x2, v2, p2 = _prw_step(i, p, x, v)
            _S["p"] = p2 if p2 is not None else p
            return x2, v2, None
        if _RRW:
            x2, v2, p2 = _rw_step(i, p, x, v)
            _S["p"] = p2 if p2 is not None else p
            return x2, v2, None
        if _RLGV:
            _hs = FRAME_DT / EVAL_SUB
            _ln = []
            for _sb in range(EVAL_SUB):
                _SIM_T[0] = (i + _sb / EVAL_SUB) * FRAME_DT
                _NF_HOLD[0] = "want"                     # 격자·노드특징을 붙잡아 MALA 평가에서 재사용 (학습과 같다)
                with torch.enable_grad(), dt_scope(_hs):     # 일반 롤아웃과 같은 문맥 (망 파라미터 기울기는 꺼져 있다)
                    _, _, _, _, _dp, _, _, _, _, _ = step_once(d, t0 + i, gsel, p, x, v, need_J=False)
                _ens, _NF_HOLD[0] = _NF_HOLD[0], None
                _U0 = _dp.detach()
                _Z0 = torch.zeros_like(_U0)
                if _RLGV == "inertial":
                    _e0 = _ens[0]
                    _uin = DM.node_inertial(x, v, _LSC["mass"], _e0[3], _e0[4], _e0[1].shape[0],
                                            d["cfg"]["g"], _hs)[0]
                    _Z0[:, :3] = (_uin - _U0[:, :3]).detach()
                _s0 = _lgv_eval(d, gsel, p, x, v, _S["F_ip"], _LSC, _hs, _ens, _U0, _Z0, True, t=t0 + i)
                _ox, _ov, _op, _oF, _odl = _lgv_step(d, gsel, p, x, v, _S["F_ip"], _LSC, _hs, _ens, _U0, _s0,
                                                     _NOD_M[0], t=t0 + i)
                _lst = _PLGV["last"]
                with torch.no_grad():
                    _S["F_ip"] = phys_resid.plastic_step(_oF, _odl).detach()
                    if _ROLLDUMP is not None and _FCUM[0] is not None:
                        _FCUM[0] = (_lst["J"].detach().float() @ _FCUM[0]).detach()
                x, v = _ox.detach(), _ov.detach()
                p = _op if _op is not None else p
                _LACC.append(_lst["acc"])
                _ln.append(f"수락 {_lst['acc']} 백트래킹 {_lst['nbt']} E {_lst['E0']:.4e}->{_lst['E1']:.4e} "
                           f"(ΔE {100 * (_lst['E1'] / max(_lst['E0'], 1e-30) - 1):+.2f}%) eta {_lst['eta']:.3e}")
                del _ens
            _S["p"] = p
            print(f"    [랑제빈 롤아웃] 프레임 {i + 1}/{L}  " + " | ".join(_ln)
                  + f"  누적수락률 {100 * np.mean(_LACC):.1f}%  누적 {time.time() - _RO_T0:.1f} s", flush=True)
            return x, v, None
        if a.ov_roll > 0:
            # 출력만 최적화: 서브스텝·jvp 없이 프레임마다 변수를 내린다
            # 서브스텝: 한 프레임을 EVAL_SUB 번에 나눠 스텝마다 변수를 새로 내린다 (dt = FRAME_DT / EVAL_SUB)
            with dt_scope(FRAME_DT / EVAL_SUB):
                x2 = x
                if _NODE_TGT is not None and i > 0:
                    # --ov_obj node: 목표 실행과 같은 상태에서 노드 출력을 맞춘다 (격자가 같아야 노드가 대응한다)
                    x2 = _NODE_TGT["pred"][i - 1].to(dev, x2.dtype)
                    v = _NODE_TGT["v"][i - 1].to(dev, v.dtype)
                    _S["F_ov"] = _NODE_TGT["F"][i - 1].to(dev, _S["F_ov"].dtype)
                for _sb in range(EVAL_SUB):
                    _SIM_T[0] = (i + _sb / EVAL_SUB) * FRAME_DT       # --cfg_bc 시간창 (롤아웃 시작 = 0)
                    x2, v, _S["F_ov"] = _ov_frame(d, t0 + i, gsel, p, x2, v, _S["F_ov"])
                    if a.pg_vdamp:
                        v = _pg_vdamp(v, d, FRAME_DT / EVAL_SUB)
                    elif a.cfg_bc and float(d["cfg"].get("grid_v_damping_scale", 1.1)) < 1.0:
                        v = v * float(d["cfg"]["grid_v_damping_scale"])   # PG 와 같이 서브스텝마다 곱한다
            _S["v_last"] = v
            if _ROLL_IP is not None:
                _lg = _OV_LOG[-EVAL_SUB:]
                _ROLL_IP.append(dict(frame=i + 1, E=[q[1] for q in _lg], E_init=[q[0] for q in _lg]))
            print(f"    [출력만최적화 진행] 프레임 {i + 1}/{L}  서브스텝 {EVAL_SUB}  누적 {time.time() - _RO_T0:.1f} s", flush=True)
            if _ROLLDUMP is not None and _FCUM[0] is not None:
                _FCUM[0] = _S["F_ov"].detach().float()          # 출력만 최적화가 실제로 굴린 F (렌더용)
        elif _ROLL_IP is not None:
            # AF_ROLL_IP: 망 롤아웃의 매 서브스텝에서 증분 포텐셜(장벽 포함, det 마스크 없음 -- 출력만 최적화와 같은 식)과
            # 출력잔차를 잰다. F 는 정답 t0 의 F 에서 시작해 plastic_step 으로 전진한다
            dmg_e, idx_e, fe_r = _S["dmg"], _S["idx"], _S["fe"]
            _hs = FRAME_DT / EVAL_SUB
            _cfg = d["cfg"]; _m = MASS[gsel]; _vol = _pvol(_m, _cfg)
            _gv = torch.tensor(_cfg["g"], device=dev, dtype=torch.float32)
            _ng, _gl = int(_cfg["n_grid"]), float(_cfg.get("grid_lim", 2.0))
            _nrm = float(_m.sum()) * (EXT ** 2) / (_hs ** 2)
            _Es, _Rs = [], []
            for _sb in range(EVAL_SUB):
                with torch.enable_grad(), dt_scope(_hs):
                    DM.PROBE[0] = "want"
                    x2, p2, v2, _, _, _, dmg_e, idx_e, fe_r, _jf = step_once(
                        d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                        idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
                    DM.PROBE[0] = None
                    _fmi = (free_mask(d, x2.shape[0], dev, gsel, x, t0 + i) if a.control else None)
                    if _BC_LAST[0] is not None:
                        _am = _BC_LAST[0][1].reshape(-1) > 0.5
                        _fmi = (~_am) if _fmi is None else (_fmi & ~_am)
                    _E, _dl, _Ftr, _ = ip_of(x, x2 - x, v, _S["F_ip"], _m, _vol, _cfg, _hs, _ng, _gl,
                                             g=_gv, norm=_nrm, free=_fmi, jac=_jf)
                    _r = float("nan")
                    if DM.PROBE[1]:
                        _pz = DM.PROBE[1][0]
                        _gU, = torch.autograd.grad(_E * _nrm, _pz, retain_graph=True)
                        _mN, = torch.autograd.grad((_m.unsqueeze(-1) * x2).sum(), _pz)
                        _mN = _mN[:, 0]
                        _sup = _mN > 1e-9 * _mN.max().clamp_min(1e-30)
                        _r = float(_gU[:, :3].norm(dim=-1)[_sup].mean() * (_hs ** 2) / _mN[_sup].mean() / EXT)
                    DM.PROBE[1] = None
                    _Es.append(float(_E)); _Rs.append(_r)
                with torch.no_grad():
                    _S["F_ip"] = phys_resid.plastic_step(_Ftr, _dl).detach()
                if _ROLLDUMP is not None and _FCUM[0] is not None and _jf is not None:
                    _FCUM[0] = (_jf.detach().float() @ _FCUM[0]).detach()
                x, p, v = x2.detach(), p2.detach(), _pg_vdamp(v2.detach(), d, _hs)
            x2 = x
            _S.update(p=p, dmg=dmg_e, idx=idx_e, fe=fe_r)
            _ROLL_IP.append(dict(frame=i + 1, E=_Es, rout=_Rs))
        else:
            dmg_e, idx_e, fe_r = _S["dmg"], _S["idx"], _S["fe"]
            with torch.enable_grad(), dt_scope(FRAME_DT / EVAL_SUB):
                for _sb in range(EVAL_SUB - 1):
                    x, p, v, _, _, _, dmg_e, idx_e, fe_r, _ = step_once(
                        d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                        idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
                    x, p, v = x.detach(), p.detach(), _pg_vdamp(v.detach(), d, FRAME_DT / EVAL_SUB)
                x2, p, v, _, _, _, dmg_e, idx_e, fe_r, _jf = step_once(
                    d, t0 + i, gsel, p, x, v, need_J=False, dmg=dmg_e,
                    idx_prev=idx_e, x0=x0e, p0=p0e, fe=fe_r)
            x2 = x2.detach(); p = p.detach(); v = _pg_vdamp(v.detach(), d, FRAME_DT / EVAL_SUB)
            _S.update(p=p, dmg=dmg_e, idx=idx_e, fe=fe_r)
            if _ROLLDUMP is not None and _FCUM[0] is not None and _jf is not None:
                # 렌더용: 망이 낸 변형장의 **실제 야코비안**을 누적한 F (이웃 추정 아님)
                _FCUM[0] = (_jf.detach().float() @ _FCUM[0]).detach()
        if _PROF_ON:
            torch.cuda.synchronize()
            print(f"    [프레임] {i + 1}/{L} 누적 {time.time() - _RO_T0:.2f} s | "
                  + " ".join(f"{k} {v:.2f}" for k, v in sorted(_PROF.items(), key=lambda z: -z[1])[:6]), flush=True)
        return x2, v, None

    def _fm(i, x2):
        return (free_mask(d, x2.shape[0], x2.device, gsel, x2, t0 + i)
                if a.control else slice(None))

    def _on_frame(i, x2, gt, _info, x_still):
        if _TUN_LOG is not None and _TUN_CUR[0] is not None:
            with torch.no_grad():
                _np, _dp, _rT = _TUN_CUR[0]
                _tet = torch.unique(_rT, dim=0)                     # 활성 사면체 [K,4] (노드 번호)
                _rec = {"frame": int(i) + 1, "tets": int(_tet.shape[0])}
                for _bc in (d["cfg"].get("boundary_conditions") or []):
                    if _bc.get("type") != "surface_collider":
                        continue
                    _pt = torch.tensor(_bc["point"], dtype=_np.dtype, device=_np.device)
                    _nr = torch.tensor(_bc["normal"], dtype=_np.dtype, device=_np.device)
                    _nr = _nr / _nr.norm().clamp_min(1e-12)
                    _s0 = ((_np - _pt) * _nr).sum(-1)[_tet]         # 스텝 시작 꼭지점 부호 거리 [K,4]
                    _s1 = ((_np + _dp[:, :3] - _pt) * _nr).sum(-1)[_tet]   # 스텝 끝 (반경 채널 제외)
                    _thru = (_s0 > 0).any(1) & (_s1 < 0).all(1)
                    # 그 사면체에 든 입자 수 (_rT 는 입자마다 자기 사면체의 네 노드)
                    _ps0 = ((_np - _pt) * _nr).sum(-1)[_rT]
                    _ps1 = ((_np + _dp[:, :3] - _pt) * _nr).sum(-1)[_rT]
                    _thru_p = (_ps0 > 0).any(1) & (_ps1 < 0).all(1)
                    _rec.update(tunnel=int(_thru.sum()),
                                tunnel_particles=int(_thru_p.sum()), particles=int(_rT.shape[0]),
                                below_end_all=int((_s1 < 0).all(1).sum()),
                                below_end_any=int((_s1 < 0).any(1).sum()),
                                above_start_any=int((_s0 > 0).any(1).sum()))
                    break
            with open(_TUN_LOG, "a") as _fT:
                _fT.write(json.dumps(_rec) + "\n")
        if _INV_LOG is not None and _INV_CUR[0] is not None:
            with torch.no_grad():
                _dI, _rI = _INV_CUR[0]
                _bI = _dI <= 0
                _nt = int(torch.unique(_rI[_bI], dim=0).shape[0]) if bool(_bI.any()) else 0
                _ntot = int(torch.unique(_rI, dim=0).shape[0])
                _Fc = _S.get("F_ov") if _S.get("F_ov") is not None else _FCUM[0]
                _nc = int((torch.linalg.det(_Fc.float()) <= 0).sum()) if _Fc is not None else -1
            with open(_INV_LOG, "a") as _fI:
                _recI = {"frame": int(i) + 1, "inv_tets": _nt, "tets": _ntot,
                         "inv_particles_step": int(_bI.sum()), "inv_particles_cum": _nc,
                         "det_min": float(_dI.min())}
                if DM.C1_LAST[0] is not None:          # --interp c1wf: 마지막 평가(받아들인 해)의 Bernstein det 계수
                    _cI = DM.C1_LAST[0][0].detach()
                    _recI.update(c1_coef_min=float(_cI.min()), c1_coef_neg=int((_cI <= 0).sum()),
                                 c1_subtets=int(_cI.shape[0] * _cI.shape[1]))
                    # 요소 말림 (det>0 이어도 휘어진 요소 경계 면이 스스로 교차): 면당 12 삼각형 근사의 변-삼각형 교차
                    _, _, _dpI, _wtI, _loI, _laI, _nnI, _uqI = DM.C1_LAST[0]
                    _ncl, _npl = WF.curl_check(_dpI.detach(), _wtI, _loI, _laI, _nnI, _uqI)
                    _recI.update(c1_curled=_ncl, c1_curl_pairs=_npl)
                _oI = _OV_CUR_OPT[0]
                if _oI is not None and getattr(_oI, "stats", None):
                    _recI["opt"] = dict(_oI.stats, phase1=_C1_P1[0])
            with open(_INV_LOG, "a") as _fI:
                _fI.write(json.dumps(_recI) + "\n")
        # 평가에서도 손잡이가 명령대로 끌리는지 본다. 학습 경로만 보고 "손잡이는
        # 정확하다" 고 판단했다가, 평가에서는 ctrl_id 가 저장 표본에 없어 대체
        # 입자가 중심이 되는 것을 놓쳤다.
        _RO_CTRL.append((_CTRL_ERR[0], _CTRL_ERR[1], _CTRL_ERR[2]))
        F_ov = _S["F_ov"]
        if _ROLLDUMP is not None:
            _ROLLDUMP.append((x2.detach().cpu(), gt.detach().cpu()))
            if _S.get("v_last") is not None:
                _VDUMP.append(_S["v_last"].detach().float().cpu())
            if _FCUM[0] is not None:
                _FDUMP.append(_FCUM[0].float().cpu() if os.environ.get("AF_ROLL_DUMP_F") == "32" else _FCUM[0].half().cpu())
            if _NODE_LAST[0] is not None:
                _NODEDUMP.append(tuple(q_.cpu() for q_ in _NODE_LAST[0]))
            if F_ov is not None:
                # 입자별 변형 스칼라. C = FᵀF (우 코시-그린), E = (C−I)/2
                # (그린-라그랑주). 정지 상태면 det 1, trC 3, ‖E‖ 0 이다.
                _C = F_ov.transpose(-1, -2) @ F_ov
                _E = 0.5 * (_C - torch.eye(3, device=_C.device,
                                           dtype=_C.dtype))
                # C_zz 는 처음에 z 를 향하던 섬유의 **신축 제곱**이다
                # (sqrt(C_zz) < 1 이면 세로로 찌그러진 것이다). psi 는 그
                # 입자의 탄성 에너지 밀도다.
                _psi, _ = phys_resid.psi_of(F_ov, d["cfg"], FRAME_DT)
                _DETDUMP.append(torch.stack([
                    torch.linalg.det(F_ov),
                    _C.diagonal(dim1=-2, dim2=-1).sum(-1),
                    _E.reshape(_E.shape[0], -1).norm(dim=-1),
                    _C[:, 2, 2],
                    _psi,
                ], -1).detach().cpu())
        if a.metrics:
            cds.append(chamfer(x2, gt) / (EXT ** 2))
            ems.append(emd(x2, gt, t0 * 1000 + i) / EXT)
            cds_s.append(chamfer(x_still, gt) / (EXT ** 2))
            ems_s.append(emd(x_still, gt, t0 * 1000 + i) / EXT)

    # 롤아웃 루프와 지표 정의는 라이브러리 하나 (lb_rollout 과 같다)
    errs, stills = DM.rollout_eval(
        _step, x, v, lambda i: take(d["x"][t0 + i + 1], gsel), L, EXT,
        fm_fn=_fm, on_frame=_on_frame)
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
    if _RPRW and _PR["rec"]:
        _rr = _PR["rec"]
        print(f"  [파라미터 표본 요약] 기저 {_PR['kind']}:{_PR['size']} {'흐름' if _PR['flow'] else '덧셈'} r_s {_PR['r']:g} eps {_PR['eps']:g} "
              f"시드 {_PR['seed']}  표본 {len(_rr)}  신축 max|σ−1| 평균(표본 평균) {np.mean([q['strain_mean'] for q in _rr]):.3f} "
              f"(표본별 평균 최소 {min(q['strain_mean'] for q in _rr):.3f}, 최대의 최대 {max(q['strain_max'] for q in _rr):.3f})  "
              f"변위 평균 {np.mean([q['du_mean'] for q in _rr]):.2f}% EXT (최대 {max(q['du_max'] for q in _rr):.2f}%)  "
              f"바닥 d<=0 입자 (합) {sum(q['floor_bad'] for q in _rr)}  det<=0 사면체 (합) {sum(q['tet_bad'] for q in _rr)}  "
              f"det<=0 입자 (합) {sum(q['part_bad'] for q in _rr)}  d최소 {min((q['d_min'] for q in _rr if q['d_min'] is not None), default=float('nan')):.3e}  "
              f"det 사면체 최소 {min(q['detJ_tet_min'] for q in _rr):.3f} 최대 {max(q['detJ_tet_max'] for q in _rr):.3f}  "
              f"서브스텝 평균 {np.mean([q['K'] for q in _rr]):.1f} 최대 {max(q['K'] for q in _rr)}  목표 미달 표본 "
              f"{sum((q['strain_mean'] < _PR['s_mean']) or (q['strain_max'] < _PR['s_max']) for q in _rr)}  "
              f"접촉 입자 비율 평균 {100 * np.mean([q['contact_frac'] for q in _rr]):.2f}% 바닥 띠 접촉 비율 평균 "
              f"{100 * np.mean([q['contact_band_frac'] for q in _rr]):.1f}% (최소 {100 * min(q['contact_band_frac'] for q in _rr):.1f}%)  "
              f"바닥 띠 신축 평균 {np.nanmean([q['band_strain_mean'] for q in _rr]):.3f}  "
              f"후보 위반 서브스텝 (합) {sum(q['n_sub_viol'] for q in _rr)}  반감 40 번 실패로 멈춘 표본 {sum(len(q['acc_viol']) for q in _rr)}", flush=True)
        if os.environ.get("AF_ROLL_PRW_LOG"):
            json.dump({"tag": d.get("tag", ""), "t0": t0, "spec": _RPRW, "flow": _PR["flow"], "info": _PR["info"],
                       "frames": _rr}, open(os.environ["AF_ROLL_PRW_LOG"], "w"))
            print(f"  [파라미터 표본 기록] {os.environ['AF_ROLL_PRW_LOG']}", flush=True)
        _PR["rec"] = []
    if _RRW and _RW["rec"]:
        _rr = _RW["rec"]
        print(f"  [랜덤 걸음 요약] r {_RW['r']:g} eps {_RW['eps']:g} 시드 {_RW['seed']}  프레임 {len(_rr)}  "
              f"바닥 위반 (프레임 합) {sum(q['floor_bad'] for q in _rr)}  위반 프레임 {sum(q['floor_bad'] > 0 for q in _rr)}  "
              f"뒤집힘 사면체 (프레임 합) {sum(q['detJ_bad_tets'] for q in _rr)}  위반 프레임 {sum(q['detJ_bad_tets'] > 0 for q in _rr)}  "
              f"‖Δ‖_G 범위 {min(q['gnorm'] for q in _rr):.4f}~{max(q['gnorm'] for q in _rr):.4f}  "
              f"입자 변위 평균 {np.mean([q['du_mean'] for q in _rr]):.4f}% EXT (최대 {max(q['du_max'] for q in _rr):.4f}%)  "
              f"d최소 {min((q['d_min_after'] for q in _rr if q['d_min_after'] is not None), default=float('nan')):.3e}  "
              f"누적detF최소 {min(q['detF_cum_min'] for q in _rr):.4f}", flush=True)
        if os.environ.get("AF_ROLL_RW_LOG"):
            json.dump({"tag": d.get("tag", ""), "t0": t0, "r": _RW["r"], "eps": _RW["eps"], "seed": _RW["seed"],
                       "frames": _rr}, open(os.environ["AF_ROLL_RW_LOG"], "w"))
            print(f"  [랜덤 걸음 기록] {os.environ['AF_ROLL_RW_LOG']}", flush=True)
        _RW["rec"] = []
    if _ROLL_IP:
        json.dump({"tag": d.get("tag", ""), "t0": t0, "sub": EVAL_SUB, "ov": a.ov_roll > 0,
                   "errs": [float(q) for q in errs], "frames": _ROLL_IP}, open(os.environ["AF_ROLL_IP"], "w"))
        print(f"  [증분포텐셜 기록] {os.environ['AF_ROLL_IP']}  프레임 {len(_ROLL_IP)}", flush=True)
        del _ROLL_IP[:]
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
_seen_tags = set()
for tag, d in TR + held:
    if tag in _seen_tags:             # 풀 모드는 홀드아웃을 학습 목록에도 넣는다 -- 같은 궤적을 두 번 굴리지 않는다
        continue
    _seen_tags.add(tag)
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
        _VDUMP.clear()
        _FDUMP.clear()
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
                        # 손잡이 입자 색인. 렌더가 **칸마다 자기 입자로** 중심을
                        # 잡게 한다 -- ctrl_pos 하나만 넘기면 두 칸 모두 PG 의
                        # 손잡이 경로를 그려, 우리 손잡이가 PG 처럼 진동하는
                        # 것처럼 보인다 (실측: 우리 손잡이 속도 표준편차 0).
                        "ctrl_id": d.get("ctrl_id"),
                        "t0": t0, "tag": tag,
                        "nodes": _NODEDUMP or None,
                        # [T,N,5] = (det F, tr C, ‖E‖_F, C_zz, psi)
                        "fscal": (torch.stack(_DETDUMP) if _DETDUMP else None),
                        # 같은 양을 **교사 궤적의 F** 로도 계산해 둔다. 없으면
                        # 렌더가 두 칸을 우리 값으로 칠해 비교가 성립하지 않는다.
                        "gt_fscal": (_gt_fscal(d, t0, L) if _DETDUMP else None),
                        # 풀 루프가 전역 EXT 를 씬 값으로 덮어쓰므로 **그 궤적의**
                        # 것을 쓴다 (덤프를 렌더할 때 길이 단위가 된다)
                        "EXT": float(d.get("_ext", EXT)),
                        # [L,N,3,3] half: 망 변형장 야코비안 누적 (AF_ROLL_DUMP_F=1 일 때)
                        "F": (torch.stack(_FDUMP) if _FDUMP else None),
                        # [L,N,3] 프레임 경계 속도 (출력만 최적화, 서브스텝 안정성 측정용)
                        "v": (torch.stack(_VDUMP) if _VDUMP else None)}, _rd)
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
_SUM = DM.summarize(rows)        # 전체 창 평균 (lb_rollout 과 같은 정의)
print(f"\n[요약] 전체 창 평균 {100*_SUM['err_mean']:.3f}% "
      f"(정지 {100*_SUM['still_mean']:.3f}%, 비 "
      f"{_SUM['ratio']:.2f}) "
      f"-- 비가 1 보다 작아야 도움이 된 것이다", flush=True)
# 최종 평가가 어느 단계의 가중치인가 (추론이면 체크포인트 단계)
_FINAL_STEP = step0 if a.infer else max(int(a.iters), int(step0))
if TBW is not None:
    DM.tb_write_eval(TBW, rows, _FINAL_STEP, prefix="eval")
    print(f"[TB] 최종 평가 (단계 {_FINAL_STEP}) -> {_TBD}", flush=True)
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
               step=_FINAL_STEP, summary=_SUM,
               minutes=(time.time() - t_start) / 60,
               loss_hist=hist[::20], rollout=rows),
          open(os.path.join(a.out, f"{a.tag}.json"), "w"), indent=1,
          ensure_ascii=False)
print(f"[저장] {a.out}", flush=True)
print("DEFORM_OK")
