"""Fit the anchor set to MPM, letting its support and its size both change.

The fixed-neighbour fit halved the one-step error and left the rollout where it
was -- 12.50% to 12.30% against MPM's particles on uniform impulses, worse on
the other two families. Two things it could not do, both structural rather than
a matter of more iterations: an anchor could only redistribute weight among the
eight Gaussians assigned to it at the start, and there were always exactly 512
anchors wherever the error happened to be.

Here the support is the region G(x) > c with weight G(x) - c, so membership
follows the parameters continuously, and anchors are split where the fit pushes
hardest and dropped where they hold nothing.

The loss moves to Gaussian space. It has to: with the anchor count changing
there is no fixed anchor-space target to compare against, and a rollout is
scored on particles anyway. So the state is projected onto whatever anchors
currently exist, stepped one coarse frame, skinned back out, and compared with
MPM's own particles.
"""
from __future__ import annotations

import math
import argparse
import os
import subprocess
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch
from tqdm import tqdm

from anchorflow import scene_setup
from anchorflow.anchor_fit import det3
from anchorflow.anchor_sparse import AnchorSparse, Traj
from anchorflow.streams import rand_rot

ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dreamphysics", default="/workspace/DreamPhysics")
ap.add_argument("--n_fit", type=int, default=115,
                 help="MPM trajectories to fit on. The student that imitates this "
                      "simulator was trained on 125 and its divergence was traced to "
                      "the narrowness of the force distribution, not to model size; "
                      "the fit was being run on three.")
ap.add_argument("--n_check", type=int, default=10)
ap.add_argument("--rt_w", default="1,1,1,0",
                 help="왕복 손실의 항별 가중 x,v,F,C. MPM 상태는 (x, v, F, C) 넷이고 "
                      "lift 가 앵커에서 그 넷을 모두 만들어 MPM 을 재시작시킨다. 위치만 "
                      "채점하면 나머지는 자유롭게 틀려도 된다 -- F 는 가우시안의 방향과 "
                      "크기를 결정해 렌더링에 직접 들어가고, F 가 어긋나면 DAgger 가 lift 로 "
                      "재시작시키는 MPM 라벨부터 오염된다.\n"
                      "C 는 기본에서 뺀다: lift 안에서 C = Fdot·F^-1 이고 F 는 앵커 위치가, "
                      "Fdot 은 앵커 속도가 정하므로, x·v·F 를 채점하면 C 는 이미 결정된다 "
                      "-- 독립적인 정보를 더하지 않는다.\n"
                      "각 항은 그 양의 전형적 크기로 나눠 무차원으로 만든 뒤 더한다.")
ap.add_argument("--frame_state", action="store_true",
                help="앵커 상태에 회전벡터 u 와 로그신축 s 를 추가한다. F 가 앵커 "
                     "위치장의 미분(형상 매칭)이 아니라 그 상태에서 조립되고, 위치 "
                     "복호도 x = sum_a w p_a + F(u,s)(X - r) 로 바뀐다. 인코더는 "
                     "frame_encode.FrameState.encode_joint -- 피팅과 같은 목적함수 "
                     "위에서 (p,u,s) 를 함께 푸는 가우스-뉴턴이라, 포락선 정리로 "
                     "출력을 detach 하고 복호만 미분해도 그래디언트가 정확하다. "
                     "형상 매칭은 F 를 항등원의 99.4%%까지밖에 못 담고, 그래서 왕복 "
                     "손실의 96%%가 F 항이었다(doc/frame_state.md).")
ap.add_argument("--fs_encoder", default="closed", choices=("closed", "gn"),
                help="프레임 인코더. 'closed' 는 가우시안별 극분해로 목표를 쪼갠 뒤 "
                     "각 공간에서 최소제곱 역을 푸는 닫힌 형태다 -- 촐레스키 두 번, "
                     "반복 없음. 인코더를 통과해 그대로 미분하므로 포락선 정리가 "
                     "필요 없고, 겹침이 늘수록 안쪽 풀이가 나빠져 그래디언트가 "
                     "틀어지는 되먹임도 없다. 'gn' 은 F 목적함수 위의 가우스-뉴턴으로 "
                     "F 잔차는 더 낮지만(0.617 대 0.895) 그 전제를 짊어진다.")
ap.add_argument("--frame_kind", default="F", choices=("F", "us"),
                help="프레임 상태의 형태. 'F' 는 앵커가 3x3 을 그대로 들고 선형 "
                     "블렌딩한다 -- 인코더가 평범한 최소제곱 한 번이고, 감김도 제약도 "
                     "없어 학생이 배울 상태의 스텝 변화 꼬리가 30 배(us 는 162 배)다. "
                     "F 잔차도 0.291 대 0.713 으로 낫다. 대가는 부피 -- 성분별 평균이라 "
                     "0.31%% 의 가우시안에서 한 축이 0.2 아래로 눌린다. "
                     "**기하와 학생의 frame_kind 는 반드시 같아야 한다** -- 다르면 "
                     "기하가 다른 복호기 기준으로 맞춰진다.")
ap.add_argument("--fs_loss", default="us", choices=("us", "F"),
                help="프레임 상태의 변형 오차를 어디서 재는가. 'us' 는 회전벡터와 "
                     "로그신축을 **따로** 잰다 -- 인코더(닫힌 형태)가 정확히 그 두 "
                     "공간의 최소제곱이므로 인코더와 목적함수가 같은 것을 푼다. "
                     "'F' 는 조립된 F 행렬의 9 성분 차이를 잰다: 물리적으로 재고 싶은 "
                     "양에 더 가깝지만 닫힌 형태 인코더가 그 최소해가 아니다. "
                     "두 항 모두 계수는 팔 길이 r_ref 로, 회전 오차든 신축 오차든 "
                     "만드는 위치 오차가 그 팔에 비례하기 때문이다.")
ap.add_argument("--fs_neff_max", type=float, default=0.0,
                help="가우시안당 유효 앵커 수의 상한. 0 이면 끈다. 프레임 상태는 F 를 "
                     "위치장에서 유도하지 않고 들고 다니므로, 형상 매칭이 가지고 있던 "
                     "'넓히면 F 가 뭉개진다'는 내장 브레이크가 없다 -- 실측으로 F 잔차가 "
                     "폭에 대해 거의 평평하고(-0.9%%) 위치 항만 크게 좋아져(-31%%) "
                     "부풀기가 순이득이 된다. 그래서 평활화를 직접 벌한다. "
                     "n_eff = 1/sum_a w_ga^2 는 지분이 실제로 몇 개 앵커에 나뉘는가이고, "
                     "지지 안에 들어오기만 한 짝을 세는 것과 달리 뭉개짐 자체를 잰다. "
                     "초기 기하가 7.9, 형상 매칭이 수렴한 곳이 7.5 이므로 10 이면 "
                     "정상 구간에서는 0 이다.")
ap.add_argument("--fs_neff_w", type=float, default=1e-4,
                help="그 힌지의 가중. 목표를 넘을 때만 물므로 사실상 제약에 가깝다.")
ap.add_argument("--fs_smax", type=float, default=0.0,
                help="앵커 반경의 **소프트** 상한 (sim.radius 배수). 0 이면 끈다. "
                     "clamp_() 의 하드 상한(s_hi=4x)은 경계에서 그래디언트가 0 이라 "
                     "거기 붙은 앵커는 스케일 학습이 멈춘다. 이쪽은 넘어선 만큼 "
                     "제곱으로 물므로 매끄럽고, 넘어설 값어치가 있으면 넘어간다. "
                     "초기값이 1.0, 형상 매칭이 수렴한 곳이 0.75 이므로 1.0 이면 "
                     "정상 구간에서 거의 0 이다.")
ap.add_argument("--fs_smax_w", type=float, default=1e-2, help="그 소프트 상한의 가중")
ap.add_argument("--fs_sreg", type=float, default=0.0,
                help="앵커 스케일의 L2 정규화. lambda * <(log_s - log R)^2> 로 **양쪽** "
                     "에서 기준 반경 R(sim.radius) 쪽으로 끌어당긴다. --fs_smax 가 "
                     "상한을 넘을 때만 무는 한쪽 힌지인 것과 달리, 이쪽은 이미 작은 "
                     "앵커까지 기준으로 끌어올리므로 정상 구간에도 편향을 준다. "
                     "형상 매칭이 실제로 수렴한 곳이 0.75R 이라 그만큼은 눌리는 셈이다.")
ap.add_argument("--fs_gn", type=int, default=6, help="프레임 인코더의 가우스-뉴턴 반복")
ap.add_argument("--fs_cg", type=int, default=20, help="그 안쪽 켤레기울기 반복")
ap.add_argument("--objective", default="roundtrip",
                 choices=("roundtrip", "dynamics"),
                 help="기하가 최적화하는 것.\n"
                      "roundtrip: MPM 상태를 앵커로 접었다 다시 편 잔차. 동역학이 "
                      "들어가지 않으므로 순수하게 **표현력**을 맞춘다. 앵커 위치·크기·"
                      "방향이 정하는 것이 바로 그것이고, 굴러가는 일은 학생과 시뮬레이터의 "
                      "몫이다. 인코더(project_ls)가 미분 가능해진 뒤에야 가능해졌다 -- "
                      "그 전에는 앵커로 가는 기울기가 끊겨 있었다.\n"
                      "dynamics: 한 번 접은 뒤 --unroll 프레임을 굴려 매 프레임 MPM 과 "
                      "비교. 표현력과 시뮬레이터 거동이 섞인다.")
ap.add_argument("--fixed_eval", type=int, default=0,
                 help="목적함수를 **고정된** 창 N 개에서도 평가한다. 학습 루프가 "
                      "찍는 loss 는 iteration 마다 다른 궤적·다른 시작 프레임을 "
                      "뽑으므로, 궤적별 변위가 150 배 벌어지는 이 데이터에서는 "
                      "같은 파라미터라도 값이 수백 배 달라진다 -- 내려가는지 "
                      "올라가는지 자체를 읽을 수 없다. 같은 창으로 재면 그 분산이 "
                      "사라지고 학습 여부가 드러난다.")
ap.add_argument("--mp_kmax", type=int, default=32)
ap.add_argument("--mp_rmin", type=float, default=1.0,
                 help="다중 포크 반경의 하한, 앵커 간격의 배수. 1.0 아래는 "
                      "학습에 나오지 않던 구석이다.")
ap.add_argument("--impulse_range", type=float, default=16.0)
ap.add_argument("--impulse_uniform", action="store_true",
                 help="세기를 로그균등 대신 같은 구간에서 균등하게 뽑는다.")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--dt_mult", type=int, default=40)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--c", type=float, default=0.25,
                 help="the kernel value that bounds an anchor's support. Smaller "
                      "reaches further and costs more pairs.")
ap.add_argument("--base_force", type=float, nargs=3, default=None,
                 help="the impulse the trajectory distribution is drawn around. "
                      "Defaults to the config's particle_impulse; a scene driven "
                      "by gravity alone (DreamPhysics's ball) has none, and the "
                      "fit still needs something to excite it with.")
ap.add_argument("--sh_degree", type=int, default=3,
                 help="spherical-harmonic degree the PLY was trained with. The "
                      "ficus scene is 3; DreamPhysics's ball is 0, and the "
                      "loader asserts rather than adapting.")
ap.add_argument("--quad", type=int, default=0,
                 help="quadratic shape matching: a Gaussian's neighbourhood is "
                      "allowed an affine PLUS quadratic deformation instead of "
                      "affine alone. The extra freedom is still DERIVED from the "
                      "anchor positions, so nothing new can drift -- which is "
                      "what ruled out carrying F. Costs the fused kernel (the "
                      "kernel is written for the linear basis) and a 9x9 solve "
                      "per Gaussian at every refresh.")
ap.add_argument("--oriented", type=int, default=0,
                 help="anchors carry an orientation and an angular velocity, and "
                      "spin under torque. Their own second moment enters the "
                      "shape matching, so a Gaussian gets its local frame from "
                      "the anchors themselves instead of inferring it from where "
                      "they sit -- the inference that recovers 17%% of MPM's F. "
                      "Three degrees of freedom in SO(3) with a restoring "
                      "torque, not the nine unconstrained ones a carried F had.")
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--softmax_w", action="store_true",
                help="가중치를 잘린 가우시안(G-c) 대신 가우시안당 최근접 k 개 위 "
                     "softmax 로. 붙는 앵커 수가 k 로 고정되고, 경계 밖 앵커에도 "
                     "그래디언트가 간다")
ap.add_argument("--softmax_k", type=int, default=16)
ap.add_argument("--polar_iters", type=int, default=6)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--accum", type=int, default=1,
                 help="how many batches to average before stepping. The graph of "
                      "an unrolled sample is 30 GB on the torch path, so --batch "
                      "cannot be raised past 2 on a 48 GB card; accumulating N "
                      "batches costs the same memory and cuts the gradient noise "
                      "by sqrt(N), which is the actual point of a bigger batch.")
ap.add_argument("--lr_pos", type=float, default=3e-4)
ap.add_argument("--lr_scale", type=float, default=1e-2)
ap.add_argument("--lr_quat", type=float, default=1e-2)
ap.add_argument("--warmup", type=int, default=80)
ap.add_argument("--refresh_every", type=int, default=20,
                 help="iterations between rebuilding the candidate pairs. The list "
                      "is padded to stay a superset in between; missing a pair is a "
                      "wrong loss, not a rough one.")
ap.add_argument("--densify_every", type=int, default=60)
ap.add_argument("--densify_until", type=float, default=0.7,
                 help="fraction of the run after which the anchor set is left alone")
ap.add_argument("--split_frac", type=float, default=0.05)
ap.add_argument("--prune_share", type=float, default=0.05)
ap.add_argument("--max_anchors", type=int, default=1024)
ap.add_argument("--dagger_every", type=int, default=40)
ap.add_argument("--dagger_traj", type=int, default=2)
ap.add_argument("--dagger_frames", type=int, default=25)
ap.add_argument("--dagger_stride", type=int, default=3)
ap.add_argument("--dagger_frac", type=float, default=0.5)
ap.add_argument("--dagger_unroll", type=int, default=1,
                 help="how many coarse frames of MPM to store behind each DAgger "
                      "state. At 1 -- what it has always been -- a DAgger sample "
                      "is one frame of supervision however long --unroll is, so "
                      "half of every batch has been the kind of signal that does "
                      "not reach a sixty-frame rollout. Raising it costs MPM at "
                      "collection time and a longer sample at training time, and "
                      "buys a DAgger state that is supervised like a trajectory "
                      "one. 0 follows --unroll.")
ap.add_argument("--dagger_cap", type=float, default=3.0)
ap.add_argument("--dagger_pool_max", type=int, default=512,
                 help="states to keep, oldest evicted. Each is a full particle state, "
                      "so an uncapped pool over a long run is tens of gigabytes. Held "
                      "on the CPU in half precision and left out of the state file: it "
                      "refills in a few collections, and writing a gigabyte every save "
                      "would cost more than regenerating it.")
ap.add_argument("--no_geom_init", action="store_true")
ap.add_argument("--params", default="shape",
                 choices=["shape", "stiff", "all"],
                 help="what the fit is allowed to move. 'shape' is where the anchors "
                      "are, which way they point and how far they reach; 'stiff' is a "
                      "per-anchor stiffness multiplier and nothing else, which is the "
                      "parameterisation an earlier attempt failed to fit at all "
                      "because the loss it used was meaningless; 'all' is both.")
ap.add_argument("--lr_stiff", type=float, default=3e-2)
ap.add_argument("--reg", type=float, default=0.0,
                 help="pull the parameters back toward where they started. The knob "
                      "on how much the discretisation is allowed to become: at zero "
                      "the fit reaches its best rollout a third of the way through "
                      "and then doubles it while the loss keeps falling.")
ap.add_argument("--init_from", default=None,
                 help="start from a saved fit rather than from the sampled anchors. "
                      "With --iters 0 this scores an existing one on the same "
                      "held-out set as everything else, which is the only way three "
                      "parameterisations fitted at different times become comparable.")
ap.add_argument("--out", default=None)
ap.add_argument("--state", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--save_every", type=int, default=10)
ap.add_argument("--traj_cache", default=None)
ap.add_argument("--r2", default=None)
ap.add_argument("--eval_every", type=int, default=40)
ap.add_argument("--cfl_frac", type=float, default=0.05,
                 help="how far an anchor may travel in one substep, as a fraction of "
                      "the anchor spacing. The starting discretisation sits at 1.4%%; "
                      "the configurations the fit blew up on reach 50%%.")
ap.add_argument("--lambda_cfl", type=float, default=1.0)
ap.add_argument("--loss", choices=["window", "mwrmsd"], default="mwrmsd",
                 help="how a frame is scored. 'window' divides by how far MPM "
                      "moved from the start of that window, so whichever window "
                      "happened to move least is amplified most -- and the fit "
                      "specialises to that. Measured against a common reference: "
                      "no fit 21.4%, window-normalised fit 29.4-33.0% (worse than "
                      "not fitting), mwrmsd fit 17.4-20.9%. Same data, same model, "
                      "only the normalisation differs, and it flips the sign of "
                      "what training is worth. 'window' is kept for reproducing "
                      "the older numbers.")
ap.add_argument("--acc_blend", type=float, default=0.0,
                 help="how much of the deformation gradient to take from one "
                      "carried forward in time, MPM style, rather than from shape "
                      "matching against rest. Shape matching sees nothing finer "
                      "than the anchor neighbourhood and averages with weights "
                      "fixed at rest; on ficus the F it produces recovers 17% of "
                      "MPM's deviation and the forces are off by 158x. Zero keeps "
                      "the old behaviour.")
ap.add_argument("--lambda_acc", type=float, default=0.0,
                 help="weight on matching MPM's frame-to-frame acceleration. The "
                      "position term alone lets the fit reach the right place by a "
                      "rougher route: on ficus it halves mwRMSD while impulse "
                      "irregularity moves away from MPM's. Zero keeps the old loss.")
ap.add_argument("--unroll", type=int, default=1,
                 help="coarse frames per training sample. One frame is what the fit "
                      "has always optimised and it does not transfer: the one-step "
                      "error halves while the sixty-frame rollout does not move. "
                      "Unrolling makes the loss measure what is actually wanted, at "
                      "the cost of that many times the compute per sample.")
ap.add_argument("--encoder", default="ls", choices=("ls",),
                 help="앵커로 접는 방법. skin 의 최소제곱 역(project_ls) 하나만 쓴다. 예전 기본값이던 가중평균('avg')은 디코더의 전치이지 역이 아니라, 표현 가능한 상태를 왕복시켜도 1~2%%를 잃었다 -- 인용해 온 '표현 하한'의 2/3가 그 편향이었다(doc/encoder_project.md). 선택지에서 뺐고, AnchorSparse.project 는 옛 수치를 재현할 때를 위해 라이브러리에만 남겨 둔다.")
ap.add_argument("--grad_frames", type=int, default=0,
                 help="truncate the gradient to the last N coarse frames of the "
                      "unroll. The LOSS is unchanged -- all --unroll frames are "
                      "still scored -- but the state is detached before each "
                      "earlier frame, so no gradient travels more than N frames "
                      "back. --unroll changes the horizon and the chain length "
                      "together; this changes only the chain, which is what "
                      "separates a gradient pathology from a loss whose optimum "
                      "really does look like this. 0 leaves the full chain.")
ap.add_argument("--eval_rollout", type=int, default=0,
                 help="score the held-out set on a full rollout at each evaluation "
                      "rather than on one step, so the number being tracked is the "
                      "number being asked for")
ap.add_argument("--edge_force", action="store_true",
                 help="앵커 쌍이 직접 주고받는 학습되는 중심력을 켠다. 표현력이 "
                      "아니라 동역학이 병목이라는 측정 결과에 대응하는 항이다.")
ap.add_argument("--edge_only", action="store_true",
                 help="가우시안 응력 경로를 끄고 앵커 간선력만 쓴다. 해석적 선형 "
                      "스프링을 물리항에 최소제곱으로 맞춰 출발점을 잡는다.")
ap.add_argument("--anchor_stress", action="store_true",
                 help="변형구배와 응력을 앵커 이웃만으로 계산한다. 중심력과 달리 "
                      "전단에 저항한다. --edge_only 와 함께 쓰면 가우시안은 힘 "
                      "계산에서 완전히 빠진다.")
ap.add_argument("--frame_dyn", action="store_true",
                 help="앵커가 회전·신축을 상태로 들고 F 를 거기서 조립한다. F 를 "
                      "위치의 미분으로 얻는 한 표현 잔차가 40%% 아래로 안 내려가는데, "
                      "이 표현은 31%% 까지 간다(측정). --anchor_stress 를 함께 켠다.")
ap.add_argument("--frame_I", type=float, default=0.0,
                 help="log(프레임 관성). 작을수록 (o,s) 가 F_shape 를 빨리 따라간다.")
ap.add_argument("--lam_couple", type=float, default=0.0,
                 help="log(결합 강성). F^frame 을 앵커 그래프의 F^shape(p) 로 "
                      "끌어당기는 항. 이것이 앵커 위치에 힘을 준다.")
ap.add_argument("--astress_k", type=int, default=16)
ap.add_argument("--astress_eig", type=float, default=0.1,
                 help="앵커 응력의 릿지 비율. 이웃이 거의 한 평면에 놓인 앵커는 "
                      "B 가 한 방향으로 랭크를 잃는데, 0.02 로는 그 방향을 못 받친다.")
ap.add_argument("--astress_jmax", type=float, default=20.0)
ap.add_argument("--finv_ridge", type=float, default=1e-6,
                 help="F^-T 의 릿지 비율. 거의 납작해진 요소에서 부피항이 폭주하는 "
                      "것을 막는다. 1e-3 이면 F^-T 가 1e3 을 넘지 않는다.")
ap.add_argument("--mass_freeze", type=float, default=0.02,
                 help="질량이 중앙값의 이 비율 아래인 앵커를 적분에서 제외한다. "
                      "질량을 부풀리는 --mass_floor 와 달리 질량 분포를 안 건드린다.")
ap.add_argument("--mass_floor", type=float, default=0.0,
                 help="앵커 질량의 바닥, 중앙값 대비 비율. 질량이 거의 0 인 앵커가 "
                      "이웃에게서 멀쩡한 힘을 받아 1e14 로 가속하는 것을 막는다.")
ap.add_argument("--astress_polar", type=int, default=8,
                 help="앵커 응력의 극분해 뉴턴 반복 수. 커널 호출의 절반을 "
                      "차지하지만 줄이면 회전 추정이 거칠어진다.")
ap.add_argument("--compile", action="store_true",
                 help="앵커 응력 forward 를 torch.compile 로 묶는다. 이 경로는 "
                      "연산이 아니라 커널 호출 수에 묶여 있어 효과가 크다.")
ap.add_argument("--nan_trace", action="store_true",
                 help="서브스텝마다 중간값의 유한성을 검사해, 무한이 처음 나타난 "
                      "양과 서브스텝 번호를 로그에 남긴다.")
ap.add_argument("--nan_rollback", action="store_true",
                 help="손실이 무한해지면 그냥 건너뛰지 않고 마지막 정상 파라미터로 "
                      "되돌린 뒤 학습률을 낮춘다. 건너뛰기만 하면 파라미터가 그대로라 "
                      "다음 이터도 같은 자리에서 터져 영영 빠져나오지 못한다.")
ap.add_argument("--nan_lr_decay", type=float, default=0.5)
ap.add_argument("--nan_lr_floor", type=float, default=0.02,
                 help="되돌림이 반복되어 학습률이 초기값의 이 비율 아래로 내려가면 멈춘다")
ap.add_argument("--cfl_max", action="store_true",
                 help="cfl 벌점을 앵커 평균이 아니라 최댓값으로 매긴다.")
ap.add_argument("--lr_astress", type=float, default=3e-3,
                 help="힘 스텐실 폭 log_h 와 강성 log_ka 의 학습률. 스키닝의 "
                      "log_s/quat 과는 별개 파라미터다.")
ap.add_argument("--edge_k", type=int, default=16)
ap.add_argument("--edge_hidden", type=int, default=32)
ap.add_argument("--lr_edge", type=float, default=1e-3)
ap.add_argument("--no_guards", action="store_true",
                 help="run without the scale bounds, the polar ridge, the scatter "
                      "floor or the skip, and stop at the first non-finite quantity "
                      "with a substep-by-substep replay. For finding out what the "
                      "failure is rather than surviving it.")
ap.add_argument("--keep_c", action="store_true",
                 help="궤적 캐시에 MPM 의 C 까지 저장한다. 왕복 손실은 C 를 채점하지 "
                      "않으므로(x·v·F 가 정하는 종속량) 기본은 F 만 저장해 용량을 "
                      "절반으로 줄인다. fine_loss 를 쓸 때만 필요하다.")
ap.add_argument("--n_fine", type=int, default=40,
                 help="trajectories for which MPM's own F and C are stored as well, so "
                      "a fine-step sample can restart MPM from its own state instead of "
                      "from a lifted one. Eighteen more floats per particle per frame is "
                      "377 MB a trajectory, so this is not all of them.")
ap.add_argument("--fine_steps", type=int, default=0,
                 help="score every SUBSTEP over this many, instead of every coarse "
                      "frame over --unroll of them. 480 differentiable substeps to "
                      "produce twelve numbers of supervision is what made an iteration "
                      "cost two minutes; MPM steps at the same 1e-4, so the two can be "
                      "walked together and compared throughout. The horizon then has to "
                      "come from DAgger rather than from rolling forward, which is what "
                      "it is for.")
ap.add_argument("--fine_end", type=int, default=0,
                 help="anneal the fine horizon from --fine_steps down to this over the "
                      "run. A long horizon shows error compounding, which is what a "
                      "short one cannot see and what the fit exists to remove; a short "
                      "one is cheap. Starting long and ending short spends the expensive "
                      "steps where the discretisation is still far off.")
ap.add_argument("--beta1", type=float, default=0.9,
                 help="Adam's momentum. The gradient between two samples has a "
                      "cosine of 0.15 (exe/probe_grad_noise.py), so what reaches "
                      "the step is mostly which trajectory was drawn; a longer "
                      "momentum averages more of that away before it is used.")
ap.add_argument("--beta2", type=float, default=0.999,
                 help="Adam's second moment. With a loss that ranges over 0.06 "
                      "to 4.19 between samples, a short window rescales the step "
                      "by whatever the last few draws happened to be.")
ap.add_argument("--clip", type=float, default=1.0,
                 help="gradient norm clip; 0 disables. At 1.0 it binds on every "
                      "iteration, which means the norm carries no information "
                      "into the step at all.")
ap.add_argument("--lr_warmup", type=int, default=0,
                 help="ramp every learning rate linearly over this many "
                      "iterations, so the first steps -- taken from the noisiest "
                      "point, before Adam's moments mean anything -- are small.")
ap.add_argument("--lr_cosine", type=int, default=0,
                 help="anneal every learning rate to zero over --iters on a "
                      "cosine. Without it the step size never shrinks, and once "
                      "the gradient is noise the parameters random-walk at full "
                      "stride: at the defaults that is 14%% of the anchor "
                      "spacing, 28%% of the anchor size and a factor 2.1 in "
                      "stiffness over 600 iterations.")
ap.add_argument("--ema", type=float, default=0.0,
                 help="evaluate an exponential moving average of the parameters "
                      "rather than the parameters themselves. Averages the walk "
                      "out without touching the optimisation; 0 disables.")
ap.add_argument("--lr_decay_at", type=int, default=0,
                 help="drop every learning rate by --lr_decay_by at this iteration. "
                      "Five fits have reached their best rollout between 125 and 250 "
                      "and never improved after, whatever the regularisation or the "
                      "length; if a smaller step keeps moving, that plateau is the "
                      "optimiser's and not the discretisation's.")
ap.add_argument("--lr_decay_by", type=float, default=0.1)
ap.add_argument("--grad_log", default=None,
                 help="write the gradient norm per parameter group, and the cosine "
                      "between consecutive iterations, to this file. A cosine near "
                      "zero means the gradient coming back through 480 stiff substeps "
                      "is noise, which is a different problem from a bad objective.")
ap.add_argument("--final_rollout", type=int, default=3,
                 help="impulses to roll out fully against MPM at the end")
args = ap.parse_args()

sys.path.insert(0, args.dreamphysics)
import warp as wp

from anchorflow.mpm_teacher import MPMTeacher

dev = "cuda"
torch.manual_seed(0)
wp.init()

sc = scene_setup.build(args.ply, args.config, args.n_anchors, args.K, device=dev,
                        frozen_weights=True, rot_fallback=True,
                        eig_floor=args.eig_floor, sh_degree=args.sh_degree)
T = MPMTeacher(sc)
base = None
for bc in sc.cfg.get("boundary_conditions", []):
    if bc["type"] == "particle_impulse":
        base = torch.tensor(bc["force"], device=dev)
if args.base_force is not None:
    base = torch.tensor(args.base_force, device=dev)
if base is None:
    raise SystemExit("this scene has no particle_impulse; pass --base_force")

fit = AnchorSparse(sc, c=args.c, eig_floor=args.eig_floor,
                    polar_iters=args.polar_iters, cfl_frac=args.cfl_frac,
                    quad=bool(args.quad),
                    oriented=bool(args.oriented),
                    softmax_w=bool(args.softmax_w),
                    softmax_k=args.softmax_k).to(dev)
fit.acc_blend = args.acc_blend
GRID_LIM = 2.0
# mass per Gaussian, for the mass-weighted objective. sc.volume is zero on the
# ones no material was assigned to, which is what we want: they ride along.
# the loss compares the material Gaussians only, so the weights have to be the
# same subset -- sc.volume covers the whole cloud
MASS_G = sc.volume[sc.keep].clone() if sc.volume.shape[0] != int(sc.keep.sum()) \
    else sc.volume.clone()
if args.no_guards:
    fit.s_lo, fit.s_hi, fit.polar_ridge = 1e-9, 1e9, 0.0
if args.anchor_stress:
    from anchorflow.anchor_stress import AnchorStress
    fit.astress = AnchorStress(k=args.astress_k, eig_floor=args.astress_eig, dev=dev)
    fit.astress.J_max = args.astress_jmax
    _n = fit.astress.rebuild(fit)
    with torch.no_grad():
        _c = fit.prepare()
        _p = fit.pos + 0.02 * fit.sim_radius * torch.randn_like(fit.pos)
        _f = fit.force(_p, *_c[:5])
        _g = fit.astress(_p, fit.pos)
        _s = float((_f * _g).sum() / (_g * _g).sum().clamp(min=1e-30))
        _rel = float((_f - _s * _g).norm() / _f.norm().clamp(min=1e-30))
    fit.astress.polar_iters = args.astress_polar
    fit.astress._owner = [fit]
    if args.compile:
        try:
            # mode="reduce-overhead"(CUDA graphs)는 서브스텝이 이전 출력을
            # 물고 있어서 "overwritten by a subsequent run" 으로 죽는다.
            fit.astress.forward = torch.compile(fit.astress.forward)
            print("[setup] 앵커 응력에 torch.compile 적용")
        except Exception as _e:
            print(f"[setup] torch.compile 실패, 그대로 간다: {type(_e).__name__}")
    print(f"[setup] 앵커 응력 간선 {_n}개 (앵커당 {_n/max(fit.M,1):.1f}), "
          f"극분해 {args.astress_polar}회, 물리항 대비 배율 {_s:.4g}, "
          f"상대 잔차 {_rel:.3f}")

if args.edge_force:
    from anchorflow.edge_force import EdgeForce
    fit.edge = EdgeForce(k=args.edge_k, hidden=args.edge_hidden, dev=dev)
    n_e = fit.edge.rebuild(fit.pos, fit.sim_radius, fit.dt)
    print(f"[setup] 학습되는 앵커 간선 {n_e}개 "
          f"(앵커당 {2*n_e/max(fit.M,1):.1f}), 출력 0 초기화")
    # 앵커 응력이 이미 바닥을 깔고 있으면 스프링은 군더더기다. 스프링은
    # 응력항이 아예 없을 때(중심력만 쓸 때)의 출발점으로만 쓴다.
    if args.edge_only and not args.anchor_stress:
        fit.edge.spring = True
        with torch.no_grad():
            _c = fit.prepare()
            _p = fit.pos + 0.02 * fit.sim_radius * torch.randn_like(fit.pos)
            _ke = fit.edge.calibrate(_p, fit.force(_p, *_c[:5]))
            _g = fit.edge.spring_basis(_p) * _ke
            _f = fit.force(_p, *_c[:5])
            _rel = float((_f - _g).norm() / _f.norm().clamp(min=1e-30))
        print(f"[setup] 스프링 강성 {_ke:.4g}, 물리항과의 상대 잔차 {_rel:.3f}")

if args.nan_trace:
    # 평상시에는 꺼두고, 손실이 무한해진 그 표본만 no_grad 로 다시 돌리며 켠다
    fit.nan_trace = False
    print("[setup] NaN/Inf 추적 준비 (발생 시 해당 표본만 재현하며 켠다)")
fit.finv_ridge = args.finv_ridge
fit.mass_floor = args.mass_floor
fit.mass_freeze = args.mass_freeze
if args.mass_floor > 0:
    print(f"[setup] 앵커 질량 바닥 = 중앙값 x {args.mass_floor}")
if args.mass_freeze > 0:
    print(f"[setup] 질량 < 중앙값 x {args.mass_freeze} 인 앵커는 적분에서 제외")
if args.cfl_max:
    fit.cfl_agg = "max"
    print("[setup] cfl 벌점을 최댓값으로 매긴다")
if args.frame_dyn:
    from anchorflow.anchor_frame import FrameDynamics
    assert fit.astress is not None, "--frame_dyn 은 --anchor_stress 가 필요하다"
    fit.fdyn = FrameDynamics(dev).size_to(fit.M, args.lam_couple, args.frame_I)
    fit.edge_only = True
    print(f"[setup] 프레임 동역학 켬 (앵커별 파라미터 4x{fit.M}, "
          f"결합강성 {float(fit.fdyn.log_lam.exp().mean()):.3f})")

if args.edge_only:
    fit.edge_only = True
    print("[setup] 가우시안 응력 경로 끔 -- 힘 계산에 가우시안이 참여하지 않는다")

SHAPE = ("pos", "log_s", "quat", "log_amp")
STIFF = ("log_k",)
TRAIN = SHAPE if args.params == "shape" else (
    STIFF if args.params == "stiff" else SHAPE + STIFF)
if args.edge_only and "log_k" in TRAIN:
    # 가우시안 응력 경로가 꺼져 있으면 log_k 는 어떤 힘에도 닿지 않는다.
    TRAIN = tuple(n for n in TRAIN if n != "log_k")
    print("[setup] --edge_only 이므로 log_k 는 학습에서 제외 (힘에 안 닿음)")
LRS = {"pos": args.lr_pos, "log_s": args.lr_scale, "quat": args.lr_quat,
       "log_k": args.lr_stiff, "log_amp": args.lr_scale}


print(f"[setup] fitting {args.params}: {', '.join(TRAIN)}")
print(f"[setup] {fit.M} anchors, {fit.N} material Gaussians, support at "
      f"{fit.mahal_radius:.2f} sigma, {fit.pair_g.shape[0]} pairs "
      f"({fit.pair_g.shape[0] / fit.N:.1f} anchors per Gaussian)")
if args.no_guards:
    fit.B_ref.zero_()
    fit.set_B_ref = lambda *a, **k: 0.0
if args.init_from:
    _b = torch.load(args.init_from, map_location=dev, weights_only=False)
    fit._rebuild(_b["pos"].to(dev), _b["quat"].to(dev), _b["log_s"].to(dev),
                  _b["log_k"].to(dev) if "log_k" in _b else None)
    print(f"[init] {args.init_from} at iteration {_b.get('iter')}, {fit.M} anchors")
elif not args.no_geom_init:
    thin = fit.init_from_geometry()
    s_ = fit.log_s.exp()
    print(f"[init] oriented; axis ratio median "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, {thin} left round, "
          f"{fit.pair_g.shape[0]} pairs")


_LSF = {"w": None, "fac": None}


def ls_fac(cache):
    """the Cholesky of C^T C for this rest state, reused across one iteration"""
    key = cache[0].data_ptr(), int(fit.M), int(fit.pair_g.shape[0])
    if _LSF["w"] != key:
        _LSF["w"], _LSF["fac"] = key, fit.ls_factor(cache)
    return _LSF["fac"]


from anchorflow.frame_encode import polar_target as frame_polar_target

_FS = [None, None]


def frame_state():
    """짝 구조가 바뀌면(densify/prune) 다시 만든다."""
    key = (int(fit.pair_g.shape[0]), int(fit.M), fit.pair_g.data_ptr())
    if _FS[0] != key:
        from anchorflow.frame_encode import FrameState
        _FS[0], _FS[1] = key, FrameState(fit.pair_g, fit.pair_a, fit.N, fit.M, MASS_G)
    return _FS[1]


def enc(x, cache):
    return fit.project_ls(x, cache, ls_fac(cache))


def enc_v(v, cache):
    return fit.project_v_ls(v, cache, ls_fac(cache))


@torch.no_grad()
def mpm_states(force, keep_fc=False):
    """MPM's own particles and velocities, frame by frame.

    With keep_fc, its deformation gradient and affine velocity too. Those are
    what MPM needs to be restarted, and storing them is the difference between
    walking alongside MPM from its own state and having to reconstruct one by
    lifting ours -- which is lossy, and which MPM refuses outright often enough
    that three samples in four were being thrown away.
    """
    cache = fit.prepare()
    dv = fit.impulse_dv(force, cache)
    v0 = torch.zeros(fit.N, 3, device=dev).index_add_(
        0, fit.pair_g, cache[0].unsqueeze(-1) * dv[fit.pair_a])
    T._set(T.pos_m.clone(), v0.contiguous(), T.eye.clone(), torch.zeros_like(T.eye))
    xs = [T.pos_m.to(torch.float16).cpu()]
    vs = [v0.to(torch.float16).cpu()]
    fs_ = [T.eye.to(torch.float16).cpu()] if keep_fc else None
    cs_ = ([torch.zeros_like(T.eye).to(torch.float16).cpu()]
           if (keep_fc and args.keep_c) else None)
    for _ in range(args.frames):
        for k in range(args.dt_mult):
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if (k + 1) % 8 == 0 and not T._in_domain():
                return None
        xs.append(T.solver.export_particle_x_to_torch().to(torch.float16).cpu())
        vs.append(T.solver.export_particle_v_to_torch().to(torch.float16).cpu())
        if keep_fc:
            fs_.append(T.solver.export_particle_F_to_torch().reshape(-1, 9)
                       .to(torch.float16).cpu())
            if cs_ is not None:
                cs_.append(T.solver.export_particle_C_to_torch().reshape(-1, 9)
                           .to(torch.float16).cpu())
    out = [Traj(torch.stack(xs)), Traj(torch.stack(vs))]
    if keep_fc:
        # C 를 안 담을 때도 자리를 채워, 길이 4 를 보는 호출부가 그대로 동작한다
        out += [Traj(torch.stack(fs_)),
                Traj(torch.stack(cs_)) if cs_ is not None else None]
    return tuple(out)


def draw(g, n):
    """임펄스는 (포크 개수 K, 반경 r) 로만 결정된다.

    옛 계열 분할 -- 균일 / 힘장 / 포크를 따로 뽑던 것 -- 은 제거했다. 평가 격자가
    K x r 두 축이고 학생도 같은 계열로 학습하므로, 기하를 다른 분포로 맞추면 그
    학생의 수치를 무엇에 귀속시킬지 알 수 없게 된다. 이 계열 하나가 셋을 모두
    포함한다: K=1 에 r 이 물체 크기면 균일, K 가 크면 힘장에 해당하는 국소 구조,
    K=1 에 r 이 작으면 포크.

    K 와 r 은 각각 로그균등 -- 둘 다 자릿수를 걸쳐 변하므로 스케일이 중요하다.
    """
    out = []
    while len(out) < n:
        uk = torch.rand(1, device=dev, generator=g).item()
        kk = max(1, int(round(args.mp_kmax ** uk)))
        ur = torch.rand(1, device=dev, generator=g).item()
        lo, hi = sc.sim.radius * args.mp_rmin, sc.extent
        rad = lo * ((hi / lo) ** ur)
        us = torch.rand(1, device=dev, generator=g).item()
        _lo, _hi = 0.5, 0.5 * args.impulse_range
        _f = (_lo + us * (_hi - _lo)) if args.impulse_uniform \
            else (_lo * ((_hi / _lo) ** us))
        mag = base.norm().item() * _f
        out.append(sc.random_multi_poke(g, kk, rad, mag))
    return out[:n]


KEY = (f"{args.n_fit}_{args.n_check}_{args.frames}_{args.dt_mult}_{args.c}"
       f"_fc{args.n_fine}_mp{args.mp_kmax}_{args.mp_rmin}"
       + (f"_unif{args.impulse_range:g}" if args.impulse_uniform else ""))
FIT = CHK = FORCES = None
if args.traj_cache and args.r2 and not os.path.exists(args.traj_cache):
    # the trajectories are twenty minutes of MPM and they were not being
    # mirrored, so an instance going away cost them every time even though the
    # fit state itself survived. Pulled back before anything else is decided.
    print(f"[data] fetching {os.path.basename(args.traj_cache)} from {args.r2}",
          flush=True)
    os.system(f"rclone copy {args.r2}/{os.path.basename(args.traj_cache)} "
               f"{os.path.dirname(args.traj_cache) or '.'} 2>/dev/null")
if args.traj_cache and os.path.exists(args.traj_cache):
    blob = torch.load(args.traj_cache, map_location=dev, weights_only=False)
    if blob.get("key") == KEY:
        FIT, CHK, FORCES = blob["fit"], blob["chk"], blob["forces"]
        print(f"[data] {args.traj_cache}")
if FIT is None:
    g1 = torch.Generator(device=dev); g1.manual_seed(4242)
    ff = draw(g1, args.n_fit)
    FIT = [s for s in (mpm_states(f, keep_fc=(i < args.n_fine))
                       for i, f in enumerate(tqdm(ff, desc="MPM fit", ncols=90)))
           if s is not None]
    g2 = torch.Generator(device=dev); g2.manual_seed(31337)
    fc = draw(g2, args.n_check)
    CHK = [s for s in (mpm_states(f) for f in tqdm(fc, desc="MPM check", ncols=90))
           if s is not None]
    FORCES = {"fit": ff[:len(FIT)], "chk": fc[:len(CHK)]}
    if args.traj_cache:
        torch.save({"fit": FIT, "chk": CHK, "forces": FORCES, "key": KEY}, args.traj_cache)
        if args.r2:
            print(f"[data] mirroring the trajectories to {args.r2}", flush=True)
            os.system(f"rclone copy {args.traj_cache} {args.r2} 2>/dev/null &")
print(f"[data] {len(FIT)} fit and {len(CHK)} held-out MPM trajectories")


def one_step(x0, v0, cache):
    """MPM particle state -> one coarse frame of this simulator -> particles,
    and how far the substeps were from being able to carry it"""
    p = enc(x0, cache)
    v = enc_v(v0, cache)
    p, _, cfl = fit.rollout(p, v, args.dt_mult, cache)
    return fit.gaussian_pos(p, cache), cfl


# 항별 고정 스케일. 한 번만 잡고 학습 내내 바꾸지 않는다.
#   dt_c   속도 오차를 한 코어스 프레임 적분하면 그만큼 위치가 어긋난다
#   r_ref  디코드가 x = cc + F·(Xc - rc) 이므로 dF 가 만드는 위치 오차는 dF·(Xc - rc).
#          그 팔 길이의 RMS 다. rc 는 그 가우시안을 잡은 앵커들의 가중 중심(정지 상태).
DT_C = args.dt_mult * sc.sub_dt
with torch.no_grad():
    _rc0 = fit.prepare()[1]
    R_REF = float((fit.Xc - _rc0).norm(dim=-1).pow(2).mean().sqrt())
    del _rc0
print(f"[setup] 왕복 손실 스케일: dt_c {DT_C:.5g}, r_ref {R_REF:.5g}, "
      f"grid_lim {GRID_LIM:g}", flush=True)
RT_W = [float(x) for x in args.rt_w.split(",")]
while len(RT_W) < 4:
    RT_W.append(0.0)


def _mw(e2):
    """질량가중 RMS -- 위치 항과 같은 축약."""
    return ((MASS_G * e2).sum() / MASS_G.sum()).sqrt()


def roundtrip_loss(x0, v0, fc0, cache):
    """MPM 상태 (x, v, F) 를 앵커로 접었다 편 잔차.

    lift 가 앵커에서 (x, v, F, C) 를 모두 만들어 MPM 을 재시작시키므로, 표현력을
    재려면 위치만으로는 모자라다. C 는 Fdot·F^-1 이라 x·v·F 가 정하는 종속량이다.

    **항별 정규화는 전부 고정 상수다.** 각 오차를 "그것이 유발하는 위치 오차" 로
    환산한 뒤 grid_lim 으로 나눈다:

        dx -> dx                    위치 그대로
        dv -> dv · dt_c             한 코어스 프레임을 적분한 만큼
        dF -> dF · r_ref            앵커가 뻗는 거리만큼 (r_ref = RMS |Xc - rc|)

    표본마다 달라지는 양(그 프레임의 RMS 속도 같은 것)으로 나누면 안 된다 -- 거의
    정지한 프레임에서 분모가 작아져 그 표본만 증폭되고, 학습이 거기에 특화된다.
    같은 함정이 평가 지표(자체 변위 정규화)와 학습 손실(창별 정규화) 양쪽에서
    이미 한 번씩 나왔다.
    """
    w_, rc_, q_, Binv_, blocked_, _ = cache
    if args.frame_state and fc0 is not None:
        return frame_roundtrip(x0, v0, fc0, cache)
    p = enc(x0, cache)
    v = enc_v(v0, cache) if RT_W[1] or RT_W[3] else None
    total = 0.0
    if RT_W[0]:
        got = fit.gaussian_pos(p, cache)
        total = total + RT_W[0] * _mw((got - x0).pow(2).sum(-1)) / GRID_LIM
    if RT_W[1] or RT_W[2] or RT_W[3]:
        xl, vl, Fl, Cl = fit.lift(p, v if v is not None else torch.zeros_like(p), cache)
        if RT_W[1]:
            total = total + RT_W[1] * _mw((vl - v0).pow(2).sum(-1)) * DT_C / GRID_LIM
        if fc0 is not None:
            F0, C0 = fc0
            F0 = F0.reshape(-1, 9).to(xl.dtype)
            C0 = None if C0 is None else C0.reshape(-1, 9).to(xl.dtype)
            if RT_W[2]:
                total = total + RT_W[2] * _mw((Fl - F0).pow(2).sum(-1)) * R_REF / GRID_LIM
            if RT_W[3] and C0 is not None:
                # C 는 dF/dt · F^-1 이라 dt_c·r_ref 로 위치 오차 단위가 된다
                total = total + RT_W[3] * _mw((Cl - C0).pow(2).sum(-1)) * (
                    DT_C * R_REF / GRID_LIM)
    return total


def _frame_penalty(w_):
    """부풀림 브레이크. 프레임 상태는 F 를 들고 다녀 형상 매칭이 가지고 있던
    '넓히면 F 가 뭉개진다'는 내장 제동이 없다 -- 넣지 않으면 짝이 3.3M 에서 13.8M 로
    분다. 두 복호 경로가 같은 것을 쓰도록 함수로 뺐다."""
    out = 0.0
    if args.fs_neff_max > 0:
        w2 = torch.zeros(fit.N, device=dev, dtype=w_.dtype).index_add_(
            0, fit.pair_g, w_ * w_)
        neff = 1.0 / w2.clamp(min=1e-20)
        ex = (neff / args.fs_neff_max - 1.0).clamp(min=0)
        out = out + args.fs_neff_w * ((MASS_G * ex.pow(2)).sum() / MASS_G.sum())
    if args.fs_sreg > 0:
        out = out + args.fs_sreg * (
            fit.log_s - float(torch.tensor(fit.sim_radius).log())).pow(2).mean()
    if args.fs_smax > 0:
        r = fit.log_s.exp() / fit.sim_radius
        out = out + args.fs_smax_w * (r / args.fs_smax - 1.0).clamp(min=0).pow(2).mean()
    return out


def frame_roundtrip(x0, v0, fc0, cache):
    """앵커가 (p, v, u, s) 를 들 때의 왕복 손실.

    인코더는 grad 없이 (p, u, s) 를 푼 뒤 detach 한다. 포락선 정리다 --
    L(theta) = min_z Phi(z, theta) 의 미분은 최적점에서의 편미분과 같으므로,
    z* 를 상수로 두고 **복호만** 다시 미분하면 그래디언트가 정확하다. 이것이
    성립하려면 인코더가 푸는 목적함수가 여기서 재는 것과 같아야 하고, 그래서
    encode_joint 는 F 만이 아니라 (x, F) 를 함께 푼다.

    고정 앵커는 예외다. 안쪽 풀이는 그 앵커들의 p 를 fit.pos 에 고정하고 움직이지
    않는데, **그 제약 자체가 학습 대상 파라미터에 의존한다.** 그러면 포락선 정리의
    전제(제약 집합이 theta 와 무관)가 깨지고 라그랑주 항이 빠진다 -- 실측으로
    |dPhi/dz| 가 초기의 0.317 배에서 더 안 떨어졌고(반복을 25 배 늘려도 그대로),
    그게 정확히 고정 부분공간의 그래디언트였다. 방향미분도 1.5~1.7 배 어긋났다.
    고정 항목을 detach 된 사본이 아니라 **살아 있는 fit.pos** 에서 다시 채워 넣으면
    그 항이 그대로 복원된다.

    속도 항만은 예외다. 속도 복호 v = sum_a w v_a + Fdot·(X - r) 의 Fdot 은
    앵커 속도의 형상 매칭이라 (u, s) 와 무관하고, project_v_ls 가 그 복호의
    정확한 최소제곱이다 -- 그 항에 대해서도 포락선 정리가 따로 성립한다.
    """
    FS = frame_state()
    w_, rc_ = cache[0], cache[1]
    Yg = fit.Xc - rc_
    F0 = fc0[0].reshape(-1, 9).to(w_.dtype)
    F0m = F0.view(-1, 3, 3)
    c_x = RT_W[0] / GRID_LIM
    c_F = RT_W[2] * R_REF / GRID_LIM
    if args.frame_kind == "F":
        # 자유 F: 인코더도 복호도 선형이라 통째로 미분 가능하다. 목표가 F^MPM
        # 자체이므로 극분해도 필요 없다.
        pj, Fa = FS.encode_closed_F(x0, F0m, w_, Yg,
                                     fixed=fit.fixed, p_fix=fit.pos)
        xh, Ff = FS.decode_x_F(pj, Fa, w_, Yg)
        total = 0.0
        if RT_W[0]:
            total = total + RT_W[0] * _mw((xh - x0).pow(2).sum(-1)) / GRID_LIM
        if RT_W[2]:
            total = total + RT_W[2] * _mw(
                (Ff.reshape(-1, 9) - F0).pow(2).sum(-1)) * R_REF / GRID_LIM
        if RT_W[1]:
            vl = fit.lift(pj, enc_v(v0, cache), cache)[1]
            total = total + RT_W[1] * _mw((vl - v0).pow(2).sum(-1)) * DT_C / GRID_LIM
        return total + _frame_penalty(w_)
    if args.fs_encoder == "closed":
        # 인코더가 닫힌 형태라 detach 하지 않는다 -- 그냥 통과해서 미분한다.
        # 극분해 목표는 F^MPM 만의 함수라 상수이고, theta 의존은 전부 선형 풀이에
        # 들어 있다.
        pj, uj, sj = FS.encode_closed(x0, F0m, w_, Yg,
                                       fixed=fit.fixed, p_fix=fit.pos)
    else:
        with torch.no_grad():
            pj, uj, sj, _ = FS.encode_joint(
                x0, F0m, w_.detach(), Yg.detach(), c_x=c_x, c_F=c_F,
                fixed=fit.fixed, p_fix=fit.pos.detach(),
                iters=args.fs_gn, cg_iters=args.fs_cg)
        pj = torch.where(fit.fixed.unsqueeze(-1), fit.pos, pj)
    xh, Ff = FS.decode_x(pj, uj, sj, w_, Yg)
    total = 0.0
    if RT_W[0]:
        total = total + RT_W[0] * _mw((xh - x0).pow(2).sum(-1)) / GRID_LIM
    if RT_W[2]:
        if args.fs_loss == "us":
            # 회전과 신축을 각자의 공간에서 따로 잰다. 목표 (u*, s*) 는 F^MPM 만의
            # 함수라 상수다. 계수는 둘 다 r_ref -- 회전 오차가 만드는 위치 오차가
            # |du x (X-r)|, 신축 오차가 |ds·(X-r)| 로 같은 팔에 비례한다.
            with torch.no_grad():
                ug_t, sg_t = frame_polar_target(F0m)
            ug, sg = FS.blend(uj, sj, w_)
            total = total + RT_W[2] * (
                _mw((ug - ug_t).pow(2).sum(-1)) + _mw((sg - sg_t).pow(2).sum(-1))
            ) * R_REF / GRID_LIM
        else:
            total = total + RT_W[2] * _mw(
                (Ff.reshape(-1, 9) - F0).pow(2).sum(-1)) * R_REF / GRID_LIM
    if RT_W[1]:
        vl = fit.lift(pj, enc_v(v0, cache), cache)[1]
        total = total + RT_W[1] * _mw((vl - v0).pow(2).sum(-1)) * DT_C / GRID_LIM
    total = total + _frame_penalty(w_)
    return total


def unrolled(X, V, t, n, cache):
    """n coarse frames from one MPM state, scored against MPM at every frame.

    The simulator runs on its own output after the first frame, which is the
    regime it will be used in and the one a single step says nothing about.
    Each frame is divided by how far MPM moved from the start, so a later frame
    is not weighted down for having drifted further.
    """
    fit.reset_carried()
    p = enc(X[t], cache)
    v = enc_v(V[t], cache)
    loss = pen = 0.0
    hi = min(n, X.shape[0] - 1 - t)
    # a frame earlier than this hands the next one a detached state, so no
    # gradient travels more than --grad_frames frames back. The loss is
    # untouched: every frame is still scored.
    cut = hi - args.grad_frames if args.grad_frames else 0
    # the last two frames of each, for the acceleration term below
    prev2 = prev1 = None
    acc = 0.0
    n_acc = 0
    for j in range(hi):
        if cut > 0 and j < cut:
            p, v = p.detach(), v.detach()
        p, v, cfl = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        d = (X[t + j + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
        if args.loss == "mwrmsd":
            # the metric itself: mass-weighted RMS against a fixed length scale.
            # The window normalisation below amplifies whichever window happened
            # to move least, which is how a fit ends up specialised to one
            # excitation and worse than no fit at all on another.
            e2 = (got - X[t + j + 1]).pow(2).sum(-1)
            loss = loss + ((MASS_G * e2).sum() / MASS_G.sum()).sqrt() / GRID_LIM
        else:
            loss = loss + (got - X[t + j + 1]).norm(dim=-1).mean() / d
        pen = pen + cfl
        # Matching positions frame by frame says nothing about how the motion
        # gets there, and the fit exploits that: on ficus it cuts mwRMSD from
        # 8.4% to 3.3% while the second difference of total momentum -- what
        # i-PhysGaussian reports as impulse irregularity -- goes the wrong way,
        # 0.26 to 0.33 against MPM's own 0.071. The stepper is buying position
        # with jerk. This scores the second difference against MPM's, in the
        # same per-frame normalisation the position term uses.
        if args.lambda_acc > 0:
            if prev2 is not None:
                a_got = got - 2 * prev1 + prev2
                a_ref = X[t + j + 1] - 2 * X[t + j] + X[t + j - 1]
                acc = acc + (a_got - a_ref).norm(dim=-1).mean() / d
                n_acc += 1
            prev2, prev1 = prev1, got
    if args.lambda_acc > 0 and n_acc:
        loss = loss + args.lambda_acc * acc / n_acc
    return loss / max(hi, 1), pen / max(hi, 1)



def fine_loss(x0, v0, cache, n_sub, fc0=None):
    """n SUBSTEPS from one state, scored against MPM at every one of them.

    The unrolled loss above walks twelve coarse frames -- 480 differentiable
    substeps -- and compares at twelve of them. That is 480 forward evaluations
    producing twelve numbers of supervision, with a gradient that has to travel
    back through all of them, and it is the reason a fit iteration cost two
    minutes.

    MPM's own step is the same 1e-4 as this simulator's, so the two can be walked
    side by side and compared at every substep instead. The same compute then
    yields forty times the supervision, and a short horizon becomes affordable:
    what the long unroll was buying is states the simulator reaches on its own,
    and DAgger supplies those directly rather than by rolling forward from an MPM
    state every iteration.

    MPM is restarted from the lifted anchor state rather than from its own stored
    frame, because the cache holds coarse frames only and storing substeps would
    be 490 GB. The lift reproduces MPM's continuation to 2e-4 of a 0.59 span
    (exe/verify_mpm_teacher.py), and it is what makes a DAgger state usable at
    all -- so both sources of start state go through the same door.
    """
    p = enc(x0, cache)
    v = enc_v(v0, cache)
    with torch.no_grad():
        if fc0 is not None:
            # MPM's own recorded state: exact, and never refused
            T._set(x0.contiguous(), v0.contiguous(),
                   fc0[0].contiguous(), fc0[1].contiguous())
        else:
            x, vx, F, C = fit.lift(p, v, cache)
            if not (torch.isfinite(x).all() and x.min() > T.margin
                    and x.max() < T.grid_lim - T.margin):
                return None, None
            det = det3(F.reshape(-1, 3, 3))
            if det.min() < 0.05 or det.max() > 20.0:
                return None, None
            T._set(x, vx, F, C)

    loss = pen = 0.0
    for _ in range(n_sub):
        p, v, cfl = fit.rollout(p, v, 1, cache)
        with torch.no_grad():
            T.solver.p2g2p(None, sc.sub_dt, device=T.wp_dev)
            if not T._in_domain():
                return None, None
            tgt = T.solver.export_particle_x_to_torch()
        got = fit.gaussian_pos(p, cache)
        # against how far MPM has moved from the start, so an early substep is
        # not weighted down for having barely moved
        d = (tgt - x0).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - tgt).norm(dim=-1).mean() / d
        pen = pen + cfl
    return loss / n_sub, pen / n_sub


@torch.no_grad()
def step_error(sets, every=10):
    """비유한 표본은 세어서 빼고 평균 낸다.

    초기 기하에서는 변위가 큰 후반 프레임 몇 개가 40 서브스텝 안에 발산한다
    (측정: fit 60 표본 중 2, chk 60 중 1 -- exe/probe_project_inverse.py 계열의
    확인). 단순 평균이면 그 한 건이 열 전체를 nan 으로 만들어, 피팅이 실제로
    내려가고 있는데도 지표를 읽을 수 없었다. 버린 개수를 함께 돌려주어 발산이
    줄고 있는지도 보이게 한다.
    """
    cache = fit.prepare()
    tot, n, bad = 0.0, 0, 0
    for X, V, *_ in sets:
        for t in range(0, X.shape[0] - 1, every):
            got, _ = one_step(X[t], V[t], cache)
            d = (X[t + 1] - X[t]).norm(dim=-1).mean().clamp(min=1e-12)
            e = ((got - X[t + 1]).norm(dim=-1).mean() / d).item()
            if e != e or e in (float("inf"), float("-inf")):
                bad += 1
                continue
            tot += e; n += 1
    return (100 * tot / max(n, 1)), bad


@torch.no_grad()
def rollout_error(sets):
    """the whole trajectory, from rest, as the evaluation actually asks for it"""
    cache = fit.prepare()
    tot = []
    for X, V, *_ in sets:
        p, v = enc(X[0], cache), enc_v(V[0], cache)
        out = [fit.gaussian_pos(p, cache)]
        for _ in range(args.frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            out.append(fit.gaussian_pos(p, cache))
        G = torch.stack(out)
        Xg = torch.stack([X[k] for k in range(args.frames + 1)])
        span = (Xg - Xg[0]).norm(dim=-1).max().clamp(min=1e-12)
        tot.append(((G - Xg).norm(dim=-1).mean(-1) / span).mean().item())
    return 100 * sum(tot) / max(len(tot), 1)


# scoring every fit trajectory costs more than training on them: 106 of them at
# one sample per ten frames is 636 rollouts of forty substeps, ten minutes an
# evaluation. A fixed subset says the same thing about whether the fit is moving
N_REPORT = 10

# 고정 평가 창: 시작할 때 한 번 뽑아 두고 계속 같은 것을 쓴다
FIXED = []
if args.fixed_eval > 0:
    _g = torch.Generator(device="cpu"); _g.manual_seed(20260905)
    _hi = max(1, args.frames - args.unroll)
    for _ in range(args.fixed_eval):
        FIXED.append((int(torch.randint(len(FIT), (1,), generator=_g).item()),
                      int(torch.randint(_hi, (1,), generator=_g).item())))
    print(f"[진단] 고정 평가 창 {len(FIXED)}개", flush=True)


@torch.no_grad()
def fixed_loss(cache):
    """같은 창에서만 잰 목적함수. 학습 루프의 loss 와 정의는 같고 표본만 고정."""
    tot, n = 0.0, 0
    for i, t in FIXED:
        X, V = FIT[i][0], FIT[i][1]
        if args.objective == "roundtrip":
            fit.reset_carried()
            ent = FIT[i]
            # C 를 안 담은 캐시에서는 ent[3] 가 None 이다(왕복 손실이 C 를 안 쓰므로 기본)
            fc = ((ent[2][t], ent[3][t] if ent[3] is not None else None)
                  if len(ent) == 4 else None)
            c = roundtrip_loss(X[t], V[t], fc, cache)
        else:
            fit.reset_state()
            c, _ = unrolled(X, V, t, args.unroll, cache)
        if torch.isfinite(c):
            tot += float(c); n += 1
    return (tot / n) if n else float("nan"), len(FIXED) - n


def drift():
    """파라미터가 초기값에서 얼마나 움직였는가. 0 이면 학습률이나 마스크 문제.

    init 리포트는 기준점 P0/S0/Q0 가 잡히기 전에 불린다 -- 그때는 이동이 0 이다.
    """
    if "P0" not in globals():
        return 0.0, 0.0, 0.0
    h = sc.sim.radius
    return (float(((fit.pos - P0) / h).norm(dim=-1).mean()),
            float((fit.log_s - S0).abs().mean()),
            float((fit.quat - Q0).abs().mean()))



@torch.no_grad()
def report(tag):
    with ema_weights():
        return _report(tag)


@torch.no_grad()
def _report(tag):
    a, abad = step_error(FIT[:N_REPORT])
    b, bbad = step_error(CHK)
    extra = ""
    if args.eval_rollout:
        extra = f"   rollout {rollout_error(CHK):6.2f}%"
    drop = f"   발산 {abad}/{bbad}" if (abad or bbad) else ""
    diag = ""
    if FIXED:
        fl, fbad = fixed_loss(fit.prepare())
        dp, ds, dq = drift()
        diag = (f"\n        [진단] 고정창 목적함수 {fl:.5f}"
                + (f" ({fbad} 발산)" if fbad else "")
                + f"   이동 |Δpos|/h {dp:.4f}, |Δlog_s| {ds:.4f}, |Δquat| {dq:.4f}")
    print(f"  [{tag}] one-step  fit {a:7.1f}%   held out {b:7.1f}%{extra}{drop}   "
          f"{fit.M} anchors, {fit.pair_g.shape[0]} pairs{diag}", flush=True)
    return b


POOL = {"x": [], "v": [], "tgt": []}
# how many coarse frames of MPM sit behind a DAgger state
N_DAG = args.unroll if args.dagger_unroll == 0 else max(1, args.dagger_unroll)


def unrolled_pool(x0, v0, tgts, cache):
    """the unrolled loss, but from a state the simulator reached on its own.

    Same scoring as unrolled(): every frame compared, each divided by how far
    MPM had moved from the start by then. The difference is only where the
    first state came from -- here it is one the simulator produced, which is
    the distribution it will actually be used in.
    """
    p = enc(x0, cache)
    v = enc_v(v0, cache)
    loss = pen = 0.0
    n = tgts.shape[0]
    cut = n - args.grad_frames if args.grad_frames else 0
    for j in range(n):
        if cut > 0 and j < cut:
            p, v = p.detach(), v.detach()
        p, v, cfl = fit.rollout(p, v, args.dt_mult, cache)
        got = fit.gaussian_pos(p, cache)
        tj = tgts[j].to(dev, torch.float32)
        d = (tj - x0).norm(dim=-1).mean().clamp(min=1e-12)
        loss = loss + (got - tj).norm(dim=-1).mean() / d
        pen = pen + cfl
    return loss / n, pen / n


@torch.no_grad()
def collect_dagger():
    cache = fit.prepare()
    cap = args.dagger_cap * max((X[args.frames] - X[0]).norm(dim=-1).max().item()
                                 for X, *_ in FIT[:8])
    added = skipped = 0
    for i in range(args.dagger_traj):
        X, V, *_ = FIT[i % len(FIT)]
        p, v = enc(X[0], cache), enc_v(V[0], cache)
        for t in range(args.dagger_frames):
            gp = fit.gaussian_pos(p, cache)
            if not torch.isfinite(p).all() or (gp - X[0]).norm(dim=-1).max() > cap:
                break
            if t % args.dagger_stride == 0:
                x, vx, F, C = fit.lift(p, v, cache)
                ok = torch.isfinite(x).all() and x.min() > T.margin and \
                    x.max() < T.grid_lim - T.margin
                tg = []
                if ok and not args.fine_steps:
                    # the coarse loss compares against stored frames; the fine
                    # one walks MPM alongside and needs no target, which also
                    # saves forty MPM substeps per state collected
                    T._set(x, vx, F, C)
                    for _ in range(N_DAG):
                        if T._advance(1, args.dt_mult) is None:
                            ok = False
                            break
                        tg.append(T.solver.export_particle_x_to_torch()
                                  .to(torch.float16).cpu())
                if ok:
                    POOL["x"].append(x.to(torch.float16).cpu())
                    POOL["v"].append(vx.to(torch.float16).cpu())
                    if not args.fine_steps:
                        POOL["tgt"].append(torch.stack(tg))
                    added += 1
                else:
                    skipped += 1
    # oldest out first, so the pool tracks where the simulator is now rather than
    # where it was at the start
    for k_ in POOL:
        while len(POOL[k_]) > args.dagger_pool_max:
            POOL[k_].pop(0)
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
    return added, skipped


def make_opt():
    for n, prm in fit.named_parameters():
        prm.requires_grad_(n in TRAIN or n.startswith("edge.")
                           or n.startswith("astress.") or n.startswith("fdyn."))
    groups = [{"params": [getattr(fit, n)], "lr": LRS[n]} for n in TRAIN]
    # 간선망은 맨 뒤에 붙인다. 학습률 스케줄은 TRAIN 과 zip 으로 돌아가므로
    # 이 그룹만 상수 학습률로 남는데, 새 항이 초반에 죽지 않도록 그 편이 낫다.
    if fit.astress is not None:
        groups.append({"params": list(fit.astress.parameters()), "lr": args.lr_astress})
    if fit.fdyn is not None:
        groups.append({"params": list(fit.fdyn.parameters()), "lr": args.lr_astress})
    if fit.edge is not None:
        groups.append({"params": list(fit.edge.parameters()), "lr": args.lr_edge})
    return torch.optim.Adam(groups, betas=(args.beta1, args.beta2))


opt = make_opt()
grad_accum = torch.zeros(fit.M, device=dev)
STATE = args.state or (args.out + ".state" if args.out else None)
LR0 = {n: LRS[n] for n in TRAIN}
EMA = {}


def ema_update():
    """track an average of the parameters, restarted when their shape changes"""
    if not args.ema:
        return
    for n in TRAIN:
        p_ = getattr(fit, n).detach()
        if n not in EMA or EMA[n].shape != p_.shape:
            EMA[n] = p_.clone()
        else:
            EMA[n].mul_(args.ema).add_(p_, alpha=1 - args.ema)


class ema_weights:
    """evaluate with the average, then put the live parameters back"""

    def __enter__(self):
        self.saved = None
        if args.ema and EMA:
            self.saved = {n: getattr(fit, n).detach().clone() for n in TRAIN}
            with torch.no_grad():
                for n in TRAIN:
                    getattr(fit, n).copy_(EMA[n])

    def __exit__(self, *a):
        if self.saved is not None:
            with torch.no_grad():
                for n in TRAIN:
                    getattr(fit, n).copy_(self.saved[n])


# the mirror is the backup, so a mirror that has quietly stopped working is no
# backup at all. A vast host started intercepting TLS mid-run and every copy
# failed with a certificate error for an hour and a half without a word, because
# the upload was a backgrounded shell command with stderr sent to /dev/null. It
# stays asynchronous -- training should not wait on an upload -- but the previous
# copy's exit status is now collected before the next one starts.
_mirror_proc = None
_mirror_fails = 0


def _mirror(path):
    global _mirror_proc, _mirror_fails
    if _mirror_proc is not None:
        rc = _mirror_proc[0].poll()
        if rc is None:
            pass
        elif rc != 0:
            _mirror_fails += 1
            err = _mirror_proc[0].stderr.read().decode(errors="replace").strip()
            print(f"\n[mirror] FAILED to copy {os.path.basename(_mirror_proc[1])} "
                  f"to {args.r2} (rclone exit {rc}, {_mirror_fails} in a row)\n"
                  f"[mirror] {err.splitlines()[-1] if err else 'no output'}\n"
                  f"[mirror] this run is NOT backed up", flush=True)
        else:
            if _mirror_fails:
                print(f"\n[mirror] recovered after {_mirror_fails} failures", flush=True)
            _mirror_fails = 0
        _mirror_proc = None
    p = subprocess.Popen(["rclone", "copy", path, args.r2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _mirror_proc = (p, path)


def save_state(it, best):
    """A state that is not finite is not a state to come back to. The previous
    run wrote one and then failed to resume from it four times in a row."""
    if not STATE:
        return
    if not all(torch.isfinite(q).all() for q in
               (fit.pos, fit.log_s, fit.quat)):
        print(f"  [state] iteration {it} is not finite; keeping the last good one",
              flush=True)
        return
    # log_amp 를 빠뜨리면 재개할 때마다 0 으로 되돌아간다 -- _rebuild 가
    # blob.get("log_amp") 로 읽고 없으면 zeros 를 넣기 때문이다. 실제로 fs_base 와
    # fs_x1 이 재개 지점마다 이 값을 잃었다.
    torch.save({"pos": fit.pos.detach(), "log_s": fit.log_s.detach(),
                 "quat": fit.quat.detach(), "log_k": fit.log_k.detach(),
                 "log_amp": fit.log_amp.detach(),
                 "iter": it, "best": best,
                 "opt": opt.state_dict(), "grad_accum": grad_accum,
                 "rng": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                 # the pool stays out: it is a gigabyte of particle states that
                 # refills in a few collections, so writing it every save would
                 # cost more than regenerating it
                 "c": args.c, "args": vars(args),
                 "edge": (fit.edge.state_dict() if fit.edge is not None else None)},
                STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    if args.r2:
        _mirror(STATE)


start_it, best = 1, None
if args.resume and STATE:
    if not os.path.exists(STATE) and args.r2:
        os.system(f"rclone copy {args.r2}/{os.path.basename(STATE)} "
                   f"{os.path.dirname(STATE) or '.'} 2>/dev/null")
    if os.path.exists(STATE):
        blob = torch.load(STATE, map_location=dev, weights_only=False)
        fit._rebuild(blob["pos"], blob["quat"], blob["log_s"], blob.get("log_k"),
                     blob.get("log_amp"))
        def _load_soft(mod, sd, tag):
            """모양이 맞는 항목만 되살린다. 앵커 수가 바뀌면 간선 버퍼와
            앵커별 파라미터의 크기가 달라지는데, 그걸로 재개 자체가
            실패하는 것보다 그 부분만 새로 잡는 편이 낫다."""
            cur = mod.state_dict()
            ok = {k: v for k, v in sd.items()
                  if k in cur and cur[k].shape == v.shape}
            mod.load_state_dict(ok, strict=False)
            print(f"  [resume] {tag}: {len(ok)}/{len(cur)} 복원")
        if fit.edge is not None and blob.get("edge"):
            fit.edge.rebuild(fit.pos, fit.sim_radius, fit.dt)
            _load_soft(fit.edge, blob["edge"], "edge")
        if fit.astress is not None and blob.get("astress"):
            fit.astress.rebuild(fit)
            _load_soft(fit.astress, blob["astress"], "astress")
        opt = make_opt(); opt.load_state_dict(blob["opt"])
        grad_accum = blob["grad_accum"].to(dev)
        torch.set_rng_state(blob["rng"].cpu()); torch.cuda.set_rng_state(blob["rng_cuda"].cpu())
        for k in POOL:
            POOL[k] = []          # not persisted; refills in a few collections
        start_it, best = blob["iter"] + 1, blob["best"]
        print(f"\n[resume] {STATE} at iteration {blob['iter']}, {fit.M} anchors, "
              f"best {best:.1f}%, {len(POOL['x'])} collected states")
if best is None:
    print(f"\n[before]")
    best = report("init")

# where the fit started, for the regulariser to refer to. Re-taken after every
# density change, since the anchors are not the same set any more
P0, S0, Q0 = fit.pos.detach().clone(), fit.log_s.detach().clone(), fit.quat.detach().clone()
t0 = time.time()
n_skip = 0
GLOG = open(args.grad_log, "a") if args.grad_log else None
if GLOG:
    GLOG.write("# iter " + " ".join(f"|g_{n}| cos_{n}" for n in TRAIN) + "\n")
PREV_G = {}

def snapshot():
    """되돌릴 지점. 파라미터만 담으므로 앵커 600 개 규모에서는 무시할 크기다."""
    snap = {n: getattr(fit, n).detach().clone() for n in TRAIN}
    for name, mod in (("astress", fit.astress), ("edge", fit.edge)):
        if mod is not None:
            snap[name] = {k: v.detach().clone() for k, v in mod.state_dict().items()}
    return snap


def restore(snap):
    with torch.no_grad():
        for n in TRAIN:
            if n in snap and getattr(fit, n).shape == snap[n].shape:
                getattr(fit, n).copy_(snap[n])
        for name, mod in (("astress", fit.astress), ("edge", fit.edge)):
            if mod is not None and name in snap:
                cur = mod.state_dict()
                ok = {k: v for k, v in snap[name].items()
                      if k in cur and cur[k].shape == v.shape}
                mod.load_state_dict(ok, strict=False)


LR_SCALE = 1.0
last_good = snapshot()
n_rollback = 0
bar = tqdm(range(start_it, args.iters + 1), desc="fit", ncols=90)
for it in bar:
    if args.lr_cosine or args.lr_warmup:
        f = 1.0
        if args.lr_cosine:
            # cosine from the given rate to zero across the whole run
            f = 0.5 * (1.0 + math.cos(math.pi * min(it / max(args.iters, 1), 1.0)))
        if args.lr_warmup:
            f *= min(1.0, it / args.lr_warmup)
        for gp, n in zip(opt.param_groups, TRAIN):
            gp["lr"] = LR0[n] * f
    if args.lr_decay_at and it == args.lr_decay_at:
        for gp in opt.param_groups:
            gp["lr"] *= args.lr_decay_by
        print(f"\n  [lr] it={it}: every rate multiplied by {args.lr_decay_by}",
              flush=True)
    if it > start_it and args.refresh_every and it % args.refresh_every == 0:
        fit.refresh()
    if args.densify_every and it % args.densify_every == 0 and \
            it <= args.densify_until * args.iters:
        dead, split = fit.densify_and_prune(grad_accum, args.split_frac,
                                             args.prune_share,
                                             max_anchors=args.max_anchors)
        # the parameters are new tensors, so Adam's moments no longer refer to
        # anything; kept simple by restarting them rather than reindexing
        opt = make_opt()
        for gp in opt.param_groups:
            gp["lr"] *= LR_SCALE
        last_good = snapshot()          # 앵커 수가 바뀌었으니 옛 스냅샷은 못 쓴다
        grad_accum = torch.zeros(fit.M, device=dev)
        P0 = fit.pos.detach().clone(); S0 = fit.log_s.detach().clone()
        Q0 = fit.quat.detach().clone()
        print(f"\n  [density] it={it}: -{dead} +{split} -> {fit.M} anchors, "
              f"{fit.pair_g.shape[0]} pairs", flush=True)
    last_sample = None
    if args.dagger_every and (it == start_it or it % args.dagger_every == 0):
        a_, s_ = collect_dagger()
        print(f"\n  [dagger] it={it}: +{a_} states ({s_} MPM could not answer for), "
              f"pool {len(POOL['x'])}", flush=True)

    frac = min(1.0, it / max(args.warmup, 1))
    hi = max(2, int(frac * (args.frames - 1)))
    # the fine horizon, annealed if asked. Geometric rather than linear: what
    # matters is the ratio between what a sample sees and what a rollout does,
    # and that is a scale
    n_sub_now = args.fine_steps
    if args.fine_steps and args.fine_end:
        u = min(1.0, (it - 1) / max(args.iters - 1, 1))
        n_sub_now = max(1, int(round(args.fine_steps
                                      * (args.fine_end / args.fine_steps) ** u)))
    opt.zero_grad(set_to_none=True)
    acc_loss, acc_pen, acc_n, skipped = 0.0, 0.0, 0, 0
    for _accum in range(args.accum):
      # rebuilt per accumulation step: prepare() is differentiable and every
      # sample's graph hangs off it, so one backward frees what the next needs
      cache = fit.prepare()
      loss, pen, bad_sample, n_ok = 0.0, 0.0, None, 0
      for _ in range(args.batch):
          src, fc0 = None, None
          if POOL["x"] and torch.rand(1).item() < args.dagger_frac:
              # a DAgger state has one labelled frame after it and nothing more, so
              # it stays a single step whatever the unroll length is
              j = torch.randint(len(POOL["x"]), (1,)).item()
              x0 = POOL["x"][j].to(dev, torch.float32)
              v0 = POOL["v"][j].to(dev, torch.float32)
              # [n,N,3] once a DAgger state carries more than one frame; kept
              # on the CPU until the frame is needed, as the trajectories are
              tgt = POOL["tgt"][j] if POOL["tgt"] else None
              if tgt is not None:
                  if tgt.dim() == 3 and tgt.shape[0] == 1:
                      tgt = tgt[0]          # one frame stored, one frame used
                  if tgt.dim() == 2:
                      tgt = tgt.to(dev, torch.float32)
          else:
              ent = FIT[torch.randint(len(FIT), (1,)).item()]
              X, V = ent[0], ent[1]
              t = torch.randint(hi, (1,)).item()
              x0, v0, tgt = X[t], V[t], X[t + 1]
              src = (X, V, t)
              # MPM's own state at that frame, when it was kept: then the fine loss
              # walks alongside MPM from where MPM actually was, with no lift
              if len(ent) == 4:
                  fc0 = (ent[2][t], ent[3][t] if ent[3] is not None else None)
          if args.fine_steps:
              fit.reset_state()
              contrib, cfl = fine_loss(x0, v0, cache, n_sub_now, fc0)
              if contrib is None:      # MPM cannot be asked from here
                  continue
          elif src is None and tgt is not None and tgt.dim() == 3 and tgt.shape[0] > 1:
              # a DAgger state with more than one frame behind it: same loss as
              # a trajectory sample, from a state the simulator itself reached
              fit.reset_state()
              last_sample = ("pool", x0, v0, tgt)
              contrib, cfl = unrolled_pool(x0, v0, tgt, cache)
          elif args.objective == "roundtrip":
              # 접었다 편 잔차. 굴리지 않으므로 CFL 도 없다.
              fit.reset_carried()
              contrib = roundtrip_loss(x0, v0, fc0, cache)
              cfl = torch.zeros((), device=dev)
          elif src is not None and args.unroll > 1:
              X_, V_, t_ = src
              fit.reset_state()
              last_sample = ("traj", X_, V_, t_)
              contrib, cfl = unrolled(X_, V_, t_, args.unroll, cache)
          else:
              got, cfl = one_step(x0, v0, cache)
              d = (tgt - x0).norm(dim=-1).mean().clamp(min=1e-12)
              contrib = (got - tgt).norm(dim=-1).mean() / d
          if args.no_guards and not torch.isfinite(contrib):
              bad_sample = (x0, v0)
          loss = loss + contrib
          pen = pen + cfl
          n_ok += 1
      if n_ok == 0:
        # every sample in the batch started somewhere MPM will not answer for --
        # a lifted state outside its grid, or a deformation gradient no material
        # is in. Nothing to learn from, and dividing by the batch would leave a
        # float where a tensor is expected
        continue
      loss, pen = loss / n_ok, pen / n_ok
      total = loss + args.lambda_cfl * pen
      if args.reg > 0:
        # measured against the anchor spacing and against unit scale, so one
        # number covers parameters that do not share units
        h = sc.sim.radius
        r = ((fit.pos - P0) / h).pow(2).mean() + (fit.log_s - S0).pow(2).mean() \
            + (fit.quat - Q0).pow(2).mean() + fit.log_k.pow(2).mean() \
            + fit.log_amp.pow(2).mean()
        if fit.astress is not None:
            r = r + fit.astress.reg()
        total = total + args.reg * r
      if not torch.isfinite(total):
        # one bad sample -- a rollout that ran away, a configuration the polar
        # factor cannot handle -- should cost that iteration, not the run
        if args.nan_trace and last_sample is not None:
            # 추적은 여기서, 미분 경로 밖에서 한다. 체크포인팅은 순전파와
            # 재계산이 같은 텐서를 내놓을 것을 요구하는데 추적기는 상태를 갖고
            # 있어 그렇지 못하고, 실제로 "saved metadata != recomputed metadata"
            # 로 죽었다. no_grad 로 같은 표본을 한 번 더 돌리면 그 문제가 없고
            # 메모리도 들지 않는다.
            with torch.no_grad():
                fit.nan_trace = True
                fit.trace_reset()
                try:
                    if last_sample[0] == "pool":
                        unrolled_pool(last_sample[1], last_sample[2],
                                       last_sample[3], cache)
                    else:
                        unrolled(last_sample[1], last_sample[2], last_sample[3],
                                  args.unroll, cache)
                except Exception as _e:
                    print(f"  [nan] 재현 중 예외 {type(_e).__name__}: {_e}",
                          flush=True)
                fit.nan_trace = False
        if args.nan_trace:
            rep = fit.nan_report()
            where = ", ".join(f"{k}@서브스텝{v}" for k, v in
                              sorted(rep.items(), key=lambda kv: kv[1])) or "추적 없음"
            mag = "  ".join(f"{k}={v:.3e}" for k, v in fit.mag_report().items())
            who = ", ".join(f"{k}#{v}" for k, v in fit.who_report().items())
            print(f"\n  [nan] it={it}: loss={float(total)}\n"
                  f"        최초 발생: {where}\n"
                  f"        문제 원소: {who or '없음'}\n"
                  f"        전 구간 크기: {mag}", flush=True)
            if fit.astress is not None and fit.who_report():
                a_ = min(fit.who_report().values())
                with torch.no_grad():
                    nb = (fit.astress.eb[fit.astress.ea == a_]).tolist()
                    print(f"        앵커#{a_}: h={float(fit.astress.h[a_]):.4e} "
                          f"(h0={float(fit.astress.h0[a_]):.4e}) "
                          f"k={float(fit.astress.ka[a_]):.3f} "
                          f"vol={float(fit.astress.vol[a_]):.3e} "
                          f"이웃 {len(nb)}개", flush=True)
        skipped = 1
        break
      # backward here rather than at the end: the graph of this batch is freed
      # before the next one is built, which is the whole reason accumulation
      # buys a bigger effective batch on a card that cannot hold one
      (total / args.accum).backward()
      # 역전파가 끝나면 그래프가 해제된다. 프레임 상태를 들고 있으면 다음 누적
      # 스텝의 순전파가 그 텐서를 물고 가서 "backward a second time" 이 난다.
      fit.reset_state()
      acc_loss += float(loss); acc_pen += float(pen); acc_n += 1
    if acc_n == 0:
        n_skip += 1
        if args.nan_rollback:
            restore(last_good)
            LR_SCALE *= args.nan_lr_decay
            n_rollback += 1
            for gp in opt.param_groups:
                gp["lr"] *= args.nan_lr_decay
            print(f"  [nan] it={it}: 마지막 정상 지점으로 되돌림, "
                  f"학습률 x{args.nan_lr_decay} (누적 x{LR_SCALE:.4f}), "
                  f"{n_rollback}번째", flush=True)
            if LR_SCALE < args.nan_lr_floor:
                print(f"  [nan] 학습률이 초기값의 {args.nan_lr_floor} 아래로 "
                      f"내려갔다. 여기서 멈춘다.", flush=True)
                break
        continue
    loss = torch.tensor(acc_loss / acc_n)
    pen = acc_pen / acc_n
    if not skipped:
      if True:
        with torch.no_grad():
            if fit.pos.grad is not None:
                grad_accum += torch.nan_to_num(fit.pos.grad).norm(dim=-1)
                fit.pos.grad[fit.fixed] = 0
        if all(getattr(fit, n).grad is None or
               torch.isfinite(getattr(fit, n).grad).all() for n in TRAIN):
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [getattr(fit, n) for n in TRAIN], args.clip)
            opt.step()
            fit.clamp_()
            ema_update()
            if args.nan_rollback:
                last_good = snapshot()
        else:
            skipped = 1
    if GLOG and not skipped:
        parts = []
        for n in TRAIN:
            g_ = getattr(fit, n).grad
            if g_ is None:
                parts += ["nan", "nan"]
                continue
            g_ = g_.detach().reshape(-1)
            c = float("nan")
            # densify_and_prune changes the anchor count, and a gradient of a
            # different length is a gradient for a different parameter set --
            # there is no angle between them to report
            if n in PREV_G and PREV_G[n].shape == g_.shape:
                c = float((g_ * PREV_G[n]).sum()
                          / (g_.norm() * PREV_G[n].norm()).clamp(min=1e-30))
            PREV_G[n] = g_.clone()
            parts += [f"{float(g_.norm()):.6g}", f"{c:.4f}"]
        GLOG.write(f"{it} " + " ".join(parts) + "\n")
        GLOG.flush()
    n_skip += skipped
    if skipped and args.no_guards:
        # what a survivable run hides: which quantity went first, and whether the
        # step that produced it grew every substep or spiked once
        print(f"\n[first failure] iteration {it}, loss {loss.item()}", flush=True)
        if bad_sample is not None:
            x0, v0 = bad_sample
        w_, rc_, q_, Binv_, blocked_, mass_ = cache
        with torch.no_grad():
            F_, _ = fit.deformation(enc(x0, cache), w_, rc_, q_, Binv_, blocked_)
            det = torch.linalg.det(F_)
            print(f"  mass: min {mass_.min():.3e}, at the clamp "
                  f"{int((mass_ <= 1.0000001e-12).sum())} of {fit.M}")
            print(f"  detF: min {det.min():.4f}, inverted {int((det <= 0).sum())} of "
                  f"{fit.N}, extents min {fit.log_s.exp().min():.3e}")
            m_ = mass_.unsqueeze(-1)
            keep_ = (~fit.fixed).unsqueeze(-1).to(torch.float32)
            p_, v_ = enc(x0, cache), enc_v(v0, cache)
            print(f"  {'substep':>8} {'|a| max':>12} {'|v| max':>12} {'detF min':>10}")
            for k_ in range(args.dt_mult):
                a_ = fit.force(p_, w_, rc_, q_, Binv_, blocked_) / m_
                v_ = (v_ + fit.dt * a_) * fit.damping * keep_
                p_ = p_ + fit.dt * v_
                F2, _ = fit.deformation(p_, w_, rc_, q_, Binv_, blocked_)
                if k_ < 4 or k_ % 4 == 0 or not torch.isfinite(p_).all():
                    print(f"  {k_:8d} {a_.norm(dim=-1).max():12.3e} "
                          f"{v_.norm(dim=-1).max():12.3e} "
                          f"{torch.linalg.det(F2).min():10.4f}")
                if not torch.isfinite(p_).all():
                    break
        break
    bar.set_postfix(loss=f"{loss.item():.3f}", cfl=f"{float(pen):.2e}", M=fit.M,
                     win=hi, skip=n_skip)
    if it % args.eval_every == 0 or it == args.iters:
        fit.reset_state()
        with torch.no_grad():
            b = report(f"it {it}")
            if args.eval_rollout:
                # the one-step error keeps falling while the rollout doubles, so
                # keeping the best by one-step keeps the wrong checkpoint
                b = rollout_error(CHK)
        if args.out and b < best:
            best = b
            # args go in too: the runs that produced the fitted sets in use
            # saved only parameters, so months later there was no way to tell
            # what --iters, --reg or DAgger setting had produced them
            torch.save({"pos": fit.pos.detach().cpu(), "log_s": fit.log_s.detach().cpu(),
                         "quat": fit.quat.detach().cpu(),
                         "log_k": fit.log_k.detach().cpu(),
                         "log_amp": fit.log_amp.detach().cpu(),
                         "astress": (fit.astress.state_dict() if fit.astress is not None else None),
                         "runtime": {"mass_floor": args.mass_floor,
                                      "mass_freeze": args.mass_freeze,
                                      "finv_ridge": args.finv_ridge,
                                      "cfl_agg": fit.cfl_agg,
                                      "astress_eig": args.astress_eig,
                                      "astress_jmax": args.astress_jmax,
                                      "edge_only": bool(args.edge_only)},
                         "c": args.c, "iter": it,
                         "edge": (fit.edge.state_dict() if fit.edge is not None else None),
                         "eig_floor": args.eig_floor, "score": float(b),
                         "args": vars(args)}, args.out)
        save_state(it, best)
    elif it % args.save_every == 0:
        save_state(it, best)

print(f"\n[done] {time.time() - t0:.0f}s, {fit.M} anchors, {n_skip} iterations skipped "
      f"as non-finite")

# ---- the number the project actually asks for ------------------------------
with torch.no_grad():
    print(f"\n[rollout] {args.frames} frames against MPM's particles, "
          f"{args.final_rollout} held-out impulses")
    cache = fit.prepare()
    # the simulator this replaces, on the same impulses: without it the number
    # above is only comparable to other runs of this script
    print(f"  {'impulse':>8} {'error':>9} {'final':>9} {'motion':>8} {'8-NN sim':>10}")
    tot, tot0 = [], []
    for i, (X, V) in enumerate(CHK[:args.final_rollout]):
        f0 = FORCES["chk"][i]
        p0, v0, g0 = sc.anchor_canonical.clone(), sc.initial_velocity(f0), sc.pos.clone()
        base_out = [g0[fit.mat].clone()]
        for _ in range(args.frames):
            p0, v0, g0 = sc.explicit_step(p0, v0, g0, args.dt_mult)
            base_out.append(g0[fit.mat].clone())
        G0 = torch.stack(base_out)
        Xb = torch.stack([X[k] for k in range(args.frames + 1)])
        span0 = (Xb - Xb[0]).norm(dim=-1).max().clamp(min=1e-12)
        e0 = ((G0 - Xb).norm(dim=-1).mean(-1) / span0).mean().item()
        tot0.append(e0)
        p = enc(X[0], cache)
        v = enc_v(V[0], cache)
        out = [fit.gaussian_pos(p, cache)]
        for _ in range(args.frames):
            p, v, _ = fit.rollout(p, v, args.dt_mult, cache)
            out.append(fit.gaussian_pos(p, cache))
        G = torch.stack(out)
        Xg = torch.stack([X[k] for k in range(args.frames + 1)])
        span = (Xg - Xg[0]).norm(dim=-1).max().clamp(min=1e-12)
        e = (G - Xg).norm(dim=-1).mean(-1) / span
        tot.append(e.mean().item())
        print(f"  {i:8d} {100*e.mean():8.2f}% {100*e[-1]:8.2f}% "
              f"{100*(G - G[0]).norm(dim=-1).max()/span:7.0f}% {100*e0:9.2f}%")
    print(f"  {'mean':>8} {100*sum(tot)/max(len(tot),1):8.2f}% {'':>9} {'':>8} "
          f"{100*sum(tot0)/max(len(tot0),1):9.2f}%")
    s_, k_ = fit.log_s.exp(), fit.log_k.exp()
    print(f"\n[params] {fit.M} anchors, {fit.pair_g.shape[0]} pairs, axis ratio "
          f"{(s_.max(-1).values / s_.min(-1).values).median():.2f}, "
          f"size {s_.mean():.4f}, stiffness {k_.min():.3f}..{k_.max():.3f} "
          f"(median {k_.median():.3f})")
if args.out:
    # 마지막 반복의 기하를 반드시 남긴다. --out 은 점수가 개선될 때만 쓰이므로
    # 3000 을 완주해도 파일은 마지막 개선 시점(실측 2050 / 2400 / 2900)에 멈춰 있고,
    # 그 위에 학생을 올리면 서로 다른 반복 수의 기하를 비교하게 된다 -- 이 사고가
    # 세 번 났다. 최종본은 별도 이름으로 따로 저장한다.
    fin = args.out.replace(".pt", "_final.pt")
    torch.save({"pos": fit.pos.detach().cpu(), "quat": fit.quat.detach().cpu(),
                 "log_s": fit.log_s.detach().cpu(), "log_k": fit.log_k.detach().cpu(),
                 "log_amp": fit.log_amp.detach().cpu(),
                 "astress": (fit.astress.state_dict() if fit.astress is not None else None),
                 "runtime": {"mass_floor": args.mass_floor,
                              "mass_freeze": args.mass_freeze,
                              "finv_ridge": args.finv_ridge,
                              "cfl_agg": fit.cfl_agg,
                              "astress_eig": args.astress_eig,
                              "astress_jmax": args.astress_jmax,
                              "edge_only": bool(args.edge_only)},
                 "c": args.c, "eig_floor": args.eig_floor, "iter": args.iters,
                 "args": vars(args)}, fin)
    print(f"[saved ] {args.out} (최고점) / {fin} (최종 반복 {args.iters})")
