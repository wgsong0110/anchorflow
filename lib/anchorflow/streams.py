"""Training data from an object that keeps being hit, instead of restarts.

Every trajectory in the original scheme began at the canonical rest state, took
one impulse, and decayed. That covers a narrow set of states -- what a single
kick reaches before it dies out -- and the later steps of every trajectory look
alike, near rest. Running continuously and delivering impulses along the way
lets kicks land on an object that is still moving, which reaches configurations
no single impulse produces.

Lives here rather than in the training script so the renderer shows the data
the training actually sees, rather than a second implementation of it.
"""
from __future__ import annotations

import torch


def rand_rot(gen, device):
    q, r = torch.linalg.qr(torch.randn(3, 3, device=device, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def draw_impulse(sc, base_force, gen, impulse_range=4.0, field=False,
                  sigma_lo=None, sigma_hi=None):
    raise NotImplementedError(
        "임펄스 계열 분할(균일 / 힘장 / 포크)은 제거했다. 이 프로젝트의 임펄스는 "
        "(포크 개수 K, 반경 r) 로만 결정되는 다중 포크 하나뿐이다 -- "
        "Scene.random_multi_poke 를 쓸 것. 기하 피팅·학생 학습·평가가 모두 "
        "같은 계열에서 뽑혀야 서로 견줄 수 있다.")

def draw_field_shape(sc, gen, sigma_lo=None, sigma_hi=None):
    raise NotImplementedError(
        "임펄스 계열 분할(균일 / 힘장 / 포크)은 제거했다. 이 프로젝트의 임펄스는 "
        "(포크 개수 K, 반경 r) 로만 결정되는 다중 포크 하나뿐이다 -- "
        "Scene.random_multi_poke 를 쓸 것. 기하 피팅·학생 학습·평가가 모두 "
        "같은 계열에서 뽑혀야 서로 견줄 수 있다.")

def stream(sc, n_steps, dt_mult, base_force, gen, cap, impulse_every=20,
           impulse_range=4.0, keep_accel=True, on_step=None, field=False,
           sigma_lo=None, sigma_hi=None):
    raise NotImplementedError(
        "임펄스 계열 분할(균일 / 힘장 / 포크)은 제거했다. 이 프로젝트의 임펄스는 "
        "(포크 개수 K, 반경 r) 로만 결정되는 다중 포크 하나뿐이다 -- "
        "Scene.random_multi_poke 를 쓸 것. 기하 피팅·학생 학습·평가가 모두 "
        "같은 계열에서 뽑혀야 서로 견줄 수 있다.")
