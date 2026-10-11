"""사면체 복합체 변형 모델의 **한 스텝·롤아웃** -- 학습(exe/train_deform.py)과
추론(exe/lb_rollout.py)이 함께 쓰는 단일 구현.

train_deform 의 전역(`a`, MASS, VEL_SCALE, _DT ...)을 쓰지 않고 필요한 것을
인자로 받는다. `a` 는 train_deform 의 argparse 결과 (또는 체크포인트에 저장된
args 를 담은 네임스페이스) 이고, 여기서는 읽기만 한다.

  node_feats      입자 -> 사면체 노드 특징·간선 (P2G 구조)
  node_inertial   노드별 관성 예측자 h·v_n + h²g (--net_inertial 의 기반)
  advance         망 한 번 통과 -> 다음 위치·속도(--v_from_dt 면 jvp)·야코비안
  init_state      평가 롤아웃의 초기 (x, v)
  rollout_eval    프레임별 평균 위치 오차 / 정지 기준 (평가 지표 정의)
  EvalModel       체크포인트 하나로 모델을 세워 롤아웃만 하는 추론용 묶음
"""
from __future__ import annotations

import contextlib
import glob
import math
import os
import types

import numpy as np
import torch

from . import phys_resid
from . import simplex as SX
from . import simplex_gnn as _SGM
from .deform import bc_features
from .simplex_gnn import SimplexGNN, node_moments, scatter_to_nodes

FCAP = float(os.environ.get("AF_FEAT_CAP", "50"))   # 특징 상한 (발산 방지)
N_MAT_FULL = 13      # logE, nu, xi, logρ, log1p(항복), φ, 종류4, g3


def _null_sec(_name):
    return contextlib.nullcontext()


def san(t):
    """특징은 O(1) 이 정상이다. 얇은 구름에서 국소 맞춤이 튀면 여기서 잘라 준다."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp(-FCAP, FCAP)


def take(t, idx, dev="cuda"):
    """궤적에서 부분표본을 떼어 GPU 로 (half 면 float 로 올린다)."""
    if t.is_cuda:
        r = t[idx]
    else:
        r = t[idx.cpu()].to(dev, non_blocking=True)
    return r.float() if r.dtype == torch.float16 else r


def extent(x0):
    """물체 크기 = 바운딩박스 대각선 길이."""
    return float((x0.max(0).values - x0.min(0).values).norm())


def traj_mass(d, dev="cuda"):
    """궤적 자기 배치·자기 밀도로 잰 입자 질량 [N_full] (d["_mass"], d["_ext"] 도 채운다)."""
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
    d["_ext"] = extent(x0)
    return d["_mass"]


def traj_F(d, frame_dt, dev="cuda", keep_gpu=True, on_rebuild=None):
    """탄성 변형구배 [T,N,3,3]. 궤적에 든 것이 전부 항등이면 위치에서 되살린다."""
    if d.get("_Fok") is None:
        _Fin = d["F"]
        _id = float((_Fin[:3].float().to(dev)
                     - torch.eye(3, device=dev)).abs().max()) < 1e-6
        if _id:
            import time
            t_ = time.time()
            Fr = phys_resid.rebuild_F(d["x"].to(dev).float(), d["cfg"], frame_dt)
            d["F"] = Fr if keep_gpu else Fr.cpu()
            if on_rebuild is not None:
                on_rebuild(time.time() - t_)
        d["_Fok"] = True
    return d["F"]


def mat_feat(cfg, no_mat=False, dev="cuda"):
    """물성 특징 [13] (no_mat 이면 [0])."""
    if no_mat:
        return torch.zeros(0, device=dev, dtype=torch.float32)
    g = torch.tensor(cfg["g"], device=dev, dtype=torch.float32) / 15.0
    _mats = ("jelly", "metal", "foam", "sand")
    _oh = [1.0 if cfg.get("material", "jelly") == m else 0.0 for m in _mats]
    return torch.cat([torch.tensor(
        [np.log(float(cfg["E"])), float(cfg["nu"]), float(cfg.get("xi", 0.0)),
         np.log(float(cfg["density"])),
         np.log1p(float(cfg.get("yield_stress", 0.0))),
         float(cfg.get("friction_angle", 0.0)) / 45.0] + _oh,
        device=dev, dtype=torch.float32), g])


def n_mat_film(a):
    """(N_MAT, N_FILM): 물성 특징 폭과 그중 FiLM 으로 넣는 폭."""
    n_mat = 0 if a.no_mat else N_MAT_FULL
    return n_mat, (n_mat if (a.mat_film and not a.no_mat) else 0)


# ---------------------------------------------------------------- 모델
def apply_arch_flags(a):
    """메시지 패싱 층 구조 스위치 (층을 만들기 **전에** 정해야 한다)."""
    _SGM._MP_NORM = not bool(a.gnn_no_norm)
    _SGM._MP_MEAN = not bool(a.gnn_sum)


def build_net(a, n_feat, ext0, frame_dt, n_film, dev="cuda"):
    """SimplexGNN. 출력 크기 scale = 0.02·ext0 (ext0 = 학습 데이터 첫 궤적의 물체 크기)."""
    if a.arch != "sgnn":
        raise SystemExit("사면체 복합체 경로만 남았다 -- --arch sgnn 을 쓸 것 "
                         "(격자 conv/unet 은 태그 grid-rqs-final 에 있다)")
    return SimplexGNN(n_feat=n_feat, hidden=a.hidden, layers=a.gnn_layers,
                      scale=0.02 * ext0, dt_cond=a.dt_cond, dt_ref=frame_dt,
                      dt_scale=a.dt_scale, n_mat=n_film).to(dev)


def ens_offsets(n):
    if n <= 1:
        return [None]
    return [tuple(((k * q) % 1.0) for q in (0.5, 0.25, 0.125))
            for k in range(n)]


def node_grid(a, x, off=None):
    return SX.grid_for_nodes(x, a.n_nodes, hz_ratio=a.hz_ratio, off=off,
                             h_fix=(a.node_h or None))


def node_feats(a, cfg, X, mass, x, v, vel_scale, grid=None, off=None,
               pt_parts=None, mat_vec=None):
    """사면체 복합체의 **노드** 특징. 입자 물리량을 자기 사면체의 barycentric
    가중으로 4 꼭짓점에 뿌려 모은다 (P2G 와 같은 구조, 전달 가중치 재사용).

    X: 정지(0 프레임) 위치 [N,3], mass: 입자 질량 [N], vel_scale: 속도 단위.
    pt_parts(hn) -> 입자별 추가 특징 목록 (fe 불변량, 손잡이 특징 등).
    mat_vec: 물성 특징 [N_MAT] (특징에 붙일 때만).

    -> (_in [M,F], npos [M,3], (lo,lat,nn), rows [N,4], lam [N,4], uniq [M],
        간선 (src,dst,cls), aux)
    """
    lo, lat, nn = grid if grid is not None else node_grid(a, x, off)
    idx, lam, _aux = SX.locate(x, lo, lat, nn)   # tau 에 무관 -- 한 번만
    if a.hex_jac:
        # 육면체 삼선형을 쓰면 원소가 **셀**이다 -- 활성 노드도 셀의 여덟
        # 꼭짓점 전부여야 한다. ci 는 nn-2 로 잘려 있어 ci+1 은 항상 격자 안이다.
        _nnl = [int(nn[k]) for k in range(3)]
        _o8 = torch.tensor([[i_, j_, k_] for i_ in (0, 1) for j_ in (0, 1)
                            for k_ in (0, 1)], device=x.device,
                           dtype=torch.long)
        _v8 = _aux[0].unsqueeze(1) + _o8.unsqueeze(0)        # [N,8,3]
        _f8 = ((_v8[..., 0] * _nnl[1] + _v8[..., 1]) * _nnl[2] + _v8[..., 2])
        _rall, uniq = SX.active_nodes(torch.cat([idx, _f8], 1))
        rows = _rall[:, :4]
    else:
        rows, uniq = SX.active_nodes(idx)      # **점유 사면체의 꼭짓점만**
    Mn = int(uniq.numel())
    npos = SX.node_pos(lo, lat, nn, uniq)
    hn = lat.s                                 # 길이 단위 = 면내 간격 h
    feat = san(node_moments(x, v / vel_scale, X, mass, rows, lam, Mn, npos, hn))
    parts = pt_parts(hn) if pt_parts is not None else []
    if parts:
        feat = torch.cat([feat, san(scatter_to_nodes(
            san(torch.cat(parts, -1)), rows, lam, Mn, mass=mass))], -1)
    extra = [bc_features(npos, cfg) / hn]
    if mat_vec is not None:
        extra.insert(0, mat_vec.reshape(1, -1).expand(Mn, mat_vec.numel()))
    _in = torch.cat([san(feat), san(torch.cat(extra, -1))], -1)
    _in = torch.nan_to_num(_in, nan=0.0, posinf=0.0,
                           neginf=0.0).clamp(-FCAP, FCAP)
    src, dst, cls = SX.edges_of(rows, uniq, nn)
    return (_in, npos, (lo, lat, nn), rows, lam, uniq, (src, dst, cls), _aux)


def node_inertial(x, v, mass, rows, lam, n_node, g, h):
    """**노드별 관성 예측자** h·v_n + h²g (barycentric 가중 질량평균 속도 v_n).

    rows/lam 은 사면체(4) 든 육면체(8) 든 열 수만큼 뿌린다.
    -> (h·v_n + h²g [M,3], (v_n [M,3], g [3]))
    """
    _mw = mass.unsqueeze(-1).to(x.dtype)
    _ws = torch.zeros(n_node, 1, device=x.device, dtype=x.dtype)
    _vs = torch.zeros(n_node, 3, device=x.device, dtype=x.dtype)
    for _j in range(rows.shape[1]):
        _w = lam[:, _j:_j + 1] * _mw
        _ws.index_add_(0, rows[:, _j], _w)
        _vs.index_add_(0, rows[:, _j], _w * v)
    _vn = _vs / _ws.clamp_min(1e-30)
    _gq = torch.as_tensor(g, device=x.device, dtype=x.dtype)
    _hq = float(h)
    return _hq * _vn + (_hq ** 2) * _gq, (_vn, _gq)


def advance(net, a, ens, x, h, cfg, *, inertial=None, node_mask=None,
            node_vel=None, ctrl_scale=1.0, mat=None, sec=None,
            dp_fast=None, dp_hook=None, pre_k=None, on_pin=None,
            post_jac=None):
    """망 한 번 통과로 한 스텝 -> (x2, v_next, dpn, Jf, corr, act).

    변형장은 "GNN + barycentric + 손잡이 혼합" 뿐이다. tau(=dt) 에 탄젠트를
    얹어 한 번 통과시키면 dPhi/dt (각 지점의 속도) 가 나온다 (--v_from_dt).

    ens: node_feats 결과 목록 (오프셋별 격자). 입자 변위는 기준 격자 하나로,
         야코비안은 오프셋 평균으로 정한다.
    inertial: 격자별 (v_n, g) -- --net_inertial 이면 망 출력에 τ·v_n + τ²g 를 더한다.
    node_mask/node_vel: 격자별 손잡이 노드와 명령 속도 (노드 변위를 덮는다).
    dp_fast/dp_hook/pre_k/on_pin/post_jac: 학습 쪽 실험 경로용 훅 (추론은 None).
    Jf = I + grad u (입자별 [N,3,3]).
    """
    sec = sec or _null_sec
    K = len(ens)
    rows0 = ens[0][3]

    def _fieldS(tau):
        _dus, _Jsum = [], None
        for _k in range(K):
            (_ink, _npk, (_lok, _latk, _nnk), _rwk, _lmk, _uqk,
             (_srk, _dsk, _clk), _lxk) = ens[_k]
            if pre_k is not None:
                pre_k(_k)
            out = (dp_fast(_npk.shape[0], x.dtype)
                   if dp_fast is not None else None)
            if out is None:
                with sec("신경망"):
                    out = net(_ink, _srk, _dsk, _clk, tau, mat=mat)
                if a.net_inertial:
                    if inertial is None or inertial[_k] is None:
                        raise SystemExit("--net_inertial 은 노드 관성 예측(--ov_init predict, 사면체)이 필요하다")
                    _vnk, _gqk = inertial[_k]
                    # τ 의 함수로 더해야 --v_from_dt 의 dΦ/dτ 에 관성 속도가 들어간다
                    out = (out[0] + tau * _vnk + (tau * tau) * _gqk,) + tuple(out[1:])
                if dp_hook is not None:
                    out = dp_hook(out)
            dpn_k = torch.nan_to_num(out[0], nan=0.0, posinf=0.0, neginf=0.0)
            if node_mask is not None and node_mask[_k] is not None:
                # **노드 변위를 명령으로 덮는다.** detach 로 기울기를 끊어
                # 그 노드는 최적화 대상에서 빠진다 (상수 경계조건이다).
                dpn_k = torch.where(
                    node_mask[_k].unsqueeze(-1),
                    (tau * ctrl_scale * node_vel[_k]).detach(), dpn_k)
                if _k == 0 and on_pin is not None:
                    on_pin(tau)
            _duk = SX.g2p_pre(_rwk, _lmk, dpn_k)
            _u1, _Jk = SX.g2p_jac_pre(_rwk, _lmk, _lxk, _latk, dpn_k)
            if post_jac is not None:
                _duk, _Jk = post_jac(_k, ens[_k], dpn_k, _duk, _Jk)
            _dus.append(_duk)
            _Jsum = _Jk if _Jsum is None else _Jsum + _Jk
            if _k == 0:
                _dpn0 = dpn_k
        # **입자 변위는 기준 격자(오프셋 0) 하나로만** 정한다. 오프셋 평균은
        # 변형구배에만 쓴다.
        _du = _dus[0]
        _Jm = _Jsum / float(K)
        q = x + _du
        _qraw = q
        _act = torch.zeros_like(q[:, :1])
        if node_mask is not None and node_mask[0] is not None:
            # 네 꼭짓점이 모두 구속된 입자는 명령만큼 간다
            _act = _act + node_mask[0][rows0].all(1).to(q.dtype).unsqueeze(-1)
        if not a.bc_soft and a.bc_mode == "project" \
                and a.bc_level == "particle":
            _duP, _fa = phys_resid.bc_project(
                x, q - x, cfg, tau, float(cfg.get("grid_lim", 2.0)),
                int(cfg["n_grid"]))
            q = x + _duP
            _act = _act + _fa.to(q.dtype).unsqueeze(-1)
        _corr = q - _qraw
        return q, (_dpn0, _Jm, _corr, _act)

    _use_dtS = a.v_from_dt and dp_hook is None and a.dt_cond
    _tau0 = torch.as_tensor(float(h), device=x.device, dtype=x.dtype)
    if _use_dtS:
        with sec("jvp(속도)"):
            (x2, _outs), (v_next, _) = torch.func.jvp(
                _fieldS, (_tau0,), (torch.ones_like(_tau0),))
    else:
        x2, _outs = _fieldS(_tau0)
        v_next = (x2 - x) / h
    Jf = torch.eye(3, device=x.device, dtype=x.dtype) + _outs[1]
    return x2, v_next, _outs[0], Jf, _outs[2], _outs[3]


# ---------------------------------------------------------------- 평가
def init_state(d, t0, gsel, frame_dt, v0_traj, dev="cuda"):
    """평가 롤아웃의 초기 (x, v). 기본 v 는 (x[t0]-x[t0-1])/dt 차분,
    --v0_traj 면 궤적에 저장된 v."""
    x = take(d["x"][t0], gsel, dev)
    v = (x - take(d["x"][max(t0 - 1, 0)], gsel, dev)) / frame_dt
    if v0_traj and "v" in d:
        v = take(d["v"][min(t0, d["v"].shape[0] - 1)], gsel, dev).to(x.dtype)
    return x, v


def frame_err(x2, gt, ext, fm=slice(None)):
    """입자 평균 위치 오차 / 물체 크기."""
    return float((x2[fm] - gt[fm]).norm(dim=-1).mean()) / ext


def rollout_eval(step_fn, x, v, gt_at, L, ext, fm_fn=None, on_frame=None):
    """L 프레임 롤아웃의 프레임별 (오차, 정지 기준).

    step_fn(i, x, v) -> (x2, v2, info); gt_at(i) -> 프레임 i+1 의 GT 위치.
    정지 기준 = 초기 위치에 그대로 둔 것의 오차. 둘 다 ext 로 나눈다.
    on_frame(i, x2, gt, info, x_still) 은 프레임마다 (덤프·렌더용).
    """
    errs, stills = [], []
    x_still = x.clone()
    for i in range(L):
        x2, v, info = step_fn(i, x, v)
        gt = gt_at(i)
        fm = fm_fn(i, x2) if fm_fn is not None else slice(None)
        errs.append(frame_err(x2, gt, ext, fm))
        stills.append(frame_err(x_still, gt, ext, fm))
        if on_frame is not None:
            on_frame(i, x2, gt, info, x_still)
        x = x2
    return errs, stills


def window_row(L, errs, stills):
    e, s = float(np.mean(errs)), float(np.mean(stills))
    return dict(L=L, err=errs, still=stills, err_mean=e, still_mean=s,
                ratio=e / max(s, 1e-12))


def summarize(rows):
    """rows[tag]["windows"][t0] -> 전체 창 평균 (train_deform [요약] 과 같은 정의).

    err/still 은 창 평균의 평균, ratio 는 창별 비의 평균이다.
    """
    r_all = [(w["err_mean"], w["still_mean"]) for r in rows.values()
             for w in r["windows"].values()]
    if not r_all:
        return None
    return dict(n=len(r_all),
                err_mean=float(np.mean([x for x, _ in r_all])),
                still_mean=float(np.mean([y for _, y in r_all])),
                ratio=float(np.mean([x / max(y, 1e-12) for x, y in r_all])))


def tb_write_eval(tbw, rows, step, prefix="eval"):
    """요약 + 궤적별 스칼라 (백분율). 이름은 세 스크립트가 같게 쓴다."""
    s = summarize(rows)
    if s is None:
        return
    tbw.add_scalar(f"{prefix}/err_pct", 100 * s["err_mean"], step)
    tbw.add_scalar(f"{prefix}/still_pct", 100 * s["still_mean"], step)
    tbw.add_scalar(f"{prefix}/ratio", s["ratio"], step)
    for tag, r in rows.items():
        for t0, w in r["windows"].items():
            k = f"{prefix}_traj/{tag}_t{t0}"
            tbw.add_scalar(f"{k}/err_pct", 100 * w["err_mean"], step)
            tbw.add_scalar(f"{k}/still_pct", 100 * w["still_mean"], step)
            tbw.add_scalar(f"{k}/ratio",
                           w["err_mean"] / max(w["still_mean"], 1e-12), step)
    tbw.flush()


# ---------------------------------------------------------------- 추론 묶음
# 오래된 체크포인트의 args 에 없을 수 있는 키 (train_deform 의 기본값)
_ARG_DEFAULTS = dict(
    arch="sgnn", hidden=128, gnn_layers=8, gnn_no_norm=False, gnn_sum=False,
    n_nodes=32, hz_ratio=1.0 / 6 ** 0.5, node_h=0.0, hex_jac=False,
    fe_state=False, control=False, net_inertial=False, v_from_dt=False,
    dt_cond=False, dt_scale=False, dt_sub=1, eval_dt_sub=0, ens_off=0,
    bc_soft=False, bc_mode="project", bc_level="node", ov_init="predict",
    no_mat=False, mat_film=False, v0_traj=False, smooth_nb=0,
    cell_poly=False, ov_roll=0, refps=False, hold_traj=None, pool=True)


def ckpt_args(ck):
    return types.SimpleNamespace(**{**_ARG_DEFAULTS, **(ck.get("args") or {})})


def train_ref(a):
    """train_deform 이 모델을 세울 때 본 첫 학습 궤적 (TR[0]) 파일 경로.

    망 출력 크기(0.02·EXT)와 속도 단위(EXT/frame_dt) 가 이 궤적의 0 프레임
    물체 크기로 정해진다 (체크포인트의 "EXT" 는 풀 루프가 덮어써 믿을 수 없다).
    """
    files = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    if not files:
        raise SystemExit(f"학습 데이터가 없다: {a.data} -- 체크포인트에 "
                         f"net_ext0 가 없어 다시 재야 한다")
    hold = set((a.hold_traj or "").split(",")) - {""}
    tr = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in hold]
    return (tr or files)[0]


class EvalModel:
    """체크포인트 하나로 모델을 세워 **롤아웃만** 한다 (학습·풀·통계 재계산 없음).

      M = EvalModel("MAIN_best.pt")
      x, v = M.init(d, t0)
      x, v, J = M.frame(d, x, v)        # 한 프레임 (서브스텝이면 그 수만큼 망 통과)

    fast=True 면 지원되는 설정에서 lib/anchorflow/fast_roll.FastStep (동기화 없는
    고정 크기 + 손으로 쓴 탄젠트 + CUDA 그래프) 로 같은 식을 돈다. 지원 밖이면
    알리고 느린 경로를 쓴다. fast_graph=False 는 같은 계산을 즉시 실행으로,
    fast_triton=False 는 입자 단위 융합 커널 대신 같은 식의 torch 연산으로.
    """

    UNSUPPORTED = ("control", "fe_state", "hex_jac", "cell_poly", "refps")

    def __init__(self, ckpt, dev="cuda", fast=False, fast_graph=True,
                 fast_triton=True, fast_edge_triton=True):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        a = ckpt_args(ck)
        for k in self.UNSUPPORTED:
            if getattr(a, k):
                raise NotImplementedError(f"EvalModel: --{k} 체크포인트는 지원하지 않는다")
        if int(a.smooth_nb) or int(a.ov_roll):
            raise NotImplementedError("EvalModel: --smooth_nb / --ov_roll 미지원")
        self.a, self.dev, self.step_ck = a, dev, int(ck.get("step", 0))
        if ck.get("net_ext0") is not None:
            self.ext0, self.frame_dt = float(ck["net_ext0"]), float(ck["frame_dt"])
            self.ref = "ckpt"
        else:
            self.ref = train_ref(a)
            d0 = torch.load(self.ref, map_location="cpu", weights_only=False, mmap=True)
            # **저장 dtype(half) 그대로** 잰다 -- train_deform 의 EXT 가 half 위에서
            # 계산돼 float 로 재면 1e-4 쯤 어긋난다 (망 출력 크기·속도 단위에 들어간다)
            self.ext0 = extent(d0["x"][0].to(dev))
            self.frame_dt = float(d0["cfg"]["frame_dt"])
            del d0
        self.n_mat, self.n_film = n_mat_film(a)
        apply_arch_flags(a)
        n_feat = int(ck.get("n_feat") or ck["net"]["in_mu"].numel())
        self.net = build_net(a, n_feat, self.ext0, self.frame_dt, self.n_film, dev)
        miss, unex = self.net.load_state_dict(ck["net"], strict=False)
        if unex or [k for k in miss if not k.startswith("dtfilm.")]:
            raise SystemExit(f"체크포인트 키 불일치: 없음 {miss[:5]} 모름 {unex[:5]}")
        self.net.eval()
        self.net.requires_grad_(False)     # 롤아웃은 그래프가 필요 없다 (jvp 는 순방향)
        self.vel_scale = self.ext0 / self.frame_dt
        self.sub = int(a.eval_dt_sub or a.dt_sub)
        self.h = self.frame_dt / self.sub
        del ck
        self.fast, self._fs = False, None
        if fast:
            from .fast_roll import fast_supported
            ok, why = fast_supported(a)
            if ok:
                self.fast, self.fast_graph = True, bool(fast_graph)
                self.fast_triton = bool(fast_triton)
                self.fast_edge_triton = fast_edge_triton
            else:
                print(f"[EvalModel] 빠른 경로 미지원 ({why}) -- 느린 경로로 돈다", flush=True)

    def prepare(self, d):
        """궤적별 상수 (질량, 크기, 정지 위치) 를 한 번만."""
        if "_X0" not in d:
            traj_mass(d, self.dev)
            d["_X0"] = take(d["x"][0], torch.arange(d["x"].shape[1], device=self.dev), self.dev)
        return d

    def init(self, d, t0):
        self.prepare(d)
        g = torch.arange(d["x"].shape[1], device=self.dev)
        return init_state(d, t0, g, self.frame_dt, self.a.v0_traj, self.dev)

    def step(self, d, x, v, h=None, sec=None):
        """망 한 번 -> (x2, v2, Jf). sec(name) 은 구간 시간 측정용 컨텍스트."""
        a, cfg = self.a, d["cfg"]
        sec = sec or _null_sec
        h = self.h if h is None else h
        mass = d["_mass"]
        mat_vec = (mat_feat(cfg, a.no_mat, self.dev)
                   if (self.n_mat and not self.n_film) else None)
        mat = (mat_feat(cfg, a.no_mat, self.dev).reshape(1, self.n_mat)
               if self.n_film else None)
        if self.fast:
            fs = self._fs
            # 궤적(X0·질량·cfg)과 h 에 묶인다 -- 바뀌면 새로 세운다
            if fs is None or fs.X0 is not d["_X0"] or fs.h != float(h):
                from .fast_roll import FastStep
                fs = self._fs = FastStep(self.net, a, cfg, d["_X0"], mass,
                                         self.vel_scale, h, mat_vec=mat_vec,
                                         mat=mat, graph=self.fast_graph,
                                         triton=self.fast_triton,
                                         edge_triton=self.fast_edge_triton)
            with sec("빠른스텝"):
                return fs(x, v)
        with sec("셀집계"):
            ens = [node_feats(a, cfg, d["_X0"], mass, x, v, self.vel_scale,
                              off=o, mat_vec=mat_vec)
                   for o in ens_offsets(max(int(a.ens_off), 1))]
        inert = None
        if a.ov_init == "predict":
            with sec("관성"):
                inert = [node_inertial(x, v, mass, e[3], e[4], e[1].shape[0],
                                       cfg["g"], h)[1] for e in ens]
        x2, v2, _dpn, Jf, _c, _act = advance(self.net, a, ens, x, h, cfg,
                                            inertial=inert, mat=mat, sec=sec)
        return x2, v2, Jf

    def frame(self, d, x, v, sec=None):
        """한 프레임 (서브스텝 sub 개) -> (x2, v2, J 누적)."""
        J = None
        for _ in range(self.sub):
            x, v, Jf = self.step(d, x, v, sec=sec)
            J = Jf if J is None else Jf @ J
        return x, v, J
