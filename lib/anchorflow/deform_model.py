"""사면체 복합체 변형 모델의 **한 스텝·롤아웃** -- 학습(exe/train_deform.py)과
추론(exe/lb_rollout.py)이 함께 쓰는 단일 구현.

train_deform 의 전역(`a`, MASS, VEL_SCALE, _DT ...)을 쓰지 않고 필요한 것을
인자로 받는다. `a` 는 train_deform 의 argparse 결과 (또는 체크포인트에 저장된
args 를 담은 네임스페이스) 이고, 여기서는 읽기만 한다.

  node_feats      입자 -> 사면체 노드 특징·간선 (P2G 구조)
  node_inertial   노드별 관성 예측자 h·v_n + h²g (--net_inertial 의 기반)
  ip_node_feats   --ip_feat: 관성 예측점에서 증분 포텐셜의 노드별 기울기·블록 대각 헤시안·뉴턴 걸음 (입력 특징)
  advance         망 한 번 통과 -> 다음 위치·속도(--v_from_dt 면 jvp)·야코비안
  init_state      평가 롤아웃의 초기 (x, v)
  rollout_eval    프레임별 평균 위치 오차 / 정지 기준 (평가 지표 정의)
  EvalModel       체크포인트 하나로 모델을 세워 롤아웃만 하는 추론용 묶음
  fast_step       no_grad 롤아웃용 빠른 스텝 (fast_roll: 동기화 없음·융합 커널·CUDA 그래프)
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
from . import wf as WF
from . import simplex_gnn as _SGM
from .deform import bc_features
from .simplex_gnn import SimplexGNN, node_moments, scatter_to_nodes
from .skin import gregory_skin

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


REGION_DENSITY = [False]   # --pg_region_density: PG additional_material_params 의 영역별 밀도를 질량에 (기본 꺼짐)


def region_rho(cfg, x0):
    """PG apply_additional_params 그대로: 0 프레임 위치가 상자 (point-size, point+size) 안(엄격 부등식)인
    입자의 밀도를 그 영역 값으로 -> 입자별 밀도 [N]. REGION_DENSITY 가 꺼져 있거나 영역이 없으면 None.
    PG 는 나중 영역이 앞 영역을 덮어쓴다 (순서대로 적용). E·nu 가 기본과 다른 영역은 아직 없다 (멈춘다)."""
    regs = cfg.get("additional_material_params") or []
    if not REGION_DENSITY[0] or not regs:
        return None
    rho = torch.full((x0.shape[0],), float(cfg["density"]), device=x0.device, dtype=torch.float32)
    for r in regs:
        if float(r.get("E", cfg["E"])) != float(cfg["E"]) or float(r.get("nu", cfg["nu"])) != float(cfg["nu"]):
            raise SystemExit(f"--pg_region_density: E/nu 가 다른 영역은 아직 없다 ({r})")
        q = torch.as_tensor(r["point"], device=x0.device, dtype=x0.dtype)
        sz = torch.as_tensor(r["size"], device=x0.device, dtype=x0.dtype)
        ins = ((x0 > q - sz) & (x0 < q + sz)).all(-1)
        rho[ins] = float(r["density"])
    return rho


def traj_mass(d, dev="cuda"):
    """궤적 자기 배치·자기 밀도로 잰 입자 질량 [N_full] (d["_mass"], d["_ext"] 도 채운다).
    --pg_region_density 면 영역별 밀도 (d["_rho"] 에 입자별 밀도도 남긴다)."""
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
    _rho = region_rho(c, x0)
    if _rho is None:
        d["_mass"] = ((dx_ ** 3) / cn_[fl_]) * float(c["density"])
    else:
        d["_mass"] = ((dx_ ** 3) / cn_[fl_]) * _rho
        d["_rho"] = _rho
    d["_ext"] = extent(x0)
    return d["_mass"]


def traj_F(d, frame_dt, dev="cuda", keep_gpu=True, on_rebuild=None):
    """탄성 변형구배 [T,N,3,3]. 궤적에 든 것이 전부 항등이면 위치에서 되살린다."""
    if d.get("_Fok") is None:
        _Fin = d["F"]
        _id = float((_Fin[:3].float().to(dev)
                     - torch.eye(3, device=dev)).abs().max()) < 1e-6
        if _id and not d.get("F_is_true"):            # F_is_true: 항등이 실제 값인 상태(예: 정지 초기 상태) -- 되살리지 않는다
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
                      dt_scale=a.dt_scale, n_mat=n_film,
                      n_out=(4 if getattr(a, "gregory", False) else
                             (WF.C1_NOUT if str(getattr(a, "interp", "bary")) == "c1wf" else 3))).to(dev)


def ens_offsets(n):
    if n <= 1:
        return [None]
    return [tuple(((k * q) % 1.0) for q in (0.5, 0.25, 0.125))
            for k in range(n)]


def node_grid(a, x, off=None):
    return SX.grid_for_nodes(x, a.n_nodes, hz_ratio=a.hz_ratio, off=off,
                             h_fix=(a.node_h or None))


def node_feats(a, cfg, X, mass, x, v, vel_scale, grid=None, off=None,
               pt_parts=None, mat_vec=None, ip_state=None):
    """사면체 복합체의 **노드** 특징. 입자 물리량을 자기 사면체의 barycentric
    가중으로 4 꼭짓점에 뿌려 모은다 (P2G 와 같은 구조, 전달 가중치 재사용).

    X: 정지(0 프레임) 위치 [N,3], mass: 입자 질량 [N], vel_scale: 속도 단위.
    pt_parts(hn) -> 입자별 추가 특징 목록 (fe 불변량, 손잡이 특징 등).
    mat_vec: 물성 특징 [N_MAT] (특징에 붙일 때만).
    ip_state: --ip_feat 일 때 dict(F=[N,3,3] 이 상태의 F, vol=[N], h=스텝 dt, free=[N] 또는 None[, feat=캐시 [M,18]])
              -- 특징 끝에 IP_FEAT_N 채널을 덧붙인다 (ip_node_feats).

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
    _interp = str(getattr(a, "interp", "bary") or "bary")
    rows_tet = rows                            # 간선은 언제나 사면체에서
    if _interp == "bspline":
        # **2 차 B-스플라인 보간** (--interp bspline): 입자 변위·야코비안을 주변 27 노드에서 (MPM 과 같은 가중치).
        # 활성 노드 = 사면체 꼭짓점 ∪ 입자가 쓰는 27 이웃 전부. rows/lam 은 27 열로 바꿔 돌려준다 -- 관성 예측자·
        # 손잡이(--cfg_bc) 노드 판정·노드 특징이 모두 같은 가중치를 쓴다.
        _f27, _w27, _dw27 = SX.bspline27(x, lo, lat, nn)
        uniq = torch.unique(torch.cat([idx.reshape(-1), _f27.reshape(-1)]))
        rows_tet = torch.searchsorted(uniq, idx)
        rows = torch.searchsorted(uniq, _f27)
        lam = _w27.to(x.dtype)
        _aux = SX.Aux(_aux)
        _aux.bsp = _dw27.to(x.dtype)
    elif _interp == "nodeavg":
        _aux = SX.Aux(_aux)
    elif _interp == "c1wf":
        # **Worsey–Farin C1 3 차** (--interp c1wf): 사면체·노드는 bary 와 같다. 입자마다 (종류, 조각, 값·기울기
        # 가중 28) 을 프레임마다 한 번 만들고, 점유 사면체 목록(장벽용)도 함께 든다.
        _aux = SX.Aux(_aux)
        _aux.wf = WF.particle_struct(x, lo, lat, nn, _aux, rows)
        _aux.wft = WF.tet_struct(rows, _aux.wf["ty"], lat)
    elif _interp != "bary":
        raise SystemExit(f"--interp {_interp} 모름")
    Mn = int(uniq.numel())
    npos = SX.node_pos(lo, lat, nn, uniq)
    if _interp == "nodeavg":
        # **노드 평균 기울기** (--interp nodeavg): 사면체별 grad u 를 노드에 부피 가중 평균 -> 입자로 barycentric
        _aux.nav = SX.nodeavg_struct(rows, _aux, lat, Mn)
    hn = lat.s                                 # 길이 단위 = 면내 간격 h
    _agg = str(getattr(a, "agg", "bary") or "bary")
    if _agg == "bspline":
        # **격자 B-스플라인 집계** (MPM P2G 와 같은 2 차 B-스플라인, 격자 좌표 기준 27 노드). 활성 노드가 아닌
        # 이웃은 덤프 노드(Mn 번)로 보내 버린다. 변위 사상·간선은 그대로 사면체(rows, lam) 를 쓴다.
        _y = (x - lo) @ lat.Ai.T
        _b = torch.floor(_y - 0.5)
        _f = _y - _b                                                   # [N,3] in [0.5, 1.5)
        _wq = torch.stack([0.5 * (1.5 - _f) ** 2, 0.75 - (_f - 1.0) ** 2, 0.5 * (_f - 0.5) ** 2], 1)   # [N,3(o),3(axis)]
        _o = torch.tensor([[i_, j_, k_] for i_ in range(3) for j_ in range(3) for k_ in range(3)],
                          device=x.device, dtype=torch.long)                                          # [27,3]
        _ijk = _b.long().unsqueeze(1) + _o.unsqueeze(0)                                                # [N,27,3]
        _w27 = (_wq[:, _o[:, 0], 0] * _wq[:, _o[:, 1], 1] * _wq[:, _o[:, 2], 2])                      # [N,27]
        _nnl = [int(nn[k]) for k in range(3)]
        _ok = ((_ijk >= 0) & (_ijk < torch.tensor(_nnl, device=x.device))).all(-1)
        _flat = ((_ijk[..., 0] * _nnl[1] + _ijk[..., 1]) * _nnl[2] + _ijk[..., 2])
        _pos = torch.searchsorted(uniq, _flat.clamp_min(0)).clamp(max=Mn - 1)
        _hit = _ok & (uniq[_pos] == _flat)
        _rows27 = torch.where(_hit, _pos, torch.full_like(_pos, Mn))
        _w27 = torch.where(_hit, _w27, torch.zeros_like(_w27))
        _npos1 = torch.cat([npos, npos[:1]], 0)
        feat = san(node_moments(x, v / vel_scale, X, mass, _rows27, _w27, Mn + 1, _npos1, hn))[:Mn]
        _prow, _pw = _rows27, _w27
    else:
        feat = san(node_moments(x, v / vel_scale, X, mass, rows, lam, Mn, npos, hn))
        _prow, _pw = rows, lam
    _Kr = int(getattr(a, "rel_pe", 0) or 0)
    if _Kr > 0:
        # **상대 위치 다주파수 인코딩**: 입자-노드 쌍마다 (x_p - x_j)/h 를 sin/cos(2^k π ·) (k < K) 로, 집계 가중치(질량×w)로 평균
        _Mx = int(_prow.max()) + 1
        _np2 = npos if _Mx <= Mn else torch.cat([npos, npos[:1]], 0)
        _rel = (x.unsqueeze(1) - _np2[_prow]) / hn                                                   # [N,K,3]
        _fr = (2.0 ** torch.arange(_Kr, device=x.device, dtype=x.dtype)) * math.pi
        _ang = _rel.unsqueeze(-1) * _fr                                                                # [N,K,3,Kr]
        _pe = torch.cat([torch.sin(_ang), torch.cos(_ang)], -1).flatten(2)                            # [N,K,6Kr]
        _ww = (mass.to(x.dtype).unsqueeze(1) * _pw).unsqueeze(-1)
        _r = _prow.reshape(-1)
        _num = torch.zeros(_Mx, _pe.shape[-1], device=x.device, dtype=x.dtype).index_add_(
            0, _r, (_ww * _pe).reshape(-1, _pe.shape[-1]))
        _den = torch.zeros(_Mx, 1, device=x.device, dtype=x.dtype).index_add_(0, _r, _ww.reshape(-1, 1))
        feat = torch.cat([feat, san((_num / _den.clamp_min(1e-20))[:Mn])], -1)
    parts = pt_parts(hn) if pt_parts is not None else []
    if parts:
        feat = torch.cat([feat, san(scatter_to_nodes(
            san(torch.cat(parts, -1)), rows, lam, Mn, mass=mass))], -1)
    _K = int(getattr(a, "bary_pe", 0) or 0)
    if _K > 0:
        # 셀 안 배치의 다주파수 요약: 입자 p 가 자기 사면체의 꼭짓점 j 에 갖는 barycentric 좌표 λ_pj 를
        # sin/cos(2^k π λ) (k < K) 로 인코딩해 노드 j 로 질량 가중 평균한다 -- 평균·2차 모멘트로는 안 보이는
        # 셀 내부 분포를 여러 해상도로 노드에 싣는다 (--bary_pe K).
        _fr = (2.0 ** torch.arange(_K, device=lam.device, dtype=lam.dtype)) * math.pi
        _ang = lam.unsqueeze(-1) * _fr                                   # [N,4,K]
        _pe = torch.cat([torch.sin(_ang), torch.cos(_ang)], -1)          # [N,4,2K]
        _w = mass.to(lam.dtype).reshape(-1, 1, 1).expand(-1, 4, 1)
        _r = rows.reshape(-1)
        _num = torch.zeros(Mn, 2 * _K, device=lam.device, dtype=lam.dtype).index_add_(
            0, _r, (_w * _pe).reshape(-1, 2 * _K))
        _den = torch.zeros(Mn, 1, device=lam.device, dtype=lam.dtype).index_add_(
            0, _r, _w.reshape(-1, 1))
        feat = torch.cat([feat, san(_num / _den.clamp_min(1e-20))], -1)
    extra = [bc_features(npos, cfg) / hn]
    if mat_vec is not None:
        extra.insert(0, mat_vec.reshape(1, -1).expand(Mn, mat_vec.numel()))
    _in = torch.cat([san(feat), san(torch.cat(extra, -1))], -1)
    if getattr(a, "ip_feat", False):
        if ip_state is None:
            raise RuntimeError("--ip_feat: node_feats 에 이 상태의 F (ip_state) 가 필요하다")
        _ipf = ip_state.get("feat")          # 같은 상태에서 이미 계산한 값 (학습 쪽 캐시; 결정적 함수라 같은 값)
        if _ipf is None or _ipf.shape[0] != Mn:
            _ipf = ip_node_feats(a, cfg, x, v, ip_state["F"], mass, ip_state["vol"], ip_state["h"],
                                 rows, lam, _aux, lat, Mn, free=ip_state.get("free"))
        _in = torch.cat([_in, _ipf.to(_in.dtype)], -1)
    _in = torch.nan_to_num(_in, nan=0.0, posinf=0.0,
                           neginf=0.0).clamp(-FCAP, FCAP)
    src, dst, cls = SX.edges_of(rows_tet, uniq, nn)
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
    _gq = _const_vec(g, x)
    _hq = float(h)
    return _hq * _vn + (_hq ** 2) * _gq, (_vn, _gq)


IP_FEAT_N = 18           # --ip_feat 채널: 기울기 총합 3, 탄성 3, 관성+중력 3, 블록 헤시안 대칭 6, 뉴턴 걸음 3
IPF_LAST = [None]        # 진단·단위 시험: 마지막 ip_node_feats 의 원시 값 (정규화 전)


def ip_node_feats(a, cfg, x, v, F, mass, vol, h, rows, lam, aux, lat, Mn, free=None):
    """**--ip_feat**: 노드 i 가 움직이면 증분 포텐셜이 어떻게 바뀌는지를 망 입력으로 (기울기 차단, 스텝마다 한 번).

    기준점 u₀ = 노드 관성 예측 h·v_n + h²g (node_inertial; --net_inertial 에서 망 출력 0 일 때의 노드 변위).
    IP(u) = 학습과 같은 식: g2p(barycentric) -> bc_maps(--bc_stick·--bc_stick_xn·--bc_softplus) 를 거친 입자 위치·F 로
    phys_resid.pts_ip_energy (관성·중력 + 탄성 (+ 켜진 장벽·접촉항)), 정규화 없음.

    블록 대각 헤시안 H_ii: IP 는 입자별 항의 합이고 입자 p 는 자기 사면체 꼭짓점 4 개에만 의존한다. 그래서 입자마다
    꼭짓점 변위의 **사본** U[p,j] (= u₀[rows[p,j]]) 를 변수로 두면 IP 의 헤시안이 입자별 12×12 블록 대각이 되고,
    헤시안-벡터 곱 12 회 (꼭짓점 칸 j 4 × 성분 c 3, 모든 입자 동시) 로 각 입자의 꼭짓점 대각 3×3 블록이 정확히 나온다.
    H_ii = Σ_{(p,j): rows[p,j]=i} ∂²IP_p/∂U_pj², g_i = Σ ∂IP_p/∂U_pj (노드 변수로 미분한 것과 같다 -- 단위 시험).
    jelly 탄성은 hess_mode 의 미분 가능한 닫힌 식 기울기(_FcrPsiH)로 2 차 미분한다.

    정규화: s_i = M_i/h² (M_i = Σ_p λ_pi m_p). g̃ = g/s_i/h_node (길이 -> 노드 간격 단위), H̃ = H/s_i (무차원).
    뉴턴 걸음: H̃ 대칭화 -> 고윳값 분해, λ' = max(λ, 0) + ε (ε = --ip_feat_eps; 양정부호면 (H̃+εI)⁻¹ 그대로,
    음의 고윳값은 0 으로 자른 뒤 ε) -> d = −Q diag(1/λ') Qᵀ (g/s_i) / h_node.
    -> [M, IP_FEAT_N]: g̃ 총합 3 | g̃ 탄성 3 | g̃ 관성+중력 3 | H̃ (xx,yy,zz,xy,yz,xz) 6 | d 3  (detach)
    """
    if rows.shape[1] != 4 or getattr(a, "hex_jac", False) or getattr(a, "gregory", False) \
            or str(getattr(a, "interp", "bary") or "bary") != "bary" or str(getattr(a, "agg", "bary") or "bary") != "bary":
        raise SystemExit("--ip_feat 는 사면체 barycentric (--interp bary, --agg bary, hex_jac·gregory 없음) 만 지원한다")
    dev, dt = x.device, x.dtype
    hn = float(lat.s)
    hq = float(h)
    eps = float(getattr(a, "ip_feat_eps", 1e-3) or 1e-3)
    with torch.enable_grad():
        x = x.detach()
        v = v.detach()
        F = F.detach().to(dt)
        mass = mass.detach().to(dt)
        vol = vol.detach().to(dt)
        u0, _ = node_inertial(x, v, mass, rows, lam, Mn, cfg["g"], hq)
        N = rows.shape[0]
        U = u0.detach()[rows].clone().requires_grad_(True)                      # [N,4,3] 꼭짓점 사본
        vr = torch.arange(4 * N, device=dev).view(N, 4)
        dv = U.reshape(-1, 3)
        du = SX.g2p_pre(vr, lam, dv)
        _u1, Jm = SX.g2p_jac_pre(vr, lam, aux, lat, dv)
        q, Jm = bc_maps(a, cfg, x, x + du, Jm)
        I3 = torch.eye(3, device=dev, dtype=dt)
        with phys_resid.hess_mode():
            E, _dl, _Ft, parts = phys_resid.pts_ip_energy(
                x, q - x, v, F, I3 + Jm, mass, vol, cfg, hq, int(cfg["n_grid"]), float(cfg.get("grid_lim", 2.0)),
                g=_const_vec(cfg["g"], x), norm=None, free=free, parts_grad=True)
        e_in, e_el = parts[0], parts[1]
        G, = torch.autograd.grad(E, U, create_graph=True)
        Gel = torch.autograd.grad(e_el, U, retain_graph=True, allow_unused=True)[0]
        Gin = torch.autograd.grad(e_in, U, retain_graph=True, allow_unused=True)[0]
        Gel = torch.zeros_like(U) if Gel is None else Gel
        Gin = torch.zeros_like(U) if Gin is None else Gin
        Hb = torch.zeros(N, 4, 3, 3, device=dev, dtype=dt)
        for j in range(4):
            for c in range(3):
                T = torch.zeros_like(U)
                T[:, j, c] = 1.0
                Hv = torch.autograd.grad(G, U, T, retain_graph=not (j == 3 and c == 2), allow_unused=True)[0]
                if Hv is not None:
                    Hb[:, j, :, c] = Hv[:, j, :].detach()
    G, Gel, Gin = G.detach(), Gel.detach(), Gin.detach()
    r = rows.reshape(-1)
    gn = torch.zeros(Mn, 3, device=dev, dtype=dt).index_add_(0, r, G.reshape(-1, 3))
    gel = torch.zeros(Mn, 3, device=dev, dtype=dt).index_add_(0, r, Gel.reshape(-1, 3))
    gin = torch.zeros(Mn, 3, device=dev, dtype=dt).index_add_(0, r, Gin.reshape(-1, 3))
    Hn = torch.zeros(Mn, 3, 3, device=dev, dtype=dt).index_add_(0, r, Hb.reshape(-1, 3, 3))
    Mi = torch.zeros(Mn, device=dev, dtype=dt).index_add_(0, r, (lam * mass.unsqueeze(-1)).reshape(-1))
    s = (Mi / (hq * hq)).clamp_min(1e-30)
    Ht = Hn / s[:, None, None]
    Ht = 0.5 * (Ht + Ht.transpose(-1, -2))
    gt = gn / s[:, None]                                                         # 길이
    ev, Q = torch.linalg.eigh(Ht.double())
    ev = ev.clamp_min(0.0) + eps
    step = -(Q @ ((Q.transpose(-1, -2) @ gt.double().unsqueeze(-1)) / ev.unsqueeze(-1))).squeeze(-1)
    IPF_LAST[0] = dict(u0=u0.detach(), g=gn, g_el=gel, g_in=gin, H=Hn, M=Mi, s=s, Ht=Ht, gt=gt,
                       step=step.to(dt), E=E.detach(), hn=hn, eps=eps)
    H6 = torch.stack([Ht[:, 0, 0], Ht[:, 1, 1], Ht[:, 2, 2], Ht[:, 0, 1], Ht[:, 1, 2], Ht[:, 0, 2]], -1)
    out = torch.cat([gt / hn, gel / s[:, None] / hn, gin / s[:, None] / hn, H6, step.to(dt) / hn], -1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-FCAP, FCAP).detach()


_CONST = {}


def _const_vec(vals, ref):
    """파이썬 수 목록 -> ref 와 같은 장치·형의 텐서 (한 번 만들어 재사용; 매번 만들면 호스트→장치 복사로 동기화된다)."""
    k = (tuple(float(q) for q in vals), ref.device, ref.dtype)
    if k not in _CONST:
        _CONST[k] = torch.as_tensor(k[0], device=ref.device, dtype=ref.dtype)
    return _CONST[k]


PROBE = [None, None]                 # [요청 여부 "want", 격자별 탐침 텐서 목록]
C1_LAST = [None]                     # --interp c1wf: 마지막 advance 의 (Bernstein det 계수 [T,12,84], 조각 부피 [T,12])
INPUT_HOOK = [None]                  # 진단용: (망 입력 특징, net) -> 바꾼 특징


def bc_ipg_nodes(npos, dpn, cfg):
    """i-PG collide 커널과 같은 바닥: 면 아래 격자점(노드)의 변위 세 성분을 0 으로 (반경 채널은 둔다).

    surface_collider 는 i-PG 의 dot(격자점 - point, normal) < 0 판정 그대로 (dx 띠 없음, 변위 뒤
    위치 아님). bounding_box 는 i-PG grid 갱신처럼 영역 가장자리 bound 칸 안 노드의 바깥 방향
    성분만 0 으로 한다.
    """
    keep = torch.ones(npos.shape[0], 3, dtype=torch.bool, device=npos.device)
    for bc in (cfg.get("boundary_conditions") or []):
        t = bc.get("type")
        if t == "surface_collider":
            if str(bc.get("surface", "sticky")) != "sticky":
                raise SystemExit(f"--bc_ipg: sticky 아닌 바닥({bc.get('surface')})은 아직 없다")
            pt = torch.as_tensor(bc["point"], device=npos.device, dtype=npos.dtype)
            nr = torch.as_tensor(bc["normal"], device=npos.device, dtype=npos.dtype)
            nr = nr / nr.norm().clamp_min(1e-12)
            und = ((npos - pt) * nr).sum(-1) < 0.0
            keep = keep & ~und.unsqueeze(-1)
        elif t == "bounding_box":
            gl = float(cfg.get("grid_lim", 2.0)); dx = gl / float(cfg["n_grid"])
            b = 3.0 * dx                                   # i-PG add_bounding_box: padding = 3 고정
            d3 = dpn[:, :3].detach()
            keep = keep & ~(((npos < b) & (d3 < 0)) | ((npos >= gl - b) & (d3 > 0)))
    d = torch.where(keep, dpn[:, :3], torch.zeros_like(dpn[:, :3]))
    return d if dpn.shape[-1] == 3 else torch.cat([d, dpn[:, 3:]], -1)


def bc_softplus_map(q, Jm, cfg, eps):
    """**softplus 바닥 매핑** (--bc_softplus EPS): 노드는 그대로 두고, barycentric 으로 구한 입자 위치 q 를
    바닥 법선 좌표 s = n·(q-p) 에 대해 s' = ε·log(1+exp(s/ε)) 로 옮긴다 (접선 성분은 그대로).

        q <- q + (s' - s) n,      J_map = I + (σ(s/ε) - 1) n nᵀ,      F <- J_map F

    Jm 은 grad u (단위행렬 없음, g2p_jac_pre 규약) -- 돌려줄 때도 같은 규약 (J_map (I+Jm) - I).
    s ≫ ε 에서는 softplus 가 s 를 그대로 돌려줘 정확히 항등이다 (torch softplus threshold).
    det J_map = σ(s/ε) > 0.
    """
    I3 = torch.eye(3, device=q.device, dtype=q.dtype)
    Jf = I3 + Jm
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") != "surface_collider":
            continue
        pt = _const_vec(bc["point"], q)
        nr = _const_vec(bc["normal"], q)
        nr = nr / nr.norm().clamp_min(1e-12)
        s = ((q - pt) * nr).sum(-1)
        sp = torch.nn.functional.softplus(s, beta=1.0 / eps, threshold=20.0)
        q = q + (sp - s).unsqueeze(-1) * nr
        # softplus 의 threshold(βs>20 에서 s 를 그대로) 와 같은 경계로 -- 그 너머는 위치·야코비안 모두 정확히 항등
        sg = torch.where(s / eps > 20.0, torch.ones_like(s), torch.sigmoid(s / eps))
        nn_ = nr.unsqueeze(-1) * nr.unsqueeze(0)
        Jmap = I3 + (sg - 1.0)[:, None, None] * nn_
        Jf = Jmap @ Jf
    return q, Jf - I3


def bc_stick_map(x0, q, Jm, cfg, eps, xn_term=False):
    """**미끄러짐 방지 매핑** (--bc_stick EPS, --bc_softplus 와 함께; 그 앞에 적용): 스텝 시작 입자 위치 x0,
    barycentric 결과 q(=x̃) 에서 Δ = q - x0 를 바닥 법선 n 으로 Δ_n·Δ_t 로 나누고

        q' = x0 + Δ_n + w Δ_t,   w = σ(s̃/ε),   s̃ = n·(q - p)   (스텝 끝 = q 의 법선 거리)

    접선 축소는 법선 좌표를 바꾸지 않는다 (뒤따르는 softplus 의 s 가 그대로 s̃).
    야코비안 ∂q'/∂q = n nᵀ + w (I - n nᵀ) + w(1-w)/ε · Δ_t nᵀ  (w 의 s̃ 의존 항 포함), det = w².
    xn_term=True 면 F 에 x0 의 명시적 의존 (1-w)(I - n nᵀ) 도 더한다 (q' 를 x0 의 함수로 본 전미분;
    기본은 끔 = 사용자 정의 그대로 ∂q'/∂q 만 곱한다).
    s̃/ε > 20 에서는 softplus threshold 와 같은 경계로 w=1 -- 위치·야코비안 정확히 항등.
    Jm 은 grad u (단위행렬 없음) -- 돌려줄 때도 같은 규약.
    """
    I3 = torch.eye(3, device=q.device, dtype=q.dtype)
    Jf = I3 + Jm
    for bc in (cfg.get("boundary_conditions") or []):
        if bc.get("type") != "surface_collider":
            continue
        pt = _const_vec(bc["point"], q)
        nr = _const_vec(bc["normal"], q)
        nr = nr / nr.norm().clamp_min(1e-12)
        s = ((q - pt) * nr).sum(-1)
        far = s / eps > 20.0
        sg = torch.sigmoid(s / eps)
        w = torch.where(far, torch.ones_like(s), sg)
        dw = torch.where(far, torch.zeros_like(s), sg * (1.0 - sg) / eps)
        d = q - x0
        dt_ = d - (d * nr).sum(-1, keepdim=True) * nr
        q = q - (1.0 - w).unsqueeze(-1) * dt_
        nn_ = nr.unsqueeze(-1) * nr.unsqueeze(0)
        Pt = I3 - nn_
        Jst = nn_ + w[:, None, None] * Pt + dw[:, None, None] * (dt_.unsqueeze(-1) * nr.view(1, 1, 3))
        Jf = Jst @ Jf
        if xn_term:
            Jf = Jf + (1.0 - w)[:, None, None] * Pt
    return q, Jf - I3


def bc_maps(a, cfg, x0, q, Jm):
    """g2p 직후 입자 위치 q·grad u (Jm) 에 바닥 매핑을 advance 순서대로: --bc_stick (x0 = 스텝 시작 위치) -> --bc_softplus.
    advance 와 ip_node_feats 가 같이 쓴다."""
    if float(getattr(a, "bc_stick", 0.0) or 0.0) > 0.0:
        # **미끄러짐 방지 매핑** (--bc_stick): x 는 이 스텝 시작 입자 위치. softplus 앞에 -- 접선 성분만 줄이고
        # 법선 좌표는 그대로라 softplus 의 s 가 스텝 끝(q) 의 법선 거리 그대로다
        q, Jm = bc_stick_map(x0, q, Jm, cfg, float(a.bc_stick), bool(getattr(a, "bc_stick_xn", False)))
    if float(getattr(a, "bc_softplus", 0.0) or 0.0) > 0.0:
        # **softplus 바닥 매핑** (--bc_softplus): g2p 직후 입자 위치와 야코비안에 함께 -- 위치(x2)·속도(jvp)·
        # 관성·탄성(Jf) 이 모두 매핑된 위치 기준이 된다. 노드는 사영하지 않는다
        q, Jm = bc_softplus_map(q, Jm, cfg, float(a.bc_softplus))
    return q, Jm


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
    # 출력 탐침: PROBE[0] 이 "want" 면 격자마다 노드 출력에 더할 0 텐서(기울기 받음)를 만든다. 학습 쪽이 그것으로
    # 증분 포텐셜을 노드 출력으로 미분한다 (jvp 안에서 만든 출력은 바깥 그래프와 이어지지 않아 직접 미분할 수 없다).
    if PROBE[0] == "want":
        _no = getattr(net, "n_out", 3)
        PROBE[1] = [torch.zeros(e[1].shape[0], _no, device=x.device, dtype=x.dtype, requires_grad=True) for e in ens]
    # "fixed": 바깥에서 준 텐서(출력만 최적화 기준해의 변수)를 그대로 더한다

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
                    if INPUT_HOOK[0] is not None:          # 진단: 망 입력 특징을 바꿔 넣는다 (섭동·무작위 입력)
                        _ink = INPUT_HOOK[0](_ink, net)
                    out = net(_ink, _srk, _dsk, _clk, tau, mat=mat)
                if a.net_inertial:
                    if inertial is None or inertial[_k] is None:
                        raise SystemExit("--net_inertial 은 노드 관성 예측(--ov_init predict, 사면체)이 필요하다")
                    _vnk, _gqk = inertial[_k]
                    # τ 의 함수로 더해야 --v_from_dt 의 dΦ/dτ 에 관성 속도가 들어간다
                    _io = tau * _vnk + (tau * tau) * _gqk
                    if out[0].shape[-1] > 3:                       # 반경 채널(gregory)·C1 기울기 채널(c1wf)에는 더하지 않는다
                        _io = torch.cat([_io, torch.zeros_like(_io[:, :1]).expand(-1, out[0].shape[-1] - 3)], -1)
                    out = (out[0] + _io,) + tuple(out[1:])
                if dp_hook is not None:
                    out = dp_hook(out)
            dpn_k = torch.nan_to_num(out[0], nan=0.0, posinf=0.0, neginf=0.0)
            if PROBE[0] in ("want", "fixed"):
                dpn_k = dpn_k + PROBE[1][_k]
            if node_mask is not None and node_mask[_k] is not None:
                # **노드 변위를 명령으로 덮는다.** detach 로 기울기를 끊어
                # 그 노드는 최적화 대상에서 빠진다 (상수 경계조건이다).
                _pvk = (tau * ctrl_scale * node_vel[_k]).detach()
                if dpn_k.shape[-1] == 4 and _pvk.shape[-1] == 3:
                    # --gregory: 묶는 것은 변위 3 채널뿐, 반경 채널은 그대로 (4 채널과 3 채널이 브로드캐스트되지 않는다)
                    _pvk = torch.cat([_pvk, dpn_k[:, 3:]], -1)
                elif dpn_k.shape[-1] == WF.C1_NOUT and _pvk.shape[-1] == 3:
                    # --interp c1wf: 손잡이 노드는 명령 평행이동 -> 기울기·변 가로 미분 채널은 0 으로 묶는다
                    _pvk = torch.cat([_pvk, torch.zeros_like(dpn_k[:, 3:])], -1)
                dpn_k = torch.where(node_mask[_k].unsqueeze(-1), _pvk, dpn_k)
                if _k == 0 and on_pin is not None:
                    on_pin(tau)
            if getattr(a, "bc_ipg", False):
                # **i-PG(MPM) 와 같은 바닥** (--bc_ipg): 장벽·접선 벌점 없이, 격자점(노드) 자체의
                # 위치가 면 아래인 노드의 이번 스텝 변위를 세 성분 모두 0 으로 박는다 -- collide 커널
                # (surface_type 0, sticky: grid_x*dx 로 격자점 위치를 재서 dot(offset, n) < 0 이면
                # grid_v = 0) 와 같은 판정이다. 노드 격자는 원점 고정(정수배)이라 i-PG 격자처럼 공간에
                # 고정된 점이다. 마스크는 변수와 무관한 상수라 나머지 노드의 기울기는 그대로 흐른다.
                dpn_k = bc_ipg_nodes(_npk, dpn_k, cfg)
            if getattr(a, "gregory", False):
                # 셀 안 로컬 변환 (rep_ip ours 와 같은 Gregory·방사형 스키닝) + 노드 변위 글로벌 변환
                _duk, _Jk = gregory_skin(x, _npk, _rwk, dpn_k[:, :3], dpn_k[:, 3], float(a.node_h))
            elif getattr(_lxk, "bsp", None) is not None:
                # --interp bspline: 변위·야코비안 모두 같은 27 노드 2 차 B-스플라인에서 (일관)
                _duk = SX.g2p_pre(_rwk, _lmk, dpn_k[:, :3])
                _Jk = SX.bspline_jac_pre(_rwk, _lxk.bsp, _latk, dpn_k)
            elif getattr(_lxk, "wf", None) is not None:
                # --interp c1wf: 변위·야코비안 모두 Worsey–Farin C1 3 차 요소 다항식에서 (입자 F 는 실제 기울기)
                if str(getattr(a, "c1_base", "none")) == "pl":
                    # 기울기·변 채널 = 노드 값의 PL 장에서 만든 C1 자유도 + 출력(보정). 보정 0 이면 '매끄럽게 한 PL' --
                    # 관성 예측자(값만 있다)로 시작할 때 기울기 0 인 3 차 보간이 손잡이 경계에서 접히는 것을 막는다
                    dpn_k = torch.cat([dpn_k[:, :3], dpn_k[:, 3:] + WF.pl_c1_dofs(dpn_k[:, :3], _lxk.wft, dpn_k.shape[0])], 1)
                _duk, _Jk = WF.jac_pre(_rwk, _lxk.wf, _latk, dpn_k)
                if _k == 0:
                    C1_LAST[0] = (WF.det_coefs(dpn_k, _lxk.wft, _latk), _lxk.wft["svol"], dpn_k, _lxk.wft, _lok, _latk, _nnk, _uqk)
            elif getattr(_lxk, "nav", None) is not None:
                # --interp nodeavg: 변위는 barycentric 그대로, 야코비안만 노드 평균 기울기의 barycentric 보간
                _duk = SX.g2p_pre(_rwk, _lmk, dpn_k)
                _Jk = SX.nodeavg_jac_pre(_rwk, _lmk, _lxk.nav, _latk, dpn_k)
            else:
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
        q, _Jm = bc_maps(a, cfg, x, q, _Jm)          # --bc_stick -> --bc_softplus (ip_node_feats 와 같은 함수)
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
    _tau0 = torch.full((), float(h), device=x.device, dtype=x.dtype)      # 채우기 커널 -- 호스트 복사·동기화 없음
    if _use_dtS:
        with sec("jvp(속도)"):
            (x2, _outs), (v_next, _) = torch.func.jvp(
                _fieldS, (_tau0,), (torch.ones_like(_tau0),))
    else:
        x2, _outs = _fieldS(_tau0)
        v_next = (x2 - x) / h
    Jf = torch.eye(3, device=x.device, dtype=x.dtype) + _outs[1]
    return x2, v_next, _outs[0], Jf, _outs[2], _outs[3]


def fast_step(net, a, d, vel_scale, h, n_mat, n_film, dev="cuda", **kw):
    """no_grad 롤아웃용 빠른 스텝 (fast_roll.FastStep) 을 궤적 d 에 묶어 세운다.

    EvalModel.step 과 같은 입력(물성 특징·FiLM)을 만든다. train_deform 처럼
    **살아 있는 망**을 넘겨도 된다 -- 옵티마이저의 제자리 갱신은 그래프에 그대로
    반영되고, tau 상수는 파라미터 _version 이 바뀌면 다시 만든다.
    d 에는 "_X0" (정지 위치), "_mass" 가 있어야 한다 (EvalModel.prepare 참조).
    지원 밖 설정이면 NotImplementedError (fast_roll.fast_supported).

      fs = DM.fast_step(net, a, d, VEL_SCALE, h, N_MAT, N_FILM)
      with torch.no_grad():
          x, v, Jf = fs(x, v)
    """
    from .fast_roll import FastStep
    cfg = d["cfg"]
    mat_vec = (mat_feat(cfg, a.no_mat, dev) if (n_mat and not n_film) else None)
    mat = (mat_feat(cfg, a.no_mat, dev).reshape(1, n_mat) if n_film else None)
    return FastStep(net, a, cfg, d["_X0"], d["_mass"], vel_scale, h,
                    mat_vec=mat_vec, mat=mat, **kw)


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
    dt_cond=False, dt_scale=False, dt_sub=1, eval_dt_sub=0, ens_off=0, interp="bary",
    bc_soft=False, bc_mode="project", bc_level="node", bc_ipg=False, bc_softplus=0.0, bc_stick=0.0, bc_stick_xn=False, ov_init="predict",
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
        if getattr(a, "ip_feat", False):
            raise NotImplementedError("EvalModel: --ip_feat (상태 F 가 필요) 미지원 -- train_deform --infer 로 굴릴 것")
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
                fs = self._fs = fast_step(self.net, a, d, self.vel_scale, h,
                                          self.n_mat, self.n_film, self.dev,
                                          graph=self.fast_graph,
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
