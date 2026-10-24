"""롤아웃(추론) 전용 **빠른 한 스텝** -- deform_model.EvalModel.step 과 같은 수식.

느린 경로(node_feats -> advance + torch.func.jvp) 의 한 스텝은 GPU 일보다 호스트
대기가 더 컸다 (실측 74 ms 중 GPU 43 ms, 커널 709 번). 원인은 셋이다.

  (1) 동기화: int(nn[k]) 세 번, torch.unique 세 번 (활성 노드·사면체·간선) 이
      매번 GPU 를 비우고 기다린다.
  (2) jvp: 망 전체를 두 벌(원값·탄젠트)로 돌리고, 쓰지도 않는 야코비안의
      탄젠트와 tau 에 무관한 앞부분까지 따라간다.
  (3) 커널 수: 작은 원소별 연산이 수백 개라 띄우는 비용이 쌓인다.

여기서는 수식을 바꾸지 않고 순서·자료구조만 바꾼다.

  - 활성 노드: unique 대신 **격자 표시 + 누적합**. 평탄색인 오름차순으로
    번호를 매기므로 압축 색인이 unique 의 것과 **똑같다** (노드 순서 동일).
  - 간선: 사면체·간선 unique 대신 노드마다 14 비트(간선 종류) 표를 칠하고
    누적합으로 압축한다. 집합이 같고 dst 별로 모여 나온다.
  - 크기는 **용량(capacity)** 으로 고정한다 (M, E, 격자 셀 수). 넘치면 스텝
    끝의 상태값(동기화 한 번)으로 알아채고 용량을 늘려 다시 잡고 그 스텝을
    다시 계산한다.
  - 탄젠트는 손으로 쓴다. tau 는 DtFiLM(gamma,beta) 와 출력 크기 dt_scale 로만
    들어오므로 그 앞(입력 특징·입력층)은 원값만, 그 뒤만 (원값, 탄젠트) 두
    줄로 흘린다. 탄젠트 식은 PyTorch 순방향 AD 공식과 같게 맞췄다
    (silu_backward, layer_norm_jvp, nan_to_num -> isfinite 곱).
  - 메시지 첫 선형층을 W1[h_src*e, h_dst] = W1a(h_src*e) + W1b h_dst 로 쪼개
    W1b h 는 노드에서 한 번만, 둘째 선형층은 합(평균) 뒤로 옮긴다
    (sum_e (W2 s_e + b2)/cnt = W2 (sum_e s_e / cnt) + b2, 대수적으로 같은 식).
  - 한 스텝 전체를 **CUDA 그래프** 하나로 잡아 호스트는 replay 한 번만 한다.

수치: 원값 경로의 연산은 대부분 원래와 같은 커널이라 같은 값이 나오고,
다른 것은 (a) index_add 의 atomic 순서 (원래도 실행마다 다르다), (b) 용량만큼
늘린 행렬곱의 커널 선택, (c) 쪼갠 행렬곱의 합 순서뿐이다 -- 모두 1 ulp 수준.

지원 범위 (그 밖은 느린 경로로 돌아간다 -- fast_supported 참조):
  sgnn, 격자 하나(ens_off<=1), 사면체(hex_jac 아님), node_h 고정 셀 크기,
  입자 단위 bc project 아님. net_inertial·물성 특징(mat_vec / mat FiLM)·
  dt_cond·dt_scale·v_from_dt 조합은 지원한다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import simplex as SX
from . import simplex_gnn as _SGM
from .simplex_gnn import node_centroids, node_moments, node_moments_finish
from . import fast_kernels as FK

__all__ = ["fast_supported", "FastStep"]

FCAP = None          # deform_model.FCAP (순환 import 를 피해 처음 쓸 때 채운다)

# 사슬 걸음 축 비트합 -> 양의 간선 종류 (SX.EDGE_OFFSETS 의 앞 7 개)
#   1:(1,0,0)->0  2:(0,1,0)->1  4:(0,0,1)->2  3:(1,1,0)->3  5:(1,0,1)->4
#   6:(0,1,1)->5  7:(1,1,1)->6
_BIT2CLS = [-1, 0, 1, 3, 2, 4, 5, 6]
# 24 칸 대칭 누적 -> node_moments 의 30 칸 배치
#   dxdx 위삼각 6 -> 9, 각운동량 3, dxdX 9, dXdX 위삼각 6 -> 9
_IDX30 = ([0, 1, 2, 1, 3, 4, 2, 4, 5] + [6, 7, 8] + list(range(9, 18))
          + [18 + i for i in [0, 1, 2, 1, 3, 4, 2, 4, 5]])
# 사면체 6 변 (꼭짓점 i<j)
_PI = [0, 0, 0, 1, 1, 2]
_PJ = [1, 2, 3, 2, 3, 3]


def fast_supported(a):
    """(지원 여부, 이유). a 는 체크포인트 args 네임스페이스."""
    if getattr(a, "arch", "sgnn") != "sgnn":
        return False, "arch != sgnn"
    if int(getattr(a, "ens_off", 0)) > 1:
        return False, "ens_off > 1 (격자 여러 개)"
    if getattr(a, "hex_jac", False):
        return False, "hex_jac"
    if str(getattr(a, "local_rot", "none") or "none") != "none":
        return False, "local_rot (셀 안 버블 회전은 융합 커널에 없다)"
    if getattr(a, "gregory", False):
        return False, "gregory (Gregory·방사형 스키닝은 아직 융합 커널에 없다)"
    if not float(getattr(a, "node_h", 0.0) or 0.0):
        return False, "node_h 없음 (셀 크기가 매 스텝 바뀌면 동기화가 필요하다)"
    if (not a.bc_soft) and a.bc_mode == "project" and a.bc_level == "particle":
        return False, "입자 단위 bc project"
    if getattr(a, "bc_ipg", False):
        return False, "bc_ipg (i-PG 방식 노드 바닥은 융합 커널에 없다)"
    if float(getattr(a, "bc_softplus", 0.0) or 0.0) > 0.0:
        return False, "bc_softplus (softplus 바닥 매핑은 융합 커널에 없다)"
    for k in ("control", "fe_state", "cell_poly", "refps"):
        if getattr(a, k, False):
            return False, k
    if int(getattr(a, "smooth_nb", 0)) or int(getattr(a, "ov_roll", 0)):
        return False, "smooth_nb / ov_roll"
    return True, ""


def _san(t):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp(-FCAP, FCAP)


class FastStep:
    """한 궤적(cfg, 정지 위치 X0, 질량)·한 스텝 크기 h 에 묶인 빠른 스텝.

      fs = FastStep(net, a, cfg, X0, mass, vel_scale, h)
      x2, v2, Jf = fs(x, v)          # EvalModel.step 과 같은 값

    net 은 **살아 있는 망**이어도 된다 (학습 중 no_grad 롤아웃). 그래프는 파라미터
    주소를 잡으므로 옵티마이저의 제자리 갱신은 그대로 반영되고, tau 에서만 오는
    상수(DtFiLM gamma/beta 와 그 탄젠트, 물성 FiLM)는 파라미터 _version 이 바뀌면
    다시 계산한다. 파라미터 텐서를 **새로 바꿔 끼우면** (load_state_dict 는 제자리
    복사라 괜찮다) fs.reset() 을 부를 것.

    graph=False 면 같은 계산을 즉시 실행으로 돈다 (디버그·비교용).
    clone_out=False 면 그래프의 정적 출력 버퍼를 그대로 돌려준다 -- 다음 호출에서
    덮어쓰이므로 호출자가 보관하지 않을 때만.
    """

    def __init__(self, net, a, cfg, X0, mass, vel_scale, h, *, mat_vec=None,
                 mat=None, graph=True, clone_out=True, cap=1.25, triton=True,
                 edge_triton=True):
        global FCAP
        if FCAP is None:
            from .deform_model import FCAP as _fc
            FCAP = _fc
        ok, why = fast_supported(a)
        if not ok:
            raise NotImplementedError(f"FastStep: {why}")
        self.net, self.a = net, a
        self.X0, self.mass = X0, mass
        self.dev = X0.device
        self.dtype = X0.dtype
        self.h = float(h)
        self.vel_scale = float(vel_scale)
        self.graph_on, self.clone_out, self.cap = bool(graph), bool(clone_out), float(cap)
        dev = self.dev
        # grid_for_nodes 의 h_fix 경로와 **같은 식**으로 격자를 만든다 (값 동일)
        hf = float(a.node_h)
        self.lat = SX.lattice(hf, a.hz_ratio * hf, device=dev, dtype=self.dtype)
        self.hn = self.lat.s
        self.use_dt = bool(a.v_from_dt and a.dt_cond)
        self.mat_vec = mat_vec
        self.mat = mat
        # 바운딩 상자 8 꼭짓점 선택표: 축 k 에서 (m>>k)&1 이면 b0 (아래), 아니면 b1
        self.cmask = torch.tensor([[bool((m >> k) & 1) for k in range(3)]
                                   for m in range(8)], device=dev)
        self.eye3l = torch.eye(3, device=dev, dtype=torch.long)
        self.eye3 = torch.eye(3, device=dev, dtype=self.dtype)
        self.off14 = torch.tensor(SX.EDGE_OFFSETS, device=dev, dtype=torch.long)
        self.opp14 = (torch.arange(14, device=dev) + 7) % 14
        self.bit2cls = torch.tensor(_BIT2CLS, device=dev, dtype=torch.long)
        self.PI = torch.tensor(_PI, device=dev)
        self.PJ = torch.tensor(_PJ, device=dev)
        self.idx30 = torch.tensor(_IDX30, device=dev)
        self.Ai_c = self.lat.Ai.contiguous()
        # 입자 단위 융합 커널 (Triton). 없으면 같은 식의 torch 연산판.
        self.tri = bool(triton) and FK.HAVE_TRITON and dev.type == "cuda"
        self.edge_tri = bool(edge_triton)
        # "pull": gather -> cuBLAS -> dst 구간 합 (기본), "dot": 한 커널 (tl.dot+atomic)
        self.edge_mode = edge_triton if isinstance(edge_triton, str) else "pull"
        self._bc_build(cfg)
        self._gq = torch.as_tensor(cfg["g"], device=dev, dtype=self.dtype)
        self.consts = None
        self._ver = None
        self.graph = None
        self.caps = None

    # ------------------------------------------------------------ 상수
    def _bc_build(self, cfg):
        """deform.bc_features 의 상수 텐서를 미리 (그래프 안에서는 H2D 복사 금지)."""
        dev, dt = self.dev, self.dtype
        self.bc = []
        for bc in cfg.get("boundary_conditions", []):
            t = bc["type"]
            if t == "surface_collider":
                n = torch.tensor(bc["normal"], device=dev, dtype=dt)
                n = n / n.norm().clamp(min=1e-12)
                q = torch.tensor(bc["point"], device=dev, dtype=dt)
                self.bc.append(("sc", n, q))
            elif t in ("cuboid", "enforce_particle_translation"):
                q = torch.tensor(bc["point"], device=dev, dtype=dt)
                s = torch.tensor(bc["size"], device=dev, dtype=dt)
                self.bc.append(("cb", q, s))

    def _bc_feat(self, p):
        out = []
        for kind, u, w in self.bc:
            if kind == "sc":
                n, q = u, w
                out.append(((p - q) * n).sum(-1, keepdim=True))
                out.append(n.reshape(1, 3).expand(p.shape[0], 3))
            else:
                q, s = u, w
                out.append(((p - q).abs() <= s).all(-1, keepdim=True).to(p.dtype))
        if not out:
            return torch.zeros(p.shape[0], 0, device=p.device, dtype=p.dtype)
        return torch.cat(out, -1)

    def _param_version(self):
        return sum(p._version for p in self.net.parameters())

    @torch.no_grad()
    def _make_consts(self):
        """tau 에서만 오는 값: DtFiLM (gamma, beta) 와 출력 크기 sc, 그리고 그 탄젠트.

        torch.func.jvp 를 **이 작은 부분에만** 써서 느린 경로와 같은 값을 얻는다.
        그래프 밖에서 만들어 정적 버퍼에 제자리 복사한다 (주소 고정).
        """
        net, dev, dt = self.net, self.dev, self.dtype
        tau = torch.as_tensor(self.h, device=dev, dtype=dt)
        one = torch.ones_like(tau)

        def f(t):
            outs = []
            if net.dtfilm is not None:
                g, b = net.dtfilm(t, dev)
                outs += [g.to(dt), b.to(dt)]
            if net.dt_scale:
                outs.append(net.scale * (t / net.dt_ref))
            return tuple(outs)

        if net.dtfilm is not None or net.dt_scale:
            prim, tang = torch.func.jvp(f, (tau,), (one,))
            prim, tang = list(prim), list(tang)
        c = {}
        if net.dtfilm is not None:
            c["g"], c["b"] = prim[0], prim[1]
            c["gt"], c["bt"] = tang[0], tang[1]
            prim, tang = prim[2:], tang[2:]
        if net.dt_scale:
            c["sc"], c["sct"] = prim[0], tang[0]
        else:
            c["sc"] = torch.as_tensor(float(net.scale), device=dev, dtype=dt)
            c["sct"] = torch.zeros((), device=dev, dtype=dt)
        if net.n_mat and self.mat is not None:
            g, b = net.mfilm(self.mat).chunk(2, -1)
            c["mg"], c["mb"] = 1.0 + g, b
        c["tau"] = tau
        if self.consts is None:
            self.consts = {k: v.clone() for k, v in c.items()}
        else:
            for k, v in c.items():
                self.consts[k].copy_(v)
        self._ver = self._param_version()

    def reset(self):
        """망을 바꿔 끼웠거나 설정을 바꾼 뒤 -- 상수·그래프를 처음부터."""
        self.consts, self.graph, self._ver = None, None, None

    # ------------------------------------------------------------ 한 스텝 (고정 크기)
    def _core(self, x, v, measure=False):
        """용량 크기로 고정된 한 스텝. 동기화·H2D 복사가 없어 그래프로 잡힌다.

        -> (x2, v2, Jf, stat[3] = (M, E, 격자 셀 수))
        measure=True 면 크기만 재고 돌아온다 (간선 크기 배열을 만들기 전).
        """
        Mc, Ec, Cc = self.caps
        a, lat, net, c = self.a, self.lat, self.net, self.consts
        dev = x.device
        x, v = x.contiguous(), v.contiguous()
        L = torch.long
        # ---- 격자 (SX.grid_for_nodes 의 h_fix 경로와 같은 연산; nn 은 장치에 둔다)
        lo_ = x.min(0).values
        hi_ = x.max(0).values
        ext = (hi_ - lo_).max().clamp_min(1e-12)
        pad = 0.05 * ext
        b0, b1 = lo_ - pad, hi_ + pad
        cor = torch.where(self.cmask, b0, b1)                       # [8,3]
        yb = cor @ lat.Ai.T
        imin = torch.floor(yb.min(0).values).long() - 1
        imax = torch.ceil(yb.max(0).values).long() + 1
        nn = (imax - imin + 1).clamp_min(2)
        lo = (lat.A @ imin.to(lat.A.dtype))
        ncell = nn.prod()
        y = (x - lo) @ lat.Ai.T             # 격자 좌표 (원래와 같은 행렬곱 -> 같은 값)
        hn = self.hn
        vn = v / self.vel_scale
        if self.tri:
            rows, lam, perm, comp, cs, mark, feat, etab = self._topo_tri(
                x, vn, y, nn, Mc, Cc)
            rank = None
        else:
            rows, lam, perm, rank, comp, cs, mark, etab = self._topo_torch(
                x, y, nn, Mc, Cc)
        Mt = cs[-1]
        tgt = torch.where(mark > 0, comp, Mc).clamp(0, Mc)
        uniq = torch.zeros(Mc + 1, device=dev, dtype=L)
        uniq.scatter_(0, tgt, self.cell_ar)        # Mc 칸(흡수 노드)은 아무 값이나 남는다
        # SX.node_pos 와 같은 연산
        k2 = uniq % nn[2]
        k1 = (uniq // nn[2]) % nn[1]
        k0 = uniq // (nn[1] * nn[2])
        ijk = torch.stack([k0, k1, k2], -1).to(lat.A.dtype)
        npos = ijk @ lat.A.T + lo                                   # [Mc+1,3]
        # ---- 노드 특징 (node_feats 와 같은 식; 노드 수만 용량)
        if self.tri:
            feat = _san(node_moments_finish(feat[0], feat[1], npos, hn))
        else:
            feat = _san(node_moments(x, vn, self.X0, self.mass,
                                     rows, lam, Mc + 1, npos, hn))
        extra = [self._bc_feat(npos) / hn]
        if self.mat_vec is not None:
            extra.insert(0, self.mat_vec.reshape(1, -1).expand(
                Mc + 1, self.mat_vec.numel()))
        _in = torch.cat([_san(feat), _san(torch.cat(extra, -1))], -1)
        _in = torch.nan_to_num(_in, nan=0.0, posinf=0.0,
                               neginf=0.0).clamp(-FCAP, FCAP)
        # etab[d, c] = 1 : d + OFF[c] 가 이웃. 그 이웃 -> d 간선의 종류는 opp(c).
        ecs = torch.cumsum(etab, 0, dtype=torch.int32)
        Et = ecs[-1]
        if measure:
            return torch.stack([Mt.long(), Et.long(), ncell.long()])
        epos = torch.where(etab > 0, ecs.long() - 1, Ec).clamp(0, Ec)
        ekey = torch.zeros(Ec + 1, device=dev, dtype=L)
        ekey.scatter_(0, epos, self.key_ar)
        valid = self.e_ar < Et
        self._et = Et                              # 간선 커널이 실제 간선 수만 돈다
        self._ecs = ecs                            # 노드 d 의 간선 = [ecs[14d-1], ecs[14d+13])
        dnode = torch.where(valid, ekey // 14, Mc)
        ecc = ekey % 14
        offf = (self.off14[:, 0] * nn[1] + self.off14[:, 1]) * nn[2] + self.off14[:, 2]
        nb = (uniq[dnode] + offf[ecc]).clamp(0, Cc - 1)
        src = torch.where(valid, comp[nb].clamp(0, Mc), Mc)
        dst = dnode
        cls = self.opp14[ecc]
        cnt = etab.view(Mc + 1, 14).sum(1, keepdim=True).to(x.dtype)
        # ---- 망 (원값 h, tau 탄젠트 t)
        h = net.inp((_in - net.in_mu) / net.in_sd)
        if "mg" in c:
            h = c["mg"] * h + c["mb"]
        t = None
        h, t = self._dtmod(h, t, 0)
        for i, blk in enumerate(net.body):
            h, t = self._mp(blk, h, t, src, dst, cls, cnt)
            h, t = self._dtmod(h, t, i + 1)
        o_raw = F.linear(h, net.out.weight, net.out.bias)
        o = o_raw * c["sc"]
        ot = None
        if self.use_dt:
            ot = F.linear(t, net.out.weight) * c["sc"] + o_raw * c["sct"]
        if a.net_inertial:
            vn = self._node_vel(x, v, rows, lam, Mc + 1)
            tau = c["tau"]
            o = o + tau * vn + (tau * tau) * self._gq
            if ot is not None:
                ot = ot + vn + (tau + tau) * self._gq
        dp = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
        dpt = (ot * torch.isfinite(o)) if ot is not None else None
        stat = torch.stack([Mt.long(), Et.long(), ncell.long()])
        if self.tri:
            # 융합 커널: 위치·속도(탄젠트)·야코비안을 한 번에
            x2 = torch.empty_like(x)
            v2 = torch.empty_like(x)
            Jf = torch.empty(x.shape[0], 3, 3, device=dev, dtype=x.dtype)
            FK.g2p(rows, lam, perm, dp.contiguous(),
                   None if dpt is None else dpt.contiguous(), x, self.Ai_c,
                   x2, v2, Jf)
            if dpt is None:
                v2 = (x2 - x) / self.h
            return x2, v2, Jf, stat
        # ---- 입자로 (g2p_pre / g2p_jac_pre 와 같은 식)
        dpc = dp[rows]                                              # [N,4,3]
        du = (lam.unsqueeze(-1) * dpc).sum(1)
        x2 = x + du
        d_ = (dpc[:, 1:] - dpc[:, :-1]).transpose(1, 2)
        Dy = d_.gather(2, rank.unsqueeze(1).expand(-1, 3, -1))
        G = Dy @ lat.Ai.to(Dy.dtype)
        Jf = self.eye3 + G / 1.0
        if dpt is not None:
            v2 = (lam.unsqueeze(-1) * dpt[rows]).sum(1)
        else:
            v2 = (x2 - x) / self.h
        return x2, v2, Jf, stat

    def _topo_torch(self, x, y, nn, Mc, Cc):
        """locate + 활성 노드 + 간선표 -- torch 연산판 (SX.locate 와 같은 연산)."""
        dev = x.device
        ci = y.floor().long()
        ci = torch.minimum(ci.clamp_min(0), nn - 2)
        f = (y - ci).clamp(0.0, 1.0)
        perm = torch.argsort(f, dim=1, descending=True)
        rank = torch.argsort(perm, dim=1)
        sv = f.gather(1, perm)
        lam = torch.stack([1.0 - sv[:, 0], sv[:, 0] - sv[:, 1],
                           sv[:, 1] - sv[:, 2], sv[:, 2]], -1)
        dirs = self.eye3l[perm]
        verts = torch.cat([torch.zeros_like(dirs[:, :1]),
                           dirs.cumsum(1)], 1) + ci.unsqueeze(1)    # [N,4,3]
        flat = ((verts[..., 0] * nn[1] + verts[..., 1]) * nn[2] + verts[..., 2])
        flat = flat.clamp(0, Cc - 1)               # 넘치면 결과는 버리지만 메모리는 안전하게
        # 활성 노드: 격자 표시 + 누적합 (= unique 의 오름차순 압축 색인)
        mark = torch.zeros(Cc, device=dev, dtype=torch.int32)
        mark.index_fill_(0, flat.reshape(-1), 1)
        cs = torch.cumsum(mark, 0, dtype=torch.int32)
        comp = cs.long() - 1
        rows = comp[flat].clamp(0, Mc)                              # [N,4]
        # 간선: 노드별 14 비트 표 (사면체 6 변 x 양방향). 걸음 r 의 축 비트 누적.
        # (cumsum 은 안쪽 축 3 칸짜리라도 느린 스캔 커널이 돌아 더하기로 푼다)
        bits = torch.bitwise_left_shift(torch.ones_like(perm), perm)
        z0 = torch.zeros_like(bits[:, 0])
        Bc = torch.stack([z0, bits[:, 0], bits[:, 0] + bits[:, 1],
                          bits[:, 0] + bits[:, 1] + bits[:, 2]], 1)   # [N,4]
        ecl = self.bit2cls[Bc[:, self.PJ] - Bc[:, self.PI]]          # [N,6]
        keys = torch.cat([rows[:, self.PI] * 14 + ecl,
                          rows[:, self.PJ] * 14 + ecl + 7], 1)
        etab = torch.zeros((Mc + 1) * 14, device=dev, dtype=torch.int32)
        etab.index_fill_(0, keys.reshape(-1), 1)
        return rows, lam, perm, rank, comp, cs, mark, etab

    def _topo_tri(self, x, vn, y, nn, Mc, Cc):
        """같은 일을 융합 커널 셋으로 (+ 노드 1·2 차 누적까지).

        -> rows[int32], lam, perm(축순열 압축), comp, cs, mark, (g1, g2), etab
        """
        dev, n = x.device, x.shape[0]
        flat = torch.empty(n, 4, device=dev, dtype=torch.int32)
        lam = torch.empty(n, 4, device=dev, dtype=x.dtype)
        perm = torch.empty(n, device=dev, dtype=torch.int32)
        mark = torch.zeros(Cc, device=dev, dtype=torch.int32)
        FK.locate_mark(y.contiguous(), nn.contiguous(), flat, lam, perm, mark)
        cs = torch.cumsum(mark, 0, dtype=torch.int32)
        comp = cs.long() - 1
        rows = torch.empty(n, 4, device=dev, dtype=torch.int32)
        g1 = torch.zeros(Mc + 1, 11, device=dev, dtype=x.dtype)
        etab = torch.zeros((Mc + 1) * 14, device=dev, dtype=torch.int32)
        FK.p2g_first(flat, cs, rows, lam, self.mass, x, self.X0, vn, perm, g1,
                     etab, Mc)
        # 중심 둘레 2 차 누적 (node_moments 의 g2; 대칭 둘은 위 삼각만 더해 거울로)
        _, cx, cX, cv = node_centroids(g1)
        c9 = torch.cat([cx, cX, cv], 1).contiguous()
        g2s = torch.zeros(Mc + 1, 24, device=dev, dtype=x.dtype)
        FK.p2g_second(rows, lam, self.mass, x, self.X0, vn, c9, g2s)
        g2 = g2s[:, self.idx30]
        return rows, lam, perm, comp, cs, mark, (g1, g2), etab

    def _node_vel(self, x, v, rows, lam, n_node):
        """deform_model.node_inertial 의 v_n (질량가중 barycentric 평균 속도)."""
        _mw = self.mass.unsqueeze(-1).to(x.dtype)
        _ws = torch.zeros(n_node, 1, device=x.device, dtype=x.dtype)
        _vs = torch.zeros(n_node, 3, device=x.device, dtype=x.dtype)
        for _j in range(rows.shape[1]):
            _w = lam[:, _j:_j + 1] * _mw
            _ws.index_add_(0, rows[:, _j], _w)
            _vs.index_add_(0, rows[:, _j], _w * v)
        return _vs / _ws.clamp_min(1e-30)

    def _dtmod(self, h, t, j):
        """SimplexGNN._dtmod 과 그 탄젠트: d(g h + b) = g' h + g t + b'."""
        c = self.consts
        if "g" not in c:
            return h, t
        j = min(j, c["g"].shape[0] - 1)
        g, b = c["g"][j], c["b"][j]
        hn = g * h + b
        if not self.use_dt:
            return hn, None
        gt, bt = c["gt"][j], c["bt"][j]
        tn = (gt * h + bt) if t is None else (gt * h + g * t + bt)
        return hn, tn

    def _mp(self, blk, h, t, src, dst, cls, cnt):
        """_MPLayer.forward 와 그 탄젠트.

        msg 첫 선형층을 W1a(h_src*e) + (W1b h + b1)_dst 로 쪼개고, 둘째 선형층은
        집계 뒤로 옮긴다 (선형이라 sum/평균과 순서를 바꿔도 같은 식).
        원값과 탄젠트는 [*, 2, H] 로 붙여 행렬곱·gather·index_add 를 한 번씩만.
        """
        H = h.shape[1]
        W1, b1 = blk.msg[0].weight, blk.msg[0].bias
        W2, b2 = blk.msg[2].weight, blk.msg[2].bias
        W1a, W1b = W1[:, :H], W1[:, H:]
        two = t is not None
        if self.tri and self.edge_tri:
            # 융합 커널: 간선 쪽 (gather·임베딩 곱·W1a·+P·SiLU·탄젠트·dst 합) 한 번
            ht = torch.stack([h, t if two else h], 1).contiguous()  # [M,2,H]
            P = F.linear(ht.view(-1, H), W1b).view(-1, 2, H)
            P[:, 0] += b1
            Sn = torch.empty_like(ht)
            if self.edge_mode == "dot":
                Sn.zero_()
                FK.edge_msg(ht, P.contiguous(), src, dst, cls, self._et, blk.emb.weight,
                            W1, Sn, two)
            else:
                # gather·임베딩 곱 (Triton) -> W1a 행렬곱 (cuBLAS fp32) -> dst 별 연속
                # 구간을 모아 +P·SiLU·탄젠트·합 (Triton, atomic 없음)
                nt = 2 if two else 1
                xs = torch.empty(src.shape[0], nt, H, device=h.device, dtype=h.dtype)
                FK.edge_gather(ht, src, cls, self._et, blk.emb.weight, xs, nt)
                Z = F.linear(xs.view(-1, H), W1a).view(-1, nt, H)
                FK.edge_pull(Z, P.contiguous(), self._ecs, Sn, nt)
            if _SGM._MP_MEAN:
                Sn = Sn / cnt.clamp_min(1.0).unsqueeze(-1)
                agg = F.linear(Sn[:, 0], W2, b2)
            else:
                agg = F.linear(Sn[:, 0], W2) + cnt * b2
            aggt = F.linear(Sn[:, 1], W2) if two else None
        elif two:
            e = blk.emb.weight[cls]                                 # [E,H]
            ht = torch.stack([h, t], 1)                             # [M,2,H]
            xs = ht[src] * e.unsqueeze(1)                           # [E,2,H]
            Z = F.linear(xs.view(-1, H), W1a).view(-1, 2, H)
            P = F.linear(ht.view(-1, H), W1b).view(-1, 2, H)
            P[:, 0] += b1
            Z = Z + P[dst]
            z, zt = Z[:, 0], Z[:, 1]
            S = torch.empty_like(Z)
            S[:, 0] = F.silu(z)
            S[:, 1] = torch.ops.aten.silu_backward(zt, z)
            Sn = torch.zeros(h.shape[0], 2, H, device=h.device, dtype=h.dtype)
            Sn.index_add_(0, dst, S)
            if _SGM._MP_MEAN:
                Sn = Sn / cnt.clamp_min(1.0).unsqueeze(-1)
                agg = F.linear(Sn[:, 0], W2, b2)
            else:
                agg = F.linear(Sn[:, 0], W2) + cnt * b2
            aggt = F.linear(Sn[:, 1], W2)
        else:
            e = blk.emb.weight[cls]
            z = F.linear(h[src] * e, W1a) + F.linear(h, W1b, b1)[dst]
            Sn = torch.zeros_like(h).index_add_(0, dst, F.silu(z))
            if _SGM._MP_MEAN:
                agg = F.linear(Sn / cnt.clamp_min(1.0), W2, b2)
            else:
                agg = F.linear(Sn, W2) + cnt * b2
        W3, b3 = blk.upd[0].weight, blk.upd[0].bias
        W4, b4 = blk.upd[2].weight, blk.upd[2].bias
        u = F.linear(torch.cat([h, agg], -1), W3, b3)
        r = F.linear(F.silu(u), W4, b4)
        if two:
            ut = F.linear(torch.cat([t, aggt], -1), W3)
            rt = F.linear(torch.ops.aten.silu_backward(ut, u), W4)
        nrm = blk.nrm
        if isinstance(nrm, torch.nn.LayerNorm):
            y, mean, rstd = torch.native_layer_norm(r, nrm.normalized_shape,
                                                    nrm.weight, nrm.bias, nrm.eps)
            if two:
                # FunctionsManual.cpp layer_norm_jvp 와 같은 식
                mean_t = rt.mean(-1, keepdim=True)
                rstd_t = -rstd.pow(3) * ((rt - mean_t) * (r - mean)).mean(-1, keepdim=True)
                yt = ((rt - mean_t) * rstd + (r - mean) * rstd_t) * nrm.weight
        else:
            y = r
            yt = rt if two else None
        return h + y, ((t + yt) if two else None)

    # ------------------------------------------------------------ 용량·그래프
    def _set_caps(self, Mc, Ec, Cc):
        dev = self.dev
        self.caps = (int(Mc), int(Ec), int(Cc))
        self.cell_ar = torch.arange(Cc, device=dev)
        self.key_ar = torch.arange((Mc + 1) * 14, device=dev)
        self.e_ar = torch.arange(Ec + 1, device=dev)
        self.graph = None

    def _grow(self, st):
        M, E, C = st
        Mc, Ec, Cc = self.caps
        k = self.cap
        self._set_caps(max(Mc, int(M * k) + 64), max(Ec, int(E * k) + 256),
                       max(Cc, int(C * k) + 1024))

    @torch.no_grad()
    def _measure(self, x, v):
        """첫 호출: 넉넉한 용량으로 한 번 즉시 실행해 크기를 잰다."""
        n = x.shape[0]
        self._set_caps(4 * n + 64, 1, 1 << 24)
        st = self._core(x, v, measure=True).tolist()
        self._set_caps(int(st[0] * self.cap) + 64, int(st[1] * self.cap) + 256,
                       int(st[2] * 1.5) + 1024)

    @torch.no_grad()
    def _capture(self, x, v):
        self.x_in = x.detach().clone()
        self.v_in = v.detach().clone()
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):                  # 예열 (cuBLAS 핸들·작업공간)
                self._core(self.x_in, self.v_in)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self._core(self.x_in, self.v_in)
        self.graph = g

    def _ok(self, st):
        M, E, C = st
        Mc, Ec, Cc = self.caps
        return M <= Mc and E <= Ec and C <= Cc

    @torch.no_grad()
    def __call__(self, x, v):
        if self.consts is None or self._param_version() != self._ver:
            self._make_consts()
        if self.caps is None:
            self._measure(x, v)
        for _ in range(8):
            if self.graph_on:
                if self.graph is None:
                    self._capture(x, v)
                self.x_in.copy_(x)
                self.v_in.copy_(v)
                self.graph.replay()
                out = self.out
            else:
                out = self._core(x, v)
            st = out[3].tolist()                # 스텝당 동기화 한 번 (용량 확인)
            if self._ok(st):
                break
            self._grow(st)                       # 넘쳤다 -- 늘려서 이 스텝을 다시
        else:
            raise RuntimeError(f"FastStep: 용량을 늘려도 맞지 않는다 {st} {self.caps}")
        self.last_stat = st
        x2, v2, Jf = out[:3]
        if self.clone_out and self.graph_on:
            return x2.clone(), v2.clone(), Jf.clone()
        return x2, v2, Jf
