"""절 CM91 묶음 wolfh11 (11-09 위임): 뒤집힘 막기를 몸 전체 보폭이 아니라 노드별 보폭으로 (새 실행기 RTSolver·MatRT: 레고·wolf·ficus·bread).

배경 (절 CM89): 전역 뒤집힘 필터 L 은 뒤집힘을 0 으로 만들지만 H8L 은 뉴턴 반복 94 중 67 번 몸 전체 보폭을 잘라 모래가 탑처럼 섬.
정의 = 옛 실행기 노드별 보폭 (절 CM29 (a)·CM30 (b)·CM32 fi, 수박 공통 글자 --ws_bar 4 --ws_fi; 커널 lib/anchorflow/fem_wmsparse.py WS_SRC·_inv_q 글자 그대로,
       라운드 순서는 fem_wmfrac2.newton (fi) 그대로). 새 문턱·클램프 없음 (노드 k 의 60 상한은 옛 커널 ws_nk·ws_bk 글자 그대로).
  뉴턴 갱신 rt_lsup 자리에서, 방향 T δ (aw_tdir):
  (0) 원래 보폭 s_o = SC[14] (에너지 선탐색·장벽 실현 가능; 온 걸음 판 full 1 이면 1). 노드 k 시작 = k_o (s_o = 2^-k_o) -- 옛 fi 의 '온 보폭' 자리
      (옛 수박 실행기에는 에너지 선탐색이 없어 시작이 0; 여기서는 원래 보폭이 몸 전체 선탐색이라 그대로 둠).
  (1) 라운드 R + 1 = 5 번: 후보 y_n = ŷ + T (η + 2^-kn δ) (ws_cand) 에서 뒤집힘 판정 (aw_inv_q 와 같은 식, 걸음 끝 변위 --ls_end 포함) 을 어기는
      가우스 점 q (k_q > 0) 의 27 노드에 kn + k_q, 장벽 판이면 관통할 활성 표본점 (rt_bar_kfe 와 같은 식) 의 27 노드에도 kn + k_q (ws_nk·ws_bk), 노드마다 최대 (ws_merge).
      -> 각 노드 보폭 = 그 노드가 걸친 가우스 점이 뒤집히지 않는 최대 2^-k (노드마다 보폭이 달라 방향이 바뀌므로 확인 라운드).
  (2) 최종 확인에서 아직 어기는 점이 있으면 그 반복은 전역 보폭 = min(s_o, 전역 뒤집힘 필터) (절 CM89 L 과 같은 값) 로 되돌림 (ws_fb; 옛 정의의 되돌림).
  (3) 갱신 = rt_lsup 와 같은 식, 보폭만 노드별 2^-kn (s_o = 0 이면 0). SC[14] = 노드 보폭 최소 (덜 풀린 걸음 판정 cj_min 이 읽음; 옛 정의 ws_sc 와 같음).
  시작점 (--y0u --y0l, 절 CM89 (나)) 은 옛 정의대로 전역 β 그대로 (옛 실행기의 노드별 보폭은 뉴턴 갱신만).
훅: fem_wolfh10 과 같은 자리 (ART.build_src, RTSolver.k 의 rt_lsup, RTSolver.ctl(1), alloc).
기록: w11info.json, --w11dg 1 이면 w11.jsonl (프레임마다 [전역 필터였다면: 자를 반복, 최대 k, 시작에서 뒤집힌 점-회, 호출]·[시작점 같은 넷]·
      [노드별: -, 되돌림 반복, 최대 노드 k, 자른 노드 합 (kn > k_o), 반복 수, 최대 전역 필터 k, -, -]).
"""
import json
import os

import numpy as np
import torch

from . import fem_wolfh10 as W10
from .fem_wolfh10 import P, _nr, _end

ST = dict(mode=0, dg=0, out=".", nsrc=0, nalloc=0, nnewton=0, nfull=0, nstart=0, nframe=0, nbar=0, R=4, err=None)
MARK = "// ---- 절 CM91 wolfh11"
LOG = [None]
R_ROUNDS = 4                                                                          # 옛 --ws_bar 4 (fi 는 라운드 하나 더)

W11_SRC = r'''
// 절 CM91: 원래 보폭 s_o 저장 + 노드 k 시작 = k_o
__global__ void w11_init(const int* IP, const double* SC, double* so, int* kn, int* kn2, int full) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    double s = full ? 1.0 : SC[14];
    if (n == 0) so[0] = s;
    if (n >= IP[9]) return;
    int k = (s > 0.0) ? -ilogb(s) : 0; if (k < 0) k = 0;
    kn[n] = k; kn2[n] = 0;
}
// 되돌림 전역 k = min(s_o, 전역 필터) 의 k (w10_comb 뒤 SC[14])
__global__ void w11_kg(const double* SC, int* kg) { double s = SC[14]; int k = (s > 0.0) ? -ilogb(s) : 0; kg[0] = k < 0 ? 0 : k; }
// SC[14] = 노드 보폭 최소 (s_o = 0 이면 0)
__global__ void w11_sc(double* SC, const double* so, const int* km, int* st) { SC[14] = (so[0] > 0.0) ? (double)ldexpf(1.f, -km[0]) : 0.0; st[2] = max(st[2], km[0]); }
// 갱신 = rt_lsup (full 0) 과 같은 식, 보폭만 노드별 2^-kn
__global__ void w11_lsup(const int* IP, const double* so, const int* kn, const float* sk, const int* pm, float n0, float n1, float n2, const float* yh, float* eta, const float* dl, float* y) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float st = (so[0] > 0.0) ? ldexpf(1.f, -kn[n]) : 0.f;
    float e[3]; for (int a=0;a<3;a++) { e[a] = eta[n*3+a] + st*dl[n*3+a]; eta[n*3+a] = e[a]; }
    float t[3]; Tapply(sk[n], pm[n], n0, n1, n2, e, t);
    for (int a=0;a<3;a++) y[n*3+a] = yh[n*3+a] + t[a];
}
'''


def _cut(src, name):
    i = src.index("__global__ void " + name + "(")
    j = src.index("\n}\n", i) + 3
    return src[:i] + src[j:]


def _ws_src():
    from .fem_wmsparse import WS_SRC, _inv_q
    s = WS_SRC
    s = _cut(s, "ws_init")                                                          # 시작은 w11_init (원래 보폭 k_o)
    s = _cut(s, "ws_lsup")                                                          # 옛 압축 판 (AO_ACT) 자리 -- 새 실행기는 w11_lsup
    s = _cut(s, "ws_sc")                                                            # w11_sc (s_o = 0 처리)
    s = s.replace("WS_G void ws_zero", "__global__ void ws_zero")
    i = s.index("__global__ void __launch_bounds__(128) ws_bk(")                     # 장벽 판에만 (AF_BX·AF_BACT)
    j = s.index("\n}\n", i) + 3
    s = s[:i] + "#ifdef AF_BAR\n" + s[i:j] + "#endif\n" + s[j:]
    return _inv_q() + s


def add_src(src):
    if not isinstance(src, str) or MARK in src:
        return src
    if "void qnodes(" not in src or "void Tapply(" not in src:
        return src
    ST["nsrc"] += 1
    s = W10.add_src(src)                                                            # 절 CM89 AW_SRC (aw_*, 시작점 고침) + w10_comb
    s = s.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n" + MARK + "\n" + _ws_src() + W11_SRC + "\n}\n"


def _dump():
    try:
        with open(os.path.join(ST["out"], "w11info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out, mode=1, dg=0):
    from . import fem_an_rt as ART
    ST.update(mode=int(mode), dg=int(dg), out=out, R=R_ROUNDS)
    if ST["dg"]:
        LOG[0] = open(os.path.join(out, "w11.jsonl"), "w")
    ob = ART.build_src

    def build_src(src, *a, **kw):
        return ob(add_src(src), *a, **kw)
    ART.build_src = build_src
    R = ART.RTSolver
    oa, ok, oc, os_ = R.alloc, R.k, R.ctl, R.step

    def alloc(s, need):
        r = oa(s, need)
        I = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.int32)
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.w10k = I(2); s.w10st = I(8)
        s.w10b = torch.zeros(2, device=s.dev, dtype=torch.float64); s.w10bn = torch.zeros(2, device=s.dev, dtype=torch.float64)
        s.w10td = Z(s.capN * 3); s.w10ez = Z(s.capN * 3); s.w10yb = Z(s.capN * 3); s.w10dy = Z(s.capN * 3)
        s.wskn = I(s.capN); s.wskn2 = I(s.capN); s.wskq = I(s.capQc * 8); s.wskk = I(2)
        s.wskg = I(1); s.wskm = I(1); s.wsfl = I(64); s.w11kb = I(1)
        s.wsy1 = Z(s.capN * 3); s.wsdd = Z(s.capN * 3)
        s.wsst = I(8); s.w11so = torch.zeros(1, device=s.dev, dtype=torch.float64)
        ST["nalloc"] += 1
        return r

    def on(s):
        return getattr(s, "wskn", None) is not None and getattr(s, "Fq", None) is not None

    def k(s, name, n, args, b=128):
        if name == "rt_lsup" and on(s):                                                # args: IP SC sk pm n0 n1 n2 yh eta dl y full
            i32, f = np.int32, np.float32
            IP, capN, capQ = s.IP, s.capN, s.capQc * 8
            full = int(args[11])
            F3 = (args[4], args[5], args[6])
            aend = _end(s)
            ok(s, "w11_init", capN, (P(IP), args[1], P(s.w11so), P(s.wskn), P(s.wskn2), i32(full)), 256)
            ok(s, "aw_tdir", capN, (args[0], args[2], args[3], *F3, args[9], P(s.w10td)), 256)
            # 전역 필터 (절 CM89 L 의 값) -- 되돌림 보폭과 진단 (전역이었다면 자를 반복)
            ok(s, "aw_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), args[10], P(s.w10td), P(s.w10k)) + aend, 128)
            ok(s, "aw_pick", 1, (P(s.w10k), P(s.w10bn), P(s.w10st)), 1)
            ok(s, "w10_comb", 1, (args[1], P(s.w10bn), i32(full)), 1)
            ok(s, "w11_kg", 1, (args[1], P(s.wskg)), 1)
            ok(s, "rt_zero_i", 64, (i32(64), P(s.wsfl)), 64)
            bar = bool(getattr(s, "bar", False)) and not getattr(s, "nobk", False)
            if bar:
                FL = s.floor
                thv = f(aend[1])
                bargs = lambda r_, sc: (P(IP), P(s.cb), P(s.BW), args[10], P(s.wsy1), P(s.U), thv, *F3, f(FL["PN"]), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc))

            def check(r_, sc):
                ok(s, "ws_cand", capN, (P(IP), P(s.wskn), args[2], args[3], *F3, args[7], args[8], args[9], P(s.w10td), P(s.wsy1), P(s.wsdd)), 256)
                ok(s, "rt_zero_i", capQ, (i32(capQ), P(s.wskq)), 256)
                ok(s, "rt_zero_i", 2, (i32(2), P(s.wskk)), 32)
                ok(s, "ws_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), args[10], P(s.wsdd), P(s.wskk)) + aend + (P(s.wskq),), 128)
                ok(s, "ws_nk", capQ, (P(IP), P(s.cb), P(s.wskq), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc)), 128)
                if bar:
                    ok(s, "ws_bk", capQ, bargs(r_, sc), 128)
            RR = R_ROUNDS + 1                                                          # fi: 첫 라운드 (원래 보폭 후보) + 확인 R
            for r_ in range(RR):
                check(r_, 1)
                ok(s, "ws_merge", capN, (P(IP), P(s.wskn), P(s.wskn2)), 256)
            check(RR, 0)
            ok(s, "rt_zero_i", 1, (i32(1), P(s.wskm)), 32)
            ok(s, "ws_fb", capN, (P(IP), P(s.wsfl[2 * RR:]), P(s.wskg), P(s.w11kb), P(s.wskn), P(s.wskm), P(s.wsst)), 256)
            ok(s, "w11_sc", 1, (args[1], P(s.w11so), P(s.wskm), P(s.wsst)), 1)
            ST["nnewton"] += 1
            ST["nfull"] += full
            ST["nbar"] += int(bar)
            return ok(s, "w11_lsup", capN, (P(IP), P(s.w11so), P(s.wskn), args[2], args[3], *F3, args[7], args[8], args[9], args[10]), 256)
        return ok(s, name, n, args, b)

    def ctl(s, op, it=0):
        if op == 1 and on(s):                                                          # 시작점: 절 CM89 (나) 그대로 (전역 β)
            i32, f = np.int32, np.float32
            IP, capN, capQ = s.IP, s.capN, s.capQc * 8
            nr = _nr(s)
            mn = getattr(s, "mn", None)
            mthr = f(getattr(s, "mthr", 0.0))
            ok(s, "aw_y0z_a", capN, (P(IP), P(s.sk), P(s.pm), *nr, P(s.yh), P(s.U), P(s.y), P(s.w10ez), P(s.w10yb), P(s.w10dy), P(mn), mthr, i32(1)), 256)
            ok(s, "aw_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.w10yb), P(s.w10dy), P(s.w10k)) + _end(s), 128)
            ok(s, "aw_pick", 1, (P(s.w10k), P(s.w10b), P(s.w10st[4:])), 1)
            ok(s, "aw_y0z_b", capN, (P(IP), P(s.w10b), P(s.sk), P(s.pm), *nr, P(s.yh), P(s.w10ez), P(s.eta), P(s.y), P(mn), mthr, i32(1), P(s.w10yb), P(s.w10dy)), 256)
            ST["nstart"] += 1
        r = oc(s, op, it)
        if op == 0 and on(s):                                                          # 프레임 시작: 통계 비움 (그래프 안)
            ok(s, "rt_zero_i", 1, (np.int32(8), P(s.w10st)), 32)
            ok(s, "rt_zero_i", 1, (np.int32(8), P(s.wsst)), 32)
        return r

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if on(s):
            ST["nframe"] += 1
            if LOG[0] is not None:
                try:
                    w = s.w10st.cpu().tolist(); v = s.wsst.cpu().tolist()
                    LOG[0].write(json.dumps(dict(f=ST["nframe"], glob=w[:4], start=w[4:], node=v)) + "\n"); LOG[0].flush()
                except Exception as e:                                                 # 진단만 (결과 불변)
                    ST["err"] = repr(e)
            if ST["nframe"] in (1, 2, 5) or ST["nframe"] % 25 == 0:
                _dump()
        return r

    R.alloc, R.k, R.ctl, R.step = alloc, k, ctl, step
    print(f"[절 CM91] wolfh11 켬 -- 모드 {ST['mode']} (뉴턴 노드별 뒤집힘 보폭 = 옛 --ws_bar {R_ROUNDS} --ws_fi 정의, 시작점 전역 β = 절 CM89), 진단 {ST['dg']}", flush=True)
    _dump()
