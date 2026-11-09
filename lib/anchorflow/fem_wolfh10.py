"""절 CM89 묶음 wolfh10 (11-09 위임): 옛 실행기 (수박, AORTSolver) 의 뒤집힘 없는 보폭 필터 --ls_inv 를 새 실행기 (RTSolver·MatRT: 레고·wolf·ficus·bread) 에 이식.

배경 (절 CM87): H8 wolf 납작 = 바닥 접촉 층 가우스 점 수천 개가 뒤집혀 (J −1 까지) 입자 det F 중앙 0.03, 부피 80 % 붕괴.
  옛 실행기는 --ls_inv (절 AS) 로 뉴턴 보폭이 가우스 점 J > 0 을 지키고, 수박 공통 글자 --ls_end (걸음 끝 변위도) · --y0u --y0l (시작점도) 를 씀. 새 실행기에는 없음.
정의 (옛 실행기 lib/anchorflow/fem_ao_rt.py AW_SRC 커널 글자 그대로 -- 새 문턱·클램프 없음):
  (가) 뉴턴 보폭 (--ls_inv + --ls_end): 갱신 rt_lsup 직전, 방향 T δ (aw_tdir) 로 가우스 점 (V_q > 0) 마다 J(α) = det((I + A(y + α T δ)) F_q) 와
       걸음 끝 변위 U_e = e1 y + e0 U (HHT e1 = 1/(1−αf), e0 = −αf/(1−αf)) 의 J 가 [0, 2^-k] 전체에서 > 0 인 가장 작은 k (aw_inv_q) -> 모든 점 최대 k -> 보폭 2^-k (aw_pick).
       보폭 = min(이 필터, 원래 보폭 (장벽 실현 가능 rt_kstep · 에너지 선탐색)); 원래 온 걸음 (full 1) 이면 필터 보폭 그대로 (옛 rt_lsup full = 0 if lsinv).
       시작에서 이미 J ≤ 0 인 점은 제약에 넣지 않고 셈만 (옛 정의 그대로).
  (나) 시작점 (--y0u --y0l): 걸음 시작점 다 정한 뒤 (s.ctl(1) 직전) 기준 y_b = 걸음 시작 변위 U 쪽 (aw_y0z_a; 고착 노드 기준은 절 CM85 H8 고침 η_z = √k P (U − ŷ) 꼴,
       --y0l 이라 거의 빈 고정 노드도 기준에서) 에서 시작점 쪽으로 같은 규칙의 최대 2^-k = β, η = η_z + β (η − η_z) (aw_y0z_b; β = 1 이면 손대지 않음).
  옛 판의 국소화 (--ws_bar·--ws_fi, 수박) 와 X2 신호 (절 CM85 S) 는 옮기지 않음 -- 전역 필터 정의 그대로.
훅 (실행기 runpy 순간, 가장 바깥): ART.build_src 에 AW_SRC 를 skipify 전에 넣음 (허용치 조기 종료를 지킴), RTSolver.k 의 rt_lsup 앞, RTSolver.ctl(1) 앞, alloc 버퍼.
기록: w10info.json (설치·훅 수), --w10dg 1 이면 w10.jsonl (프레임마다 [뉴턴: 줄인 반복, 최대 k, 시작에서 뒤집힌 점-회, 호출] + [시작점: 같은 넷]).
"""
import json
import os

import numpy as np
import torch

ST = dict(mode=0, dg=0, out=".", nsrc=0, nalloc=0, nnewton=0, nfull=0, nstart=0, nframe=0, err=None)
MARK = "// ---- 절 CM89 wolfh10"
LOG = [None]

# 절 CM85 H8 시작점 기준 고침 (fem_speed12 A_NEW1 의 앞줄, S 신호 없이)
A_OLD = "    float isk = (s > 0.f) ? 1.f/s : 0.f;\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];"
A_NEW = "    float isk = s;   // 절 CM85 (가) H8: 기준 η_z = √k P dd (T η_z = k P dd) -- 1/√k 로 나누지 않음\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];"

W10_SRC = r'''
// 절 CM89: 뉴턴 보폭 = 원래 보폭 (full 0) 과 뒤집힘 필터 보폭의 작은 쪽, 원래 온 걸음 (full 1) 이면 필터 보폭
__global__ void w10_comb(double* SC, const double* b, int full) { SC[14] = full ? b[0] : fmin(SC[14], b[0]); }
'''


def _src():
    from . import fem_ao_rt as AORT
    aw = AORT.AW_SRC
    assert aw.count(A_OLD) == 1, "wolfh10: aw_y0z_a 자리"
    return "\n" + MARK + "\n" + aw.replace(A_OLD, A_NEW) + W10_SRC


def add_src(src):
    if not isinstance(src, str) or MARK in src:
        return src
    if "void qnodes(" not in src or "void Tapply(" not in src:
        return src
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + _src() + "\n}\n"


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


def _dump():
    try:
        with open(os.path.join(ST["out"], "w10info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _nr(s):
    fl = getattr(s, "floor", None)
    nr = fl["NR"] if fl else [0.0, 0.0, 1.0]
    return [np.float32(q) for q in nr]


def _end(s):
    """옛 _aw_end (--ls_end) 와 같은 계수: HHT/GA y = (1−αf) U_{n+1} + αf U_n (새 실행기 rt_ga_step0·rt_ga_end 도 같은 꼴)."""
    if getattr(s, "ga", None):
        af = s.ga_c["af"]
        return (P(s.U), np.float32(1.0 / (1.0 - af)), np.float32(-af / (1.0 - af)))
    return (P(s.U), np.float32(1.0), np.float32(0.0))


def install(out, mode=1, dg=0):
    from . import fem_an_rt as ART
    ST.update(mode=int(mode), dg=int(dg), out=out)
    if ST["dg"]:
        LOG[0] = open(os.path.join(out, "w10.jsonl"), "w")
    ob = ART.build_src

    def build_src(src, *a, **kw):
        return ob(add_src(src), *a, **kw)
    ART.build_src = build_src
    R = ART.RTSolver
    oa, ok, oc, os_ = R.alloc, R.k, R.ctl, R.step

    def alloc(s, need):
        r = oa(s, need)
        Z = lambda *sh: torch.zeros(*sh, device=s.dev, dtype=torch.float32)
        s.w10k = torch.zeros(2, device=s.dev, dtype=torch.int32)
        s.w10st = torch.zeros(8, device=s.dev, dtype=torch.int32)
        s.w10b = torch.zeros(2, device=s.dev, dtype=torch.float64)
        s.w10bn = torch.zeros(2, device=s.dev, dtype=torch.float64)
        s.w10td = Z(s.capN * 3); s.w10ez = Z(s.capN * 3); s.w10yb = Z(s.capN * 3); s.w10dy = Z(s.capN * 3)
        ST["nalloc"] += 1
        return r

    def on(s):
        return getattr(s, "w10td", None) is not None and getattr(s, "Fq", None) is not None

    def k(s, name, n, args, b=128):
        if name == "rt_lsup" and on(s):                                                # args: IP SC sk pm n0 n1 n2 yh eta dl y full
            i32 = np.int32
            IP, capN, capQ = s.IP, s.capN, s.capQc * 8
            full = int(args[11])
            ok(s, "aw_tdir", capN, (args[0], args[2], args[3], args[4], args[5], args[6], args[9], P(s.w10td)), 256)
            ok(s, "aw_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), args[10], P(s.w10td), P(s.w10k)) + _end(s), 128)
            ok(s, "aw_pick", 1, (P(s.w10k), P(s.w10bn), P(s.w10st)), 1)
            ok(s, "w10_comb", 1, (args[1], P(s.w10bn), i32(full)), 1)
            ST["nnewton"] += 1
            ST["nfull"] += full
            args = tuple(args[:11]) + (i32(0),)
        return ok(s, name, n, args, b)

    def ctl(s, op, it=0):
        if op == 1 and on(s):                                                          # 걸음 시작점 다 정한 뒤 (뉴턴 고리 직전, 확장 w8_free 앞)
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
        return r

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if on(s):
            ST["nframe"] += 1
            if LOG[0] is not None:
                try:
                    w = s.w10st.cpu().tolist()
                    LOG[0].write(json.dumps(dict(f=ST["nframe"], newton=w[:4], start=w[4:])) + "\n"); LOG[0].flush()
                except Exception as e:                                                 # 진단만 (결과 불변)
                    ST["err"] = repr(e)
            if ST["nframe"] in (1, 2, 5) or ST["nframe"] % 25 == 0:
                _dump()
        return r

    R.alloc, R.k, R.ctl, R.step = alloc, k, ctl, step
    print(f"[절 CM89] wolfh10 켬 -- 모드 {ST['mode']} (뉴턴 보폭·시작점 뒤집힘 필터 --ls_inv --ls_end --y0u --y0l, 옛 실행기 AW_SRC 그대로), 진단 {ST['dg']}", flush=True)
    _dump()
