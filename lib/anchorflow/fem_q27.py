"""절 CM101 묶음 q27 (사용자 11-10 지시): 요소 적분을 칸마다 가우스 3x3x3 (27 점) 으로 -- 이차 B-스플라인 칸의 정의 그대로 완전 적분.

발견 (조정자, 균일 격자 선형 탄성): 칸마다 2x2x2 적분은 강체 6 개 밖 영 에너지 (모래시계) 모드를 만든다 (3x3x3 칸 덩어리 30 개 대 3x3x3 적분 6 개,
1 칸 144 대 45). 이 모듈은 두 실행기 (새 fem_an_rt · 옛 fem_ao_rt (fem_an_rt_v0 바탕)) 의 고운 층 가우스 점을 27 점으로 바꾼다.

정의 (2x2x2 의 일반화, 새 문턱·클램프 없음):
  - 칸 국소 좌표 u ∈ [0, 1)^3 (칸 가운데 ½). 1 차원 가우스 3 점 ξ = ½ + {−½√(3/5), 0, +½√(3/5)}, 무게 w = {5, 8, 5}/18. 27 점 o = 9 i + 3 j + k,
    무게 W_o = w_i w_j w_k (합 1; 2x2x2 의 ⅛ 자리).
  - 점 몫 칸 (입자 -> 가우스 점, 2x2x2 의 팔분): 축마다 무게 길이로 나눈 세 칸 [0, 5/18) · [5/18, 13/18) · [13/18, 1).
    가우스 점 상태 (F 의 회전·로그 신장 평균, 뒤집힘, 질량·부피, 장벽 표본점, 옛 실행기 logJp·손상 상태) 는 그 몫 칸 입자에서 (2x2x2 와 같은 식).
  - V_q·m_q: 점유 가중 판 (κ = --gq_oct) = (1 − κ) 몫 칸 실제 + κ W_o × 칸 합 (2x2x2 의 '칸 합 / 8' 자리에 W_o), κ < 0 판 = W_o × 칸 합.
  - 보간 교체 (--fp_interp · --lj_interp, 입자 <- 가우스 점): 2x2x2 의 '칸 안 삼선형 (가우스 점 사이 선형, 바깥은 끝 점 값)' 을
    3 점 격자로 -- 축마다 이웃한 두 가우스 점 사이 선형 (u < ½ 이면 ξ0–ξ1, 아니면 ξ1–ξ2), 바깥 (u < ξ0, u > ξ2) 은 끝 점 값 (2x2x2 와 같은 규칙).
  - 거친 멀티그리드 층 (전처리기) 은 2x2x2 그대로 (풀리는 방정식은 고운 층, 전처리기는 근사); 고운 -> 거친 모으기는 27 점 위치로.
구현: 칸당 가우스 점 자리 32 (27 + 빈 자리 5, 부피·질량·기저 0) -- q = 32 c + o. 8 레인 묶음 (seg8·head) 커널은 그대로 (한 칸 = 8 레인 네 묶음, 노드 원자 합이라
  부분 합이 같은 결과). 용량: 칸 용량 capQc 를 4 배로 잡아 (가우스 점 버퍼 capQc×8 = 칸×32) 파이썬 쪽 할당을 그대로 둠, 칸 넘침 검사는 '칸 × 4 > capQc'.
  CUDA 소스는 cp.RawModule 을 감싸 커널 이름별로 고침 (고칠 곳 수를 셈 -- 어긋나면 오류). 커널 목록은 갈무리 (--q27cap) 에서 확인한 고운 층 커널 전부.
"""
import hashlib
import inspect
import json
import os
import sys
import traceback

import numpy as np

ST = dict(mode=0, cap=0, dg=0, exe=None, out=".", nsrc=0, nmod=0, ntr=0, kern={}, err=None, solver=None, nalloc=0)
SQ35 = 0.5 * (0.6 ** 0.5)
XI3 = [0.5 - SQ35, 0.5, 0.5 + SQ35]
W3 = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]


def log(msg):
    print("[절 CM101 q27] " + msg, flush=True)


def _info():
    try:
        with open(os.path.join(ST["out"], "q27info.json"), "w") as f_:
            json.dump({k: v for k, v in ST.items()}, f_, ensure_ascii=False, default=str, indent=1)
    except Exception:
        pass


def _caller():
    for fr in inspect.stack()[2:8]:
        fn = os.path.basename(fr.filename)
        if fn != "fem_q27.py":
            return "%s:%d" % (fn, fr.lineno)
    return "?"


def _capture(code):
    d = os.path.join(ST["out"], "q27src")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(code.encode()).hexdigest()[:10]
    fn = os.path.join(d, "%03d_%s.cu" % (ST["nsrc"], h))
    if not any(x.endswith(h + ".cu") for x in os.listdir(d)):
        with open(fn, "w") as f_:
            f_.write(code)
    with open(os.path.join(d, "manifest.txt"), "a") as f_:
        f_.write("%d %s %s %d\n" % (ST["nsrc"], h, _caller(), len(code)))


def _hook_rawmodule():
    import cupy as cp
    orig = cp.RawModule

    def RawModule(*a, **kw):
        code = kw.get("code") if "code" in kw else (a[0] if a else None)
        if isinstance(code, str):
            ST["nsrc"] += 1
            if ST["cap"]:
                _capture(code)
            if ST["mode"]:
                code2 = transform(code)
                if "code" in kw:
                    kw["code"] = code2
                else:
                    a = (code2,) + tuple(a[1:])
            _info()
        return orig(*a, **kw)

    cp.RawModule = RawModule


def _hook_solvers():
    """첫 풀이기 생성 때 격자 행렬·재질·점유 가중 κ·입자 부피·질량을 q27solver.pt 에 (모드 분석 exe/q27_eig.py 용, 결과 불변)."""
    import torch
    clss = []
    try:
        from . import fem_an_rt as ART
        clss.append(ART.RTSolver)
    except Exception:
        pass
    try:
        from . import fem_ao_rt as AORT
        clss += [AORT.AORTSolver, AORT.RTAdapt]
    except Exception:
        pass
    for C in clss:
        if getattr(C.__init__, "_q27", False):
            continue
        oi = C.__init__

        def init(s, *a, _oi=oi, _C=C, **kw):
            if ST["solver"] is None:
                try:
                    names = ["x", "v", "F", "m", "vol", "A", "Ai", "mu", "lam"]
                    d = {n: a[i] for i, n in enumerate(names) if i < len(a)}
                    sv = dict(cls=_C.__name__, goct=kw.get("goct", None))
                    for n in ("m", "vol", "A", "Ai", "mu", "lam"):
                        v = d.get(n)
                        sv[n] = v.detach().cpu() if torch.is_tensor(v) else v
                    torch.save(sv, os.path.join(ST["out"], "q27solver.pt"))
                    ST["solver"] = dict(cls=_C.__name__, goct=kw.get("goct", None), N=int(d["x"].shape[0]) if "x" in d else None,
                                        A=[float(q) for q in d["A"].reshape(-1).tolist()] if "A" in d and torch.is_tensor(d["A"]) else None,
                                        mu=(float(d["mu"]) if not torch.is_tensor(d.get("mu")) else "tensor"), lam=(float(d["lam"]) if not torch.is_tensor(d.get("lam")) else "tensor"))
                    _info()
                except Exception as e:
                    ST["err"] = "solver_info: " + repr(e)
                    _info()
            return _oi(s, *a, **kw)

        init._q27 = True
        C.__init__ = init


def install(out, mode, cap, dg, exe):
    ST.update(mode=int(mode), cap=int(cap), dg=int(dg), exe=exe, out=out)
    print("[절 CM101] q27 켬 mode %d cap %d dg %d 실행기 %s" % (ST["mode"], ST["cap"], ST["dg"], exe), flush=True)
    try:
        _hook_rawmodule()
        _hook_solvers()
        if ST["mode"]:
            _install_q27()
    except Exception as e:
        ST["err"] = repr(e) + " " + traceback.format_exc()[-1500:]
        _info()
        raise
    _info()


# ---------------------------------------------------------------- 27 점 판 (mode 1)
def transform(code):
    raise NotImplementedError("q27 mode 1 아직")


def _install_q27():
    raise NotImplementedError("q27 mode 1 아직")
