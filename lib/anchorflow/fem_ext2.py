"""절 CM84 묶음 ext2 (11-08 위임): 공통 후보 H7 = H5 + S_구속 + C + [κ 로 고르는 (확장 T1E2 + P1X 강성 × κ)].

배경 (절 CM83): 확장 B-스플라인 T1E2 (절 CM81) 는 wolf (모래, κ 0) 를 맞추나 (f50 505~510), κ 1 재질 (레고·ficus·bread·수박) 에서는
  T1 내부 칸 정의가 겉층 전체를 외삽으로 묶어 잠금 (레고 발산, ficus 잎 진동 사라짐, bread·수박 오차 악화, 수박 깨짐 0).
정의 H7 (새 문턱 없음 -- 재질 정의에서 나오는 κ = 2σ_t/(σ_t+σ_c) (절 CM75, fem_lightg4.kappa_of 그대로) 만 씀):
  - 노드 κ_n = 노드 받침 (매듭 칸 [n−2, n]³) 안 점유 칸 재질 κ 의 최솟값.
    근거: 확장 묶기와 P1X 강성 빼기는 둘 다 '이 노드 자유도를 이웃 칸과 탄성으로 이을 근거가 있는가' 의 답 -- 받침 안 칸 하나라도 인장 강도 0 이면
    (모래) 그 칸에서 몸이 갈라져 흩어질 수 있으므로 그 노드는 흐름 쪽 (κ 가 작은 쪽) 처리를 받음. 최솟값은 κ 0 칸이 하나라도 닿은 노드를 모래로 보는 것.
  - κ_n = 0 노드: 외부 노드면 확장으로 묶음 (T1E2, 절 CM81.0 그대로), P1X 강성은 면 κ 배 (= 0; 절 CM77 (K)).
  - κ_n > 0 노드: 확장 안 함 (원래 자유도), P1X 강성 × κ (κ 1 이면 × 1.0 = 비트 같음) -- 곧 κ 1 재질은 H5 (+ S_구속 + C) 와 같은 계산.
  지금 다섯 씬은 모두 재질 하나 (씨 영역 β 는 fem_lightg4 가 거절) 라 κ_n = 씬 κ 하나: wolf 0 -> 확장 켬, 나머지 1 -> 확장 커널을 하나도 안 띄움.
  칸마다 κ 가 다른 씬 (섞인 재질) 은 노드별 켬 표시가 필요한데 지금 확장 커널 (fem_wolfh8.W8_SRC) 은 씬 단위라 이 모듈은 거절 (kappa_of 가 NotImplementedError).
구현: 두 실행기의 alloc 을 확장 설치 전·후 것으로 갈라 κ 로 고름 -- κ 0 이면 확장 판 alloc (w8b 등 버퍼 + 켬 표시), 아니면 설치 전 alloc 그대로
  (w8b 없음 -> fem_wolfh8._on · fem_ext1 on 이 거짓 -> 분류·접기·펼치기·기록 커널을 하나도 안 띄움). 옛 실행기 (수박 AORTSolver) 는 첫 alloc 이
  재질 속성 (mat·prm) 을 두기 전 (super().__init__ 안) 이라 __init__ 인자 (mat·prm·wf) 를 먼저 붙여 둠 (속성 이름 _x2kw, 실행기 계산은 안 씀).
  (K) P1X 강성 × κ 는 절 CM77 fem_wolfh6 (K) 그대로 (κ 1 -> 비트 같음).
기록: x2info.json (판마다 실행기·재질·κ·결정·alloc 수).
"""
import json
import os
import types

ST = dict(out=".", dec=[], nalloc_ext=0, nalloc_base=0, D=2, T=1, X=0)


def _kap(s):
    from . import fem_lightg4 as L4
    k = s.__dict__.get("_x2k")
    if k is not None:
        return k
    src = s
    if not hasattr(s, "_mat") and "mat" not in s.__dict__ and "_x2kw" in s.__dict__:
        src = types.SimpleNamespace(**s.__dict__["_x2kw"])                                      # 옛 실행기 첫 alloc: 재질 속성 전 -> __init__ 인자
    k, info = L4.kappa_of(src)
    k = float(k)
    s._x2k = k
    info = dict(info, kappa=k, solver=type(s).__name__, ext=(k == 0.0), from_init_args=src is not s)
    ST["dec"].append(info)
    _dump()
    print(f"[절 CM84 ext2] {type(s).__name__} 재질 {info.get('name')} -> κ {k:g} ({info.get('rule')}) -> "
          f"{'확장 T1E2 켬 + P1X 강성 × κ' if k == 0.0 else '확장 안 함 (H5 그대로, P1X 강성 × κ)'}", flush=True)
    return k


def install(out, D=2, T=1, X=0, ck=0):
    """실행기 runpy 순간 (가장 바깥). fem_wolfh6 (K) 설치는 감싸개가 따로."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_ext1 as X1
    ST.update(out=out, D=int(D), T=int(T), X=int(X))
    R, C = ART.RTSolver, AORT.AORTSolver
    a0R, a0C = R.alloc, C.alloc
    X1.install(out, D=D, T=T, X=X, ck=ck)
    a1R, a1C = R.alloc, C.alloc
    oiC = C.__init__

    def initC(s, *a, **kw):
        s._x2kw = dict(mat=int(kw.get("mat", 2)), prm=kw.get("prm"), wf=dict(kw.get("wf") or {}))
        return oiC(s, *a, **kw)

    def allocR(s, need):
        if _kap(s) == 0.0:
            ST["nalloc_ext"] += 1
            return a1R(s, need)
        ST["nalloc_base"] += 1
        return a0R(s, need)

    def allocC(s, need):
        return (a1C if _kap(s) == 0.0 else a0C)(s, need)

    R.alloc, C.alloc, C.__init__ = allocR, allocC, initC
    print(f"[절 CM84] ext2 켬 -- H7: 노드 κ (받침 칸 κ 최솟값) 0 이면 확장 T1E2 (차수 {D} 내부 칸 정의 {T}) + P1X 강성 × κ, 아니면 H5 그대로", flush=True)
    import atexit
    atexit.register(_dump)


def _dump():
    try:
        with open(os.path.join(ST["out"], "x2info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
