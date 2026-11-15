"""절 CM75 묶음 lightg4 (11-08 위임): 가벼운 노드 튐 4 -- 절 CM73 (A) 가벼움 벌점 몫에 그 칸 재질의 응집 (인장) 강도 몫 κ 를 곱함.

면 가중 (절 CM73 (A), 'P1X 밖만'): fw = 1[P1X] + (1 − 1[P1X]) · r · κ · λ_A,  r = γ̄_L/γ̄,  κ = 재질 응집 몫 (문턱·클램프 없음, 재질 매개변수에서만).
  P1X 강성 (1[P1X]) 은 그대로, 새 몫 (r λ_A) 에만 κ 를 곱한다.
κ 정의: 정수압 축 (p = −tr σ / 3) 위 인장 꼭짓점 강도 σ_t 와 압축 꼭짓점 강도 σ_c 로
  κ = 2 σ_t / (σ_t + σ_c)     (= σ_t 를 두 강도의 평균으로 나눈 값; 인장·압축이 대칭인 재질 = 1, 인장 강도 0 = 0)
  - 탄성 (항복 없음; 레고·ficus): 항복면이 없어 σ ↦ −σ 대칭 -> κ = 1.
  - von Mises (PG metal·plasticine = bread, jelly 탄성): 항복 함수가 압력과 무관 (f(σ) = f(−σ)) -> 대칭 -> κ = 1.
  - NACC (GF 수박): 항복면 (3/2)(1+2β) s:s + M² (p + β p0)(p − p0) = 0 의 압력 축 꼭짓점 p = p0 (압축) · p = −β p0 (인장) -> σ_t/σ_c = β,
    κ = 2β/(1+β) (수박 config β 1 -> κ = 1). β 는 NACC 원래 정의에서 '응집 (cohesion) 계수' 그 자체.
  - Drucker-Prager 모래 (PG sand = wolf): 응집력 항 없음 (c = 0; 반환 사상은 tr ε ≥ 0 이면 꼭짓점 = 응력 0) -> σ_t = c cot φ = 0, σ_c = ∞ (뚜껑 없음) -> κ = 0.
  각 씬은 재질 하나 (수박 기본 판은 씨 영역 β 없이 β 1 하나) 라 칸 κ = 씬 κ -- 스칼라로 계산해 r 에 곱한다 (씨 영역 --seedb 처럼 칸마다 다르면 이 모듈은 거절).
  κ = 0 이면 가벼움 몫 전체가 0 이므로 lightg3 모듈이 아무 커널도 띄우지 않게 한다 (m̄ = 0 인 레고·ficus 와 같은 '아무것도 안 함 (P1X 그대로)' 경로 = G'' 와 같은 계산).
  κ = 1 이면 r · 1.0 = r (부동소수 그대로) 이라 절 CM73 (A) 와 같은 계산.
구현: lib/anchorflow/fem_lightg3.py 의 _on (씬 결정) · reff (가중에 쓰는 r) 를 덮어씀 -- 커널·진단·검사 (lg3fd) 는 lightg3 것 그대로 (검사 참값도 같은 reff 를 씀).
기록: l4info.json (재질 번호·이름·σ_t·σ_c·κ·결정).
"""
import json
import math
import os
import sys

ST = dict(kap=None, info=None, out=".")
MATN = {None: "탄성 (재질 판 아님)", 0: "jelly (PG 탄성)", 1: "metal (PG von Mises)", 2: "sand (PG Drucker-Prager)", 3: "foam", 5: "plasticine (PG von Mises + 손상)", 7: "watermelon (GF NACC)"}


def _matno(s):
    m = getattr(s, "_mat", None)                                                                # MatRT (새 실행기 재질 판)
    if m is None:
        mm = getattr(s, "mat", None)                                                            # AORTSolver (옛 실행기)
        m = mm if isinstance(mm, int) else None
    return m


def kappa_of(s):
    """(κ, 기록 dict). 재질 매개변수에서만 (상태 무관)."""
    m = _matno(s)
    inf = float("inf")
    if m is None or m == 0:
        return 1.0, dict(mat=m, name=MATN.get(m), rule="항복 없음 -> σ ↦ −σ 대칭", st="∞", sc="∞")
    if m in (1, 3, 5):      # 절 CM186: 재질 3 (foam, PG viscoplasticity_return_mapping_with_StVK) 도 von Mises 항복 (압력 무관) -> 1·5 와 같은 대칭 규칙
        ys = None
        if getattr(s, "ysp", None) is not None and s.ysp.numel():
            ys = float(s.ysp[0])
        return 1.0, dict(mat=m, name=MATN.get(m), rule="von Mises: 압력 무관 항복 -> 대칭", st="∞ (정수압)", sc="∞ (정수압)", yield_stress=ys)
    if m == 2:
        return 0.0, dict(mat=m, name=MATN.get(m), rule="DP 응집력 c = 0 (PG 반환 사상에 응집 항 없음): σ_t = c cot φ = 0, σ_c = ∞", st=0.0, sc="∞", cdp=float(getattr(s, "cdp", float("nan"))))
    if m == 7:
        wf = getattr(s, "wf", None) or {}
        if wf.get("seedb") or "--seedb" in sys.argv:
            raise NotImplementedError("lightg4: 씨 영역 β (칸마다 다른 κ) 는 아직 없음")
        prm = s.prm.detach().double().cpu()
        beta, kb, xi = float(prm[5]), float(prm[2]), float(prm[6])
        lj0 = float(s.ljp[0]) if getattr(s, "ljp", None) is not None and s.ljp.numel() else -0.04
        p0 = kb * (1e-5 + math.sinh(xi * max(-lj0, 0.0)))
        return 2.0 * beta / (1.0 + beta), dict(mat=m, name=MATN.get(m), rule="NACC 꼭짓점: σ_c = p0, σ_t = β p0 -> κ = 2β/(1+β)", beta=beta, p0=p0, st=beta * p0, sc=p0, logJp0=lj0, xi=xi, bulk=kb)
    raise NotImplementedError(f"lightg4: 재질 {m} 의 κ 정의 없음")


def install(out):
    """실행기 runpy 순간, lightg3 설치 뒤에 부름."""
    from . import fem_lightg3 as L3
    assert L3.GL > 0.0, "lightg4: lightg3 (--l3 γ̄_L) 가 켜져 있어야 함"
    ST["out"] = out
    o_on, o_reff = L3._on, L3.reff

    def _on(s):
        k, info = kappa_of(s)
        ST.update(kap=float(k), info=info)
        base = bool(o_on(s))
        on = base and k > 0.0
        info.update(kappa=float(k), mbar=float(getattr(s, "mthr", 0.0) or 0.0), lightg3_on=base, on=on, r=L3.ratio(), r_used=(o_reff() * k))
        with open(os.path.join(out, "l4info.json"), "w") as f_:
            json.dump(info, f_, ensure_ascii=False, default=str)
        print(f"[절 CM75 lightg4] 재질 {info.get('name')} -> κ {k:g} ({info.get('rule')}); m̄ > 0 {base} -> "
              f"{'가중 r κ λ (r ' + format(o_reff(), 'g') + ' x κ ' + format(k, 'g') + ')' if on else '아무것도 안 함 (P1X 그대로)'}", flush=True)
        return on

    def reff():
        r = o_reff()
        return r * ST["kap"] if ST["kap"] is not None else r

    L3._on = _on
    L3.reff = reff
    print(f"[절 CM75] lightg4 켬 -- 가벼움 몫에 재질 응집 몫 κ = 2σ_t/(σ_t + σ_c) 곱 (정의 {L3.FR} {L3.FRN[L3.FR]}, γ̄_L {L3.GL:g}, 범위 {L3.SC})", flush=True)
