"""절 CM87 묶음 wolfh9 (11-09 위임): H8 wolf 가 정답보다 낮게 깔림 -- κ 0 (모래) 씬 P1X 강성 무게를 0 대신 재질에서 나오는 값으로.

진단 (절 CM87.1): H8 (= H7 wolf 계산; 확장 T1E2 + P1X 강성 × κ, 모래 κ 0 -> 강성 0) 은 f30 부터 입자가 부피를 잃음
  (1 cm 복셀당 입자 정답 2.23 -> H8 f50 12.5, 점유 부피 85 L -> 15 L). 확장 없이 강성만 0 (절 CM77 SCK) 도 같은 꼴
  (입자 det F 중앙 f30 0.27 · f35 0.09), 강성 있는 H4 는 det F 1.00 -- 부피를 지키던 것이 P1X 강성.
정의 (새 문턱·클램프 없음; fem_wolfh6 (K) 가 k ← κ k 로 0 을 만든 뒤 그 값을 바꿈, κ > 0 씬은 손대지 않음 = 비트 같음):
  --w9 1 : κ 0 이면 k ← k_0 (P1X 강성 그대로; 확장 T1E2 는 그대로 켬) -- '강성 0 이 부피 붕괴의 몫' 시험.
  --w9 2 : κ 0 이면 k ← α_DP k_0, α_DP = √(2/3) · 2 sin φ / (3 − sin φ) (PG Drucker-Prager 마찰 계수; wolf φ 30° -> 0.3266).
           근거: 모래는 인장·응집 강도 0 (κ 0) 이나 압력 p 아래 편차 응력을 α p 까지 버팀 -- 이웃 칸을 잇는 강성을 0 이 아니라
           마찰이 옮길 수 있는 몫으로. φ 는 재질 매개변수 (cfg friction_angle; 실행기 cdp = (3λ+2μ) α / (2μ) 에서 거꾸로).
기록: w9info.json.
"""
import json
import math
import os

ST = dict(mode=0, out=".", kap=None, k0=None, k1=None, w=None, alpha=None, phi=None, n=0)


def _dump():
    try:
        with open(os.path.join(ST["out"], "w9info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def install(out, mode):
    from . import fem_ghost6 as G6
    from . import fem_lightg4 as L4
    from . import fem_wolfh6 as W6
    ST.update(mode=int(mode), out=out)
    prev = G6._k_of

    def _k_of(s):
        first = not getattr(s, "_w9d", False)
        k = prev(s)
        if first:
            s._w9d = True
            kap, info = L4.kappa_of(s)
            ST.update(kap=float(kap), name=info.get("name"))
            if float(kap) == 0.0 and W6.ST.get("k0") is not None:
                k0 = float(W6.ST["k0"])
                mu, lam, cdp = float(s.mu), float(s.lam), float(getattr(s, "cdp", float("nan")))
                al = cdp * 2.0 * mu / (3.0 * lam + 2.0 * mu)                                  # cdp = (3λ+2μ) α / (2μ)
                sp = 3.0 * al / (al + 2.0 * math.sqrt(2.0 / 3.0))                             # α = √(2/3) 2 sp / (3 − sp) 의 역
                w = 1.0 if ST["mode"] == 1 else al
                s._gpk = k0 * w
                ST.update(k0=k0, k1=float(s._gpk), w=w, alpha=al, phi=math.degrees(math.asin(sp)), cdp=cdp)
                print(f"[절 CM87 wolfh9] 모드 {ST['mode']}: 재질 {info.get('name')} κ 0 -> P1X 강성 k {k} -> {s._gpk:.6g} "
                      f"(= k_0 {k0:.6g} x {w:.4g}; α_DP {al:.4g}, φ {ST['phi']:.2f}°)", flush=True)
            else:
                print(f"[절 CM87 wolfh9] 재질 {info.get('name')} κ {float(kap):g} -> 손대지 않음 (k {k})", flush=True)
            ST["n"] += 1
            _dump()
        return s._gpk

    G6._k_of = _k_of
    print(f"[절 CM87] wolfh9 켬 -- 모드 {ST['mode']} (1 κ 0 씬 P1X 강성 그대로, 2 × α_DP)", flush=True)
    _dump()
