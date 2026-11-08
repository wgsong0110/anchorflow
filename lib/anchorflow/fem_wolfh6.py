"""절 CM77 묶음 wolfh6 (11-08 위임): wolf 가 정답보다 높게 쌓임 6 -- P1X 강성의 재질 응집 몫 (K) + 가벼운 자유 노드 갱신의 연속 후퇴 오일러 섞기 (L).

배경 (절 CM70): S_구속 + C 위에서 남은 방출 뒤 매달림은 ghost penalty 강성 (P1X, 걸음 변위의 법선 미분 점프 탄성 결합) -- 빼면 낙하 몫이 정답 수준이 되나
  가벼운 노드가 튀어 이탈·메모리 발산 (2/2). 같은 벌점이 가벼운 노드 튐도 막고 있었음.
정의 (문턱·클램프 없음):
  (K) AF_W6_K=1: P1X 강성 k = γ̄ (2μ+λ) h 에 재질 응집 몫 κ = 2σ_t/(σ_t+σ_c) (절 CM75, lib/anchorflow/fem_lightg4.kappa_of 그대로) 를 곱함 -- k ← κ k.
      고스트 질량 k_M 은 그대로 (절 CM70: 질량만 0 은 높이를 안 바꿈). 수박 NACC β 1 · bread von Mises · 레고·ficus 탄성 κ = 1 (k × 1.0 = 비트 같음),
      wolf DP c 0 κ = 0 (응집력 없는 모래 -> 강성 벌점 0). 근거 = 절 CM75: 법선 미분 점프 강성은 이웃 칸을 탄성으로 묶는 것인데, 인장 강도가 0 인 재질
      (모래) 은 칸 사이가 갈라져 흩어지는 것이 물리이므로 묶을 근거가 없음.
  (L) AF_W6_L=1: 걸음 끝 갱신에서 자유 노드 (고정 표시 fixd 아님, 저질량 고정 pm 2 아님, 이 걸음 후퇴 오일러 노드 (--ue_loc S · H5) 아님) 마다
      가벼움 l = m̄/(m + m̄) (절 CM69 와 같은 l; m = 노드 질량, m̄ = 그 씬 저질량 문턱 s.mthr) 으로 HHT 갱신과 후퇴 오일러 갱신을 섞음:
        U1 = (1 − l) U1_HHT + l y,   v1 = (1 − l) v1_HHT + l (y − U)/h,   a1 = (1 − l) a1_HHT
      (y = 이 걸음 뉴턴 해, U1_HHT = U + h v + h² ((½−β) a + β a1_HHT), v1_HHT = α (y − z0)). 무거운 노드 (m ≫ m̄) 는 l → 0 = HHT 그대로, m = m̄ 이면 반반,
      m → 0 이면 후퇴 오일러. 절 CM58 H 계열 (충돌 걸음 몸 전체 후퇴 오일러) 의 노드별 연속판 = 가벼운 노드에만 연속으로 수치 감쇠 (후퇴 오일러는
      스펙트럼 반경 ρ∞ = 0 의 L-안정, HHT ρ 0.8 은 고주파를 덜 죽임).
      근거: 튐 노드는 m̄ 바로 위의 가벼운 자유 노드 (절 CM64·CM67: 질량 비 1e-3~5e-2, 87 %) -- 질량 행렬이 작아 그 노드의 국소 진동수 ω ~ √(k/m) 가 크고
      ω h ≫ 1 이 되는 자리. Newmark 계열에서 ω h ≫ 1 성분은 ρ∞ 로만 줄어 사다리꼴에 가까운 HHT 는 되튐을 남기고, 후퇴 오일러는 ω h ≫ 1 성분을
      1/(1 + ω h) 로 죽인다. 가벼움 l 은 ω h 가 큰 정도의 단조 대리값 (문턱 없음) 이라 '큰 ω h 노드에만 L-안정 갱신' 이 됨.
      m̄ = 0 인 씬 (레고·ficus: 저질량 문턱 없음) 은 l ≡ 0 -> 원래 커널 그대로 (설치 안 함).
  (L') --w6x 1: 저질량 고정 노드 (m < m̄, 구속 받침 밖 k_구속 = 1) 의 걸음 고정 자리를 자기 예측 Ũ (절 CM70 C 는 구속 노드만 바꿈) 대신 이웃 26 자유 노드
      (m ≥ m̄, 구속 아님) 의 예측 증분 (Ũ_j − U_j; 앞 걸음 해의 속도·가속도로 정해짐) 질량 가중 평균으로 외삽 (0 차): ŷ_n = y_n = U_n + Σ_j m_j (Ũ_j − U_j) / Σ_j m_j.
      이웃 자유 노드가 없으면 그대로 Ũ. 확장 B-스플라인 (Höllig 2001, 작은 받침 기저 계수를 안쪽 계수의 외삽으로 묶음) 의 0 차판 -- 고정 노드가 몸과 따로
      탄도로 떨어져 칸 안 가짜 변형 (절 CM70: 구속 몸 가짜 압축과 같은 꼴) 을 만드는 것을 막음. 구속 노드 (k_구속 < 1) 는 C 그대로.
훅: 실행기 runpy 순간 (다른 감싸개 다 설치 뒤, 가장 바깥) -- (K) ghost6._k_of 감쌈, (L) ART.build_src·AORT._src 의 결과 (--ue_loc 갱신 커널 *_ufl 과 hhtc2 가
  덧붙인 *_hc) 에 섞기 판 *_ufl_l6·*_hc_l6 (같은 글자 + 마지막 인자 mn·pm·m̄) 를 덧붙이고, s.k 가 걸음 끝 갱신 (rt_ga_end_ufl·ao_stepend_ufl; 장벽 씬은 hhtc2 가
  *_hc 로 바꿈) 을 띄우기 직전 s.fn[*_ufl]·s.fn[*_hc] 를 '같은 인자 + (mn, pm, m̄) 로 *_l6 을 띄우는' 호출체로 둠 (모듈이 바뀌면 다시). --ue_loc 없는 판은 안 걸림 (연기에서 확인).
기록: w6info.json (κ·k 전후·m̄·설치 여부·커널 수).
"""
import json
import os

import numpy as np

ST = dict(X=0, nfl=0, K=0, L=0, out=".", nsrc=0, nhl=0, kap=None, k0=None, k1=None, mbar=None, Lon=None, said=False, nlaunch=0)
KERS = (("rt_ga_end_hc", True), ("rt_ga_end_ufl", True), ("ao_stepend_hc", False), ("ao_stepend_ufl", False))   # (원래 이름, 새 실행기) -> 이름 + "_l6"
VK_LINE = "        vk[n*3+a] = v1*damp;\n"
LOOP = "    for (int a=0;a<3;a++) {\n"


def _hlify_one(k, nm, rt):
    nh = nm + "_l6"
    k = k.replace(nm + "(", nh + "(", 1)
    i = k.index(") {")                                                                          # 서명 끝 -- 인자 셋 덧붙임
    k = k[:i] + ", const float* l6mn, const int* l6pm, float l6mb" + k[i:]
    assert k.count(LOOP) == 1 and k.count(VK_LINE) == 1, ("wolfh6 고리·vk 줄", nm, k.count(LOOP), k.count(VK_LINE))
    assert "w3_" not in k and " uf_" in k and " um" in k, ("wolfh6: 절 CM39 W3 섞기 판 위는 안 함 / uf_·um 없음", nm)
    fx = "!fx_ && " if rt else ""
    pre = ("    float l6_ = 0.f;                                                          // 절 CM77 (L): 자유 노드 가벼움 l = m̄/(m + m̄) 로 HHT·후퇴 오일러 섞기\n"
           f"    if (l6mb > 0.f && {fx}!(uf_ && um != 2) && !(l6pm && l6pm[n] == 2)) {{ float m6_ = fmaxf(l6mn[n], 0.f); l6_ = l6mb/(m6_ + l6mb); }}\n")
    i = k.index(LOOP)
    k = k[:i] + pre + k[i:]
    bl = ("        if (l6_ > 0.f) { float vb6_ = (y[n*3+a] - u)/dt; v1 = l6_*vb6_ + (1.f - l6_)*v1; U[n*3+a] = l6_*y[n*3+a] + (1.f - l6_)*U[n*3+a]; a1 *= (1.f - l6_); }   // 절 CM77 (L)\n")
    i = k.index(VK_LINE)
    return k[:i] + bl + k[i:]


def hlify(src):
    add = []
    for nm, rt in KERS:
        key = "__global__ void " + nm + "("
        if key not in src:
            continue
        i = src.index(key)
        j = src.index("\n}\n", i) + 3
        add.append(_hlify_one(src[i:j], nm, rt))
        ST["nhl"] += 1
    if not add:
        return src
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n".join(add) + "\n}\n"


LX_DEV = r'''
// ---- 절 CM77 (L'): 저질량 고정 노드 자리 = 이웃 26 자유 노드 예측 증분의 질량 가중 평균 (lib/anchorflow/fem_wolfh6.py)
__device__ __forceinline__ float w6_nbr(const int* IP, int n, const float* U, const float* Ut, const float* mn, float mthr, const float* kb, float* ex) {
    ex[0] = 0.f; ex[1] = 0.f; ex[2] = 0.f;
    if (kb && kb[n] < 1.f) return 0.f;                                          // 구속 받침 노드는 절 CM70 C 그대로
    int nb0 = IP[6], nb1 = IP[7], nb2 = IP[8];
    int i0 = n/(nb1*nb2), j0 = (n/nb2)%nb1, k0 = n%nb2; float sw = 0.f;
    for (int di=-1;di<=1;di++) for (int dj=-1;dj<=1;dj++) for (int dk=-1;dk<=1;dk++) {
        if (!di && !dj && !dk) continue;
        int i = i0+di, j = j0+dj, k = k0+dk;
        if (i < 0 || j < 0 || k < 0 || i >= nb0 || j >= nb1 || k >= nb2) continue;
        int q = (i*nb1 + j)*nb2 + k;
        if (q >= IP[9]) continue;
        float m = mn[q];
        if (!(m >= mthr && m > 0.f) || (kb && kb[q] < 1.f)) continue;
        sw += m; for (int a=0;a<3;a++) ex[a] += m*(Ut[q*3+a] - U[q*3+a]);
    }
    if (sw > 0.f) for (int a=0;a<3;a++) ex[a] /= sw;
    return sw;
}
'''
FL_KEYS = ("mt_floor_n", "ao_floor_n")


def lxify(src):
    """floor_n 의 저질량 고정 줄 'if (mn[n] < mthr) { ... float yy = E; ...' 를 이웃 외삽으로 (구속 노드·이웃 없음은 E 그대로)."""
    for nm in FL_KEYS:
        key = "__global__ void " + nm + "("
        if key not in src or "w6_nbr(" in src[src.index(key):src.index("\n}\n", src.index(key))]:
            continue
        i = src.index(key)
        j = src.index("\n}\n", i) + 3
        k = src[i:j]
        a0 = "if (mn[n] < mthr) {"
        assert k.count(a0) == 1, ("wolfh6 (L'): 저질량 고정 줄", nm, k.count(a0))
        b0 = k.index(a0) + len(a0)
        k = k[:b0] + " float ex6_[3]; float sw6_ = w6_nbr(IP, n, U, Ut, mn, mthr, kb, ex6_);   // 절 CM77 (L')\n       " + k[b0:]
        c0 = k.index("float yy = ", b0)
        c1 = k.index(";", c0)
        E = k[c0 + len("float yy = "):c1]
        assert E.lstrip().startswith("Ut[n*3+a]"), ("wolfh6 (L'): yy 식", nm, E)
        k = k[:c0] + "float yy = (sw6_ > 0.f) ? (U[n*3+a] + ex6_[a]) : (" + E + ")" + k[c1:]
        src = src[:i] + LX_DEV + k + src[j:]
        ST["nfl"] += 1
    return src


def P(t):
    return np.uint64(0 if t is None else t.data_ptr())


class _Launch:
    """s.fn[*_ufl·*_hc] 자리에 두는 호출체: 같은 격자·블록·인자 + (mn, pm, m̄) 로 *_l6 을 띄움."""

    def __init__(self, s, nh):
        self.s, self.mod, self.f = s, s.mod, s.mod.get_function(nh)

    def __call__(self, grid, block, args):
        s = self.s
        pm = getattr(s, "pm", None)
        ST["nlaunch"] += 1
        return self.f(grid, block, tuple(args) + (P(s.mn), P(pm), np.float32(ST["mbar_of"](s))))


def _mbar(s):
    return float(getattr(s, "mthr", 0.0) or 0.0)


def install(out, K=0, L=0, X=0):
    """실행기 runpy 순간 (다른 감싸개 뒤) 에 부름."""
    ST.update(K=int(K), L=int(L), X=int(X), out=out, mbar_of=_mbar)
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    if ST["K"]:
        from . import fem_ghost6 as G6
        from . import fem_lightg4 as L4
        _ok = G6._k_of

        def _k_of(s):
            first = not getattr(s, "_w6kd", False)
            _ok(s)
            if first:
                kap, info = L4.kappa_of(s)
                s._w6kd = True
                k0 = float(s._gpk)
                s._gpk = s._gpk * float(kap)
                ST.update(kap=float(kap), k0=k0, k1=float(s._gpk), kinfo=info)
                print(f"[절 CM77 wolfh6] (K) 재질 {info.get('name')} κ {kap:g} ({info.get('rule')}): P1X 강성 k {k0:.6g} -> {float(s._gpk):.6g} (고스트 질량 k_M {float(getattr(s, '_g3kM', 0.0)):.6g} 그대로)", flush=True)
                _dump()
            return s._gpk
        G6._k_of = _k_of
    if ST["X"]:
        o_bx, o_sx = ART.build_src, AORT._src
        ART.build_src = lambda *a, **kw: lxify(o_bx(*a, **kw))
        AORT._src = lambda *a, **kw: lxify(o_sx(*a, **kw))
    if ST["L"]:
        o_b, o_s = ART.build_src, AORT._src
        ART.build_src = lambda *a, **kw: hlify(o_b(*a, **kw))
        AORT._src = lambda *a, **kw: hlify(o_s(*a, **kw))
        for cls in (ART.RTSolver, AORT.AORTSolver):
            _wrap_k(cls)
    print(f"[절 CM77] wolfh6 켬 -- (K) P1X 강성 × κ {ST['K']} · (L) 가벼운 자유 노드 HHT·후퇴 오일러 섞기 {ST['L']} · (L') 저질량 고정 노드 자리 이웃 외삽 {ST['X']}", flush=True)
    import atexit
    atexit.register(_dump)


def _wrap_k(cls):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name in ("rt_ga_end_ufl", "ao_stepend_ufl"):
            mb = _mbar(s)
            if ST["Lon"] is None:
                ST.update(Lon=bool(mb > 0.0), mbar=mb)
                print(f"[절 CM77 wolfh6] (L) m̄ {mb:.6g} -> {'섞기 판 ' + name + '_l6 · ' + name.replace('_ufl', '_hc') + '_l6' if mb > 0.0 else '아무것도 안 함 (m̄ 0, 원래 커널)'}", flush=True)
                _dump()
            if mb > 0.0:                                                                # 걸음 끝 갱신 커널 (hhtc2 가 *_hc 로 바꾸든 장벽 없는 씬이라 *_ufl 그대로든) 을 섞기 판으로
                fn = s.fn
                for nm in (name, name.replace("_ufl", "_hc")):
                    cur = dict.get(fn, nm)
                    if not (isinstance(cur, _Launch) and cur.mod is s.mod and cur.s is s):
                        try:
                            fn[nm] = _Launch(s, nm + "_l6")
                        except Exception as e_:                                          # *_hc 가 없는 모듈 (hhtc2 꺼짐) -- *_ufl 만
                            if nm == name:
                                raise
                            ST.setdefault("miss", []).append(f"{nm}: {type(e_).__name__}")
        return ok(s, name, n, args, b)
    cls.k = k


def _dump():
    try:
        d = {k: v for k, v in ST.items() if k not in ("mbar_of",)}
        with open(os.path.join(ST["out"], "w6info.json"), "w") as f_:
            json.dump(d, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
