"""[substep_fix 11-15] 서브스텝 수 N 이 커질 때 발산하는 문제의 수정 감싸개 (새 플래그, 기본 0 = 소스 그대로, N=1 은 결과 불변).

원인 (확인, 절 CM184): 뉴턴 조기 종료 허용치 --tolN 은 '걸음 시작 잔차 |g0| 에 대한 상대값' 인데 |g0| = M |Ũ - U| / τ^2 ∝ 1/h (관성 예측 이동 / 걸음^2) 라서
절대 힘 허용치가 N 에 비례해 느슨해진다 (N 배). 프레임당 N 걸음이 오차를 쌓으므로 프레임 속도 오차는 N·h·(g/m) 로 N 에 비례해 커진다.
접촉 (바닥 장벽) 걸음에서 덜 풀린 걸음이 연쇄되어 (보폭 최소 < 1, ‖g‖² 가 1 에 가까움) 큰 N 에서 몸이 바닥을 뚫고 나감.
수정: 허용치를 걸음 길이에 비례시킴 (메모리 'Newton tol ∝ dt' 와 같은 원리) -- 유효 허용치 = tolN · (h / frame_dt) = tolN / N.

플래그 (모두 감싸개 앞쪽에, 숫자 하나; 0 = 끔):
  --ssf_tol 1     뉴턴 허용치 tolN 에 1/N 을 곱함 (실행기 RTSolver 생성 직후, N = K)
  --ssf_split D   프레임에서 무너진 걸음 (덜 풀림 ∧ 끝 잔차 ‖g‖²/F² > --ssf_bigT (기본 1 = 잔차가 힘 크기 자체보다 큼), 또는 비유한·메모리 초과) 이 있으면 입자 상태를 되돌려 걸음 수 K 를 두 배로 다시 풂 (최대 2^D 배; 실행기 --ksplit 을 K>1 기준으로 일반화)
  --ssf_det 1     (기본 1) 프레임 끝 입자 det F <= 0 도 무너진 프레임으로 (뒤집힌 상태는 풀이 실패의 확실한 표지)
  --ssf_bec 1     매 걸음 끝 갱신을 국소 후퇴 오일러 (--ue be --ue_loc 비트 와 같이; HHT 는 자유 노드만) -- 접촉 노드의 HHT 가속도 외삽이 큰 N 발산의 원인 (절 CM186)
  --ssf_kmax 256  다시 풀 때 걸음 수 상한 (기본 256; 그래프 캡처 메모리)
  --ssf_tolp 1    PCG 허용치 tolP 에도 1/N 곱함 (선택)
  --ssf_fpic M    [d192 11-16] (기본 0 = 소스 그대로) 재질 판 + --fpi_el 1 (소성 재질은 --fp_interp 꺼짐) 에서, M 1 = 받아들인 프레임이 끝 판정 무너짐 (최대 깊이에서도 덜 풀림) 이면,
                  M 2 = 다시 푼 프레임이면 그 프레임만 입자 F·재질 상태를 프레임 끝 가우스 점 보간 (옛 판 --fp_interp 와 같은 커널) 으로 교체. 탄성 재질은 아무 일 없음.
  --ssf_fpg J0    [d192] (기본 0 = 끔) 재질 판 + --fpi_el 1 에서 받아들인 프레임마다 입자 det F 가 (J0, 1/J0) 밖인 입자만 F·재질 상태를 프레임 끝 가우스 점 보간으로 교체
                  (범위 안 입자는 그대로 = 그런 입자가 없는 판은 코드 경로·결과 그대로)
  --ssf_detj J0   [d192] (기본 0 = 그대로) 무너짐 판정의 입자 det F 문턱 (det F <= J0 이면 무너짐 -> K 두 배 다시 풂)
  --ssf_fpik K    (기본 0) --ssf_fpic 를 받아들인 프레임의 걸음 수 K 가 이 값 이상일 때만
  --ssf_cgk P     [d192 11-16] (기본 0 = 소스 그대로) PCG 반복 상한 cg 를 실행기 서브스텝 수 N (--K) 에 맞춰: cg_N = min(--ssf_cgmax, max(cg, round(cg · N^P))).
                  N = 1 이면 cg 그대로 (다시 풀기로 K 가 커져도 그대로) = 코드 경로 같음. --ssf_cgm 1 (기본) 이면 재질 판 (소성 재질 실행기) 에만, 0 이면 모든 판
  --ssf_tsum 1    (wolffix 11-15) 다시 푼 프레임의 시간 s.last_ms = 버린 시도들 + 마지막 시도 합 (실시간 판정용; 0 = 마지막 시도만, 옛 동작)
사용: python exe/ssfix_wrap.py --ssf_tol 1 exe/stabhex_wrap.py ... exe/fem_an_rt_sim.py ... (바깥 첫 감싸개로)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
C = dict(tol="0", tolp="0", kex="0", split="0", bigT="1.0", wrx="1", dumpfail="", det="1", bec="0", tsum="0", kmax="256", fpic="0", fpik="0", cgk="0", cgmax="64", cgm="1", fpg="0", detj="0")
while argv and argv[0].startswith("--ssf_"):
    C[argv[0][6:]] = argv[1]; del argv[:2]
_orig = runpy.run_path; _done = [False]


def _install():
    from anchorflow import fem_an_rt as ART
    oi = ART.RTSolver.__init__

    def init(s, *a, **kw):
        oi(s, *a, **kw)
        K = max(int(s.K), 1)
        if float(C["tol"]) != 0 and s.tolN >= 0:
            s.tolN = s.tolN / K ** float(C["tol"])
        if float(C["tolp"]) != 0 and s.tolP >= 0:
            s.tolP = s.tolP / K ** float(C["tolp"])
        if float(C["kex"]) != 0 and getattr(s, "floor", None):
            s.floor["KEX"] = s.floor["KEX"] * float(C["kex"])
        print(f"[substep_fix] KEX {getattr(s, 'floor', None) and s.floor['KEX']} tolN {s.tolN:.4g} tolP {s.tolP:.4g} (K {K}, ssf_tol {C['tol']}, ssf_tolp {C['tolp']})", flush=True)
    ART.RTSolver.__init__ = init
    if float(C["cgk"]) != 0:                                                           # --ssf_cgk P: PCG 상한을 실행기 서브스텝 수 N (= --K) 에 맞춤 (N 1 은 다시 풀기 포함 그대로)
        oinit_c = ART.RTSolver.__init__

        def init_c(s, *a, **kw):
            oinit_c(s, *a, **kw)
            s._cg0 = int(s.cg)
            if int(float(C["cgm"])) == 0 or hasattr(type(s), "mat_g2p"):               # 재질 판 (fem_mat_rt.MatRT) -- matk 는 부모 생성자 뒤에 정해지므로 클래스로
                N_ = max(int(s.K), 1)
                s.cg = int(min(int(float(C["cgmax"])), max(s._cg0, round(s._cg0 * N_ ** float(C["cgk"])))))
            print(f"[d192 cgk] N {s.K} PCG 상한 {s._cg0} -> {s.cg} (P {C['cgk']}, 재질 판만 {C['cgm']}, 재질 판 {hasattr(type(s), 'mat_g2p')})", flush=True)
        ART.RTSolver.__init__ = init_c
    if float(C["bec"]) != 0:                                                           # --ssf_bec 1: 걸음 끝 갱신 선택자 cjl 을 매 걸음 1 로 -> --ue be --ue_loc <비트> 의 국소 후퇴 오일러 (L-안정) 가 덜 풀린 걸음뿐 아니라 매 걸음 그 노드 집합 (장벽 걸린 표본점 노드 등) 에
        okc = ART.RTSolver.k; BS = dict(k=0)

        def kc(s, name, n, args, b=128):
            r = okc(s, name, n, args, b)
            if name == "rt_reset":
                BS["k"] = 0
            if name == "cj_end":
                s.cjl[BS["k"]: BS["k"] + 1].fill_(1)
                BS["k"] += 1
            return r
        ART.RTSolver.k = kc
    D = int(float(C["split"]))
    if D > 0 or C["dumpfail"] or float(C["fpic"]) != 0 or float(C["fpg"]) != 0:                                                                          # --ssf_split D: 프레임 단위 걸음 수 두 배 다시 풀기 (최대 2^D 배), 기준 K = 실행기 --K
        import torch
        ostep = ART.RTSolver.step
        oinit2 = ART.RTSolver.__init__

        def init2(s, *a, **kw):
            oinit2(s, *a, **kw)
            s.kcap = max(int(getattr(s, "kcap", 0) or 0), min(int(s.K) << D, max(int(s.K), int(float(C["kmax"])))))   # 진단·상태 버퍼는 걸음 수 상한으로 (실행기 --ksplit 의 S.kcap 과 같은 자리); 절 CM190: 다시 풀기는 kmax 를 안 넘으므로 버퍼도 kmax 까지 (K 128 x 8 = 1024 칸 버퍼가 재질 장면 f1 잘못된 메모리 접근)
            s.alloc(None)
        ART.RTSolver.__init__ = init2
        T = float(C["bigT"]); ST = dict(retry=0, frames=0, hist={}, fpic=0)

        def _fpi_rescue(s, J0=None):
            """[d192] --ssf_fpic: 재질 판에서 --fp_interp 가 꺼진 경우 (--fpi_el 1 의 소성 재질) 만, 받아들인 프레임의 입자 F·재질 상태를
            그 프레임 끝 가우스 점 값 (마지막 걸음 커밋 RM 뒤) 의 칸 안 삼선형 보간으로 교체 (= 그 프레임만 옛 판 --fp_interp 의 G2P 뒤 단계와 같은 커널·같은 인자).
            탄성 재질·--fp_interp 켠 판은 아무것도 안 함."""
            import numpy as np
            if not getattr(s, "matk", 0) or getattr(s, "fpi", False) or not hasattr(s, "_src0"):
                return
            P = ART.P
            if getattr(s, "_fpm", None) is None:
                s._fpm = ART.cp.RawModule(code=ART.build_src(s._src0, s.adapt, s.goct, s.floor, fk=getattr(s, "fk", False), hspec=getattr(s, "hspec", False), ncomp=getattr(s, "ncomp", False),
                                                            gqm=getattr(s, "gqm", False), fpi=True, cj=getattr(s, "cjj", False), cue=bool(getattr(s, "cue", 0)), uel=getattr(s, "uel", 0),
                                                            ufw=getattr(s, "ufw", False), ls0=getattr(s, "ls0", False), c2=getattr(s, "c2", None)), options=("--std=c++14",))
            M = s._fpm; i32, f = np.int32, np.float32
            capQ = s.capQc * 8; N = s.N; IP = s.IP; K = int(s.K)
            Fe = torch.zeros(capQ, 28, device=s.dev, dtype=torch.float32)
            Uk = s.Ust[K - 1].contiguous()
            mod0, fn0 = s.mod, s.fn
            _sv = {nm: getattr(s, nm, None) for nm in ("Fe", "fiJ0", "fiJX", "fiJp0", "fiD")}  # 절 CM92 finterp 훅 (모드 0 = fpi_p 결과 그대로) 이 fpi_p 에 인자 넷을 더하므로 그 판의 버퍼를 잠깐 둠
            _Z = lambda n_: torch.zeros(n_, device=s.dev, dtype=torch.float32)
            s.Fe, s.fiJ0, s.fiJX, s.fiJp0, s.fiD = Fe, _Z(capQ), _Z(capQ), _Z(N), _Z(N * 8)
            s.mod, s.fn = M, {}                                                         # 같은 실행기 s.k 경로 (옛 판 mat_g2p 와 같은 발사) 로 이 모듈의 커널을 부름
            try:
                if s.aq:
                    s.k("aq_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), i32(s._mat), P(s.prm), P(s.Sq), f(s.dt)))
                else:
                    s.k("ao_commit", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), f(s.cdp)))
                s.k("fpi_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(Uk), P(Fe)))
                if J0 is None:                                                          # 입자 전부 교체
                    tF, tys, tlj, tal = s.F, getattr(s, "ysp", None), getattr(s, "ljp", None), getattr(s, "alp", None)
                else:                                                                   # --ssf_fpg: 사본에 보간을 받아 J 가 범위 밖인 입자만 교체
                    tF = s.F.clone()
                    tys, tlj, tal = ((s.ysp.clone(), s.ljp.clone(), s.alp.clone()) if s.aq else (None, None, None))
                s.k("fpi_p", N, (i32(N), P(IP), P(s.fx), P(s.pcd), P(s.cidx), P(Fe), i32(0), i32(0), P(tF),
                                 P(s.Sq if s.aq else None), P(tys if s.aq else None), P(tlj if s.aq else None), P(tal if s.aq else None)))
            finally:
                s.mod, s.fn = mod0, fn0
                for nm, v_ in _sv.items():
                    setattr(s, nm, v_)
            torch.cuda.synchronize()
            if J0 is not None:
                J = torch.linalg.det(s.F[:N].reshape(-1, 3, 3).double())
                m = ~((J > J0) & (J < 1.0 / J0))
                nm = int(m.sum())
                if nm:
                    s.F[:N].view(-1, 9)[m] = tF[:N].view(-1, 9)[m]
                    if s.aq:
                        for d_, t_ in ((s.ysp, tys), (s.ljp, tlj), (s.alp, tal)):
                            d_[:N][m] = t_[:N][m]
                    ST["fpgn"] = ST.get("fpgn", 0) + nm; ST["fpgf"] = ST.get("fpgf", 0) + 1
                    print(f"[d192 fpg] 프레임 {ST['frames']} K {K}: J 범위 밖 입자 {nm} -> 가우스 점 보간 (J 최소 {float(J.min()):.3g} 최대 {float(J.max()):.3g})", flush=True)
                return
            ST["fpic"] += 1
            print(f"[d192 fpic] 프레임 {ST['frames']} K {K}: 입자 F·상태 = 가우스 점 보간 (재질 판)", flush=True)

        def collapsed(s, K):
            if not bool(torch.isfinite(s.x).all()):
                return True
            if float(C["det"]) != 0 and float(torch.linalg.det(s.F.reshape(-1, 3, 3)).min()) <= float(C["detj"]):   # [d192] --ssf_detj J0 (기본 0 = 그대로)   # 입자 변형 기울기 뒤집힘 (det F <= 0): 물질이 뒤집힌 것은 불가능한 상태
                return True
            fl = s.cjl[:K] > 0
            big = ~(s.cjg[:K] <= T)
            stall = s.cjs[:K] <= 0.0                                                    # 선탐색이 받아들인 보폭이 하나도 없는 반복이 있는 걸음 (뉴턴이 전혀 못 나감)
            return bool((fl & big).any())

        def step(s, t):
            ST["frames"] += 1
            bk = s._bk(); K0 = int(s.K); h = float(s.h); cm0 = s.cap_mul
            oom = False
            try:
                redo = ostep(s, t)
            except torch.OutOfMemoryError:                                               # 발산한 입자가 노드 상자를 키워 용량 다시 할당이 메모리 초과 -> 무너진 프레임으로 취급
                oom = True; redo = False
            dp = 0; tw = 0.0
            first = ST["frames"] == 1                                                  # 프레임 0 (정지에서 시작하는 첫 풀이) 은 비유한·메모리 초과가 아니면 다시 풀지 않음
            while dp < D and (K0 << (dp + 1)) <= int(float(C["kmax"])) and (oom or (collapsed(s, int(s.K)) and not first) or (first and not bool(torch.isfinite(s.x).all()))):
                tw += float(getattr(s, "last_ms", 0.0) or 0.0) if not oom else 0.0
                dp += 1; ST["retry"] += 1
                if oom:
                    torch.cuda.empty_cache()
                s._rs(bk)
                if oom or s.cap_mul != cm0:                                             # 용량이 늘었을 수 있음 -> 처음 배수로 다시 할당, 캐시된 그래프 (옛 용량) 버림
                    s.cap_mul = cm0; s.alloc(None); s.graphs.clear(); s.graph = None
                oom = False
                if hasattr(s, "wrx") and float(C["wrx"]) != 0:                          # 다시 풀 때는 절 CM38 X3 (고착 상한 + 속도 수준 접촉) 를 켬 (실행기 --wrx_col 1 의 다시 풀기와 같음)
                    from anchorflow import fem_wmrx as WRX
                    WRX.on(s, 1)
                ART.set_K_an(s, K0 << dp, h)
                try:
                    redo = ostep(s, t) or redo
                except (torch.OutOfMemoryError, RuntimeError) as ex_:                    # 그래프 캡처 메모리 부족 (CUDA error: out of memory) 도 같이
                    if not isinstance(ex_, torch.OutOfMemoryError) and "out of memory" not in str(ex_):
                        raise
                    oom = True
            if C["dumpfail"] and dp == D and (oom or collapsed(s, int(s.K))) and not ST.get("dumped"):   # 최대 깊이에서도 무너진 프레임의 시작 상태 (재현용, 진단)
                ST["dumped"] = True
                torch.save(dict(bk=[q.cpu() for q in bk], t=float(t), K0=K0, fi=ST["frames"] - 1), C["dumpfail"])
                print(f"[substep_fix] 끝까지 무너진 프레임 시작 상태 저장 {C['dumpfail']} (프레임 {ST['frames'] - 1})", flush=True)
            ST["hist"][dp] = ST["hist"].get(dp, 0) + 1
            fpm = int(float(C["fpic"]))
            if fpm and (oom is False) and (dp > 0 if fpm == 2 else collapsed(s, int(s.K))) and int(s.K) >= int(float(C["fpik"])):
                _fpi_rescue(s)
            if float(C["fpg"]) > 0 and oom is False and bool(torch.isfinite(s.x).all()):   # --ssf_fpg J0: 받아들인 프레임마다 J 범위 밖 입자만
                J_ = torch.linalg.det(s.F[:s.N].reshape(-1, 3, 3).double())
                if not bool(((J_ > float(C["fpg"])) & (J_ < 1.0 / float(C["fpg"]))).all()):
                    _fpi_rescue(s, float(C["fpg"]))
            if dp and float(C["tsum"]) != 0:
                s.last_ms = float(s.last_ms) + tw
            if dp:
                print(f"[substep_fix] 프레임 t={t:.4f} 무너짐 -> K {K0} 에서 {K0 << dp} 로 다시 풂 (깊이 {dp}; 끝 판정 무너짐 {oom or collapsed(s, int(s.K))})", flush=True)
                ART.set_K_an(s, K0, h)
                if hasattr(s, "wrx") and float(C["wrx"]) != 0:
                    from anchorflow import fem_wmrx as WRX
                    WRX.on(s, 0)
                if oom:
                    s.x.fill_(float("nan"))
            return redo
        ART.RTSolver.step = step
        import atexit
        atexit.register(lambda: print(f"[substep_fix] 프레임 {ST['frames']} 다시 풂 {ST['retry']} 깊이별 {ST['hist']}" + (f" fpic {ST['fpic']}" if int(float(C['fpic'])) else "") + (f" fpg 프레임 {ST.get('fpgf', 0)} 입자 {ST.get('fpgn', 0)}" if float(C['fpg']) else ""), flush=True))


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        if float(C["tol"]) != 0 or float(C["tolp"]) != 0 or float(C["kex"]) != 0 or float(C["split"]) != 0 or C["dumpfail"] or float(C["bec"]) != 0 or float(C["fpic"]) != 0 or float(C["cgk"]) != 0 or float(C["fpg"]) != 0:
            _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
