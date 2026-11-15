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
  --ssf_tolp 1    PCG 허용치 tolP 에도 1/N 곱함 (선택)
  --ssf_tsum 1    (wolffix 11-15) 다시 푼 프레임의 시간 s.last_ms = 버린 시도들 + 마지막 시도 합 (실시간 판정용; 0 = 마지막 시도만, 옛 동작)
사용: python exe/ssfix_wrap.py --ssf_tol 1 exe/stabhex_wrap.py ... exe/fem_an_rt_sim.py ... (바깥 첫 감싸개로)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
C = dict(tol="0", tolp="0", kex="0", split="0", bigT="1.0", wrx="1", dumpfail="", det="1", bec="0", tsum="0")
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
    if D > 0 or C["dumpfail"]:                                                                          # --ssf_split D: 프레임 단위 걸음 수 두 배 다시 풀기 (최대 2^D 배), 기준 K = 실행기 --K
        import torch
        ostep = ART.RTSolver.step
        oinit2 = ART.RTSolver.__init__

        def init2(s, *a, **kw):
            oinit2(s, *a, **kw)
            s.kcap = max(int(getattr(s, "kcap", 0) or 0), int(s.K) << D)               # 진단·상태 버퍼는 걸음 수 상한으로 (실행기 --ksplit 의 S.kcap 과 같은 자리)
            s.alloc(None)
        ART.RTSolver.__init__ = init2
        T = float(C["bigT"]); ST = dict(retry=0, frames=0, hist={})

        def collapsed(s, K):
            if not bool(torch.isfinite(s.x).all()):
                return True
            if float(C["det"]) != 0 and float(torch.linalg.det(s.F.reshape(-1, 3, 3)).min()) <= 0.0:   # 입자 변형 기울기 뒤집힘 (det F <= 0): 물질이 뒤집힌 것은 불가능한 상태
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
            except torch.OutOfMemoryError:                                              # 발산한 입자가 노드 상자를 키워 용량 다시 할당이 메모리 초과 -> 무너진 프레임으로 취급
                oom = True; redo = False
            dp = 0; tw = 0.0
            while dp < D and (oom or collapsed(s, int(s.K))):
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
                except torch.OutOfMemoryError:
                    oom = True
            if C["dumpfail"] and dp == D and (oom or collapsed(s, int(s.K))) and not ST.get("dumped"):   # 최대 깊이에서도 무너진 프레임의 시작 상태 (재현용, 진단)
                ST["dumped"] = True
                torch.save(dict(bk=[q.cpu() for q in bk], t=float(t), K0=K0, fi=ST["frames"] - 1), C["dumpfail"])
                print(f"[substep_fix] 끝까지 무너진 프레임 시작 상태 저장 {C['dumpfail']} (프레임 {ST['frames'] - 1})", flush=True)
            ST["hist"][dp] = ST["hist"].get(dp, 0) + 1
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
        atexit.register(lambda: print(f"[substep_fix] 프레임 {ST['frames']} 다시 풂 {ST['retry']} 깊이별 {ST['hist']}", flush=True))


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        if float(C["tol"]) != 0 or float(C["tolp"]) != 0 or float(C["kex"]) != 0 or float(C["split"]) != 0 or C["dumpfail"] or float(C["bec"]) != 0:
            _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
