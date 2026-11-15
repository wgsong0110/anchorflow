"""[substep_fix 11-15] 서브스텝 수 N 이 커질 때 발산하는 문제의 수정 감싸개 (새 플래그, 기본 0 = 소스 그대로, N=1 은 결과 불변).

원인 (확인, 절 CM184): 뉴턴 조기 종료 허용치 --tolN 은 '걸음 시작 잔차 |g0| 에 대한 상대값' 인데 |g0| = M |Ũ - U| / τ^2 ∝ 1/h (관성 예측 이동 / 걸음^2) 라서
절대 힘 허용치가 N 에 비례해 느슨해진다 (N 배). 프레임당 N 걸음이 오차를 쌓으므로 프레임 속도 오차는 N·h·(g/m) 로 N 에 비례해 커진다.
접촉 (바닥 장벽) 걸음에서 덜 풀린 걸음이 연쇄되어 (보폭 최소 < 1, ‖g‖² 가 1 에 가까움) 큰 N 에서 몸이 바닥을 뚫고 나감.
수정: 허용치를 걸음 길이에 비례시킴 (메모리 'Newton tol ∝ dt' 와 같은 원리) -- 유효 허용치 = tolN · (h / frame_dt) = tolN / N.

플래그 (모두 감싸개 앞쪽에, 숫자 하나; 0 = 끔):
  --ssf_tol 1     뉴턴 허용치 tolN 에 1/N 을 곱함 (실행기 RTSolver 생성 직후, N = K)
  --ssf_tolp 1    PCG 허용치 tolP 에도 1/N 곱함 (선택)
사용: python exe/ssfix_wrap.py --ssf_tol 1 exe/stabhex_wrap.py ... exe/fem_an_rt_sim.py ... (바깥 첫 감싸개로)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
argv = sys.argv[1:]
C = dict(tol="0", tolp="0")
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
        print(f"[substep_fix] tolN {s.tolN:.4g} tolP {s.tolP:.4g} (K {K}, ssf_tol {C['tol']}, ssf_tolp {C['tolp']})", flush=True)
    ART.RTSolver.__init__ = init


def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        if float(C["tol"]) != 0 or float(C["tolp"]) != 0:
            _install()
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
_orig(sys.argv[0], run_name="__main__")
