"""절 CM146: 잔차 노름 판정 선탐색 `--rls 1` (기본 0 = 옛 판). 에너지 (탄성 포텐셜) 대신 반환 사상 뒤 실제 기울기 잔차 ||g|| 의 Armijo 감소로 뉴턴 보폭을 판정한다
(비결합 소성은 포텐셜이 아니므로 잔차가 맞는 판정 기준). 이 판은 에너지 선탐색 (--ls_n)·--en_mo·--descent 를 끈 명령 위에서만 (exe/cm140_gen.sh 토큰 rl 이 뺌).
정의 (걸음 안, 반복 k): 기준점 (eta_b, g2_b = ||g(eta_b)||^2, 기준 방향 d_b = 그 점의 뉴턴 방향), 시험 보폭 a (처음 1). 반복마다 현재 점 (= eta_b + a d_b) 의 ||g||^2 = g2 (rt_ctl 3 의 SC[1]) 를
  받아들임 <=> sqrt(g2) <= (1 - c1 a) sqrt(g2_b)  (c1 = 1e-4, 표준 Armijo 상수)  -> 기준점 = 현재 점, 기준 방향 = 이 반복의 뉴턴 방향 (dl 그대로), a = 1.
  거절 -> a <- a/2 (반씩 되돌림), 이 반복 방향을 기준점에서 새 시험 점 eta_b + a d_b 로 가는 변위로 바꿈 (PCG 가 현재 점에서 푼 방향은 버림).
  첫 반복 (k = 0) 은 기준점 설정만. 허용치로 끝난 반복 (CT[0]) 은 그대로. 임계값·클램프 없음 (보폭 줄임은 Armijo 가 정하고, 하한은 걸음 끝).
모드 2 (절 CM173, 새 정의): 같은 Armijo 를 전역 비가중 ‖g‖² (Σ|g|², 구속면 근처 큰 힘이 지배해 먼 쪽 국소 잔차가 안 보임) 대신 관성 계량 ‖g‖²_{Mi⁻¹} = Σ |g_n|²/Mi_n (Mi = meff/τ², 노드 가속도 잔차 노름 = 뉴턴 감소량의 계량) 로 잰다.
  --xit_from 의 판정 2 (‖g‖²/F² > 1 이면 계속; cj_g) 도 같은 계량 (F² = 반복 0 의 ‖T f0‖²_{Mi⁻¹}, T f = g − Mi η) 으로 바꾸고, 이 반복 점이 Armijo 에서 거절될 점이면 판정 2 를 '아직 안 풀림' (∞) 으로 둔다 (잔차가 늘어난 점에서 반복을 멈추지 않음). 상수·문턱은 새로 안 넣음 (c1·1 은 기존).
설치: fem_an_rt.RTSolver.ctl 의 op 3 (|g|² 직후) 에서 g2 를 저장, op 7 (PCG 끝, 보폭 쓰기 전) 에서 판정·방향 바꿈."""
import numpy as np
import torch

MODE = [0]
C1 = 1e-4


W_MARK = "// ---- 절 CM173 rls 2"
W_SRC = r"""
// ---- 절 CM173 rls 2: 관성 계량 잔차 노름 (읽기만): out[0] += |g|²/Mi, out[1] += |g − Mi η|²/Mi (= |T f|²/Mi), g = −r0
__global__ void cm173_w(const int* IP, const float* meff, float itau2, const float* r0, const float* eta, double* out) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float Mi = meff[n]*itau2; if (!(Mi > 0.f)) return;
    double g2 = 0.0, f2 = 0.0;
    for (int a=0;a<3;a++) { double g = -(double)r0[n*3+a]; double tf = g - (double)Mi*(double)eta[n*3+a]; g2 += g*g; f2 += tf*tf; }
    atomicAdd(&out[0], g2/(double)Mi); atomicAdd(&out[1], f2/(double)Mi);
}
"""


def _patch(code):
    if W_MARK in code or "__global__ void rt_grad(" not in code:
        return code
    i = code.index("__global__ void rt_grad(")
    return code[:i] + W_MARK + "\n" + W_SRC + code[i:]


def install(mode):
    MODE[0] = int(mode)
    from . import fem_an_rt as ART
    _wrap(ART.RTSolver)
    if MODE[0] == 2:
        _install_w(ART)
    return {ART.RTSolver}


def _install_w(ART):
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = _patch(kw["code"])
        elif a:
            a = (_patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    R = ART.RTSolver
    ok = R.k
    octl = R.ctl

    def ctl(s, op, it=0):
        if op == 2:
            s._r2_it = it
        return octl(s, op, it)

    def k(s, name, n, args, b=128):
        r = ok(s, name, n, args, b)
        if MODE[0] == 2 and getattr(s, "adapt", True):
            if name == "rt_grad":
                if not hasattr(s, "_r2_out") or s._r2_out.device != s.dl.device:
                    s._r2_out = torch.zeros(2, device=s.dl.device, dtype=torch.float64); s._r2_F = torch.zeros(1, device=s.dl.device, dtype=torch.float64)
                s._r2_out.zero_()
                ok(s, "cm173_w", s.capN, (np.uint64(s.IP.data_ptr()), np.uint64(s.meff.data_ptr()), np.float32(float(args[2])), np.uint64(s.g0buf.data_ptr()), np.uint64(s.eta.data_ptr()),
                                          np.uint64(s._r2_out.data_ptr())), 256)
                if getattr(s, "_r2_it", 0) == 0:
                    s._r2_F.copy_(s._r2_out[1:2])
            elif name == "cj_g" and hasattr(s, "_r2_out"):
                kk = int(args[3])
                rr = torch.where(s._r2_F > 0, s._r2_out[0:1] / s._r2_F, torch.zeros_like(s._r2_F))
                if getattr(s, "_r2_it", 0) > 0 and hasattr(s, "_rl_g2b"):                 # Armijo 가 거절할 점 ((1 − c1 a)‖g_b‖ 초과) 은 '풀렸다' 로 안 침: 판정 2 를 계속 (cj_end 의 덜 풀린 걸음 판정도 같은 값)
                    acc = s._rl_g2 <= (1.0 - C1 * s._rl_a) ** 2 * s._rl_g2b
                    rr = torch.where(acc, rr, torch.full_like(rr, 3.0e38))
                s.cjg[kk:kk + 1].copy_(torch.where(rr == rr, torch.clamp(rr, max=3.0e38), torch.full_like(rr, 3.0e38)).float())
        return r
    R.k, R.ctl = k, ctl


def _wrap(c):
    octl = c.ctl
    oalloc = c.__dict__.get("alloc")

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if MODE[0] == 0 or getattr(s, "_rl_in", False) or not getattr(s, "adapt", True):
            return r
        if op == 3:
            _bufs(s); s._rl_g2.copy_(s._r2_out[0:1] if (MODE[0] == 2 and hasattr(s, "_r2_out")) else s.SC[1:2])
        elif op == 7:
            s._rl_in = True
            try:
                _step(s, it)
            finally:
                s._rl_in = False
        return r
    c.ctl = ctl
    if oalloc is not None:
        def alloc(s, need):
            r = oalloc(s, need)
            if MODE[0]:
                s._rl_n = None; _bufs(s)
            return r
        c.alloc = alloc


def _bufs(s):
    n = s.dl.numel()
    if getattr(s, "_rl_n", None) == n:
        return
    dev = s.dl.device
    s._rl_n = n
    s._rl_eb = torch.zeros(n, device=dev, dtype=torch.float32); s._rl_db = torch.zeros(n, device=dev, dtype=torch.float32)
    s._rl_g2 = torch.zeros(1, device=dev, dtype=torch.float64); s._rl_g2b = torch.zeros(1, device=dev, dtype=torch.float64)
    s._rl_a = torch.ones(1, device=dev, dtype=torch.float64)


def _step(s, it):
    _bufs(s)
    d = s.dl.view(-1); e = s.eta.view(-1)
    run = (s.CT[0] == 0)
    if it == 0:
        s._rl_eb.copy_(torch.where(run, e, s._rl_eb)); s._rl_db.copy_(torch.where(run, d, s._rl_db))
        s._rl_g2b.copy_(torch.where(run, s._rl_g2, s._rl_g2b)); s._rl_a.copy_(torch.where(run, torch.ones_like(s._rl_a), s._rl_a))
        return
    a = s._rl_a
    acc = s._rl_g2 <= (1.0 - C1 * a) ** 2 * s._rl_g2b                 # sqrt 양변: ||g|| <= (1 - c1 a) ||g_b||
    if not hasattr(s, "_rl_hist"):
        s._rl_hist = torch.zeros(16 * int(s.nt), 6, device=a.device, dtype=torch.float64)      # 절 CM173 기록 (결과 불변): 반복별 [노름², 기준 노름², 보폭 a, 받음, 돈 반복, 반복 번호]
    hi = max(getattr(s, "_cw_k", 0), 0) * int(s.nt) + it
    s._rl_hist[hi].copy_(torch.cat([s._rl_g2, s._rl_g2b, a, acc.double(), run.double().view(1), torch.full((1,), float(it), device=a.device, dtype=torch.float64)]))
    a_new = torch.where(acc, torch.ones_like(a), a * 0.5)
    trial = s._rl_eb + a_new.float() * s._rl_db - e                    # 거절 시 새 시험 점으로 가는 변위
    d_new = torch.where(acc, d, trial)
    eb_new = torch.where(acc, e, s._rl_eb); db_new = torch.where(acc, d, s._rl_db); g2b_new = torch.where(acc, s._rl_g2, s._rl_g2b)
    d.copy_(torch.where(run, d_new, d))
    s._rl_eb.copy_(torch.where(run, eb_new, s._rl_eb)); s._rl_db.copy_(torch.where(run, db_new, s._rl_db))
    s._rl_g2b.copy_(torch.where(run, g2b_new, s._rl_g2b)); s._rl_a.copy_(torch.where(run, a_new, a))
