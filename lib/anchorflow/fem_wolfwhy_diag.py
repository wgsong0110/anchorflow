"""절 CM25 진단 --nrec (새 실행기 exe/fem_an_rt_sim.py, 재질 판 MatRT·탄성 판 공통; 결과 불변).

옛 실행기 (수박, fem_ao_rt) 의 --nrec (절 CM23.3) 과 같은 기록을 새 실행기에:
- 걸음마다 시작점 규칙: 실현 가능 k0 (rt_bar_kfe, U -> ŷ 에서 활성 표본점 d > 0 인 최소 k) 와 최종 고른 k (--y0eb 면 장벽 에너지 규칙 뒤; 기존 Y0K).
- 뉴턴 반복마다: 반복 시작 ‖g‖/F (DG [2i] SC[1], F² = CD[2] = 걸음 첫 반복의 |T f|² -- 허용치 규칙 ‖g‖ ≤ εN F 와 같은 척도), 돌았는지·허용치로 멈췄는지, PCG 반복·상한 도달 (CT 사본),
  보폭 (선탐색 없는 장벽 판 = 장벽 실현 가능 보폭 2^-bk, 기존 BSK; 다른 판은 SC[14]).
- 걸음 끝 ‖g‖/F: 마지막 뉴턴 뒤 y 에서 요소 (mat_elem / rt_elem) + 장벽 + rt_grad 를 다시 불러 계산 -- 덮어쓰는 풀이 버퍼 (f·B·H·SC·r·z·p·dl·Tp·r0·Dinv·장벽 노드 스칼라·진단 플래그) 는 사본으로 되돌림.
- 걸음 끝 장벽 노드 기울기 법선 합 Σ G_n (= 장벽 힘 크기, 걸음 끝 y).
그래프 안에서는 복사·같은 커널 재호출만 (AF_SKIP 이 풀린 ctl(8) 뒤), 읽기는 프레임 뒤 호스트에서.
"""
import numpy as np
import torch

ON = [False]


def alloc(s):
    """RTSolver.alloc 끝에서 (ON 일 때만): 걸음·반복별 작은 사본 + 되돌림 사본."""
    if not ON[0]:
        return
    s.nrec = True
    dev = s.dev
    KK = 4 * max(s.K, s.Kc or 0, getattr(s, "kcap", 0))
    s.NR_Y0F = torch.zeros(KK, 4, device=dev, dtype=torch.int32)                   # 걸음별 실현 가능 k0 (rt_bar_kfe 직후 kk)
    s.NR_CT = torch.zeros(KK * s.nt * 2, 16, device=dev, dtype=torch.int32)         # ctl(3) 뒤 [2i], ctl(7) 뒤 [2i+1]
    s.NR_CD = torch.zeros(KK, 4, device=dev, dtype=torch.float64)                  # 걸음 끝 CD (CD[2] = F²)
    s.NR_SC = torch.zeros(KK, 64, device=dev, dtype=torch.float64)                 # 걸음 끝 다시 잰 SC (1 = ‖g‖², 56 = |T f|²)
    s.NR_GB = torch.zeros(KK, s.capN, device=dev, dtype=torch.float32)             # 걸음 끝 장벽 노드 기울기 (법선 스칼라)
    s.NR_bk = {}
    for nm in ("f", "B", "H", "H30", "SC", "r", "z", "p", "dl", "Tp", "g0buf", "Dinv"):
        t = getattr(s, nm, None)
        if t is not None:
            s.NR_bk[nm] = torch.empty_like(t)
    for nm in ("bGn", "bKn", "bcf", "bpg", "bph"):
        t = getattr(s, nm, None)
        if isinstance(t, torch.Tensor):
            s.NR_bk[nm] = torch.empty_like(t)
    from . import fem_wolfprop_diag as PRD
    PRD.alloc(s)                                                                    # 절 CM26 --prec (PRD.ON 일 때만)


def y0f(s, kstep):
    if getattr(s, "nrec", False):
        s.dcopy(s.NR_Y0F[kstep], s.kk)


def ct(s, kstep, it, j):
    if getattr(s, "nrec", False):
        s.dcopy(s.NR_CT[2 * (kstep * s.nt + it) + j], s.CT)


def end(s, kstep, sg, capQ, capN, it2, BA, nr, barE):
    """ctl(8) 뒤 (AF_SKIP 0): 걸음 끝 y 의 ‖g‖ 다시 계산 -> NR_SC[kstep], 장벽 노드 기울기 -> NR_GB[kstep]; 풀이 버퍼 되돌림."""
    if not getattr(s, "nrec", False):
        return
    from .fem_an_rt import P
    from . import fem_wolfprop_diag as PRD
    i32, f = np.int32, np.float32
    IP = s.IP
    s.dcopy(s.NR_CD[kstep], s.CD)
    for nm, b in s.NR_bk.items():
        s.dcopy(b, getattr(s, nm))
    s.zf(s.f, 9, 3); s.zf(s.B, 9, 9)
    s.k("rt_sc_clear", 1, (P(s.SC), i32(0), i32(60)), b=1)
    if s.matk:
        s.mat_elem(kstep, -1, 10 ** 6, capQ)                                       # it = −1 -> 마지막 반복 표시 아님 (cnt 안 셈)
        PRD.fel(s, kstep, s.nt)                                                        # 절 CM26 진단: 걸음 끝 요소 기울기
    else:
        s.k("rt_elem", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), f(s.mu), f(s.lam), i32(1 | 2 | 8), P(s.SC[10:]), P(s.f), P(s.H), i32(capQ), P(s.B)))
    if barE:
        assert not s.bnode, "--nrec: 노드 장벽 판 미지원"
        s.zf(s.bGn); s.zf(s.bKn)
        s.k("rt_bar_q", capQ, (P(IP), P(s.cb), P(s.BW), P(s.Vq), P(s.y), P(s.U), f(sg["thinv"]), *BA, i32(1), P(s.bGn), P(s.bKn), P(s.SC[10:])))
        if getattr(s, "det", False):
            s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bpg), P(s.bGn)), b=256)
            s.k("ao_gath_q", capN, (P(IP), P(s.cidx), P(s.BW), P(s.bph), P(s.bKn)), b=256)
        s.dcopy(s.NR_GB[kstep], s.bGn)
        PRD.bar(s, kstep, s.nt)
        s.k("rt_bar_n", capN, (P(IP), P(s.bGn), P(s.bKn), *[f(q) for q in nr], P(s.f), P(s.B)), b=256)
    s.k("rt_grad", capN, (P(IP), P(s.meff), it2, P(s.sk), P(s.pm), *[f(q) for q in nr], P(s.eta), P(s.f), P(s.B), P(s.Dinv), P(s.r), P(s.z), P(s.p), P(s.dl), P(s.Tp), P(s.SC), P(s.g0buf)), b=256)
    s.dcopy(s.NR_SC[kstep], s.SC)
    PRD.grad(s, kstep, s.nt)                                                       # 절 CM26 진단: 걸음 끝 y·η·잔차 (되돌리기 전)
    for nm, b in s.NR_bk.items():
        s.dcopy(getattr(s, nm), b)


def frame(s, tolN):
    """프레임 뒤 (호스트): fem.jsonl nr."""
    K, nt = s.nstage, s.nt
    DG = s.DG.cpu().numpy(); CT = s.NR_CT.cpu().numpy(); CD = s.NR_CD.cpu().numpy(); SCe = s.NR_SC.cpu().numpy()
    Y0K = s.Y0K[:K, 0].cpu().tolist(); Y0F = s.NR_Y0F[:K, 0].cpu().tolist()
    BSK = s.BSK[:K * nt, 0].cpu().tolist()
    fbar = (not s.ls) and getattr(s, "bar", False) and not getattr(s, "vres", 0.0)
    nn = int(s.IP[9])
    steps = []
    for k in range(K):
        F2 = float(CD[k][2])
        F = float(np.sqrt(F2)) if F2 > 0 else None
        its = []
        for i in range(nt):
            a_, b_ = 2 * (k * nt + i), 2 * (k * nt + i) + 1
            ran = int(CT[a_][0]) == 0
            stop = (not ran) and (i == 0 or int(CT[2 * (k * nt + i - 1)][0]) == 0)
            g = float(np.sqrt(max(DG[a_][1], 0.0))) if (ran or stop) else None
            r = dict(i=i, ran=ran, stop=bool(stop), gF=(g / F if (g is not None and F) else None))
            if ran:
                r.update(nP=int(CT[b_][3] - CT[a_][3]), cap=int(CT[b_][5] - CT[a_][5]), st=float(DG[b_][14]) if not fbar else float(2.0 ** -BSK[k * nt + i]), bk=(BSK[k * nt + i] if fbar else None))
            its.append(r)
        ge = float(np.sqrt(max(SCe[k][1], 0.0)))
        gb = float(s.NR_GB[k][:nn].double().sum()) if getattr(s, "bar", False) else None
        steps.append(dict(k0f=(Y0F[k] if s.y0ut and getattr(s, "bar", False) else None), y0k=(Y0K[k] if s.y0ut and getattr(s, "bar", False) else None), F=F,
                          it=its, gFend=(ge / F if F else None), conv=bool(F and ge <= tolN * F) if tolN > 0 else None, Gbar=gb))
    return dict(steps=steps, tolN=tolN)
