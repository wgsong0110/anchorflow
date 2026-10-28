"""FEM 실시간 판 단독 시뮬레이터 (사용자 10-28 승인 계획, doc/fem_realtime.md 절 H) -- lib/anchorflow/fem_rt.py.

train_deform FEM 판 (AF_FEM_TOL=1e-2 AF_FEM_FAST=tet AF_FEM_TETQ=1 AF_FEM_FIX_NEWTON=1, PCG 블록 야코비 rtol 1e-6 상한 10000, |λ| 투영 바닥 1e-6,
LM, K float32, 고유분해 float32) 과 같은 식을 상태를 GPU 에 둔 채 돈다. 오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 L 프레임 평균
(train_deform [롤아웃] 과 같은 지표, EXT = 바운딩박스 대각선 DM.extent(x0)).

  --floor particle : 기존 바닥 정의 (입자 softplus·stick 매핑, fem_fast.FemFrame.tetq_*)
  --floor node     : 노드 단위 장벽 (+ --k_st 고착) (fem_rt.NodeFrame) -- 사용자 승인 정의 변경
  --solver pcg     : 블록 야코비 PCG (기존)
           mg      : Kuhn 격자 기하 멀티그리드 V(ν,ν) 전처리 PCG
           pdpcg   : PD 행렬 (질량 + w·2μ V_T 라플라시안) 미리 분해 (cholespy) 를 전처리로 쓴 PCG (뉴턴 방향 그대로)
           pd      : PD 반복 U <- U - α L⁻¹ ∇E (국소 = 극분해가 든 기울기, 전역 = 분해 재사용 풀이) --pd_iters 회, 아르미호 선탐색
출력 <out>/: fem.jsonl (프레임마다 시간·반복·오차·입자 최소 높이), summary.json, roll_fem.pt (train_deform AF_ROLL_DUMP 와 같은 형식, F float32).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
os.environ.setdefault("AF_FEM_FAST_K", "manual")
os.environ.setdefault("AF_FEM_FAST_NEAR", "analytic")
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_fast as FF                                               # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--floor", choices=["particle", "node"], default=os.environ.get("AF_FEM_FLOOR", "particle"))
ap.add_argument("--eps_sp", type=float, default=0.01)
ap.add_argument("--eps_st", type=float, default=0.01)
ap.add_argument("--kappa", type=float, default=1.0)
ap.add_argument("--dhat", type=float, default=0.0, help="0 이면 0.5·grid_lim/n_grid (기존 --bc_dhat 기본)")
ap.add_argument("--k_st", type=float, default=1000.0, help="노드 고착 세기 (관성 m/h² 배수, 0 = 미끄러짐)")
ap.add_argument("--nf_clear", choices=["min", "lam"], default="min", help="노드 여유 c_n: min = 받치는 입자 최저 높이, lam = min s_p/λ_pn")
ap.add_argument("--nf_stick", choices=["min", "lam"], default="min", help="고착 무게 w_n: min = 1-σ(c_n/ε), lam = max_p λ_pn (1-σ(s_p/ε))")
ap.add_argument("--nf_mfloor", type=float, default=0.0, help="장벽·고착 무게 질량 하한 (꽉 찬 노드 질량 ρ·v_node 의 배수)")
ap.add_argument("--solver", choices=["pcg", "mg", "pdpcg", "pd", "vbd"], default="pcg")
ap.add_argument("--newton", type=int, default=1)
ap.add_argument("--tol", type=float, default=1e-2, help="AF_FEM_TOL: |p|∞/(h·EXT) < tol 이면 수렴 (걸음 안 함)")
ap.add_argument("--pcg_max", type=int, default=10000)
ap.add_argument("--rtol", type=float, default=1e-6)
ap.add_argument("--ck", type=int, default=50, help="PCG 수렴 확인 간격")
ap.add_argument("--eig_floor", type=float, default=1e-6)
ap.add_argument("--mg_nu", type=int, default=2)
ap.add_argument("--mg_omega", type=float, default=0.6)
ap.add_argument("--mg_nmin", type=int, default=400)
ap.add_argument("--mg_f32", action="store_true", help="멀티그리드 전처리를 float32 로")
ap.add_argument("--pd_w", type=float, default=1.0, help="PD 라플라시안 무게 = pd_w · 2μ")
ap.add_argument("--pd_iters", type=int, default=10)
ap.add_argument("--pd_reuse", type=int, default=1, help="1 = 노드 집합이 같으면 분해 재사용")
ap.add_argument("--pd_floor_diag", type=int, default=1, help="1 = 바닥 장벽·고착 곡률의 등방 대각을 L 에 (분해 때 값)")
ap.add_argument("--proj_gate", action="store_true", help="배치 Cholesky 로 양정치인 사면체는 고유분해 투영을 건너뛴다")
ap.add_argument("--graph", action="store_true", help="PCG 반복 한 번을 CUDA 그래프로 잡아 재생")
ap.add_argument("--fast", action="store_true", help="정렬 없는 격자 압축, 자동미분 없는 E·기울기, 사면체 B 곱 float32")
ap.add_argument("--bell", action="store_true", help="CSR 대신 블록 ELL (15 점 스텐실) 곱만 (패턴 정렬 없음)")
ap.add_argument("--hess", choices=["newton", "gn"], default="newton", help="gn = 가우스-뉴턴 탄성 헤시안 (양반정, 고유분해 투영 없음; --floor node)")
ap.add_argument("--csr_bell", action="store_true", help="CSR 를 블록 ELL 에서 정렬 없이 (패턴 정렬 생략, 곱은 cuSPARSE)")
ap.add_argument("--cuda", action="store_true", help="손으로 쓴 CUDA 커널 (fem_rt_cuda): E·기울기, 헤시안+|λ| 투영+블록 ELL 합, PCG (--floor node)")
ap.add_argument("--pcg2", type=int, default=1, help="--cuda: 2 커널 PCG (1) / 5 커널 (0) / 지속 커널 (2) / 지속 f32 탐색 벡터 (3) / "
                "지속 float32 + float64 신뢰 갱신 = 혼합 정밀도 반복 개선 (4) / 지속 float32 단독 (5)")
ap.add_argument("--mix_delta", type=float, default=0.1, help="--pcg2 4: 재귀 잔차가 지난 갱신 뒤 최대의 δ 배 아래면 float64 잔차로 갱신")
ap.add_argument("--mix_cap", type=int, default=2, help="--pcg2 4/5: SM 당 상주 블록 상한")
ap.add_argument("--mix_kern", default="pcg_mix", help="--pcg2 4/5: 커널 (pcg_mix / pcg_mix_lb2·lb3·lb4 = __launch_bounds__(256, k))")
ap.add_argument("--mix_rows", type=int, default=0, help="--pcg2 4/5: 1 = 곱을 (행, 노드) 당 스레드로")
ap.add_argument("--vbd_ls", type=int, default=0, help="--solver vbd: 노드 국소 선탐색 (α = 1, 1/2, ..., 2^-(k-1) 중 국소 에너지가 줄어드는 가장 큰 α; 0 = 끔)")
ap.add_argument("--vbd_warm", choices=["none", "acc", "adapt"], default="none",
                help="--solver vbd 시작점: none = h v_n + h² g, acc = 이전 프레임 입자 가속도 (노드 평균) 로 h² a_n, adapt = VBD 논문 적응 초기화 (중력 방향 성분만 [0, |g|] 로 자름)")
ap.add_argument("--jac_sweeps", type=int, default=10)
ap.add_argument("--jac_tol", type=float, default=1e-12)
ap.add_argument("--tprof", type=int, default=-1, help="이 프레임부터 5 프레임을 torch.profiler 로 (CPU·GPU 상위 연산 표)")
ap.add_argument("--gate_tau", type=float, default=0.0, help="--cuda: H + τ|H|_F I 가 Cholesky 되면 고유분해 투영 생략 (0 = 끔)")
ap.add_argument("--dump_pred", action="store_true", help="위치 궤적만 pred.pt 로 (결과 불변 대조용)")
ap.add_argument("--lite", action="store_true", help="--floor particle --cuda: 격자 locate·사면체 ∇λ·P2G·G2P 도 커널/마스크 합으로 (FemFrame 안 거침)")
ap.add_argument("--warp", action="store_true", help="--cuda: 사면체 헤시안을 워프당 사면체 하나 (공유 메모리 병렬 야코비)")
ap.add_argument("--d3", action="store_true", help="--cuda PCG 곱 노드당 3 스레드 (기본 노드당 1)")
ap.add_argument("--near32", action="store_true", help="--cuda: 바닥 후보 헤시안 매핑 대수를 float32 로")
ap.add_argument("--true_res", action="store_true", help="--cuda: PCG 뒤 실제 잔차 |b-Ax|/|b| (float64) 기록")
ap.add_argument("--sym", action="store_true", help="--cuda --warp --pcg2 2: 블록 ELL 대칭 반쪽 (자기+양의 오프셋 7 칸) 저장 -- L2 에 들어간다")
ap.add_argument("--vbd_iters", type=int, default=10, help="--solver vbd: 반복 (8 색 가우스-자이델 한 바퀴 = 1)")
ap.add_argument("--vbd_rho", type=float, default=0.0, help="--solver vbd: 체비셰프 가속 ρ (0 = 끔)")
ap.add_argument("--vbd_maxstep", type=float, default=0.0, help="--solver vbd: 노드 한 번 이동 상한 (node_h 배수, 0 = 없음)")
ap.add_argument("--vbd_refresh", type=int, default=0, help="--solver vbd: R 반복마다 노드 블록 헤시안을 지금 상태에서 다시 (0 = 걸음 시작 값 고정)")
ap.add_argument("--mmax", type=int, default=60000)
ap.add_argument("--prof", action="store_true", help="구간마다 CUDA 동기화해 정확한 구간 시간")
ap.add_argument("--no_dump", action="store_true")
ap.add_argument("--dump_sys", default="", help="쉼표 프레임 목록: 그 프레임 첫 걸음 선형계 (블록 ELL·b) 를 <out>/sys_t###.pt 로")
ap.add_argument("--warm", type=int, default=2, help="시간 통계에서 뺄 앞 프레임 수 (컴파일·데우기)")
a = ap.parse_args()

dev = "cuda"
os.makedirs(a.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
EXT = float(d["_ext"])
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
L = min(int(a.frames), int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).float()
v = d["v"][0].to(dev).float()
F = DM.traj_F(d, h, dev)[0].to(dev).float().reshape(N, 3, 3)
GT = d["x"][1:L + 1].to(dev).float()
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
mu, lam = PR.lame(cfg["E"], cfg["nu"])
dhat = a.dhat if a.dhat > 0 else 0.5 * float(cfg.get("grid_lim", 2.0)) / int(cfg["n_grid"])
planes = [(torch.as_tensor(b["point"], device=dev, dtype=torch.float32), torch.as_tensor(b["normal"], device=dev, dtype=torch.float32))
          for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider"]
planes = [(p, n / n.norm()) for p, n in planes]
M_FULL = float(cfg["density"]) * float(torch.linalg.det(RT.SX.lattice(a.node_h, RT.SX.HZ_CUBE * a.node_h, dtype=torch.float64).A).abs())
sys_dump = {int(q) for q in a.dump_sys.split(",") if q.strip()}
print(f"[실시간 FEM] 입자 {N}, 프레임 {L}, h {h:.6f}, E {cfg['E']:g} ν {cfg['nu']:g} (μ {mu:.4g} λ {lam:.4g}), EXT {EXT:.6f}, 바닥 {a.floor} "
      f"(d̂ {dhat:g}, κ {a.kappa:g}, k_st {a.k_st:g}), 풀이 {a.solver}, 뉴턴 {a.newton}, 계측 {'동기화' if a.prof else '끝만'}", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)

PC = RTC.CudaPcg(a.mmax, dev) if a.cuda else None
if PC is not None:
    PC.d3 = a.d3
    PC.sym = int(a.sym)
    PC.mx_cap = a.mix_cap
    PC.mx_kern = a.mix_kern
    PC.mx_rows = a.mix_rows
PDF = RT.PdFactor() if a.solver in ("pdpcg", "pd") else None
PRED = torch.empty(L, N, 3, device=dev) if (not a.no_dump or a.dump_pred) else None
VD = torch.empty(L, N, 3, device=dev, dtype=torch.float16) if not a.no_dump else None
FD = torch.empty(L, N, 3, 3, device=dev, dtype=torch.float16) if not a.no_dump else None
errs = []
recs = []
v_prev = None
flog = open(os.path.join(a.out, "fem.jsonl"), "w")


def lm_resolve(mvA, D, MnH, b, mu_):
    reg = (mu_ * MnH).unsqueeze(-1).expand(-1, 3).reshape(-1)
    Dm = D + torch.diag_embed((mu_ * MnH).unsqueeze(-1).expand(-1, 3))
    Di = torch.linalg.inv(Dm)
    xx, it, res, neg = RT.pcg(lambda q: mvA(q) + reg * q, lambda r: (Di @ r.reshape(-1, 3, 1)).reshape(-1),
                              b, a.pcg_max, a.rtol, a.ck)
    return xx, it, neg


_TP = None
for i in range(L):
    if i == a.tprof:
        from torch.profiler import profile, ProfilerActivity
        _TP = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        _TP.__enter__()
    if _TP is not None and i == a.tprof + 5:
        torch.cuda.synchronize()
        _TP.__exit__(None, None, None)
        _ka = _TP.key_averages()
        print(_ka.table(sort_by="self_cpu_time_total", row_limit=30), flush=True)
        print(_ka.table(sort_by="self_cuda_time_total", row_limit=20), flush=True)
        _TP = None
    torch.cuda.synchronize()
    t_frame0 = time.perf_counter()
    tk = RT.Tick(a.prof)
    rec = dict(t=i)
    G = RTC.GridK(x, a.node_h) if a.lite else RT.Grid(x, a.node_h, fast=a.fast)
    U, ws = RT.node_inertial(x, v, m, G, g, h)                                # --ov_init predict
    if a.solver == "vbd" and a.vbd_warm != "none" and v_prev is not None:
        # VBD 시작점: h v_n + h² ã_n (ã = 이전 프레임 입자 가속도의 노드 질량 가중 평균; adapt = 중력 방향 성분만 [0, |g|] 로 자른 것 -- Chen et al. 2024)
        ap_ = (v - v_prev) / h
        if a.vbd_warm == "adapt":
            gh = g / g.norm()
            ap_ = gh * (ap_ @ gh).clamp(0.0, float(g.norm())).unsqueeze(-1)
        U = U + (h * h) * (RT.node_inertial(x, ap_, m, G, torch.zeros_like(g), 1.0)[0] - g)
    MnH = ws.double() / (h * h)
    tk("grid")
    if a.lite and a.floor == "particle":
        fr = CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U, eps_sp=a.eps_sp, eps_st=a.eps_st, stick_xn=True)
        CF.warp = a.warp
        CF.near32 = a.near32
        CF.sym = int(a.sym)
        if a.solver == "vbd":
            CF.vbd_setup(G)
        tets = G.tets
        rec["n_near"] = CF.nn
    elif a.floor == "node":
        fr = RT.NodeFrame(G, x, v, F, m, vol, cfg, h, kappa=a.kappa, dhat=dhat, k_st=a.k_st, eps_st=a.eps_st, clear=a.nf_clear,
                          m_floor=a.nf_mfloor * M_FULL, stick_w=a.nf_stick)
        if a.fast:
            fr.energy, fr.energy_grad = fr.energy_fast, fr.energy_grad_fast
        if a.cuda:
            CF = RTC.CudaFem(fr)
            CF.warp = a.warp
            CF.sym = int(a.sym)
            fr.energy, fr.energy_grad = CF.energy, CF.energy_grad
            if a.solver == "vbd":
                CF.vbd_setup(G)
        tets = G.tets
    else:
        fr = FF.FemFrame(x, v, F, m, vol, G.rows, G.lam, G.npos, None, cfg, h, eps_sp=a.eps_sp, eps_st=a.eps_st, stick_xn=True, dtype=torch.float32)
        fr.tetq_setup(U)
        tets = fr.uq
        rec["n_near"] = int(fr.tq["ni"].numel())
        if a.cuda:
            CF = RTC.CudaTetq(fr)
            fr.energy, fr.energy_grad = CF.energy, CF.energy_grad
    tk("p2g")
    csr = RT.Csr(tets, G.M) if not (a.bell or a.csr_bell or a.cuda) else None
    tk("pattern")
    steps = []
    for k in range(a.newton):
        st = {}
        if a.floor == "particle" and k > 0:
            fr.tetq_setup(U)
        if a.solver == "pd":
            # PD 반복: U <- U - α L⁻¹ ∇E (아르미호), --pd_iters 회. 수렴 판정은 뉴턴과 같은 |p|∞/(h·EXT) < tol
            E0, gg = fr.energy_grad(U)[:2]
            Es = float(E0)
            tk("eg")
            cd = None
            if a.pd_floor_diag and a.floor == "node":
                cd = torch.zeros(G.M, device=dev, dtype=torch.float64)
                for nr, cn, w in fr.planes:
                    b2 = RT.barrier3(cn + U.double() @ nr, fr.dhat)[2]
                    cd = cd + fr.kb * b2 + fr.ks * w
            VT = torch.zeros(G.T, device=dev, dtype=torch.float64).index_add_(0, G.pinv, vol.double())
            new = PDF.build(G, ws.double(), VT, h, a.pd_w * 2.0 * mu, cdiag=cd, reuse=bool(a.pd_reuse))
            st["pd_fact"] = bool(new)
            tk("pre")
            nacc, nls, pin = 0, 0, float("nan")
            for it in range(a.pd_iters):
                if it > 0:
                    E0, gg = fr.energy_grad(U)[:2]
                Ec = float(E0)
                p = PDF.solve(gg.reshape(-1).double()).reshape(G.M, 3).to(U.dtype)
                gp = float((gg.double() * p.double()).sum())
                pin = float(p.abs().max()) / (h * EXT)
                if not gp > 0:
                    break
                al = 1.0
                for _ in range(30):
                    En = float(fr.energy(U - al * p))
                    nls += 1
                    if math.isfinite(En) and En <= Ec - 1e-4 * al * gp:
                        U = U - al * p
                        nacc += 1
                        break
                    al *= 0.5
                else:
                    break
            tk("pd_iter")
            st.update(pd_iters=nacc, n_ls=nls, E0=Es, E1=float(fr.energy(U)), pin=pin)
            steps.append(st)
            break
        E0, gg = fr.energy_grad(U)[:2]
        gg = gg.to(U.dtype)
        Ec = float(E0)
        tk("eg")
        if a.cuda:
            M_ = G.M
            gi = G.gi
            slots = (tets.unsqueeze(-1) * 15 + RT.off_cls(gi[tets].unsqueeze(1) - gi[tets].unsqueeze(2))).int().contiguous()
            Ms = PC.Mmax
            PC.B.zero_()
            CF.cnt.zero_()
            CF.hess_into(U, slots, PC.B, a.eig_floor, a.jac_sweeps, a.jac_tol, Ms=Ms, gtau=a.gate_tau)
            Bs = PC.B.view(15, 9, Ms)
            if a.floor == "node":
                Dn = fr._floor_hess(U)
                Bs[0, :, :M_] += Dn.reshape(M_, 9).t().float()
            tk("hess")
            D = Bs[0, :, :M_].t().double().reshape(M_, 3, 3)
            bad = ~(torch.diagonal(D, dim1=1, dim2=2) > 0).all(1)
            D[bad] = torch.eye(3, device=dev, dtype=D.dtype)
            Dinv = torch.linalg.inv(D)
            PC.Dinv[:M_ * 9].copy_(Dinv.reshape(-1))
            nbl = RT.BEll(G.gi, dtype=torch.float32)
            PC.nb.view(15, Ms)[:, :M_] = (torch.where(nbl.valid, nbl.nb, torch.full_like(nbl.nb, -1)) if a.sym else nbl.nb).t().int()
            tk("asm")
            if a.solver == "vbd":
                if i == 0:
                    _gv = CF.vbd_grad_all(U)
                    print(f"[검산] VBD 색별 국소 기울기 합 (float32) vs 전체 기울기 (float64) 상대차 {float((_gv - gg.double()).norm() / gg.double().norm()):.2e}", flush=True)
                def _dinv(Uc, M_=M_, Ms=Ms):
                    PC.B.zero_()
                    CF.hess_into(Uc, slots, PC.B, a.eig_floor, a.jac_sweeps, a.jac_tol, Ms=Ms)
                    D_ = PC.B.view(15, 9, Ms)[0, :, :M_].t().double().reshape(M_, 3, 3)
                    if a.floor == "node":
                        D_ = D_ + fr._floor_hess(Uc.double())
                    bad_ = ~(torch.diagonal(D_, dim1=1, dim2=2) > 0).all(1)
                    D_[bad_] = torch.eye(3, device=dev, dtype=D_.dtype)
                    return torch.linalg.inv(D_)
                vfl = None
                if a.floor == "node" and fr.planes:
                    nr_, cn_, w_ = fr.planes[0]
                    vfl = dict(cn=cn_.contiguous(), w=w_.contiguous(), kb=fr.kb.contiguous(), ks=(fr.ks * (1.0 if fr.k_st > 0 else 0.0)).contiguous(),
                               n=nr_, dhat=fr.dhat)
                U = CF.vbd(U, Dinv, a.vbd_iters, rho=a.vbd_rho, maxstep=a.vbd_maxstep * a.node_h, refresh=a.vbd_refresh, dinv_fn=_dinv,
                           ls=a.vbd_ls, floor=vfl)
                tk("vbd")
                st.update(E0=Ec, E1=float(fr.energy(U)), vbd_iters=a.vbd_iters,
                          vbd_rej=(int(CF.vbd_nrej) if a.vbd_ls > 0 else None))
                tk("ls")
                steps.append(st)
                break
            b = gg.reshape(-1).double()
            if a.pcg2 in (4, 5):
                xx, it, res, negc = PC.solve_mix(M_, b, a.rtol, a.pcg_max, mode=(1 if a.pcg2 == 4 else 0), delta=a.mix_delta)
                st["mix_upd"] = PC.last_nup
            else:
                xx, it, res, negc = {3: PC.solve_coop32, 2: PC.solve_coop, 1: PC.solve2, 0: PC.solve}[a.pcg2](M_, b, a.rtol, a.pcg_max, use_graph=a.graph)
            if a.true_res:
                st["true_res"] = PC.true_res(M_, xx, b)
            negf = int(CF.cnt[1]) / G.T
            st["gate"] = int(CF.cnt[0])
            tk("pcg")
            Bd = Bs[:, :, :M_].permute(2, 0, 1).reshape(M_, 15, 3, 3)
            nbi = nbl.nb
            if a.sym:
                # LM 재풀이용 전체 블록: 음의 오프셋 칸 = 이웃의 양의 칸 전치 (무효 칸은 0)
                Bd = Bd.clone()
                for k_ in range(1, 8):
                    j_ = nbl.nb[:, 7 + k_]
                    Bd[:, 7 + k_] = torch.where(nbl.valid[:, 7 + k_, None, None], Bd[j_, k_].transpose(1, 2), torch.zeros_like(Bd[:, 0]))
            mv = lambda q: torch.einsum("mkab,mkb->ma", Bd.double(), q.reshape(M_, 3)[nbi]).reshape(-1)
        if a.cuda:
            pass
        elif True:
            if a.floor == "node":
                He, Dn = fr.hess_gn(U) if a.hess == "gn" else fr.hess(U, f32_b=a.fast)
            else:
                He, _fi = fr.hess_tetq(U)
                Dn = None
            tk("hess")
            if a.hess == "gn":
                negf = None
            else:
                He, negf = RT.project(He, a.eig_floor, gate=a.proj_gate)
            tk("proj")
            if a.bell:
                Be = RT.BEll(G.gi)
                Be.add_tets(tets, He)
                if Dn is not None:
                    Be.add_diag(Dn)
                A = Be
                D = Be.diag().clone()
                bad = ~(torch.diagonal(D, dim1=1, dim2=2) > 0).all(1)
                D[bad] = torch.eye(3, device=dev, dtype=D.dtype)
                mv = Be.mv
            elif a.csr_bell:
                Be = RT.BEll(G.gi)
                Be.add_tets(tets, He)
                if Dn is not None:
                    Be.add_diag(Dn)
                A = Be.to_csr_sorted(G.nn)
                if i == 0 and k == 0:
                    _v = torch.randn(3 * G.M, device=dev, dtype=torch.float64)
                    print(f"[검산] 정렬 없는 CSR 곱 vs 블록 ELL 곱 상대차 {float(((A @ _v.unsqueeze(-1)).squeeze(-1) - Be.mv(_v)).norm() / Be.mv(_v).norm()):.2e}", flush=True)
                D = Be.diag().clone()
                bad = ~(torch.diagonal(D, dim1=1, dim2=2) > 0).all(1)
                D[bad] = torch.eye(3, device=dev, dtype=D.dtype)
                mv = lambda q: (A @ q.unsqueeze(-1)).squeeze(-1)
            else:
                A = csr.build(He, Dn)
                D = RT.block_diag(tets, He, G.M, Dn)
                mv = lambda q: (A @ q.unsqueeze(-1)).squeeze(-1)
            Dinv = torch.linalg.inv(D)
            tk("asm")
            jac = lambda r: (Dinv @ r.reshape(-1, 3, 1)).reshape(-1)
            pre = jac
            if a.solver == "mg":
                if not (a.bell or a.csr_bell):
                    Be = RT.BEll(G.gi)
                    Be.add_tets(tets, He)
                    if Dn is not None:
                        Be.add_diag(Dn)
                MG = RT.KuhnMG(Be, nu=a.mg_nu, omega=a.mg_omega, nmin=a.mg_nmin, dtype=torch.float32 if a.mg_f32 else torch.float64,
                               A0csr=(None if a.bell else A), bell=a.bell)
                pre = MG
                st.update(mg_levels=MG.sizes, mg_drop=MG.drop)
            elif a.solver == "pdpcg":
                cd = None
                if a.pd_floor_diag and Dn is not None:
                    cd = torch.diagonal(Dn, dim1=1, dim2=2).mean(1)
                VT = fr.VT if a.floor == "node" else torch.zeros(G.T, device=dev, dtype=torch.float64).index_add_(0, G.pinv, vol.double())
                new = PDF.build(G, ws.double(), VT, h, a.pd_w * 2.0 * mu, cdiag=cd, reuse=bool(a.pd_reuse))
                st["pd_fact"] = bool(new)
                pre = PDF.solve
            tk("pre")
            b = gg.reshape(-1).double()
            if i in sys_dump and k == 0:
                Be2 = RT.BEll(G.gi)
                Be2.add_tets(tets, He)
                if Dn is not None:
                    Be2.add_diag(Dn)
                torch.save(dict(gi=G.gi.cpu(), nb=Be2.nb.cpu(), valid=Be2.valid.cpu(), B=Be2.B.cpu(), b=b.cpu(), tets=tets.cpu(), He=He.float().cpu(),
                                Dn=(Dn.cpu() if Dn is not None else None), M=G.M, t=i, mn=ws.cpu(), VT=(fr.VT.cpu() if a.floor == "node" else None),
                                gl=G.gl8.cpu(), mu=mu, h=h), os.path.join(a.out, f"sys_t{i:03d}.pt"))
                tk("dump")
            _r = RT.pcg_graph(mv, pre, b, a.pcg_max, a.rtol) if a.graph else None
            xx, it, res, negc = _r if _r is not None else RT.pcg(mv, pre, b, a.pcg_max, a.rtol, (min(a.ck, 2) if a.solver == 'mg' else a.ck))
            tk("pcg")
        p = xx.reshape(G.M, 3).to(U.dtype)
        pin = float(p.abs().max()) / (h * EXT)
        gp = float((gg * p).sum())
        st.update(E0=Ec, cg=it, cg_res=math.sqrt(res), neg_curv=negc, neg_tet=negf, pin=pin, gp=gp)
        if not (math.isfinite(pin) and math.isfinite(gp)):
            st["why"] = "nan"
            steps.append(st)
            break
        if pin < a.tol:
            st["why"] = "수렴"
            steps.append(st)
            break
        mu_, tries, acc, nls = 0.0, [], False, 0
        pd_ = p
        while True:
            if mu_ > 0:
                xm, itm, _ = lm_resolve(mv, D, MnH, b, mu_)
                pd_ = xm.reshape(G.M, 3).to(U.dtype)
            gpd = float((gg * pd_).sum())
            al, acc = 1.0, False
            if gpd > 0:
                for _ in range(30):
                    En = float(fr.energy(U - al * pd_))
                    nls += 1
                    if math.isfinite(En) and En <= Ec - 1e-4 * al * gpd:
                        acc = True
                        break
                    al *= 0.5
            tries.append(dict(mu=mu_, alpha=al, acc=acc))
            if acc:
                break
            mu_ = max(10.0 * mu_, 1e-4)
            if mu_ > 1e8:
                break
        if acc:
            U = U - al * pd_
        tk("ls")
        st.update(alpha=al if acc else 0.0, acc=acc, n_ls=nls, E1=(En if acc else Ec), lm=(tries if len(tries) > 1 else None))
        steps.append(st)
    # G2P
    if a.lite and a.floor == "particle":
        x2, v2k, F2 = CF.g2p(U, F)
    elif a.floor == "node":
        x2, F2 = fr.g2p(x, F, U)
    else:
        Uf = U.to(x.dtype)
        Up = Uf[G.rows]
        u = (G.lam.unsqueeze(-1) * Up).sum(1)
        Gp = (Up.unsqueeze(-1) * fr.gl.unsqueeze(-2)).sum(1)
        x2, Jf = fr._maps(x, x + u, torch.eye(3, device=dev) + Gp)
        F2 = Jf @ F
    v2 = (x2 - x) / h
    tk("g2p")
    torch.cuda.synchronize()
    t_frame = time.perf_counter() - t_frame0
    # ---- 측정 밖: 오차·바닥 높이·덤프 ----
    e = float((x2 - GT[i]).norm(dim=-1).mean()) / EXT
    errs.append(e)
    smin, nbelow = None, None
    if planes:
        s = ((x2 - planes[0][0]) * planes[0][1]).sum(-1)
        smin, nbelow = float(s.min()), int((s < 0).sum())
    if PRED is not None:
        PRED[i] = x2
    if VD is not None:
        VD[i] = v2.half()
        FD[i] = F2.half()
    rec.update(sec=t_frame, ms=tk.d, steps=steps, err=e, s_min=smin, n_below=nbelow, M=G.M, T=G.T, nnz=(csr.nnz if csr is not None else None))
    if PDF is not None:
        rec.update(pd_nfact=PDF.n_fact, pd_tfact=PDF.t_fact)
    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
    flog.flush()
    recs.append(rec)
    s0 = steps[0] if steps else {}
    print(f"  프레임 {i + 1}/{L} {1e3 * t_frame:.1f} ms 오차 {100 * e:.3f}% (누적 평균 {100 * sum(errs) / len(errs):.3f}%) 노드 {G.M} 사면체 {G.T} "
          f"PCG {s0.get('cg', '-')} α {s0.get('alpha', '-')} 최저 입자 {smin:+.5f} 아래 {nbelow} "
          + " ".join(f"{k_} {v_:.1f}" for k_, v_ in tk.d.items()), flush=True)
    v_prev = v
    x, v, F = x2.detach(), v2.detach(), F2.detach()

flog.close()
W = a.warm
sec = [r["sec"] for r in recs[W:]]
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
keys = sorted({k for r in recs for k in r["ms"]})
summ = dict(out=a.out, args=vars(a), frames=L, err_mean=sum(errs) / len(errs), ms_mean=1e3 * sum(sec) / max(len(sec), 1), ms_med=1e3 * med(sec),
            fps=len(sec) / max(sum(sec), 1e-9), ms_parts_med={k: med([r["ms"].get(k, 0.0) for r in recs[W:]]) for k in keys},
            ms_parts_mean={k: sum(r["ms"].get(k, 0.0) for r in recs[W:]) / max(len(sec), 1) for k in keys},
            cg_med=med([r["steps"][0].get("cg", 0) for r in recs if r["steps"]]), cg_mean=sum(r["steps"][0].get("cg", 0) for r in recs if r["steps"]) / L,
            s_min=min(r["s_min"] for r in recs if r["s_min"] is not None) if planes else None,
            n_below_max=max(r["n_below"] for r in recs if r["n_below"] is not None) if planes else None,
            frames_below=sum(1 for r in recs if (r["n_below"] or 0) > 0),
            lm_frames=sum(1 for r in recs if r["steps"] and r["steps"][0].get("lm")),
            rej_frames=sum(1 for r in recs if r["steps"] and r["steps"][0].get("acc") is False),
            pd_nfact=(PDF.n_fact if PDF else None), pd_tfact_ms=(1e3 * PDF.t_fact / max(PDF.n_fact, 1) if PDF else None),
            true_res_max=max([r["steps"][0].get("true_res") or 0.0 for r in recs if r["steps"]] or [0.0]),
            true_res_med=med([r["steps"][0].get("true_res") or 0.0 for r in recs if r["steps"]]),
            mix_upd_mean=(sum(r["steps"][0].get("mix_upd") or 0 for r in recs if r["steps"]) / L),
            vbd_rej_mean=(sum(r["steps"][0].get("vbd_rej") or 0 for r in recs if r["steps"]) / L),
            gpu=torch.cuda.get_device_name(0))
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(f"[롤아웃] L000 t0=  0: {L} 프레임, 평균 {100 * summ['err_mean']:.3f}%  프레임당 {summ['ms_mean']:.1f} ms (중앙 {summ['ms_med']:.1f}, "
      f"{summ['fps']:.2f} fps, 앞 {W} 프레임 제외) PCG 중앙 {summ['cg_med']} 최저 입자 {summ['s_min']} 아래 최대 {summ['n_below_max']} "
      f"실제 잔차 최대 {summ['true_res_max']:.2e} 갱신 평균 {summ['mix_upd_mean']:.2f}", flush=True)
print("[구간 중앙 ms] " + " ".join(f"{k} {v_:.2f}" for k, v_ in summ["ms_parts_med"].items()), flush=True)
if a.dump_pred and PRED is not None:
    torch.save({"pred": PRED.half().cpu() if False else PRED.cpu(), "EXT": EXT}, os.path.join(a.out, "pred.pt"))
    PRED = None
if PRED is not None:
    torch.save({"pred": PRED.cpu(), "gt": GT.cpu(), "x0": d["x"][0].float(), "EXT": EXT, "t0": 0, "tag": "L000",
                "F": FD.float().cpu(), "v": VD.float().cpu(), "ctrl_pos": d.get("ctrl_pos"), "ctrl_R": d.get("ctrl_R"), "ctrl_id": d.get("ctrl_id"),
                "nodes": None, "fscal": None, "gt_fscal": None}, os.path.join(a.out, "roll_fem.pt"))
    print(f"[롤아웃 덤프] {os.path.join(a.out, 'roll_fem.pt')}", flush=True)
