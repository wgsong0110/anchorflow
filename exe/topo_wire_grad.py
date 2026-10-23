"""토폴로지 추적 riem 사면체 메쉬 실행: 프레임별 목적함수 + 유클리드·리만 기울기 크기 (그림).

rep_track2 --opt riem --kappa 0 --riem_energy logbarrier --method tetmesh (b4d94f8) 의 정의 그대로:
  변수 θ = 메쉬 꼭짓점 변위 u [M,3] (정지 위치 Xn 대비, 연결·무게중심 좌표 고정)
  목적함수 f = 입자 L2 = mean_i |y_i - tgt_i|²,  y = G0 + Λu    (κ=0 -> 장벽은 목적함수에 없다)
  유클리드 기울기 g = ∇_u f = (2/N) Λᵀ(y - tgt)
  계량 G = JrᵀJr + ε I,  r_c(u) = log(clamp(J_c, 1e-6)) / √C  (셀 det 로그 장벽 잔차의 가우스-뉴턴, riem_k=1)
       ε = riem_eps · λmax(JrᵀJr), 프레임 시작에서 한 번 (실행 기록 stop[:,4] 의 값을 그대로 쓴다)
  리만 기울기 크기 = √(gᵀ G⁻¹ g)  (= G-노름 ‖G⁻¹g‖_G),  G⁻¹g 는 CG (상대 잔차 1e-8 까지)
상태: 프레임 t 의 처음 = 직전 프레임 끝 노드 (topo_wire.py analyze 가 되찾은 u_{t-1}), 끝 = u_t.
y 는 실행이 저장한 위치를 그대로 쓴다 (g 는 u 복원 오차와 무관); G 는 복원한 u 로.
"""
import argparse
import math
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--flow", required=True)
ap.add_argument("--aux", required=True)
ap.add_argument("--res", required=True)
ap.add_argument("--nodes", required=True)
ap.add_argument("--out", required=True, help="npz (그림은 같은 이름 .png)")
ap.add_argument("--stall", type=int, default=-1)
ap.add_argument("--riem_eps", type=float, default=1e-2)
ap.add_argument("--cg", type=int, default=3000)
ap.add_argument("--plot_only", action="store_true", help="계산은 건너뛰고 저장된 --out npz 로 그림만")
ap.add_argument("--font", default="/home/dkta/.fonts/NotoSansCJKkr-Regular.otf")
a = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    use = sorted((int(m), int(i)) for i, m in (l.split(",") for l in q))
    print("[GPU] 메모리 사용 (MiB):", {i: m for m, i in use}, "-> 카드", use[0][1], flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use[0][1])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
from torch.func import jvp, vjp                                      # noqa: E402
from anchorflow import simplex as sx                                 # noqa: E402

dev = "cuda"
dt = torch.float64
D = np.load(a.flow)
AX = np.load(a.aux, allow_pickle=True)
G0f = torch.as_tensor(AX["G"], dtype=torch.float32, device=dev)[torch.as_tensor(D["idx"], device=dev)].contiguous()
NG = G0f.shape[0]
S_NORM = float(AX["s"])
h = (2.0 / 100.0) * (2.0 * math.sqrt(2.0)) ** (1.0 / 3.0) / S_NORM
lo, lat, nn = sx.grid_for_nodes(G0f, 1.0, h_fix=h)
idx, lam, _ = sx.locate(G0f, lo, lat, nn)
rows, uniq = sx.active_nodes(idx)
Xn = sx.node_pos(lo, lat, nn, uniq).to(dt)
cells = torch.unique(rows, dim=0)
M, C = Xn.shape[0], cells.shape[0]
Xc0 = Xn[cells]
cD0i = torch.linalg.inv((Xc0[:, 1:] - Xc0[:, :1]).transpose(1, 2))
ii = torch.arange(NG, device=dev).repeat_interleave(4)
AT = torch.sparse_coo_tensor(torch.stack([rows.reshape(-1), ii]), lam.reshape(-1).to(dt), (M, NG)).coalesce()
Z = np.load(a.nodes)
NODES = torch.as_tensor(Z["nodes"], device=dev, dtype=dt)            # [T+1,M,3]
assert NODES.shape[1] == M
R = np.load(a.res)
Y = R["traj"]
T = Y.shape[0]
TR = D["traj"]
stop = R["stop"]
curve = R["curve"]
met = R["metrics"]
L = float(R["L"])
print(f"[메쉬] 꼭짓점 {M} 사면체 {C}  프레임 {T}  L {L:.4f}", flush=True)


def res(x):
    X = (Xn + x.view(M, 3))[cells]
    Dm = (X[:, 1:] - X[:, :1]).transpose(1, 2)
    J = torch.linalg.det(Dm @ cD0i)
    return torch.log(J.clamp_min(1e-6)) * (1.0 / math.sqrt(C))


def GN(th):
    _, vf = vjp(res, th)
    return lambda v: vf(jvp(res, (th,), (v,))[1])[0]


def lam_max(Hv, n):
    g_ = torch.Generator(device=dev).manual_seed(0)
    v = torch.randn(n, device=dev, dtype=dt, generator=g_)
    lm = 0.0
    for _ in range(60):
        v = Hv(v); lm = float(v.norm()); v = v / max(lm, 1e-300)
    return lm


def cg(Av, b, maxit, tol=1e-8):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = (r * r).sum(); nb = b.norm()
    if float(nb) == 0.0:                                             # 정답이 안 움직인 프레임: g = 0
        return x, 0, 0.0
    for k in range(maxit):
        Ap = Av(p); al = rr / (p * Ap).sum().clamp_min(1e-300)
        x += al * p; r -= al * Ap
        rr2 = (r * r).sum()
        if rr2.sqrt() < tol * nb:
            break
        p = r + (rr2 / rr.clamp_min(1e-300)) * p; rr = rr2
    return x, k + 1, float(rr2.sqrt() / nb)


out = []
if a.plot_only:
    T = 0
print("t | 상태 | f | ‖g‖ | √(gᵀG⁻¹g) | ε(기록) | λmax 재계산·ε/0.01 비 | ‖G⁻¹g‖ | J_c 최소 | CG 반복·상대잔차", flush=True)
with torch.no_grad():
    for t in range(1, T + 1):
        tgt = torch.as_tensor(TR[t], device=dev, dtype=dt)
        eps = float(stop[t - 1, 4])
        for s_, (y_np, uu) in enumerate(((Y[t - 2] if t > 1 else G0f.cpu().numpy(), NODES[t - 1]),
                                         (Y[t - 1], NODES[t]))):
            y = torch.as_tensor(y_np, device=dev, dtype=dt)
            d = y - tgt
            f = float((d * d).sum(1).mean())
            g = (2.0 / NG) * torch.sparse.mm(AT, d).reshape(-1)
            th = (uu - Xn).reshape(-1).contiguous()
            Hv = GN(th)
            lm = lam_max(Hv, th.numel()) if s_ == 0 else float("nan")
            x, k, rel = cg(lambda v: Hv(v) + eps * v, g, a.cg)
            rn = math.sqrt(max(float((g * x).sum()), 0.0))
            Xc = uu[cells]
            Jc = torch.linalg.det((Xc[:, 1:] - Xc[:, :1]).transpose(1, 2) @ cD0i)
            row = (t, s_, f, float(g.norm()), rn, eps, lm, float(x.norm()), float(Jc.min()), k, rel)
            out.append(row)
            print(f"{t:2d} | {'처음' if s_ == 0 else '끝  '} | {f:.4e} | {row[3]:.4e} | {rn:.4e} | {eps:.3e} | "
                  f"{lm:.3e} {lm * a.riem_eps / eps if s_ == 0 else float('nan'):.3f} | {row[7]:.3e} | {row[8]:.2e} | {k} {rel:.1e}",
                  flush=True)
O = np.load(a.out)["rows"] if a.plot_only else np.array(out)
T = Y.shape[0]
if not a.plot_only:
  np.savez(a.out, rows=O, curve=curve, metrics=met, L=L,
         cols=np.array(["t", "state(0 처음,1 끝)", "f", "g_euc", "g_riem", "eps", "lmax_recomp", "Ginv_g_norm",
                        "cellJ_min", "cg_iters", "cg_rel"]))
  print(f"[저장] {a.out}", flush=True)

# ---------------------------------------------------------------- 그림
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib import font_manager                                  # noqa: E402
font_manager.fontManager.addfont(a.font)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=a.font).get_name()
plt.rcParams["axes.unicode_minus"] = False
ts = np.arange(1, T + 1)
f0 = np.array([curve[curve[:, 0] == t][0, 2] for t in ts])
f1 = (met[:, 1] * L) ** 2                                            # 프레임 끝 (RMSE·L)²
S0, S1 = O[O[:, 1] == 0], O[O[:, 1] == 1]
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
ax[0].semilogy(ts, f0, "o-", ms=3, color="#1f5fbf", label="프레임 처음")
ax[0].semilogy(ts, f1, "s-", ms=3, color="#d9480f", label="프레임 끝")
ax[0].set_title("목적함수 (입자 L2 평균) -- 프레임별")
ax[0].set_xlabel("프레임"); ax[0].set_ylabel("입자 L2")
sel = sorted(set([1, 10, a.stall - 2, a.stall, a.stall + 2, 30, T] if a.stall > 0 else [1, 10, 20, 30, T]))
cm = plt.get_cmap("viridis")
for i, t in enumerate([t for t in sel if 1 <= t <= T]):
    cv = curve[curve[:, 0] == t]
    ax[1].semilogy(cv[:, 1], cv[:, 2] / cv[0, 2], color=cm(i / max(len(sel) - 1, 1)), label=f"프레임 {t}")
ax[1].set_title("프레임 안 반복별 목적함수 (처음 값으로 나눔)")
ax[1].set_xlabel("반복"); ax[1].set_ylabel("L2 / 처음 L2")
ax[2].semilogy(ts, S0[:, 3], "o-", ms=3, color="#1f5fbf", label="유클리드 ‖g‖ (처음)")
ax[2].semilogy(ts, S1[:, 3], "o--", ms=3, color="#1f5fbf", alpha=0.6, label="유클리드 ‖g‖ (끝)")
ax[2].semilogy(ts, S0[:, 4], "s-", ms=3, color="#d9480f", label="리만 √(gᵀG⁻¹g) (처음)")
ax[2].semilogy(ts, S1[:, 4], "s--", ms=3, color="#d9480f", alpha=0.6, label="리만 √(gᵀG⁻¹g) (끝)")
ax[2].set_title("기울기 크기 (변수 = 메쉬 꼭짓점 변위, G = 셀 det 로그 장벽 GN + εI)")
ax[2].set_xlabel("프레임")
for k in (0, 2):
    if a.stall > 0:
        ax[k].axvline(a.stall, color="red", lw=1.5, ls=":")
        ax[k].text(a.stall + 0.5, ax[k].get_ylim()[1], f"멈춤 {a.stall}", color="red", va="top")
for k in range(3):
    ax[k].grid(alpha=0.3, which="both"); ax[k].legend(fontsize=9)
fig.suptitle("wolf 사면체 메쉬 riem 추적 (κ=0: 목적함수 = 입자 L2 만, 장벽은 계량에만)")
fig.tight_layout()
png = a.out.replace(".npz", ".png")
fig.savefig(png, dpi=130)
print(f"[그림] {png}", flush=True)
