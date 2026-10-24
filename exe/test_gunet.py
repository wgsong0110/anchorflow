"""--gnn_layer unet (Graph U-Net 몸통, 사용자 10-24) 단위 확인. GPU 하나, 수 초.

  풀링·언풀링 형상 · 거친 간선 종류 14 안 · 자식 평균 == 직접 계산 · 0 초기화(항등) · 몸통 모든 층·스킵 결합 기울기 ·
  노드 집합이 바뀌면 계층 다시 만듦 (같은 객체면 재사용) · 정수 좌표 짝수 이동 불변 · torch.func.jvp(τ) · 파라미터 수 (mp 8 층 128 대비)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX                       # noqa: E402
from anchorflow import simplex_gnn as SG                   # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OK = [True]


def chk(name, ok, msg=""):
    print(f"[{'통과' if ok else '실패'}] {name} {msg}", flush=True)
    OK[0] &= bool(ok)


def make(N, seed, shift=0.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(N, 3, generator=g, dtype=torch.float64) * torch.tensor([0.3, 0.2, 0.25], dtype=torch.float64) + 0.5 + shift).to(dev)
    lo, lat, nn = SX.grid_for_nodes(x, 0, h_fix=0.028284)
    idx, lam, aux = SX.locate(x, lo, lat, nn)
    rows, uniq = SX.active_nodes(idx)
    src, dst, cls = SX.edges_of(rows, uniq, nn)
    return uniq, nn, src, dst, cls


NF, H = 33, 128
uniq, nn, src, dst, cls = make(6000, 1)
Mn = uniq.numel()
net = SG.GraphUNetGNN(n_feat=NF, hidden=H, levels=3, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
mp = SG.SimplexGNN(n_feat=NF, hidden=H, layers=8, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
npar = lambda m: sum(p.numel() for p in m.parameters())
print(f"층 배분 {net.plan}, 층 {len(net.body)}")
print(f"파라미터 수: Graph U-Net {npar(net)} (몸통 {npar(net.body)}, 스킵 결합 {npar(net.fuse)}) | 기존 mp 8 층 128 {npar(mp)} (몸통 {npar(mp.body)}) "
      f"| 비 {npar(net) / npar(mp):.3f}")
chk("층 수 8 (3 수준)", len(net.body) == 8)

# --- 계층 형상
lv = net.hierarchy(src, dst, cls, (uniq, nn))
for l, d in enumerate(lv):
    print(f"  수준 {l} (간격 {2 ** l}h): 노드 {d['M']}, 간선 {d['src'].numel()}, 노드당 이웃 평균 {d['src'].numel() / max(d['M'], 1):.2f}")
chk("수준별 노드 수 줄어듦", all(lv[l + 1]["M"] < lv[l]["M"] for l in range(len(lv) - 1)), str([d["M"] for d in lv]))
for l in range(len(lv) - 1):
    P = lv[l]["parent"]
    chk(f"수준 {l}->{l + 1} 부모 색인 범위·모든 거친 노드에 자식", P.shape == (lv[l]["M"],) and int(P.min()) == 0 and int(P.max()) == lv[l + 1]["M"] - 1
        and bool((lv[l + 1]["cnt"] > 0).all()), f"자식 수 최소 {int(lv[l + 1]['cnt'].min())} 최대 {int(lv[l + 1]['cnt'].max())} (최대 8)")
    chk(f"수준 {l + 1} 간선 종류 0..13", bool(((lv[l + 1]["cls"] >= 0) & (lv[l + 1]["cls"] < SX.N_EDGE_CLASS)).all()))
    chk(f"수준 {l + 1} 자기 간선 없음", bool((lv[l + 1]["src"] != lv[l + 1]["dst"]).all()))
pos0 = SG.grid_ijk(uniq, nn)
pos1 = torch.div(pos0, 2, rounding_mode="floor")
same = (pos1[lv[0]["src"]] == pos1[lv[0]["dst"]]).all(-1)
chk("부모 같음 <-> 정수좌표//2 같음", bool(((lv[0]["parent"][lv[0]["src"]] == lv[0]["parent"][lv[0]["dst"]]) == same).all()))
# 거친 간선 오프셋 = 부모 좌표 차 (종류 표와 일치)
_, cpos, (cs, cd, cc), _ = SG.coarsen_grid(pos0, src, dst)
offt = torch.tensor(SX.EDGE_OFFSETS, device=dev)
chk("거친 간선 종류 == 부모 좌표 차", bool((offt[cc] == (cpos[cd] - cpos[cs])).all()))
# 풀링 평균 직접 계산
hh = torch.randn(Mn, H, device=dev)
P = lv[0]["parent"]
agg = torch.zeros(lv[1]["M"], H, device=dev).index_add_(0, P, hh) / lv[1]["cnt"].unsqueeze(-1).float()
j = int(torch.randint(0, lv[1]["M"], (1,)))
chk("자식 평균 == 직접", float((agg[j] - hh[P == j].mean(0)).abs().max()) < 1e-5)
up = agg[P]
chk("언풀링 형상", up.shape == (Mn, H) and bool((up[P == j] == agg[j]).all()))

# --- 순전파·기울기
feat = torch.randn(Mn, NF, device=dev)
o = net(feat, src, dst, cls, 1 / 24, grid=(uniq, nn))[0]
chk("출력 형상 [M,3]", o.shape == (Mn, 3), str(tuple(o.shape)))
chk("출력 0 초기화 (항등)", float(o.abs().max()) == 0.0)
torch.nn.init.normal_(net.out.weight, std=0.01)
o = net(feat, src, dst, cls, 1 / 24, grid=(uniq, nn))[0]
o.square().sum().backward()
gl = [sum(float(p.grad.abs().sum()) for p in b.parameters() if p.grad is not None) for b in net.body]
gf = [sum(float(p.grad.abs().sum()) for p in f.parameters() if p.grad is not None) for f in net.fuse]
fin = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
chk("몸통 8 층 모두 기울기 > 0", all(g > 0 for g in gl), " ".join(f"{g:.2e}" for g in gl))
chk("스킵 결합 기울기 > 0", all(g > 0 for g in gf), " ".join(f"{g:.2e}" for g in gf))
chk("기울기 유한", bool(fin))
chk("입력층 기울기 > 0", float(net.inp.weight.grad.abs().sum()) > 0)

# --- 같은 객체면 계층 재사용, 노드 집합 바뀌면 새로
lvA = net.hierarchy(src, dst, cls, (uniq, nn))
chk("같은 객체 -> 계층 재사용", lvA is lv)
u2, n2, s2, d2, c2 = make(4500, 7, shift=0.013)
f2 = torch.randn(u2.numel(), NF, device=dev)
o2 = net(f2, s2, d2, c2, 1 / 24, grid=(u2, n2))[0]
st2 = [d["M"] for d in net._hc[2]]
chk("노드 집합 바뀜 -> 새 계층·출력 형상", o2.shape == (u2.numel(), 3) and net._hc[0] is u2 and bool(torch.isfinite(o2).all()),
    f"노드 {Mn} -> {u2.numel()}, 수준별 {st2}")
o3 = net(feat, src, dst, cls, 1 / 24, grid=(uniq, nn))[0]
chk("되돌아오면 같은 출력", float((o3 - o).abs().max()) < 1e-6)

# --- 정수 좌표 2^(수준-1) = 4 배수 이동 불변 (모든 수준에서 묶는 칸이 같다): nn 을 키워 uniq 를 (4,4,4) 만큼 민 색인으로
#     (2 만 밀면 수준 1 에서 1 칸 = 홀수 이동이 되어 수준 2 묶음이 달라진다 -- 그래서 4)
nnl = [int(nn[k]) for k in range(3)]
nnb = torch.tensor([nnl[0] + 8, nnl[1] + 8, nnl[2] + 8], device=dev)
pb = pos0 + 4
ub = (pb[:, 0] * nnb[1] + pb[:, 1]) * nnb[2] + pb[:, 2]
o4 = net(feat, src, dst, cls, 1 / 24, grid=(ub, nnb))[0]
chk("정수 좌표 (4,4,4) 이동 -> 같은 출력", float((o4 - o).abs().max()) < 1e-6, f"{float((o4 - o).abs().max()):.2e}")
pc = pos0 + torch.tensor([1, 0, 0], device=dev)
uc = (pc[:, 0] * nnb[1] + pc[:, 1]) * nnb[2] + pc[:, 2]
o5 = net(feat, src, dst, cls, 1 / 24, grid=(uc, nnb))[0]
print(f"  (참고) 홀수 (1,0,0) 이동은 묶는 칸이 달라져 출력이 달라진다: 최대차 {float((o5 - o).abs().max()):.2e}")
pd = pos0 + 2
ud = (pd[:, 0] * nnb[1] + pd[:, 1]) * nnb[2] + pd[:, 2]
o7 = net(feat, src, dst, cls, 1 / 24, grid=(ud, nnb))[0]
print(f"  (참고) (2,2,2) 이동은 수준 2 묶음만 달라진다: 최대차 {float((o7 - o).abs().max()):.2e}")

# --- 격자 원점 (lo, lat): 같은 점 집합을 격자 한 칸 민 바운딩박스로 잡아도 (imin 이 1 바뀜) 전역 좌표 묶음이 같다
xw = (torch.rand(3000, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 0.2 + 0.5).to(dev)
def build_w(xx, extra=None):
    lo_, lat_, nn__ = SX.grid_for_nodes(xx if extra is None else torch.cat([xx, extra]), 0, h_fix=0.028284)
    idx_, lam_, _ = SX.locate(xx, lo_, lat_, nn__)
    rows_, uq_ = SX.active_nodes(idx_)
    s_, d_, c_ = SX.edges_of(rows_, uq_, nn__)
    return uq_, nn__, lo_, lat_, s_, d_, c_
A_ = build_w(xw)
A_lat = A_[3]
far = (xw.min(0).values - 1.0 * A_lat.A[:, 0] * 1.0001).reshape(1, 3)   # 바운딩박스만 격자 한 칸 넓힘 (이 점은 locate 하지 않음)
B_ = build_w(xw, far)
iA = torch.round(A_[3].Ai @ A_[2]).long(); iB = torch.round(B_[3].Ai @ B_[2]).long()
fw = torch.randn(A_[0].numel(), NF, device=dev)
oA = net(fw, A_[4], A_[5], A_[6], 1 / 24, grid=(A_[0], A_[1], A_[2], A_[3]))[0]
oB = net(fw, B_[4], B_[5], B_[6], 1 / 24, grid=(B_[0], B_[1], B_[2], B_[3]))[0]
oBn = net(fw, B_[4], B_[5], B_[6], 1 / 24, grid=(B_[0], B_[1]))[0]
chk("격자 원점 반영: 바운딩박스가 바뀌어도 같은 출력", A_[0].numel() == B_[0].numel() and float((oA - oB).abs().max()) < 1e-6,
    f"imin {iA.tolist()} -> {iB.tolist()}, 최대차 {float((oA - oB).abs().max()):.2e} (원점 없이 국소 좌표면 {float((oA - oBn).abs().max()):.2e})")

# --- jvp (계층을 jvp 안에서 처음 만드는 경우 포함)
net._hc = None
tau = torch.tensor(1 / 24, device=dev)
y, ty = torch.func.jvp(lambda t_: net(feat, src, dst, cls, t_, grid=(uniq, nn))[0], (tau,), (torch.ones_like(tau),))
chk("torch.func.jvp 통과 (계층 jvp 안에서 생성)", bool(torch.isfinite(ty).all()) and float((y - o).abs().max()) < 1e-6, f"|dy/dτ| {float(ty.abs().max()):.3g}")

# --- 4 수준
net4 = SG.GraphUNetGNN(n_feat=NF, hidden=H, levels=4, dt_cond=True, dt_ref=1 / 24, dt_scale=True).to(dev)
torch.nn.init.normal_(net4.out.weight, std=0.01)
o6 = net4(feat, src, dst, cls, 1 / 24, grid=(uniq, nn))[0]
chk("4 수준 동작", bool(torch.isfinite(o6).all()), f"층 {len(net4.body)}, 파라미터 {npar(net4)}, 수준별 노드 {[d['M'] for d in net4._hc[2]]}")
print("TEST_GUNET", "OK" if OK[0] else "FAIL", flush=True)
sys.exit(0 if OK[0] else 1)
