"""절 CM156: 뒤집힌 가우스 점 (det A F_q = JX <= 0 또는 JE <= 0) 이 처음 생기는 걸음의 위치·노드.  점 덤프 (cm151_wrap, 노드 U·mn·kb 포함, JX<=0 점도 그대로).
사용: python exe/cm156_an.py <판 디렉토리 (r2: 가능)> <h5 디렉토리 (r2: 가능)> <셀 크기 m>"""
import sys, os, io, subprocess
import numpy as np, h5py
run, h5d, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
def rd(p): return io.BytesIO(subprocess.run(["rclone", "cat", p], capture_output=True, check=True).stdout) if p.startswith("r2:") else p
def gtx(f):
    with h5py.File(rd(h5d + f"/sim_{f:010d}.h5"), "r") as hh: return np.array(hh["x"]).T
off = np.array([(o // 9, (o // 3) % 3, o % 3) for o in range(27)])
X0 = gtx(0); rel = np.full(len(X0), 99)
for f in range(1, 11):
    m = (rel == 99) & (np.abs(gtx(f)[:, 2] - X0[:, 2]) > 5e-4); rel[m] = f
for f in (8, 9, 10):
    try: d = np.load(rd(run + f"/cm151_f{f}.npz"))
    except Exception: continue
    nb1, nb2 = [int(v) for v in d["nbs"]]; U = d["U"]; mn = d["mn"]; cb = d["cb"]
    q = np.nonzero(d["mq"] > 0)[0]; JX = d["JX"][q]; JE = d["JE"][q]; J0 = d["J0"][q]; x = d["xq"][q]; pc = d["pc"][q]
    fl = (JX <= 0) | (JE <= 0); near = (JX > 0) & (JX < 0.3)
    print(f"== f{f}: 점 {len(q)}, 뒤집힘 (JX<=0 또는 JE<=0) {fl.sum()} (JX<=0 {(JX<=0).sum()}), JX<0.3 (양수) {near.sum()}, J0<=0 {(J0<=0).sum()}, JX 최소 {JX.min():.3f}")
    if not fl.any(): continue
    base = cb[q >> 3]; ids = ((base[:, None, 0] + off[None, :, 0]) * nb1 + (base[:, None, 1] + off[None, :, 1])) * nb2 + (base[:, None, 2] + off[None, :, 2])
    mnn = mn[ids]; Uz = np.where(mnn > 0, U[ids][:, :, 2], np.nan); Uall = np.where((mnn > 0)[:, :, None], U[ids], np.nan)
    rng = (np.nanmax(Uz, 1) - np.nanmin(Uz, 1)) * 1e3; mag = np.nanmax(np.linalg.norm(Uall, axis=2), 1) * 1e3
    xp = gtx(f - 1); heldf = (rel >= f)
    i = np.clip(np.floor((x - np.array([0.4, 0.4, 0.4])) / 0.02).astype(int), 0, 69)
    g = np.zeros((70, 70, 70)); gh = np.zeros((70, 70, 70)); ii = np.clip(np.floor((xp - 0.4) / 0.02).astype(int), 0, 69)
    np.add.at(g, tuple(ii.T), 1); np.add.at(gh, tuple(ii[heldf].T), 1); hv = gh[tuple(i.T)] / np.maximum(g[tuple(i.T)], 1)
    for lab, mk in (("뒤집힘", fl), ("나머지", ~fl)):
        mm = np.where(mnn[mk] > 0, mnn[mk], np.inf).min(1)
        print(f"  {lab}: 높이 중앙 {np.median(x[mk, 2] - 0.48):.3f} m | 칸 입자 수 중앙 {np.median(pc[mk]):.0f} | 칸 노드 z 변위 범위 중앙 {np.nanmedian(rng[mk]):.2f} mm (셀 {h*1e3:.0f} mm) | 최대 노드 변위 {np.nanmedian(mag[mk]):.2f} mm | 최소 노드 질량 중앙 {np.median(mm):.1e} | 정답 입자 중 아직 안 풀린 몫 hv 중앙 {np.median(hv[mk]):.2f} | 칸 변위 범위 > 셀 몫 {np.nanmean(rng[mk] > h*1e3):.2f}")
