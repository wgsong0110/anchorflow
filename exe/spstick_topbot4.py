"""lego 새 바닥(spstick): 셀(사면체)별 제일 위–제일 아래 꼭지점 간격의 스텝 전후 비율 분포 + 기존 RGB 렌더 합성.
셀 야코비안(spstick_jcell.py)과 같은 덤프·같은 사면체 연결 재구성(스텝 시작 입자 위치 -> simplex.grid_for_nodes·locate·active_nodes,
덤프 npos 일치 검산). 꼭지점: 스텝 시작 노드 위치에서 z 최대/최소 (같으면 x+y 로 가름). 후 = 노드 + 그 스텝 변위 dpn (바닥 매핑 전).
위: lbours/spstick/{gt,ov,m_multi}_T.mp4 프레임 t+1, 아래: 그래프 프레임 t (스텝 t->t+1).
  PYTHONPATH=<spstick 워크트리>/lib python spstick_topbot.py --node_h 0.028284 --out DIR
"""
import argparse, glob, json, math, os
import numpy as np, torch, imageio.v2 as iio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from anchorflow import simplex as SX
fpath = None
for f in fm.findSystemFonts() + glob.glob(os.path.expanduser("~/.fonts/*")):
    if "Noto" in f and ("CJK" in f or "KR" in f):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); fpath = f; break
plt.rcParams["axes.unicode_minus"] = False
W = "/home/dkta/work"; U = f"{W}/lbours/spstick"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--trajs", default="0000,0001,0002,0003"); ap.add_argument("--mm_dir", required=True, help="roll_mm_T.pt·mm_T.mp4·gt_T.mp4·ov_T.mp4 있는 곳")
ap.add_argument("--edges_from", required=True, help="기존 L000·0004 topbot_L000.npz (축 고정)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(16)
HZ = 1.0 / math.sqrt(6.0)
RUNS = [("sp", "새 바닥 출력만 최적화", "#d9480f", "roll_ov_{T}.pt", U),
        ("mm", "새 바닥 다섯 궤적 모델", "#1c7ed6", "roll_mm_{T}.pt", None)]
LOG = open(f"{a.out}/check.txt", "w")
def say(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
DATA = {}
TRAJ = a.trajs.split(",")
CONTACT = {}
for T in TRAJ:
    for key, lab, _, pat, base in RUNS:
        p = f"{base or a.mm_dir}/{pat.format(T=T)}"
        D = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        assert D.get("nodes"), f"{p}: 노드 덤프 없음"
        if key == "sp":   # 첫 접촉: 정답(덤프 gt) 입자가 바닥 거리 0.01 안에 처음 들어온 프레임 (stick_diag.py 와 같은 기준)
            cfg = json.load(open(f"{W}/lbdata/config.json")); bc = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
            pt_ = torch.tensor(bc["point"]).float(); nr_ = torch.tensor(bc["normal"]).float(); nr_ = nr_ / nr_.norm()
            dmin = [float(((D["x0"].float() - pt_) @ nr_).min())] + [float(((D["gt"][i].float() - pt_) @ nr_).min()) for i in range(D["gt"].shape[0])]
            c = next((i for i, dm in enumerate(dmin) if dm < 0.01), None); CONTACT[T] = (c, dmin)
            say(f"[{T} 첫 접촉] 정답 프레임 {c} (바닥 거리 최소: 0 프레임 {dmin[0]:.4f}, {c} 프레임 {dmin[c] if c is not None else float('nan'):.4f}) -> 직전 스텝 {max((c or 1) - 1, 1)}")
        nodes = D["nodes"]; L = len(nodes); R1, R2, NC = [], [], []; diag_ok = 0; ntot = 0; tie = 0
        for t in tqdm(range(L), desc=f"{T} {key}"):
            xs = (D["x0"] if t == 0 else D["pred"][t - 1]).to("cuda").float()
            npos_d, dpn, _, _ = (q.to("cuda") for q in nodes[t])
            lo, lat, nn = SX.grid_for_nodes(xs, 32, hz_ratio=HZ, h_fix=a.node_h)
            idx, lam, aux = SX.locate(xs, lo, lat, nn)
            rows, uniq = SX.active_nodes(idx)
            npos = SX.node_pos(lo, lat, nn, uniq)
            if npos.shape != npos_d.shape or float((npos - npos_d.float()).abs().max()) > 1e-5:
                raise SystemExit(f"[중단] {T} {key} 스텝 {t}: 재구성 노드가 덤프와 다르다")
            tid = SX.tet_id(lo, lat, nn, aux)
            _, first = np.unique(tid.cpu().numpy(), return_index=True); first = torch.as_tensor(first, device="cuda")
            rw = rows[first]                                   # [C,4] 사슬 순서 v0..v3
            P0 = npos_d.float()[rw]; P1 = P0 + dpn[:, :3].float()[rw]
            key_ = P0[..., 2] * 1e6 + (P0[..., 0] + P0[..., 1])   # z 우선, 같으면 x+y (z 차는 격자 간격 hz 단위라 1e6 배로 충분히 분리)
            zs = P0[..., 2]; srt = torch.sort(zs, 1).values; tie += int(((srt[:, 1:] - srt[:, :-1]).abs() < 1e-9).any(1).sum())
            it = key_.argmax(1); ib = key_.argmin(1)
            diag_ok += int(((it == 3) & (ib == 0)).sum()); ntot += len(first)
            g = lambda P, i: P[torch.arange(len(i), device=P.device), i]
            d0 = (g(P0, it) - g(P0, ib)); d1 = (g(P1, it) - g(P1, ib))
            R1.append((d1.norm(dim=-1) / d0.norm(dim=-1)).cpu().numpy().astype(np.float32))
            R2.append((d1[:, 2] / d0[:, 2]).cpu().numpy().astype(np.float32))
            NC.append(len(first))
        DATA[(T, key)] = (R1, R2, np.array(NC))
        say(f"[{T} {key}] {L} 스텝, 점유 사면체 {min(NC)}~{max(NC)}; 노드 재구성 전 스텝 일치; "
            f"제일 위/아래 = Kuhn 사슬 끝 v3/v0(주대각선 양 끝) 비율 {diag_ok/ntot:.6f} ({diag_ok}/{ntot}); z 동률 있는 셀 {tie}; 원본 {p}")
# (GPU 로 재구성 -- CPU 는 floor/정렬 반올림이 학습 때(GPU)와 갈려 스텝 15 에서 노드가 달라졌다)
# 고정 축 (두 방법·두 궤적·전 스텝 공통)
EZ = np.load(a.edges_from); edges = [EZ["edges0"], EZ["edges1"]]; rng = [(float(e[0]), float(e[-1])) for e in edges]; NB = len(edges[0]) - 1
for j in range(2):
    v = np.concatenate([x for k in DATA for x in DATA[k][j]])
    say(f"[축 {j}] 기존 L000·0004 축 그대로 {rng[j][0]:.5f}~{rng[j][1]:.5f}; 이 궤적들 전체 최소 {v.min():.5f} 최대 {v.max():.5f}, 축 밖 비율 {((v<rng[j][0])|(v>rng[j][1])).mean():.2e} (끝 칸에 합침)"); cen = [0.5 * (e[1:] + e[:-1]) for e in edges]
XL = ["|p_top − p_bot| 스텝 후 / 스텝 전  (선형 축)", "Δz(top−bot) 스텝 후 / 스텝 전  (선형 축)"]
TI = ["제일 위–제일 아래 꼭지점 거리 비율", "같은 두 꼭지점의 z 간격 비율"]
font = ImageFont.truetype(fpath, 28) if fpath else None
LAB = ["정답 (PG)", "새 바닥 출력만 최적화", "새 바닥 다섯 궤적 모델"]
V = {}
TAB = open(f"{a.out}/table_topbot4.txt", "w")
TAB.write("# 궤적 스텝 방법 | 거리비 중앙 · 눌림(<1) 비율 | Δz비 중앙 · 눌림(<1) 비율   (스텝 k = 프레임 k-1→k, 첫 접촉 직전 = 정답이 바닥 0.01 안에 처음 든 프레임 c 의 직전 스텝 c-1)\n")
for T in TRAJ:
    Rr = [iio.mimread(f"{a.mm_dir}/{n}_{T}.mp4", memtest=False) for n in ("gt", "ov", "mm")]
    assert all(len(r) == 81 for r in Rr), [len(r) for r in Rr]
    L = len(DATA[(T, "sp")][0]); assert L == 80
    ST = {k: np.array([[np.median(DATA[(T, k)][j][t]), np.quantile(DATA[(T, k)][j][t], .01), np.quantile(DATA[(T, k)][j][t], .99),
                        (DATA[(T, k)][j][t] < 1).mean()] for t in range(L) for j in range(2)]).reshape(L, 2, 4) for k, *_ in RUNS}
    np.savez(f"{a.out}/topbot_{T}.npz", **{f"ST_{k}": ST[k] for k in ST}, edges0=edges[0], edges1=edges[1])
    with open(f"{a.out}/stats_{T}.txt", "w") as f:
        f.write(f"# lego {T} 셀별 위-아래 꼭지점 비율; 열: 거리비 중앙·1%·99%·<1 비율, Δz비 중앙·1%·99%·<1 비율\n")
        c = CONTACT[T][0]; kpre = max((c or 1) - 1, 1)
        for nm, k_ in (("스텝1", 1), (f"첫접촉직전(스텝{kpre})", kpre), ("스텝40", 40), ("스텝80", 80)):
            for k, *_ in RUNS:
                s_ = ST[k][k_ - 1]; TAB.write(f"{T} {nm} {k} | {s_[0,0]:.5f} · {100*s_[0,3]:.1f}% | {s_[1,0]:.5f} · {100*s_[1,3]:.1f}%\n")
        TAB.flush()
        for t in (0, 39, 79):
            for k, *_ in RUNS:
                s = ST[k][t]; f.write(f"스텝 {t+1:3d} {k} " + " ".join(f"{q:.5f}" for q in s.ravel()) + "\n")
    say(open(f"{a.out}/stats_{T}.txt").read())
    ylo = 0.5 / max(DATA[(T, k)][2].max() for k, *_ in RUNS)
    out = f"{a.out}/spstick_topbot_img_{T}.mp4"
    wr = iio.get_writer(out, fps=15, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=8)
    for t in tqdm(range(L), desc=f"{T} 영상"):
        fig, ax = plt.subplots(1, 2, figsize=(19.2, 6.4), dpi=100)
        for j in range(2):
            for r_, (k, lab, c, *_r) in enumerate(RUNS):
                v = DATA[(T, k)][j][t]; h = np.histogram(np.clip(v, *rng[j]), edges[j])[0] / len(v)
                h = np.where(h > 0, h, ylo * 0.5)
                ax[j].step(cen[j], h, where="mid", color=c, lw=1.6, label=f"{lab} (셀 {len(v)}개)")
                ax[j].fill_between(cen[j], h, ylo * 0.5, step="mid", color=c, alpha=0.10)
                s = ST[k][t, j]
                ax[j].text(0.02, 0.98 - 0.06 * r_, f"{lab}: 중앙 {s[0]:.4f} · 1% {s[1]:.4f} · 99% {s[2]:.4f} · 비율<1(눌림) {100*s[3]:.1f}%",
                           transform=ax[j].transAxes, va="top", ha="left", fontsize=9, color=c)
            ax[j].axvline(1, color="#888", lw=0.8, ls=":")
            ax[j].set_yscale("log"); ax[j].set_ylim(ylo, 1.2); ax[j].set_xlim(*rng[j])
            ax[j].set_xlabel(XL[j]); ax[j].set_ylabel("셀 비율 (로그 축)"); ax[j].set_title(TI[j], fontsize=11)
            ax[j].spines[["top", "right"]].set_visible(False)
        ax[0].legend(loc="lower left", frameon=False, fontsize=9)
        fig.suptitle(f"lego {T} · 새 바닥 · 셀(사면체)별 제일 위–제일 아래 꼭지점 간격 비율 · 스텝 {t}→{t+1}"
                     f"   (점유 사면체만, 바닥 매핑 전 셀 매핑, 축 영상 전체 고정, 범위 밖은 끝 칸)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.canvas.draw()
        g = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
        GW = g.width; tiles = [Image.fromarray(r[t + 1]) for r in Rr]
        th = int(tiles[0].height * (GW / 3) / tiles[0].width)
        top = Image.new("RGB", (GW, th + 40), "white"); d = ImageDraw.Draw(top)
        for i, im in enumerate(tiles):
            top.paste(im.resize((GW // 3, th)), (i * (GW // 3), 40))
            d.text((i * (GW // 3) + 10, 4), f"{LAB[i]} · 렌더 프레임 {t+1}", fill="black", font=font)
        cimg = Image.new("RGB", (GW, top.height + g.height), "white"); cimg.paste(top, (0, 0)); cimg.paste(g, (0, top.height))
        if cimg.height % 2: cimg = cimg.crop((0, 0, GW, cimg.height - 1))
        wr.append_data(np.asarray(cimg))
    wr.close(); say(f"[영상] {out} {L} 프레임")
TAB.close(); say(open(f"{a.out}/table_topbot4.txt").read())
print("SPTOPBOT4_OK", flush=True)
