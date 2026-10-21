"""모델 롤아웃이 멈춘 프레임 판정 + 이어 굴리기용 상태 사본 (사용자 10-21).

입력: 모델 롤아웃 덤프 (AF_ROLL_DUMP, pred·gt·x0·F[float32]), 궤적 .pt (cfg.frame_dt).
출력 (out_dir):
  motion.json  프레임별 입자 평균 이동량 |x_t - x_{t-1}| (모델·정답, EXT 로 나눈 %) 와 비
  tgt.pt       {pred, v, F}: v = (pred_t - pred_{t-1}) / frame_dt (모델 롤아웃 v 의 정의, dt_sub 1) -- AF_OV_CONT 용
판정 규칙 (고정): 모델 평균 이동량 m_t 가 같은 프레임 정답 g_t 의 10% 미만이면서 그때까지 모델 최대 이동량의 10% 미만인
첫 프레임 t, 그리고 그 뒤 끝까지 m 이 그 최대의 10% 를 넘지 않을 것 ("거의 0 으로 떨어져 다시 안 움직임").
"""
import argparse, json
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--frac", type=float, default=0.1)
ap.add_argument("--no_tgt", action="store_true", help="상태 사본은 안 쓴다 (판정 곡선만)")
a = ap.parse_args()

D = torch.load(a.dump, map_location="cpu", weights_only=False)
d = torch.load(a.data, map_location="cpu", weights_only=False)
h = float(d["cfg"]["frame_dt"]); EXT = float(D.get("EXT", 1.0))
P = D["pred"].float(); G = D["gt"].float(); x0 = D["x0"].float()
assert D.get("F") is not None and D["F"].dtype == torch.float32, "F float32 덤프가 아님"
Pp = torch.cat([x0[None], P[:-1]]); Gp = torch.cat([x0[None], G[:-1]])
m = ((P - Pp).norm(dim=-1).mean(1) / EXT * 100).numpy()
g = ((G - Gp).norm(dim=-1).mean(1) / EXT * 100).numpy()
err = ((P - G).norm(dim=-1).mean(1) / EXT * 100).numpy()
# 질량중심 이동량 (입자 평균 위치의 프레임 간 이동; 입자별 떨림은 상쇄된다)과 중앙값 이동량
cm = ((P.mean(1) - Pp.mean(1)).norm(dim=-1) / EXT * 100).numpy()
cg = ((G.mean(1) - Gp.mean(1)).norm(dim=-1) / EXT * 100).numpy()
md = ((P - Pp).norm(dim=-1).median(1).values / EXT * 100).numpy()
mg = ((G - Gp).norm(dim=-1).median(1).values / EXT * 100).numpy()
L = len(m); stop = None
for t in range(L):
    mx = m[:t + 1].max()
    if m[t] < a.frac * g[t] and m[t] < a.frac * mx and (m[t:] <= a.frac * mx).all():
        stop = t + 1           # 프레임 번호 (pred[t] = 프레임 t+1)
        break
rec = dict(dump=a.dump, frame_dt=h, EXT=EXT, frac=a.frac, stop_frame=stop,
           frames=list(range(1, L + 1)), model_move_pct=m.tolist(), gt_move_pct=g.tolist(),
           ratio=(m / np.maximum(g, 1e-12)).tolist(),
           model_com_move_pct=cm.tolist(), gt_com_move_pct=cg.tolist(), model_med_move_pct=md.tolist(), gt_med_move_pct=mg.tolist(), pos_err_pct=err.tolist(), n_particles=int(P.shape[1]))
json.dump(rec, open(f"{a.out_dir}/motion.json", "w"), indent=1)
for t in range(L):
    print(f"[이동량] 프레임 {t + 1:2d}  평균 모델 {m[t]:.4f}% 정답 {g[t]:.4f}% 비 {m[t] / max(g[t], 1e-12):.3f} | 중앙값 {md[t]:.4f}% / {mg[t]:.4f}%"
          f" | 질량중심 {cm[t]:.4f}% / {cg[t]:.4f}% 비 {cm[t] / max(cg[t], 1e-12):.3f} | 위치오차 {err[t]:.3f}%")
print(f"[멈춘 시점] 규칙(비<{a.frac}, 최대의 {a.frac} 미만, 이후 계속) -> 프레임 {stop}")
V = (P - Pp) / h
if a.no_tgt:
    print("OVCONT_PREP_OK"); raise SystemExit
torch.save({"pred": P, "v": V, "F": D["F"].float()}, f"{a.out_dir}/tgt.pt")
print(f"[상태 사본] {a.out_dir}/tgt.pt  pred {tuple(P.shape)} F {tuple(D['F'].shape)} h {h}  x0 대 정답 0 프레임 차 "
      f"{float((x0 - d['x'][0].float()).abs().max()):.2e}")
print("OVCONT_PREP_OK")
