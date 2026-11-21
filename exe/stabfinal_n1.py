"""[stabfinal CM192] 우리 판 N=1 세 번 (a·b·c) 안정 판정 + 다시 풀기 (ssfix) 로 걸음 수를 늘린 프레임 수. torch 없이 (로컬).
판정 = exe/stab1115_lib.judge 와 같은 식: 전 프레임 끝 (NaN 없음) + 운동+위치 에너지 증가 최대 (탄성 에너지 뺌) <= 1 + 바닥 관통 깊이 / L <= 0.5.
  python exe/stabfinal_n1.py   (R2 에서 res.npz·log.txt 를 ~/.cache/sfn1 로 받아 판정)
"""
import json, os, re, subprocess
import numpy as np
RX = "r2:storage/result/anchorflow/lbours/stabfinal/res"; C = os.path.expanduser("~/.cache/sfn1")
SC = [("lego", "Lego", "_f5"), ("pillow2sofa", "pillow2sofa", "_f5"), ("ship_plastic_lego", "ship", "_f7"), ("mic_metal_lego", "mic", "_f7")]
out = {}
for s, nm, tg in SC:
    os.makedirs(f"{C}/{s}", exist_ok=True)
    subprocess.run(f"rclone copyto {RX}/{s}/config.json {C}/{s}/config.json -q", shell=True)
    cfg = json.load(open(f"{C}/{s}/config.json"))
    zf = float([b for b in cfg["boundary_conditions"] if b["type"] == "surface_collider"][0]["point"][2])
    nf = int(cfg.get("frame_num", 60))
    for rep in "abc":
        d = f"{s}_N1_{rep}{tg}"; ld = f"{C}/{s}/{d}"
        subprocess.run(f"rclone copy {RX}/{s}/hex/{d} {ld} -q --include res.npz --include log.txt --include FIN", shell=True)
        if not os.path.exists(f"{ld}/res.npz"):
            out[f"{nm} {rep}"] = None; print(nm, rep, "없음"); continue
        z = np.load(f"{ld}/res.npz"); keys = [str(k) for k in z["keys"]]; R = {k: z["stats"][:, i] for i, k in enumerate(keys)}
        X = z["x"]; L = float(np.linalg.norm(X[0].max(0) - X[0].min(0)))
        fin = len(X) >= nf + 1 and np.nanmax(R["nan"]) == 0 and np.isfinite(X).all()
        E = R["ke"] + R["pe"]; ed = float(np.nanmax((E - E[0]) / abs(E[0])))
        pen = float(max((zf - X[..., 2]).max(), 0.0) / L)
        lg = open(f"{ld}/log.txt", errors="ignore").read() if os.path.exists(f"{ld}/log.txt") else ""
        mm = re.findall(r"프레임 (\d+) 다시 풂 (\d+) 깊이별 (\{[^}]*\})", lg)
        hist = {int(k): int(v) for k, v in re.findall(r"(\d+): (\d+)", mm[-1][2])} if mm else {}
        nres = sum(v for k, v in hist.items() if k > 0)
        ok = bool(fin and ed <= 1.0 and pen <= 0.5)
        out[f"{nm} {rep}"] = dict(frames=len(X) - 1, stable=ok, e_drift_max=ed, pen_depth=pen, resolved_frames=nres, depth_hist=hist)
        print(f"{nm:12s} {rep} 프레임 {len(X) - 1}/{nf} 안정 {ok} 에너지 증가 {ed:.4f} 관통/L {pen:.4f} 다시 푼 프레임 {nres} {hist}")
json.dump(out, open(f"{C}/n1.json", "w"), indent=1)
