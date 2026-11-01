"""절 AM 결과 불변 확인: 각 판의 x.pt 를 원래 솔버 판 (pg) 과 프레임마다 비교 + 시간 표.

  python exe/mpm_fast_cmp.py DIR  (DIR/<판>/x.pt, res.json; 기준 DIR/pg)
출력: DIR/cmp.json, 표 (stdout)
"""
import json
import os
import sys

import torch

D = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "pg"
RD = sys.argv[3] if len(sys.argv) > 3 else os.path.join(D, ref)                # 기준 판 폴더 (다른 묶음의 pg 를 쓸 때)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = json.load(open(os.path.join(RD, "res.json")))
EXT = R["EXT"]
xr = torch.load(os.path.join(RD, "x.pt")).to(DEV)
out = {}
names = sorted(v for v in os.listdir(D) if os.path.isfile(os.path.join(D, v, "res.json")))
print(f"| 판 | 프레임 중앙값 ms | 서브스텝 us | 배수 (pg 대비) | 정답 h5 대비 % | pg 대비 평균 % (80 평균 / 최대 프레임) | pg 대비 최대 |dx| mm |")
print("|---|---|---|---|---|---|---|")
for v in names:
    r = json.load(open(os.path.join(D, v, "res.json")))
    row = dict(frame_ms=r["frame_ms_median"], substep_us=r["substep_us_median"], err_gt=100 * r["err_mean"], prof=r["prof_ms_per_frame"],
               speedup=R["frame_ms_median"] / r["frame_ms_median"], cfl_v=r["cfl_v"], cfl_c=r["cfl_c"], vmax=max(r["vmax"]) if r["vmax"] else None)
    px = os.path.join(D, v, "x.pt")
    if os.path.isfile(px) and r["args"]["mode"] != "floor":
        x = torch.load(px).to(DEV)
        k = min(len(x), len(xr))
        d = (x[:k] - xr[:k]).norm(dim=-1)
        pf = d.mean(1) / EXT * 100
        row.update(dpg_mean=float(pf.mean()), dpg_maxframe=float(pf.max()), dpg_maxabs_mm=1e3 * float(d.max()), dpg_frames=k,
                   dpg_f10=float(pf[min(9, k - 1)]), dpg_f80=float(pf[k - 1]))
        del x
    out[v] = row
    print(f"| {v} | {row['frame_ms']:.2f} | {row['substep_us']:.1f} | {row['speedup']:.2f} | {row['err_gt']:.4f} | "
          f"{row.get('dpg_mean', float('nan')):.2e} / {row.get('dpg_maxframe', float('nan')):.2e} | {row.get('dpg_maxabs_mm', float('nan')):.3e} |")
for v in names:
    print(f"[구간] {v}: {json.dumps({k: round(x, 3) for k, x in out[v]['prof'].items()}, ensure_ascii=False)}")
r = out.get(ref, {}) or next(iter(out.values()), {})
print(f"[CFL] 최대 |v| {r.get('vmax')}, |v|max dt/dx {r.get('cfl_v')}, 탄성파 c_p dt/dx {r.get('cfl_c')}")
json.dump(out, open(os.path.join(D, "cmp.json"), "w"), ensure_ascii=False, indent=1)
