"""절 AH.8 진단: 덤프 (roll_fem.pt) 에서 우리 / 정답의 비강체 변형 -- 0 프레임 대비 질량 무가중 Kabsch 맞춤 잔차 RMS (EXT 비) 와 몸 좌표 높이 (회전 되돌린 z 범위).
사용: python exe/fem_ah_rigid.py <roll_fem.pt> [<roll_fem.pt> ...]"""
import sys

import torch

for p in sys.argv[1:]:
    d = torch.load(p, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    x0 = d["x0"].float().to(d["pred"].device) if hasattr(d["x0"], "float") else torch.as_tensor(d["x0"]).float()
    E = float(d["EXT"])
    c0 = x0.mean(0)
    q0 = x0 - c0
    h0 = float(q0[:, 2].max() - q0[:, 2].min())
    print(p, f"EXT {E:.3f} 0 프레임 높이 {h0:.3f}")
    for k, X in (("우리", d["pred"]), ("정답", d["gt"])):
        out = []
        for i in range(0, X.shape[0], 8):
            x = X[i].float()
            c = x.mean(0)
            H = q0.T @ (x - c)
            U, S, Vh = torch.linalg.svd(H)
            Dg = torch.eye(3, device=x.device)
            Dg[2, 2] = torch.sign(torch.det(Vh.T @ U.T))
            Rm = Vh.T @ Dg @ U.T
            res = ((x - c) - q0 @ Rm.T).norm(dim=-1)
            b = (x - c) @ Rm                                                        # 몸 좌표로 되돌림
            out.append(f"{i + 1}:{100 * float(res.pow(2).mean().sqrt()) / E:.2f}%/{float(b[:, 2].max() - b[:, 2].min()):.3f}")
        print(f"  {k} (프레임:잔차 RMS/EXT / 몸 좌표 높이) " + " ".join(out))
