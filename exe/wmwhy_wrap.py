"""절 CM97 묶음 wmwhy 감싸개 (진단, 결과 불변 기록만; 기본 판 안 바꿈): 수박 첫 충돌 f3~f8 원인 가르기 판들을 옛 실행기 (exe/s6/fem_ao_rt_sim.py) 에 바로 넘김.
--wsh_sv 프레임마다 입자 탄성 F 특이값·det (exe/wmK_wrap.py --wmksv 글자 그대로) 와 입자 속도 (원래 순서, f32) 를 <out>/wsh/fs_<t>.npz·vs_<t>.npz 에 저장.
감싸개 사슬 없음 (H 사슬의 W̄·R2·A2·P1X·쪼개기·예산·가속 모두 안 씀) -- 판 사이 차이는 판 목록 (exe/wmwhy_spec.txt) 의 실행기 플래그 하나씩.

사용: python exe/wmwhy_wrap.py --wyv 1 exe/s6/fem_ao_rt_sim.py <실행기 인자...>   (--wyv 0 이면 아무것도 안 바꿈)
"""
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
C = dict(v="0")
while argv and argv[0].startswith("--wy"):
    C[argv[0][4:]] = argv[1]                                                           # --wyv -> v
    del argv[:2]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)


def err(where, e_):
    import traceback
    with open(os.path.join(out, "wy_err.txt"), "a") as f:
        f.write(f"[{where}] {e_!r}\n{traceback.format_exc()}\n")


if int(float(C["v"])):
    import numpy as np
    import torch
    from anchorflow import fem_wmshape as WSHM
    o_frame = WSHM.Rec.frame

    def _det3(F):
        return (F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1]) - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
                + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]))

    def _svd3(F):
        """특이값 (큰 것부터) = √(FᵀF 고윳값), 대칭 3x3 고윳값 닫힌 식 (exe/wmK_wrap.py 와 같은 글자)."""
        A = F.transpose(1, 2) @ F
        p1 = A[:, 0, 1] ** 2 + A[:, 0, 2] ** 2 + A[:, 1, 2] ** 2
        q = (A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]) / 3
        p2 = (A[:, 0, 0] - q) ** 2 + (A[:, 1, 1] - q) ** 2 + (A[:, 2, 2] - q) ** 2 + 2 * p1
        p = torch.sqrt(p2 / 6)
        I3 = torch.eye(3, device=F.device, dtype=F.dtype)
        ps = torch.where(p > 0, p, torch.ones_like(p))
        B = (A - q[:, None, None] * I3) / ps[:, None, None]
        r = torch.clamp(_det3(B) / 2, -1.0, 1.0)
        phi = torch.acos(r) / 3
        e1 = q + 2 * p * torch.cos(phi)
        e3 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
        e2 = 3 * q - e1 - e3
        e = torch.stack([e1, e2, e3], 1)
        e = torch.where((p > 0)[:, None], e, q[:, None].expand(-1, 3))
        return torch.sqrt(torch.clamp(e, min=0.0))

    def frame(s, fi):
        r = o_frame(s, fi)
        t = fi + 1
        try:
            if t in s.sv and int(s.S.IP[13]) == 0:
                S = s.S
                Fo = s._orig(S.F.reshape(S.F.shape[0], -1)).reshape(-1, 3, 3)
                sv = torch.empty(Fo.shape[0], 3, device=Fo.device, dtype=torch.float32)
                dt_ = torch.empty(Fo.shape[0], device=Fo.device, dtype=torch.float32)
                for i in range(0, Fo.shape[0], 1 << 20):
                    Fi = Fo[i:i + (1 << 20)].double()
                    ok = torch.isfinite(Fi).flatten(1).all(1)
                    Fz = torch.where(ok[:, None, None], Fi, torch.eye(3, device=Fi.device, dtype=Fi.dtype).expand_as(Fi))
                    sv[i:i + Fi.shape[0]] = torch.where(ok[:, None], _svd3(Fz), torch.full_like(Fz[:, 0], float("nan"))).float()
                    dt_[i:i + Fi.shape[0]] = torch.where(ok, _det3(Fz), torch.full_like(Fz[:, 0, 0], float("nan"))).float()
                np.savez_compressed(os.path.join(s.out, "wsh", f"fs_{t:03d}.npz"), sv=sv.cpu().numpy(), det=dt_.cpu().numpy())
                vo = s._orig(S.v.reshape(S.v.shape[0], -1)).reshape(-1, 3)
                np.savez_compressed(os.path.join(s.out, "wsh", f"vs_{t:03d}.npz"), v=vo.float().cpu().numpy())
        except Exception as e_:
            err(f"wy f{t}", e_)
        return r
    WSHM.Rec.frame = frame
    print("[절 CM97] --wyv: wsh_sv 프레임에 입자 탄성 F 특이값·det·속도 저장 (결과 불변)", flush=True)

sys.argv = [os.path.join(HERE, argv[0]) if not os.path.isabs(argv[0]) and not os.path.exists(argv[0]) else argv[0]] + argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
