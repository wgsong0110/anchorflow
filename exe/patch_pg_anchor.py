"""MPM 입자를 **새로 샘플링한 것만** 쓰고, 가우시안은 그 입자를 따라가게 한다.

PhysGaussian 원본은 가우시안 하나하나가 곧 MPM 입자라, 껍데기 가우시안이 많은 씬은
격자당 입자 수가 통제되지 않는다 (ficus 24.9, pillow2sofa 72.2 vs wolf 9.09).
여기서는 닫힌 부피를 포아송 디스크로 균일 샘플링한 입자셋만 시뮬레이션하고,
가우시안은 초기에 가장 가까운 입자에 묶어 두었다가 그 입자의 변형을 따라간다:

    x_g = x_p + F_p r_g,   Sigma_g = F_p Sigma_g0 F_p^T,   SH 은 polar(F_p) 로 회전

격자당 입자 수는 간격 하나로 정해진다: ppc = (dx / spacing)^3.
AF_PPC 를 주면 dx 에서 역산해 간격을 자동으로 잡는다.

  python exe/patch_pg_anchor.py --pg <PG>
환경변수: AF_ANCHOR=1, AF_PPC(기본 9), AF_FILL_GRID, AF_FILL_CLOSE, AF_FILL_CACHE
"""
from __future__ import annotations

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pg", required=True)
a = ap.parse_args()

p = os.path.join(a.pg, "gs_simulation.py")
s = open(p).read()
if "AF_ANCHOR" in s:
    print("이미 패치됨")
    raise SystemExit(0)

# ---------------------------------------------------------------- 1) 입자셋 교체
# _af_os 가 없으면(psfill 패치를 안 걸었으면) 여기서 만든다
if "import os as _af_os" not in s:
    s = s.replace("from particle_filling.filling import *",
                  "from particle_filling.filling import *\nimport os as _af_os", 1)

# 앵커 모드에서는 PG 의 채움을 통째로 건너뛴다
A0 = '    filling_params = preprocessing_params["particle_filling"]'
B0 = A0 + "\n    if _af_os.environ.get(\"AF_ANCHOR\"):\n        filling_params = None"
assert A0 in s
s = s.replace(A0, B0, 1)

A1 = "    mpm_solver = MPM_Simulator_WARP(10)"
B1 = '''    if _af_os.environ.get("AF_ANCHOR"):
        import numpy as _an_np
        from scipy.spatial import cKDTree as _an_KD
        from anchorflow.psfill import poisson_fill as _an_pf

        _an_dx = material_params["grid_lim"] / material_params["n_grid"]
        _an_ppc = float(_af_os.environ.get("AF_PPC", 9.0))
        _an_sp = _an_dx / (_an_ppc ** (1.0 / 3.0))
        _an_S = transformed_pos.detach().cpu().numpy().astype("float64")
        _an_ck = _af_os.environ.get("AF_FILL_CACHE")
        if _an_ck and _af_os.path.exists(_an_ck):
            _an_L = _an_np.load(_an_ck)
            print(f"[앵커] 캐시에서 입자 {_an_L.shape[0]} 개", flush=True)
        else:
            _an_L = _an_pf(_an_S, spacing=_an_sp,
                           grid=int(_af_os.environ.get("AF_FILL_GRID", 160)),
                           close=float(_af_os.environ.get("AF_FILL_CLOSE", 0.03)),
                           surf=0.0)
            if _an_ck:
                _an_np.save(_an_ck, _an_L)
        print(f"[앵커] dx {_an_dx:.4f}, 목표 ppc {_an_ppc}, 간격 {_an_sp:.5f}, "
              f"입자 {_an_L.shape[0]} 개 (가우시안 {_an_S.shape[0]} 개는 시뮬 안 함)",
              flush=True)
        # 가우시안을 최근접 입자에 묶는다
        _an_d, _an_nb = _an_KD(_an_L).query(_an_S, k=1)
        _af_nb = torch.from_numpy(_an_nb).long().to(device)
        _af_off = (transformed_pos - torch.from_numpy(_an_L[_an_nb]).float().to(device))
        _af_cov0 = init_cov.to(device).view(-1, 6)
        _af_S0 = torch.zeros((_af_cov0.shape[0], 3, 3), device=device)
        _af_S0[:, 0, 0] = _af_cov0[:, 0]
        _af_S0[:, 0, 1] = _af_S0[:, 1, 0] = _af_cov0[:, 1]
        _af_S0[:, 0, 2] = _af_S0[:, 2, 0] = _af_cov0[:, 2]
        _af_S0[:, 1, 1] = _af_cov0[:, 3]
        _af_S0[:, 1, 2] = _af_S0[:, 2, 1] = _af_cov0[:, 4]
        _af_S0[:, 2, 2] = _af_cov0[:, 5]
        mpm_init_pos = torch.from_numpy(_an_L).float().to(device)
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        shs = init_shs
        opacity = init_opacity
        gs_num = mpm_init_pos.shape[0]
        mpm_init_vol = get_particle_volume(
            mpm_init_pos, material_params["n_grid"],
            material_params["grid_lim"] / material_params["n_grid"],
            unifrom=True).to(device=device)

''' + A1
assert A1 in s
s = s.replace(A1, B1, 1)

# ---------------------------------------------------------------- 2) 렌더
A2 = '''        if args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)'''
B2 = '''        if args.render_img and _af_os.environ.get("AF_ANCHOR"):
            import warp as _an_wp
            _xp = mpm_solver.export_particle_x_to_torch()
            _Fp = _an_wp.to_torch(mpm_solver.mpm_state.particle_F).view(-1, 3, 3)
            _F = _Fp[_af_nb]
            pos = _xp[_af_nb] + torch.einsum("nij,nj->ni", _F, _af_off)
            _Sg = _F @ _af_S0 @ _F.transpose(1, 2)
            cov3D = torch.stack([_Sg[:, 0, 0], _Sg[:, 0, 1], _Sg[:, 0, 2],
                                 _Sg[:, 1, 1], _Sg[:, 1, 2], _Sg[:, 2, 2]], -1)
            # polar(F) 를 그람-슈미트로 (SVD 는 이 크기에서 너무 느리다)
            _c0 = _F[:, :, 0]
            _c0 = _c0 / _c0.norm(dim=1, keepdim=True).clamp(min=1e-9)
            _c1 = _F[:, :, 1] - (_F[:, :, 1] * _c0).sum(1, keepdim=True) * _c0
            _c1 = _c1 / _c1.norm(dim=1, keepdim=True).clamp(min=1e-9)
            _c2 = torch.cross(_c0, _c1, dim=1)
            rot = torch.stack([_c0, _c1, _c2], -1)
        elif args.render_img:
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)'''
assert A2 in s
s = s.replace(A2, B2, 1)
open(p, "w").write(s)
print(f"패치 완료: {p}")
