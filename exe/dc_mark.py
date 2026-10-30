"""발산 셀 입자 표시용 도우미 (사용자 10-30): lb_render.py·wm_render.py 의 --mark (색칠만, 시뮬 계산과 무관).

  - sh_rgb: 3DGS 래스터라이저 computeColorFromSH 와 같은 식 (보는 방향 = 위치 − 카메라 위치, 결과 + 0.5, 0 아래 자름) -- SH 를 colors_precomp 로 바꿔도 같은 색
  - load_mark: 표시 입자 번호 -> (가우시안 쪽 번호 < NG, 채움 입자 번호 ≥ NG)
  - extra_cov: 채움 입자 (가우시안 없음) 를 그릴 등방 가우시안 공분산 F (s² I) Fᵀ
  - project_px: 화면 픽셀 위치 (3DGS ndc2Pix 와 같은 식) -> 확대 칸 중심용
"""
import numpy as np
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
RED = (1.0, 0.0, 1.0)                                                              # 자홍 (수박 속살 빨강·lego 빨간 부품과 구별)


def sh_rgb(shs, pos, campos):
    """shs [N,16,3] (GaussianModel.get_features 순서), pos [N,3], campos [3] -> rgb [N,3]."""
    d = pos - campos[None]
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    s = shs
    r = C0 * s[:, 0]
    r = r - C1 * y * s[:, 1] + C1 * z * s[:, 2] - C1 * x * s[:, 3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    r = (r + C2[0] * xy * s[:, 4] + C2[1] * yz * s[:, 5] + C2[2] * (2.0 * zz - xx - yy) * s[:, 6] + C2[3] * xz * s[:, 7] + C2[4] * (xx - yy) * s[:, 8])
    r = (r + C3[0] * y * (3 * xx - yy) * s[:, 9] + C3[1] * xy * z * s[:, 10] + C3[2] * y * (4 * zz - xx - yy) * s[:, 11]
         + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * s[:, 12] + C3[4] * x * (4 * zz - xx - yy) * s[:, 13] + C3[5] * z * (xx - yy) * s[:, 14]
         + C3[6] * x * (xx - 3 * yy) * s[:, 15])
    return (r + 0.5).clamp_min(0.0)


def load_mark(path, ng, dev):
    idx = torch.as_tensor(np.load(path).astype(np.int64), device=dev)
    return idx[idx < ng], idx[idx >= ng]


def extra_cov(F, s):
    """F [K,3,3], s 축 길이 -> 6 성분 공분산."""
    cov = (s * s) * (F @ F.transpose(1, 2))
    return torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1), cov


def project_px(pos, full_proj, W, H):
    """pos [K,3] -> 픽셀 [K,2] (full_proj = 행 벡터 규약 [4,4], 3DGS 와 같다)."""
    ph = torch.cat([pos, torch.ones_like(pos[:, :1])], 1) @ full_proj
    nd = ph[:, :3] / (ph[:, 3:4] + 1e-7)
    return torch.stack([((nd[:, 0] + 1.0) * W - 1.0) * 0.5, ((nd[:, 1] + 1.0) * H - 1.0) * 0.5], 1)
