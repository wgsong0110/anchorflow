"""절 CM186: GF 저장소의 3DGS (Inria 2023 판) 학습 코드를 우리 래스터라이저 wheel (원본 3DGS API: 색·반경 2 개 반환) 에 맞춘다 -- 호환 고침만, 학습 알고리즘·하이퍼파라미터 불변.
 (1) scene/dataset_readers.py: PIL 에 np.byte(int8) -> np.uint8 (같은 바이트, 새 PIL 이 int8 을 거부)
 (2) gaussian_renderer/__init__.py: GF 판 래스터라이저는 point_xy 를 세 번째로 돌려주나 학습에 안 쓰임 -> 2 개 반환이면 None
  python patch_3dgs_train.py <gaussian-splatting 폴더>"""
import os, sys
g = sys.argv[1]
p = os.path.join(g, "scene", "dataset_readers.py"); s = open(p).read()
s = s.replace('dtype=np.byte), "RGB"', 'dtype=np.uint8), "RGB"'); open(p, "w").write(s)
p = os.path.join(g, "gaussian_renderer", "__init__.py"); s = open(p).read()
old = "    rendered_image, radii, point_xy = rasterizer("
if old in s:
    s = s.replace(old, "    _af_out = rasterizer(", 1)
    s = s.replace("        cov3D_precomp = cov3D_precomp)\n\n    # Those Gaussians",
                  "        cov3D_precomp = cov3D_precomp)\n    rendered_image, radii = _af_out[0], _af_out[1]; point_xy = _af_out[2] if len(_af_out) > 2 else None\n\n    # Those Gaussians", 1)
    s = s.replace("'point_xy' : point_xy.detach()}", "'point_xy' : point_xy.detach() if point_xy is not None else None}", 1)
    assert "_af_out[0]" in s
open(p, "w").write(s); print("[3DGS 호환 고침] 완료")
