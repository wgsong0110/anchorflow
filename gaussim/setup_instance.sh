#!/usr/bin/env bash
# GausSim 을 이 프로젝트 이미지(torch 2.5.1 / py3.11) 위에서 돌릴 수 있게 만든다.
#
# 공개 저장소가 torch 1.13 / py3.9 를 전제로 하고 여러 조각이 빠져 있어서,
# 매번 손으로 메우다 인스턴스가 죽으면 처음부터 다시 하게 된다. 한 번에 돌도록 모았다.
# 멱등하다 -- 다시 돌려도 안전하다.
#
# 메우는 것들:
#   - mmcv 1.x: CUDA 확장 휠이 없다 -> MMCV_WITH_OPS=0 (컴파일 안 함)
#   - mmcv.ops.QueryAndGroup/grouping_operation -> 순수 torch 대체 (mmcv_ops_fallback.py)
#   - dgl: PyPI 휠은 CPU 전용 -> CUDA 휠 + graphbolt 로딩 우회(torch 2.5 용 .so 없음)
#   - torch._six (torch 2.x 에서 삭제)
#   - PointCloudViewer: 저장소에 없는 클래스인데 임포트만 되어 있다
#   - blender 카메라 경로가 Camera(static_img_path=...) 를 안 넘긴다
set -uo pipefail
W=${W:-/workspace}
GS=$W/GausSim_ICCV2025
AF=$W/anchorflow

echo "[1/7] 저장소"
[ -d "$GS" ] || git clone -q --depth 1 https://github.com/ftbabi/GausSim_ICCV2025 "$GS"
[ -d "$AF" ] || git clone -q --depth 1 https://github.com/wgsong0110/anchorflow "$AF"
(cd "$AF" && git pull -q 2>/dev/null || true)

echo "[2/7] 파이썬 패키지 (컴파일 없음)"
pip install -q "warp-lang==0.10.1" h5py PyMCubes pymeshlab plyfile kornia 2>&1 | tail -1
MMCV_WITH_OPS=0 pip install -q "mmcv==1.7.2" 2>&1 | tail -1
python - <<'PY'
import importlib.util as u
print("mmcv", "OK" if u.find_spec("mmcv") else "실패")
PY

echo "[3/7] dgl CUDA 휠"
# dgl 2.1 은 graphbolt 를 통해 torchdata.datapipes 를 요구한다. torchdata 0.11+ 에서
# datapipes 가 빠졌으므로 그 API 가 남아 있는 마지막 버전을 고정한다.
pip install -q "torchdata==0.7.1" 2>&1 | tail -1
python -c "import dgl" 2>/dev/null || \
  pip install -q dgl -f https://data.dgl.ai/wheels/cu121/repo.html 2>&1 | tail -1
python - <<'PY'
# 이 휠은 torch 2.2.1 까지의 graphbolt .so 만 담고 있다. GausSim 은 graphbolt 를
# 안 쓰므로 로딩만 건너뛴다. 실제로 호출하면 그때 터지게 둔다.
import dgl, os
p = os.path.join(os.path.dirname(dgl.__file__), "graphbolt", "__init__.py")
s = open(p).read()
if "GRAPHBOLT_SKIP" not in s:
    s = s.replace("def load_graphbolt():", """def load_graphbolt():
    import warnings as _w
    _w.warn("graphbolt 로딩 건너뜀 (torch 2.5 용 .so 없음)")
    return
    # GRAPHBOLT_SKIP""", 1)
    open(p, "w").write(s)
    print("graphbolt 우회 적용")
PY
python -c "import dgl,torch; g=dgl.graph(([0,1],[1,2])).to('cuda'); print('dgl CUDA', dgl.__version__, g.device)"

echo "[4/7] mmgs 설치"
(cd "$GS" && pip install -q -e . 2>&1 | tail -1)

echo "[5/7] 소스 패치"
cp "$AF/gaussim/mmcv_ops_fallback.py" "$GS/mmgs/utils/"
for f in mmgs/models/backbones/meshgraphnet_hie.py mmgs/models/heads/acc_decoder.py mmgs/models/utils/dgl_graph.py; do
  sed -i "s|^from mmcv.ops import QueryAndGroup, grouping_operation|from mmgs.utils.mmcv_ops_fallback import QueryAndGroup, grouping_operation  # mmcv._ext 없이|" "$GS/$f"
done
grep -rl "torch._six" "$GS" --include=*.py | while read f; do
  sed -i "s/from torch._six import inf/from torch import inf  # torch 2.x 에서 torch._six 가 사라졌다/" "$f"
done
sed -i "s|^from mmgs.utils import PointCloudViewer|# from mmgs.utils import PointCloudViewer  # 저장소에 없는 클래스, 쓰이지도 않는다|" \
  "$GS/mmgs/models/simulators/gs_simulator_hierarchy.py"
python - "$GS" <<'PY'
import sys, os
p = os.path.join(sys.argv[1], "mmgs/datasets/multiview_video_dataset.py")
s = open(p).read()
old = """                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    img_hw=img_hw,"""
new = """                    FoVy=FovY,
                    FoVx=FovX,
                    img_path=img_path,
                    # blender 경로는 이 인자를 안 넘긴다 -- colmap 경로만 시험된 흔적.
                    static_img_path=img_path,
                    img_hw=img_hw,"""
if old in s:
    open(p, "w").write(s.replace(old, new, 1))
    print("static_img_path 주입")
else:
    print("static_img_path 이미 적용")
PY
cp "$AF"/gaussim/*.py "$GS/tools/" 2>/dev/null

echo "[6/7] 임포트 확인"
cd "$GS" && PYTHONPATH="$GS:$W/PhysGaussian/gaussian-splatting" python - <<'PY'
from mmgs.models import build_simulator
from mmgs.datasets import build_dataset
print("mmgs 임포트 OK")
PY

echo "[7/7] 완료"
