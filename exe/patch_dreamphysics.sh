#!/usr/bin/env bash
# Trim the DreamPhysics clone for anchorflow's GNN (no-MPM) path on the instance.
#  - drop `import tinycudann as tcnn` from utils/threestudio_utils.py: SVDGuidance
#    only uses parse_version/cleanup/get_device/C (none need tcnn) -> no
#    tiny-cuda-nn build required.
# The training path also avoids utils.decode_param (warp/MPM) by construction.
set -euo pipefail
DP=${1:?usage: patch_dreamphysics.sh /path/to/DreamPhysics}
sed -i '/import tinycudann as tcnn/d' "$DP/utils/threestudio_utils.py"
# diffusers>=0.28 renamed the SVD pipeline's image_processor -> video_processor
# (svd_guidance only uses pil_to_numpy/numpy_to_pt, inherited by VideoProcessor).
sed -i 's/self\.pipe\.image_processor/self.pipe.video_processor/' \
    "$DP/video_distillation/svd_guidance.py"
echo "patched $DP (removed tinycudann import; image_processor->video_processor)"
