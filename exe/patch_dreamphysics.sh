#!/usr/bin/env bash
# Trim the DreamPhysics clone for anchorflow's GNN (no-MPM) path on the instance.
#  - drop `import tinycudann as tcnn` from utils/threestudio_utils.py: SVDGuidance
#    only uses parse_version/cleanup/get_device/C (none need tcnn) -> no
#    tiny-cuda-nn build required.
# The training path also avoids utils.decode_param (warp/MPM) by construction.
set -euo pipefail
DP=${1:?usage: patch_dreamphysics.sh /path/to/DreamPhysics}
sed -i '/import tinycudann as tcnn/d' "$DP/utils/threestudio_utils.py"
echo "patched $DP (removed tinycudann import)"
