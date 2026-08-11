#!/usr/bin/env bash
# Everything an instance needs to run the anchor simulator against PhysGaussian's
# MPM: the solver, the trained ficus, and the CUDA step kernel.
#
# This existed only as a sequence of commands typed into one instance, which
# meant a second instance for a side-by-side run had to be rebuilt from memory
# and the two were not guaranteed to match. Idempotent: safe to re-run, and
# skips anything already in place.
#
#   bash exe/mpm_setup.sh
#
# Needs R2 credentials in the environment (R2_ACCESS_KEY / R2_SECRET /
# R2_ENDPOINT) or an existing rclone config, and gh authenticated for the CUDA
# release download.
set -uo pipefail
WS="${WS:-/workspace}"
AF="$WS/anchorflow"
DP_COMMIT="${DP_COMMIT:-45025ab}"
fail=0

echo "[setup] anchorflow"
if [ -d "$AF/.git" ]; then
    git -C "$AF" pull -q && echo "  pulled $(git -C "$AF" rev-parse --short HEAD)"
else
    git clone -q https://github.com/wgsong0110/anchorflow "$AF" && echo "  cloned"
fi

echo "[setup] rclone"
if [ -n "${R2_ACCESS_KEY:-}" ] && [ ! -f ~/.config/rclone/rclone.conf ]; then
    mkdir -p ~/.config/rclone
    printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\n' \
        "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
fi
if rclone lsd r2:storage >/dev/null 2>&1; then
    echo "  OK"
else
    echo "  BROKEN -- the trained ficus and the checkpoint backups both need this"
    fail=1
fi

echo "[setup] PhysGaussian MPM (DreamPhysics @ $DP_COMMIT)"
# pinned: the solver's parameter API changed across versions, and this project's
# region-varying material only reaches it through set_parameters_dict here
if [ ! -d "$WS/DreamPhysics/.git" ]; then
    git clone -q https://github.com/tyhuang0428/DreamPhysics "$WS/DreamPhysics"
fi
git -C "$WS/DreamPhysics" fetch -q origin 2>/dev/null
git -C "$WS/DreamPhysics" checkout -q "$DP_COMMIT" 2>/dev/null
echo "  at $(git -C "$WS/DreamPhysics" rev-parse --short HEAD)"

echo "[setup] SC-GS (scene.gaussian_model reads the PLY)"
[ -d "$WS/SC-GS/.git" ] || git clone -q --depth 1 https://github.com/yihua7/SC-GS "$WS/SC-GS"
echo "  OK"

echo "[setup] python packages"
# warp is pinned: 1.0.2 is what the solver's kernels are written against.
# Nothing here compiles -- these are wheels
python3 -c "import warp" 2>/dev/null || pip install -q warp-lang==1.0.2
python3 -c "import h5py" 2>/dev/null || pip install -q h5py
python3 -c "import warp, h5py; print(f'  warp {warp.__version__}, h5py {h5py.__version__}')" \
    || { echo "  MISSING"; fail=1; }

echo "[setup] trained ficus"
PLY="$WS/ficus_whitebg-trained/point_cloud/iteration_60000/point_cloud.ply"
if [ -s "$PLY" ]; then
    echo "  already here"
else
    rclone copy r2:storage/datasets/anchorflow/ficus_whitebg-trained \
        "$WS/ficus_whitebg-trained" 2>&1 | tail -1
fi
[ -s "$PLY" ] && echo "  $(stat -c%s "$PLY") bytes" || { echo "  MISSING"; fail=1; }

echo "[setup] CUDA step kernel"
# The release tag carries a hash of the .cu sources, so the binary is matched to
# the code that is checked out rather than to whatever was built most recently.
# Mirrors wbuild's hashing so an instance does not need wbuild itself.
python3 - "$AF" <<'PYEOF'
import json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
prefix = json.load(open(root / ".wexec.json")).get("cuda_build", {}).get("release_tag", "cuda-build")
repo = "wgsong0110/anchorflow"
for setup_py in sorted(root.rglob("setup.py")):
    if ".git" in setup_py.parts:
        continue
    d = setup_py.parent
    files = ["./" + str(f.relative_to(root)) for f in sorted(d.rglob("*.cu"))] \
        + ["./" + str(f.relative_to(root)) for f in sorted(d.rglob("*.cuh"))]
    if not files:
        continue
    listing = subprocess.check_output(["sha256sum"] + files, cwd=root, text=True)
    h = subprocess.check_output(["sha256sum"], input=listing.encode()).decode()[:7]
    tag = f"{prefix}-{d.name}-{h}"
    so = list(d.glob("*.so"))
    if so:
        print(f"  {d.name}: {so[0].name} already present")
        continue
    r = subprocess.run(["gh", "release", "download", tag, "--repo", repo,
                        "--pattern", f"{d.name}.tar.gz", "--dir", "/tmp"],
                       capture_output=True, text=True)
    if r.returncode or not os.path.exists(f"/tmp/{d.name}.tar.gz"):
        print(f"  {d.name}: release {tag} NOT FOUND -- trigger a build (wbuild trigger)")
        sys.exit(3)
    subprocess.run(["tar", "xzf", f"/tmp/{d.name}.tar.gz", "-C", str(root)], check=True)
    os.unlink(f"/tmp/{d.name}.tar.gz")
    print(f"  {d.name}: {tag}")
PYEOF
[ $? -ne 0 ] && fail=1

echo
echo "[check] importing everything a run touches"
cd "$WS/SC-GS" && PYTHONPATH="$AF/lib:$WS/SC-GS:$WS/DreamPhysics" python3 - <<'PYEOF'
import sys
try:
    import warp; warp.init()
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from scene.gaussian_model import GaussianModel
    from anchorstep import fused_energy_force
    from anchorflow import scene_setup
    from anchorflow.mpm_teacher import MPMTeacher
    print("  OK -- MPM solver, SC-GS loader, CUDA kernel, anchorflow")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
[ $? -ne 0 ] && fail=1

echo
[ $fail -eq 0 ] && echo "[setup] ready" || { echo "[setup] INCOMPLETE -- see above"; exit 1; }
