#!/usr/bin/env bash
# One-shot instance setup for the SC-GS baseline + AnchorFlow (rotation-head
# GNN+SSM dynamics) workflow -- exe/scgs_baseline_run.sh and
# exe/train_sc_anchorflow.py. Run once per fresh instance (wired into
# .wexec.json's "setup"). Idempotent: safe to re-run.
#
# What this replaces (all previously done by hand, see 2026-07-26 session):
#   - lib/lbs CUDA .so was never synced/downloaded -> both LBS kernels
#     silently fell back to pure torch with no warning.
#   - ficus_ds_wind dataset was scp'd in by hand from the local machine.
#   - SC-GS repo clone was already automated (kept as-is below).
set -uo pipefail
WS="${WS:-/workspace}"
AF="$WS/anchorflow"

echo "[setup] SC-GS repo"
git clone --depth 1 https://github.com/yihua7/SC-GS "$WS/SC-GS" 2>/dev/null || true

echo "[setup] SC-GS idx=None bug patch (samp_hyper node-downsample step)"
# train_gui.py's samp_hyper strategy (the default deform_downsamp_strategy)
# calls deform.deform.init(...) at iteration_node_sampling and uses its
# return value as a fancy index: tensor[dynamic_mask][idx]. But init()
# returns idx=None whenever it takes its "keep_all" branch (node_num >
# init_pcl.shape[0], i.e. fewer Gaussians survived pruning by that point
# than the configured node_num -- hit on ficus_ds_wind, 2026-07-28: only 490
# points survived vs node_num=512). Indexing with a bare None is NOT a
# no-op in torch -- it inserts a new axis -- so features_dc/features_rest
# end up wrong-rank and get_features() crashes with a dim-0 mismatch a few
# lines later. Patch: treat idx=None as slice(None) (the actual keep-all
# no-op). Idempotent (skips if already patched, e.g. re-run on same instance).
python3 - <<'PYEOF'
path = "/workspace/SC-GS/train_gui.py"
with open(path) as f:
    src = f.read()
needle = "idx = self.deform.deform.init(init_pcl=original_gaussians.get_xyz[dynamic_mask], hyper_pcl=hyper_pcl[dynamic_mask], force_init=True, opt=self.opt, reset_bbox=False, feature=self.gaussians.feature)"
if "if idx is None:" in src:
    print("  already patched, skipping")
elif src.count(needle) == 1:
    replacement = needle + "\n                    if idx is None:\n                        idx = slice(None)  # keep_all path: None is not a no-op index in torch"
    src = src.replace(needle, replacement)
    with open(path, "w") as f:
        f.write(src)
    print("  patched OK")
else:
    print(f"  WARNING: expected 1 match for the idx=None patch target, found {src.count(needle)} -- SC-GS source may have changed upstream, patch NOT applied")
PYEOF

echo "[setup] anchorflow repo (self-heal if the instance wasn't pre-cloned)"
if [ ! -d "$AF/.git" ]; then
    git clone --depth 1 https://github.com/wgsong0110/anchorflow "$AF"
fi

echo "[setup] lib/lbs fused CUDA kernel (.so from the cuda-lbs release)"
LBS="$AF/lib/lbs"
mkdir -p "$LBS"
curl -sL "https://api.github.com/repos/wgsong0110/anchorflow/releases/tags/cuda-lbs" \
  | grep -oE '"browser_download_url":[^,]*\.so"' | grep -oE 'https[^"]+' \
  | while read -r u; do curl -sL "$u" -o "$LBS/$(basename "$u")"; done
PYTHONPATH="$AF/lib" python3 -c "
import lbs
print('  lbs._HAVE_CUDA          =', lbs._HAVE_CUDA)
print('  lbs._HAVE_COV_CUDA      =', lbs._HAVE_COV_CUDA)
print('  lbs._HAVE_ROT_BATCH_CUDA=', lbs._HAVE_ROT_BATCH_CUDA)
" || echo "  WARNING: lib/lbs import failed -- training will fall back to (slower) pure torch"

echo "[setup] rclone binary (baked into the ghcr.io/wgsong0110/anchorflow image
       itself -- not installed here. Previously this was apt-installed at
       instance startup and gated behind R2_ACCESS_KEY being set, so any
       instance created without that env var silently never got rclone at
       all, and every checkpoint backup failed for the whole run. Fallback
       apt-get kept below only for instances still on an older image.)"
command -v rclone >/dev/null 2>&1 || \
    (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rclone >/dev/null 2>&1)

echo "[setup] R2 credentials (from env, never committed)"
if [ -n "${R2_ACCESS_KEY:-}" ] && [ ! -f ~/.config/rclone/rclone.conf ]; then
    mkdir -p ~/.config/rclone
    printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\n' \
        "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
fi
if command -v rclone >/dev/null 2>&1; then
    echo "  rclone binary: OK"
else
    echo "  rclone binary: MISSING (apt install failed) -- R2 backups will NOT work"
fi
if [ -f ~/.config/rclone/rclone.conf ]; then
    echo "  R2 credentials: configured"
else
    echo "  R2 credentials: MISSING -- R2_ACCESS_KEY/R2_SECRET/R2_ENDPOINT were not"
    echo "    passed at instance creation. Checkpoint backups during training WILL"
    echo "    silently fail (loudly now -- see train_sc_anchorflow.py's WARNING on"
    echo "    rclone non-zero exit) until this is fixed, e.g.:"
    echo "    mkdir -p ~/.config/rclone && printf '[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\n' \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > ~/.config/rclone/rclone.conf"
fi

echo "[setup] ficus_ds_wind dataset (R2: r2:storage/datasets/anchorflow/ficus_ds_wind)"
if [ -d "$WS/ficus_ds_wind" ] && [ -n "$(ls -A "$WS/ficus_ds_wind" 2>/dev/null)" ]; then
    echo "  already present, skipping"
elif command -v rclone >/dev/null 2>&1 && [ -f ~/.config/rclone/rclone.conf ]; then
    rclone copy r2:storage/datasets/anchorflow/ficus_ds_wind "$WS/ficus_ds_wind" --progress
else
    echo "  SKIPPED: rclone not configured on this instance."
    echo "  Either pass R2_ACCESS_KEY/R2_SECRET/R2_ENDPOINT at instance creation"
    echo "  (same convention as exe/instance_setup.sh), or copy the dataset by hand:"
    echo "    scp -r -P <port> ~/r2/datasets/anchorflow/ficus_ds_wind root@<host>:$WS/ficus_ds_wind"
fi

echo "[setup] done."
