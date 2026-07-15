#!/usr/bin/env python
"""v3 front-end: single image -> multi-view video (SV4D 2.0) -> SC-GS-ready dataset.

Pipeline (all GPU steps; NOT runnable on the arm64 host):

    subject.png
      │  (1) SVD img2vid  — animate the still into a monocular driving video
      ▼      (reuses the exe/gen_video.py recipe; skipped if --video given)
    monocular video  (T frames, object-centered, 576x576)
      │  (2) SV4D 2.0  — Stability-AI/generative-models, self-contained
      ▼      scripts/sampling/simple_video_sample_4d2.py
    V synchronized novel-view videos  (V views x T frames, KNOWN azimuths)
      │      run ONCE PER ELEVATION (each --elevations_deg=E renders the variant's
      │      fixed azimuth set at that elevation) -> az x elev VIEW GRID
      │  (3) pose + pack  (per-view c2w from BOTH azimuth AND elevation)
      ▼
    dataset/  — per-view frame folders  (N_views = n_azimuths x n_elevations)
              + transforms_train.json (D-NeRF/blender: c2w + per-frame time)
              + cameras.json          (per-view K + OpenCV/OpenGL extrinsics)
              + timestamps.json        (shared per-frame times)

SV4D 2.0 output geometry (verbatim from the repo configs / sampling script):

  variant "sv4d2"        : model_path=checkpoints/sv4d2.safetensors
                           T=12 native frames, V=4 novel views
                           azimuths_deg = [0, 60, 120, 180, 240]  (view 0 = input)
                           img_size = 576
  variant "sv4d2_8views" : model_path=checkpoints/sv4d2_8views.safetensors  (DEFAULT)
                           T=5 native frames, V=8 novel views
                           azimuths_deg = [0, 30, 75, 120, 165, 210, 255, 300, 330]
                           img_size = 576

  IMPORTANT (verified against simple_video_sample_4d2.py):
    * The sampler SAVES ONLY the novel views (view_indices = arange(V)+1); the
      input view (view 0, azimuth 0) is NOT written as an mp4. So the emitted
      videos, sorted _v001.._v00V, correspond to azimuths_deg[1:] (NOT [0:V]).
    * `elevations_deg` is a SINGLE elevation per run, broadcast to every view
      (input + all novel), so all V saved novel views render at absolute
      elevation E. We therefore run the sampler ONCE PER ELEVATION and merge.

  simple_video_sample_4d2.py default n_frames=21 (native T extended
  auto-regressively); output written per-view as:
      {output_folder}/{model_name}/{base_count:06d}_v{view:03d}.mp4
  Each elevation run uses a DISTINCT output subfolder so the per-view mp4s
  (identical {model_name}/{base_count}_v{view} names) never overwrite each other.

Usage:
    python exe/gen_mv_video.py --image subject.png --out /path/to/dataset
    python exe/gen_mv_video.py --image subject.png --out ds --variant sv4d2
    python exe/gen_mv_video.py --image subject.png --out ds --elevations=-20,0,20
    python exe/gen_mv_video.py --video mono.mp4    --out ds   # skip SVD step
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# SV4D 2.0 variant table — the KNOWN camera azimuths of the generated views.
# Numbers taken from Stability-AI/generative-models configs/sv4d2*.yaml and
# scripts/sampling/simple_video_sample_4d2.py. azimuths_deg[0] is the input view.
# --------------------------------------------------------------------------- #
SV4D2_VARIANTS = {
    "sv4d2": {
        "ckpt": "sv4d2.safetensors",
        "azimuths_deg": [0.0, 60.0, 120.0, 180.0, 240.0],
        "native_frames": 12,
    },
    "sv4d2_8views": {
        "ckpt": "sv4d2_8views.safetensors",
        "azimuths_deg": [0.0, 30.0, 75.0, 120.0, 165.0, 210.0, 255.0, 300.0, 330.0],
        "native_frames": 5,
    },
}

GENMODELS_DIR = os.environ.get("GENMODELS_DIR", "/opt/generative-models")
SV4D2_SCRIPT = "scripts/sampling/simple_video_sample_4d2.py"


def novel_azimuths(variant):
    """Absolute azimuths (deg) of the SAVED novel views, in emitted (_v001..) order.

    The sampler writes only view_indices 1..V (novel views); view 0 (the input,
    azimuth 0) is never saved. So the sorted per-view mp4s map to azimuths_deg[1:].
    (The old code used azimuths_deg[0:V], off by one vs the actual outputs.)
    """
    return [float(a) for a in SV4D2_VARIANTS[variant]["azimuths_deg"][1:]]


# --------------------------------------------------------------------------- #
# (1) single image -> monocular driving video (SVD img2vid), saved as frames
# --------------------------------------------------------------------------- #
def image_to_mono_frames(image_path, frames_dir, n_frames, size, seed):
    """Animate a still image into `n_frames` object-centered 576x576 PNGs.

    Same recipe as exe/gen_video.py (SVD-XT img2vid + cpu offload to fit 24GB),
    then center-square-crop/resize so SV4D sees a centered object at `size`x`size`.
    """
    import torch
    from PIL import Image
    from diffusers import StableVideoDiffusionPipeline

    os.makedirs(frames_dir, exist_ok=True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    image = Image.open(image_path).convert("RGB").resize((1024, 576))
    gen = torch.Generator("cuda").manual_seed(seed)
    frames = pipe(image, num_frames=n_frames, decode_chunk_size=8,
                  generator=gen).frames[0]

    for i, f in enumerate(frames):
        w, h = f.size
        s = min(w, h)
        f = f.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        f = f.resize((size, size), Image.LANCZOS)
        f.save(os.path.join(frames_dir, f"{i:05d}.png"))
    print(f"[mono] SVD wrote {len(frames)} frames -> {frames_dir}")
    return len(frames)


# --------------------------------------------------------------------------- #
# (2) monocular video -> SV4D 2.0 multi-view videos (per-view mp4s)
# --------------------------------------------------------------------------- #
def run_sv4d2(input_path, out_dir, variant, n_frames, num_steps, img_size,
              elevations_list, seed, remove_bg, low_vram):
    """Invoke the real SV4D 2.0 sampler (Fire CLI) as a subprocess.

    elevations_list: PER-VIEW absolute elevations, length n_views = 1 (input v0)
    + n_novel. SV4D encodes elevation RELATIVE to v0, so a genuine per-view list
    renders each novel view at its own elevation in ONE run (a scalar broadcast
    would collapse every relative elevation to 0 -> geometrically-false grid).

    Writes per-view mp4s to {out_dir}/{model_name}/{base_count:06d}_v{view:03d}.mp4
    """
    cfg = SV4D2_VARIANTS[variant]
    model_path = os.path.join("checkpoints", cfg["ckpt"])
    script = os.path.join(GENMODELS_DIR, SV4D2_SCRIPT)

    elev_arg = "[" + ",".join(str(float(e)) for e in elevations_list) + "]"
    cmd = [
        sys.executable, script,
        f"--input_path={input_path}",
        f"--model_path={model_path}",
        f"--output_folder={out_dir}",
        f"--n_frames={n_frames}",
        f"--num_steps={num_steps}",
        f"--img_size={img_size}",
        f"--elevations_deg={elev_arg}",
        f"--seed={seed}",
    ]
    if remove_bg:
        cmd.append("--remove_bg=True")
    if low_vram:
        cmd += ["--encoding_t=1", "--decoding_t=1"]
    # NOTE: azimuths_deg defaults to None -> the script uses the variant's fixed
    # azimuths (SV4D2_VARIANTS[variant]["azimuths_deg"]). We rely on that default
    # so the emitted poses (below) match the generated views exactly.

    print("[sv4d2] running:", " ".join(cmd))
    print("[sv4d2] cwd:", GENMODELS_DIR)
    subprocess.run(cmd, cwd=GENMODELS_DIR, check=True)

    model_name = os.path.splitext(cfg["ckpt"])[0]              # "sv4d2"
    view_mp4s = sorted(glob.glob(os.path.join(out_dir, model_name, "*_v*.mp4")))
    if not view_mp4s:
        raise RuntimeError(
            f"[sv4d2] no per-view mp4s under {out_dir}/{model_name} — "
            "verify the sampler ran and the output naming "
            "({base_count:06d}_v{view:03d}.mp4).")
    print(f"[sv4d2] {len(view_mp4s)} per-view videos:")
    for p in view_mp4s:
        print("        ", p)
    return view_mp4s


def decode_mp4(path):
    """Return list of HxWx3 uint8 frames from an mp4 (imageio/ffmpeg backend)."""
    import imageio.v3 as iio
    return list(iio.imread(path))                              # (T,H,W,3)


# --------------------------------------------------------------------------- #
# (3) camera geometry — orbital poses for the known azimuths / elevation
# --------------------------------------------------------------------------- #
def look_at_c2w(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """OpenGL/blender camera-to-world (camera looks down -z, +y up).

    This is the `transform_matrix` convention consumed by D-NeRF / SC-GS.
    """
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    z = eye - target                                          # backward (+z)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def orbital_pose(azimuth_deg, elevation_deg, radius):
    """Camera on a sphere of `radius`, looking at the origin.

    SV4D orbits a normalized, origin-centered object at fixed elevation while
    azimuth steps through the KNOWN per-view angles. The absolute radius/FOV are
    a rendering convention (not emitted by SV4D); what a 4D reconstructor needs
    is that all views share ONE consistent camera model — which they do here.
    """
    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elevation_deg)
    eye = np.array([radius * np.cos(e) * np.sin(a),
                    radius * np.sin(e),
                    radius * np.cos(e) * np.cos(a)])
    return look_at_c2w(eye)


def intrinsics(fov_deg, size):
    f = 0.5 * size / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([[f, 0.0, size / 2.0],
                     [0.0, f, size / 2.0],
                     [0.0, 0.0, 1.0]])


# OpenGL(c2w) -> OpenCV(c2w): flip the y and z camera axes.
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- #
# dataset writer — SC-GS / D-NeRF compatible
# --------------------------------------------------------------------------- #
def write_dataset(out, views, n_azimuths, elevations_deg,
                  radius, fov_deg, size, fps):
    """Pack the full az x elev VIEW GRID into an SC-GS-ready dataset.

    views: list (len V = n_azimuths x n_elevations) of dicts, each
        {"azimuth_deg": float, "elevation_deg": float,
         "frames": list (len T) of HxWx3 uint8 arrays}
    Every view carries its OWN (azimuth, elevation) pose; the c2w is computed
    from BOTH angles (orbital_pose), so elevation rotates the camera up/down
    instead of the whole ring sharing one elevation.
    """
    from PIL import Image

    V = len(views)
    T = min(len(v["frames"]) for v in views)
    times = [i / (T - 1) if T > 1 else 0.0 for i in range(T)]
    K = intrinsics(fov_deg, size)
    fov_x = np.deg2rad(fov_deg)

    img_root = os.path.join(out, "images")
    os.makedirs(img_root, exist_ok=True)

    dnerf_frames, cams = [], []
    for k, vw in enumerate(views):
        az, elev = vw["azimuth_deg"], vw["elevation_deg"]
        c2w_gl = orbital_pose(az, elev, radius)          # elevation enters here
        c2w_cv = c2w_gl @ _GL2CV
        w2c_cv = np.linalg.inv(c2w_cv)

        vdir = os.path.join(img_root, f"view_{k:02d}")
        os.makedirs(vdir, exist_ok=True)
        for t in range(T):
            fp = os.path.join(vdir, f"{t:05d}.png")
            Image.fromarray(vw["frames"][t]).save(fp)
            dnerf_frames.append({
                "file_path": f"./images/view_{k:02d}/{t:05d}",   # D-NeRF: no ext
                "time": times[t],
                "transform_matrix": c2w_gl.tolist(),             # OpenGL c2w
            })

        cams.append({
            "view": k,
            "azimuth_deg": az,
            "elevation_deg": elev,
            "img_size": [size, size],
            "K": K.tolist(),
            "c2w_opengl": c2w_gl.tolist(),
            "c2w_opencv": c2w_cv.tolist(),
            "w2c_opencv": w2c_cv.tolist(),
        })

    # (a) D-NeRF / blender transforms — the format SC-GS's dynamic loader eats.
    with open(os.path.join(out, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": float(fov_x), "frames": dnerf_frames},
                  f, indent=2)
    # SC-GS also opens transforms_test.json; reuse train so the loader is happy.
    with open(os.path.join(out, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": float(fov_x), "frames": dnerf_frames},
                  f, indent=2)
    # (b) explicit per-view intrinsics + extrinsics (both conventions).
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump({"fov_deg": fov_deg, "radius": radius,
                   "n_azimuths": n_azimuths,
                   "elevations_deg": list(elevations_deg),
                   "convention": "c2w_opengl is blender transform_matrix; "
                                 "w2c_opencv is [R|t] world->cam, cam looks +z",
                   "cameras": cams}, f, indent=2)
    # (c) shared per-frame timestamps.
    with open(os.path.join(out, "timestamps.json"), "w") as f:
        json.dump({"num_frames": T, "num_views": V, "fps": fps,
                   "times": times}, f, indent=2)

    n_elev = len(elevations_deg)
    print("\n[dataset] layout ------------------------------------------------")
    print(f"  {out}/")
    print(f"    images/view_00 .. view_{V-1:02d}/   ({T} frames each)")
    print(f"    transforms_train.json  (D-NeRF: {V*T} frame entries, per-frame time)")
    print(f"    transforms_test.json   (copy of train)")
    print(f"    cameras.json           (per-view K + c2w/w2c)")
    print(f"    timestamps.json        (T={T} times in [0,1], shared across views)")
    print(f"  views V={V} ({n_azimuths} az x {n_elev} elev)  frames T={T}  "
          f"size={size}x{size}  fov={fov_deg}deg  radius={radius}")
    print(f"  elevations(deg)={list(elevations_deg)}")
    print(f"  azimuths(deg)={sorted({round(vw['azimuth_deg'], 1) for vw in views})}")
    print("---------------------------------------------------------------")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single input image -> SVD monocular video -> SV4D")
    src.add_argument("--video", help="existing monocular video (mp4/gif) or frames folder; skip SVD")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--variant", default="sv4d2_8views", choices=list(SV4D2_VARIANTS),
                    help="sv4d2_8views (8 novel views, DEFAULT) or sv4d2 (4 novel views)")
    ap.add_argument("--n_frames", type=int, default=21,
                    help="SV4D2 output frames per view (native T extended auto-regressively)")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=576)
    ap.add_argument("--elevations", default="30,-30,15,-15,30,-30,15,-15",
                    help="PER-NOVEL-VIEW elevations (deg): 1 value (broadcast) or "
                         "n_novel values, one per azimuth. SV4D runs ONCE and renders "
                         "each novel view at its own elevation relative to the input "
                         "-> genuine 3D coverage (default: 8-view up/down spread)")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--remove_bg", action="store_true", help="rembg on plain-bg input")
    ap.add_argument("--low_vram", action="store_true", help="encoding_t=1 decoding_t=1")
    # camera-model knobs (rendering convention SV4D does not emit; kept consistent
    # across all views so the reconstructor sees one pinhole model).
    ap.add_argument("--radius", type=float, default=2.0, help="orbit radius (world units)")
    ap.add_argument("--fov_deg", type=float, default=33.8,
                    help="assumed pinhole FOV for emitted intrinsics (tunable; "
                         "SV4D's orbital renders are ~30-40deg — verify vs your SC-GS scale)")
    ap.add_argument("--fps", type=float, default=10.0, help="timestamp fps metadata")
    args = ap.parse_args()

    azimuths = novel_azimuths(args.variant)     # novel-view azimuths (input az excluded)
    n_az = len(azimuths)
    # PER-VIEW novel elevations (length n_az). SV4D encodes elevation relative to
    # the input view, so a per-view list renders genuine per-view elevations in ONE
    # run — real 3D coverage. (A single scalar per run collapses all relative
    # elevations to 0, producing identical images falsely placed at diff elevations.)
    novel_elevs = [float(e.strip()) for e in args.elevations.split(",") if e.strip() != ""]
    if len(novel_elevs) == 1:
        novel_elevs = novel_elevs * n_az
    if len(novel_elevs) != n_az:
        ap.error(f"--elevations must be 1 or {n_az} values (got {len(novel_elevs)})")
    input_elev = 0.0                            # v0 (input view) assumed elevation
    print(f"[gen_mv_video] variant={args.variant}  V={n_az} novel views (ONE SV4D run)  "
          f"azimuths(deg)={azimuths}  novel_elevs(deg)={novel_elevs}")

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    # (1) obtain the monocular driving video (frames folder that SV4D accepts).
    # The SVD bootstrap is elevation-agnostic (it just animates the still), so it
    # is done ONCE and reused as the input to every per-elevation SV4D run.
    if args.image:
        mono_dir = os.path.join(work, "mono_frames")
        image_to_mono_frames(args.image, mono_dir, args.n_frames,
                             args.img_size, args.seed)
        sv4d_input = mono_dir
    else:
        sv4d_input = args.video

    # (2)+(3) run SV4D 2.0 ONCE PER ELEVATION and merge into one az x elev grid.
    # Each elevation writes to a DISTINCT subfolder so the identically-named
    # per-view mp4s ({model_name}/{base_count}_v{view}.mp4) never collide.
    sv4d_out = os.path.join(work, "sv4d2_out")
    view_mp4s = run_sv4d2(sv4d_input, sv4d_out, args.variant, args.n_frames,
                          args.num_steps, args.img_size,
                          [input_elev] + novel_elevs,        # v0 + per-view novel
                          args.seed, args.remove_bg, args.low_vram)
    # sorted _v001.._v00V <-> (azimuths[i], novel_elevs[i]).
    if len(view_mp4s) != n_az:
        print(f"[warn] got {len(view_mp4s)} views, expected {n_az} "
              f"(azimuths={azimuths}). Pairing by sorted order.")
    views = []
    for a, e, p in zip(azimuths, novel_elevs, view_mp4s):
        views.append({"azimuth_deg": a, "elevation_deg": e, "frames": decode_mp4(p)})

    # (3) pack into a posed, timestamped dataset (each view keeps its own az+elev).
    write_dataset(args.out, views, n_az, novel_elevs,
                  args.radius, args.fov_deg, args.img_size, args.fps)

    print(f"\n[gen_mv_video] done -> {args.out}  "
          f"(V={len(views)} = {n_az} az x {n_elev} elev)")


if __name__ == "__main__":
    main()
